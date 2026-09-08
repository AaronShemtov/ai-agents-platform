"""Markdown as the models write it, turned into HTML as Telegram accepts it.

The models answer in Markdown — headings, fences, `**bold**`, bullet lists — and
the sending code used to `html.escape()` that and post it with parse_mode=HTML.
Escaping is not translating: `**` and `###` are not HTML tags, so Telegram found
no markup to render and printed the source characters. Formatting had been off
the whole time in the loudest possible way.

Three properties of the Bot API decide the shape of everything below.

**The tag set is tiny and closed.** b/strong, i/em, u/ins, s/strike/del, a, code,
pre, blockquote, tg-spoiler, tg-emoji — and nothing else. There is no h1-h6, no
ul/li, no table, no hr, no br, no p. Emitting one is not a cosmetic slip: the API
answers 400 `Unsupported start tag "ul"` and the message is never delivered at
all. So headings become bold, list items become bullets, rules become a line of
box-drawing characters, and tables become a monospace grid.

**Escaping happens once, at the leaf, per context.** Escaping the source first
destroys the Markdown; escaping the output last turns our own tags into visible
`&lt;b&gt;`. Text nodes take `& < >`; attribute values take those plus `"`; code
content takes the text treatment and is never parsed as Markdown.

**Length is counted in UTF-16 code units, not characters.** An emoji is two. A
reply of 2,100 characters made of emoji is over the 4,096 limit while `len()`
says it is comfortably short.

The last one is why splitting works the way it does here. Every leaf is emitted
already wrapped in the tags that are open around it, so each segment is valid
HTML on its own and any run of segments can be packed into a message without
cutting a tag or an entity in half. `<b>a</b><b>b</b>` renders exactly like
`<b>ab</b>`, which makes that a free property rather than a compromise.

Parsing is markdown-it-py rather than regexes, and that is the point of the
dependency. Emphasis in CommonMark is decided by left/right-flanking delimiter
runs, which is precisely what keeps `MAX_TOKENS_PER_TURN`, `snake_case`, `*.py`
and `2 * 3` intact. Every regex-based attempt fails on those, and it fails
silently — the reader gets an identifier with characters missing and no way to
know. `html=False` is deliberate too: raw HTML in the source is text, so a model
explaining `<div>` gets an escaped `<div>` rather than a 400.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from markdown_it import MarkdownIt

# Telegram's own ceiling is 4096 UTF-16 code units. The margin absorbs the tag
# reopening a split adds, so a packed message cannot land just over the line.
# Deliberately the same number telegram.py has always used, so there is one
# budget in the codebase rather than two that can drift apart.
MAX_MESSAGE = 3800

# Telegram has no <hr>. Box drawing is the closest thing that survives as text.
RULE = "──────────"

# Tags we are allowed to emit. Kept as a set so the renderer can assert against
# it — a typo here would otherwise ship as an undeliverable message.
TELEGRAM_TAGS = frozenset(
    {"b", "i", "u", "s", "a", "code", "pre", "blockquote", "tg-spoiler"}
)


_ATOM = re.compile(r"&[A-Za-z]+;|&#\d+;|.", re.S)


def _literal_delimiters(state, silent: bool) -> bool:
    """Take delimiters that are not emphasis out of the parser's hands.

    A correct CommonMark parser is what keeps `MAX_TOKENS_PER_TURN` and `2 * 3`
    intact — but it is not enough, because CommonMark genuinely does read
    emphasis in two constructs this chat is full of. `*.py и *.pyc` opens on the
    asterisk before `.py` and closes on the one before `.yaml`, deleting both and
    italicising everything between. `__init__.py` is bold `init.py`. Both are
    correct Markdown and both destroy characters the reader needs.

    So two narrow rules, applied before the emphasis rule sees the run:

    * A single `*` is literal when neither neighbour is alphanumeric. That is
      exactly the glob, the multiplication operator and the bare bullet, while
      `*курсив*` still opens (next char is a letter) and still closes (previous
      char is a letter).
    * `_` never marks emphasis at all. Underscore emphasis is vanishingly rare in
      this kind of conversation and identifiers with underscores are everywhere,
      so the trade is not close.

    Showing a literal asterisk is a cosmetic loss; deleting one from a path is a
    wrong command. That asymmetry is the whole argument.
    """
    src, pos = state.src, state.pos
    ch = src[pos]
    if ch not in "*_":
        return False

    run = 0
    while pos + run < len(src) and src[pos + run] == ch:
        run += 1

    if ch == "*":
        if run != 1:
            return False  # ** is bold; let the emphasis rule have it
        prev_ch = src[pos - 1] if pos > 0 else ""
        next_ch = src[pos + 1] if pos + 1 < len(src) else ""
        if prev_ch.isalnum() or next_ch.isalnum():
            return False

    if not silent:
        state.pending += ch * run
    state.pos += run
    return True


def utf16_len(text: str) -> int:
    """Length as Telegram counts it.

    Python counts an emoji as one character; Telegram counts the surrogate pair
    as two. Measuring in code points is how a short-looking reply gets rejected
    for being too long.
    """
    return len(text.encode("utf-16-le")) // 2


def _esc(text: str) -> str:
    """Text-node escaping: & < > and nothing else."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(value: str) -> str:
    """Attribute escaping, which is not the same job as text escaping."""
    return _esc(value).replace('"', "&quot;")


def _chunk_words(text: str, size: int = 600) -> list[str]:
    """Break text into pieces of at most `size` characters, on spaces.

    `"".join(_chunk_words(t)) == t` for every input, and that is the whole point:
    an earlier version split on spaces and dropped the separator, so every split
    silently ate one space. In a JSON array of tool names that is a corrupted
    payload rather than a typo, and nothing about the message looks wrong.

    Only long text is affected; ordinary prose comes back as one piece. A word
    longer than `size` is emitted whole rather than cut, because a token without
    spaces is usually a URL or a hash and halving it makes it useless.
    """
    if not text:
        return []
    if len(text) <= size:
        return [text]

    words = text.split(" ")
    last = len(words) - 1
    pieces: list[str] = []
    current = ""
    for index, word in enumerate(words):
        # The separator travels with the word it followed, so concatenation
        # rebuilds the original exactly.
        token = word if index == last else word + " "
        if current and len(current) + len(token) > size:
            pieces.append(current)
            current = token
        else:
            current += token
    if current:
        pieces.append(current)
    return pieces


@dataclass
class Block:
    """One top-level piece of the answer, already rendered to Telegram HTML.

    `segments` are atomic: each one carries its own opening and closing tags, so
    a message can be cut between any two of them and both halves stay valid.
    `prefix`/`suffix` wrap the whole block and are repeated when the block has to
    be split — that is how a long code fence stays a code fence on both sides of
    the seam.
    """

    segments: list[str] = field(default_factory=list)
    prefix: str = ""
    suffix: str = ""
    joiner: str = ""

    def html(self) -> str:
        return self.prefix + self.joiner.join(self.segments) + self.suffix


class _Renderer:
    """Walks markdown-it's token stream and emits Blocks."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        # Open inline tags, innermost last. Every leaf is wrapped in these.
        self._open: list[str] = []
        # List nesting: one entry per open list, each ("bullet"|"ordered", counter).
        self._lists: list[list] = []
        self._quote_depth = 0

    # -- helpers ------------------------------------------------------------

    def _wrap(self, inner: str) -> str:
        """Wrap a leaf in the currently open tags, making it self-contained."""
        for tag in reversed(self._open):
            name = tag.split(" ", 1)[0]
            inner = f"<{tag}>{inner}</{name}>"
        return inner

    def _text_block(self, segments: list[str]) -> None:
        if any(s.strip() for s in segments):
            self.blocks.append(Block(segments=list(segments)))

    # -- inline -------------------------------------------------------------

    def _inline(self, token) -> list[str]:
        segments: list[str] = []
        for child in token.children or []:
            t = child.type
            if t == "text":
                # Split long prose into word-sized leaves. Each leaf carries its
                # own tags, so keeping them small is what lets the splitter cut
                # between segments instead of inside a `<b>` that spans 4,000
                # characters.
                for piece in _chunk_words(child.content):
                    segments.append(self._wrap(_esc(piece)))
            elif t == "code_inline":
                # Content is literal: escaped, never parsed as Markdown. This is
                # what keeps `MAX_TOKENS_PER_TURN` and `2 * 3` in a code span
                # exactly as written.
                segments.append(self._wrap(f"<code>{_esc(child.content)}</code>"))
            elif t in ("softbreak", "hardbreak"):
                # A softbreak is a real newline for us. Rendering it as a space
                # is what glues the three-line "AI stats" footer into one line.
                segments.append("\n")
            elif t == "strong_open":
                self._open.append("b")
            elif t == "em_open":
                self._open.append("i")
            elif t == "s_open":
                self._open.append("s")
            elif t in ("strong_close", "em_close", "s_close"):
                if self._open:
                    self._open.pop()
            elif t == "link_open":
                href = child.attrGet("href") or ""
                self._open.append(f'a href="{_esc_attr(href)}"')
            elif t == "link_close":
                if self._open:
                    self._open.pop()
            elif t == "image":
                # Telegram cannot inline an image from HTML. The alt text plus the
                # URL is strictly more useful than a dropped node.
                alt = child.content or "image"
                src = child.attrGet("src") or ""
                segments.append(
                    self._wrap(f'<a href="{_esc_attr(src)}">{_esc(alt)}</a>')
                )
            elif t in ("html_inline", "html_block"):
                # html=False means these should not appear; if they ever do, the
                # only safe reading is "the model typed a tag as prose".
                segments.append(self._wrap(_esc(child.content)))
            else:
                if child.content:
                    segments.append(self._wrap(_esc(child.content)))
        return segments

    # -- blocks -------------------------------------------------------------

    def _list_prefix(self) -> str:
        depth = len(self._lists) - 1
        indent = "  " * depth
        kind, counter = self._lists[-1]
        if kind == "ordered":
            self._lists[-1][1] = counter + 1
            return f"{indent}{counter}. "
        return f"{indent}• "

    def run(self, tokens) -> list[Block]:
        pending: list[str] | None = None
        heading = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            t = tok.type

            if t == "inline":
                seg = self._inline(tok)
                if heading:
                    # No headings in Telegram. Bold is the whole of the budget.
                    seg = [f"<b>{s}</b>" if s.strip() else s for s in seg]
                if pending is None:
                    pending = seg
                else:
                    pending.extend(seg)

            elif t == "heading_open":
                heading = True
                pending = []
            elif t == "heading_close":
                heading = False
                self._text_block(pending or [])
                pending = None

            elif t == "paragraph_open":
                # Inside a list item the bullet has already been queued by
                # list_item_open. Resetting here is what silently ate every
                # bullet and turned a list into loose paragraphs.
                if not (self._lists and pending):
                    pending = []
            elif t == "paragraph_close":
                if self._lists and pending is not None:
                    # A list item's paragraph is the item's text; the bullet was
                    # already queued by list_item_open.
                    self._text_block(pending)
                else:
                    self._text_block(pending or [])
                pending = None

            elif t == "fence" or t == "code_block":
                info = (tok.info or "").strip().split(" ")[0]
                lang = f' class="language-{_esc_attr(info)}"' if info else ""
                body = tok.content
                if body.endswith("\n"):
                    body = body[:-1]
                self.blocks.append(
                    Block(
                        prefix=f"<pre><code{lang}>",
                        suffix="</code></pre>",
                        segments=[_esc(line) for line in body.split("\n")],
                        joiner="\n",
                    )
                )

            elif t == "bullet_list_open":
                self._lists.append(["bullet", 1])
            elif t == "ordered_list_open":
                start = tok.attrGet("start")
                self._lists.append(["ordered", int(start) if start else 1])
            elif t in ("bullet_list_close", "ordered_list_close"):
                if self._lists:
                    self._lists.pop()

            elif t == "list_item_open":
                pending = [self._list_prefix()]
            elif t == "list_item_close":
                if pending:
                    self._text_block(pending)
                pending = None

            elif t == "blockquote_open":
                self._quote_depth += 1
            elif t == "blockquote_close":
                self._quote_depth -= 1

            elif t == "hr":
                self.blocks.append(Block(segments=[RULE]))

            elif t == "table_open":
                consumed, block = self._table(tokens, i)
                self.blocks.append(block)
                i += consumed
                continue

            i += 1

        if pending:
            self._text_block(pending)
        return self.blocks

    def _table(self, tokens, start: int) -> tuple[int, Block]:
        """Tables become an aligned monospace grid.

        Telegram has no <table>, and passing the pipes through leaves the
        `|---|:--:|` delimiter row visible as junk while the columns do not line
        up in a proportional font.
        """
        rows: list[list[str]] = []
        row: list[str] = []
        i = start
        depth = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "table_open":
                depth += 1
            elif tok.type == "table_close":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif tok.type == "tr_open":
                row = []
            elif tok.type == "tr_close":
                rows.append(row)
            elif tok.type == "inline":
                # Cell text only — inline markup inside a monospace grid would
                # have to survive the padding, and plain text is the honest
                # rendering of a cell.
                row.append("".join(c.content for c in (tok.children or [])))
            i += 1

        widths: list[int] = []
        for r in rows:
            for col, cell in enumerate(r):
                if col >= len(widths):
                    widths.append(0)
                widths[col] = max(widths[col], len(cell))

        lines = []
        for n, r in enumerate(rows):
            lines.append(
                "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(r)).rstrip()
            )
            if n == 0:
                lines.append("  ".join("-" * w for w in widths).rstrip())

        return i - start, Block(
            prefix="<pre>",
            suffix="</pre>",
            segments=[_esc(line) for line in lines],
            joiner="\n",
        )


def _split_block(block: Block, limit: int) -> list[str]:
    """Render one block into as many messages as it needs.

    prefix/suffix are repeated on every piece, so a code fence stays a fence and
    a quote stays a quote across the seam. A single segment longer than the limit
    is cut by characters as a last resort — that only happens for one
    unbroken run of text with no whitespace, and losing the formatting there
    beats losing the message.
    """
    out: list[str] = []
    overhead = utf16_len(block.prefix) + utf16_len(block.suffix)
    room = limit - overhead
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            out.append(block.prefix + block.joiner.join(current) + block.suffix)
            current = []
            size = 0

    for seg in block.segments:
        seg_len = utf16_len(seg) + utf16_len(block.joiner)
        if seg_len > room:
            flush()
            # Hard cut. Atoms rather than characters, so a surrogate pair is
            # never halved and an escaped entity is never cut into `&am` — which
            # Telegram either renders as garbage or rejects outright.
            piece = ""
            for atom in _ATOM.findall(seg):
                if utf16_len(piece) + utf16_len(atom) > room:
                    out.append(block.prefix + piece + block.suffix)
                    piece = ""
                piece += atom
            if piece:
                current = [piece]
                size = utf16_len(piece)
            continue
        if size + seg_len > room:
            flush()
        current.append(seg)
        size += seg_len

    flush()
    return out


def render(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Markdown in, ready-to-send Telegram HTML out, one string per message.

    Callers send each element with parse_mode=HTML and escape nothing further —
    escaping already happened, once, at the leaves.
    """
    if not text or not text.strip():
        return []

    md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
    # Before 'emphasis', so the run never reaches it — see _literal_delimiters.
    md.inline.ruler.before("emphasis", "literal_delimiters", _literal_delimiters)
    if "s_open" not in md.get_all_rules().get("inline", []):
        md.enable("strikethrough", ignoreInvalid=True)

    blocks = _Renderer().run(md.parse(text))

    messages: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        piece = block.html()
        piece_len = utf16_len(piece)
        if piece_len > limit:
            if current:
                messages.append("\n\n".join(current))
                current, size = [], 0
            messages.extend(_split_block(block, limit))
            continue
        if size + piece_len + 2 > limit and current:
            messages.append("\n\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += piece_len + 2

    if current:
        messages.append("\n\n".join(current))
    return messages
