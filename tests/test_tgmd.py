"""Markdown to Telegram HTML.

The cases here are the ones that actually break converters, grouped by how they
break. Two invariants are asserted on the output of every single test, because
both failures are silent in different ways: an unsupported tag makes the API
reject the whole message, and an unbalanced tag does the same while looking fine
in a diff.
"""

from __future__ import annotations

import html
import re

from agentcore.ui.tgmd import MAX_MESSAGE, _chunk_words, render, utf16_len

# Nothing outside this set may ever be emitted: Telegram answers 400
# `Unsupported start tag "ul"` and the reader gets nothing at all.
ALLOWED = {"b", "i", "u", "s", "a", "code", "pre", "blockquote", "tg-spoiler"}

FORBIDDEN = ("<h1", "<h2", "<h3", "<h4", "<ul", "<ol", "<li", "<table", "<tr", "<td", "<hr", "<br", "<p>")

TAG = re.compile(r"<(/?)([a-zA-Z0-9-]+)(?:\s[^>]*)?>")


def check(messages: list[str]) -> list[str]:
    """Assert the two invariants on every message, then hand them back."""
    for msg in messages:
        assert utf16_len(msg) <= MAX_MESSAGE, f"over the limit: {utf16_len(msg)}"

        stack = []
        for closing, name in TAG.findall(msg):
            assert name in ALLOWED, f"unsupported tag <{name}> in: {msg[:120]}"
            if closing:
                assert stack, f"unmatched </{name}> in: {msg[:120]}"
                assert stack.pop() == name, f"crossed tags near </{name}>"
            else:
                stack.append(name)
        assert not stack, f"unclosed {stack} in: {msg[:120]}"

        for bad in FORBIDDEN:
            assert bad not in msg, f"{bad} survived into: {msg[:120]}"
    return messages


def one(text: str) -> str:
    msgs = check(render(text))
    assert len(msgs) == 1, f"expected one message, got {len(msgs)}"
    return msgs[0]


# -- what Telegram does not have --------------------------------------------


def test_headings_become_bold():
    for hashes in ("#", "##", "###", "####"):
        out = one(f"{hashes} Основные действия")
        assert out == "<b>Основные действия</b>", out


def test_bullet_list_becomes_bullets():
    out = one("- запускает службы\n- управляет зависимостями")
    assert "•" in out
    assert "запускает службы" in out


def test_nested_list_is_indented_not_tagged():
    out = one("- outer\n  - inner\n    - deepest")
    assert out.count("•") == 3
    assert "  •" in out


def test_ordered_list_keeps_its_numbers():
    out = one("3. third\n4. fourth")
    assert "3. third" in out
    assert "4. fourth" in out


def test_horizontal_rule_survives_as_text():
    out = one("above\n\n---\n\nbelow")
    assert "─" in out


def test_table_becomes_a_monospace_grid():
    out = one("| repo | tag |\n|------|-----|\n| lead | 0.0.41 |")
    assert "<pre>" in out
    # The delimiter row must not arrive as pipes-and-colons junk.
    assert ":---" not in out
    assert "lead" in out and "0.0.41" in out


def test_blockquote_does_not_emit_a_tag_per_line():
    out = one("> first line\n> second line")
    assert out.count("<blockquote>") <= 1


# -- escaping ----------------------------------------------------------------


def test_literal_angle_bracket_in_prose_is_escaped():
    out = one("Ошибка: node <none> недоступен, и 1 < 2")
    assert "&lt;none&gt;" in out
    assert "&lt; 2" in out


def test_literal_ampersand_in_prose_is_escaped():
    out = one("Запусти make build && make test")
    assert "&amp;&amp;" in out


def test_code_block_of_html_is_escaped_not_rendered():
    out = one('```html\n<div class="card"><b>привет</b></div>\n```')
    assert "&lt;div" in out
    # <b> inside a fence must NOT become real bold — that is the silent one.
    assert "<b>привет</b>" not in out
    assert "&lt;b&gt;привет&lt;/b&gt;" in out


def test_inline_code_with_entity_stays_visible():
    out = one("Сравни `&amp;` с `&`")
    assert "&amp;amp;" in out


def test_link_url_and_text_take_different_escaping():
    out = one('Смотри [логи & трейсы](https://example.com/a?x=1&y="2")')
    # The ampersand is an entity in the attribute, and the quote cannot break out
    # of it — markdown-it percent-encodes it during URL normalisation.
    assert "x=1&amp;y=" in out
    assert '"2"' not in out
    # Anchor text is a text node, so it takes the text treatment instead.
    assert "логи &amp; трейсы" in out


def test_a_tag_typed_as_prose_is_not_markup():
    out = one("Тег <b> делает текст жирным, а </div> закрывает блок")
    assert "&lt;b&gt;" in out
    assert "&lt;/div&gt;" in out


# -- emphasis that is not emphasis ------------------------------------------


def test_screaming_snake_identifier_survives():
    out = one("Подними MAX_TOKENS_PER_TURN до 150000")
    assert "MAX_TOKENS_PER_TURN" in out
    assert "<i>" not in out


def test_snake_case_and_dunder_survive():
    out = one("Файлы __init__.py и snake_case_name() не трогай")
    assert "__init__.py" in out
    assert "snake_case_name()" in out
    assert "<b>" not in out and "<i>" not in out


def test_globs_survive():
    out = one("Игнорируй *.py и *.pyc, но оставь src/**/*.yaml")
    assert "*.py" in out
    assert "*.pyc" in out


def test_multiplication_operator_survives():
    out = one("Стоит 2 * 3 * 4 токена")
    assert "2 * 3 * 4" in out


def test_real_emphasis_still_works():
    out = one("Это **systemd**, а это *курсив*")
    assert "<b>systemd</b>" in out
    assert "<i>курсив</i>" in out


# -- structure ---------------------------------------------------------------


def test_fence_carries_its_language():
    out = one("```bash\nsystemctl status nginx\n```")
    assert '<pre><code class="language-bash">' in out
    assert "systemctl status nginx" in out


def test_fence_without_language_still_renders():
    out = one("```\nplain\n```")
    assert "<pre><code>" in out


def test_unclosed_fence_still_produces_valid_html():
    # A truncated stream must not leave an unbalanced document.
    check(render("Вот команда:\n\n```bash\nsystemctl restart nginx"))


def test_softbreaks_keep_the_stats_footer_on_three_lines():
    footer = (
        "📊 AI stats\n"
        "model=gpt-5.6-luna · steps=1 · stop=completed\n"
        "input=15,272 · output=377 · cached=0 (0%)"
    )
    out = one(footer)
    assert out.count("\n") == 2, repr(out)
    assert "AI stats" in out and "output=377" in out


def test_empty_input_sends_nothing():
    assert render("") == []
    assert render("   \n  ") == []


# -- splitting ---------------------------------------------------------------


def test_long_fence_splits_into_whole_fences():
    body = "\n".join(f"line {i} of a very long log entry" * 2 for i in range(400))
    msgs = check(render(f"```log\n{body}\n```"))
    assert len(msgs) > 1
    for m in msgs:
        assert m.startswith('<pre><code class="language-log">')
        assert m.endswith("</code></pre>")


def test_long_bold_run_splits_without_breaking_tags():
    text = "**" + " ".join(f"слово{i}" for i in range(2000)) + "**"
    msgs = check(render(text))
    assert len(msgs) > 1


def test_emoji_are_counted_as_telegram_counts_them():
    # 2,100 characters, 4,200 UTF-16 units: under len() but over the limit.
    msgs = check(render("🙂" * 2100))
    assert len(msgs) > 1, "emoji payload was not split"


def test_a_single_unbreakable_run_is_still_delivered():
    msgs = check(render("x" * 9000))
    assert len(msgs) >= 3


def test_line_longer_than_the_limit_with_no_newline():
    msgs = check(render("JSON: " + '{"k":"v"},' * 800))
    assert len(msgs) >= 2


def test_word_chunking_is_lossless():
    """The first version dropped the space it split on, once per split.

    A missing space in prose is a typo; in a JSON array of tool names it is a
    corrupted payload, and nothing about the delivered message looks wrong.
    """
    text = " ".join(f"слово{i}" for i in range(2000))
    assert "".join(_chunk_words(text)) == text
    assert "".join(_chunk_words("a b c", size=2)) == "a b c"
    assert "".join(_chunk_words("одно-длинное-слово-без-пробелов", size=5)) == (
        "одно-длинное-слово-без-пробелов"
    )


def test_a_split_paragraph_reassembles_exactly():
    text = " ".join(f"токен{i}" for i in range(3000))
    msgs = check(render(text))
    assert len(msgs) > 1
    rebuilt = "".join(html.unescape(m) for m in msgs)
    assert rebuilt == text


# -- realistic answers -------------------------------------------------------


SYSTEMCTL = """`systemctl` — команда для управления службами и состоянием **systemd** в Linux.

### Основные действия

```bash
systemctl status nginx
```

Показать состояние службы.

```bash
sudo systemctl restart nginx
```

Перезапустить службу.

### Что такое systemd

`systemd` — системный менеджер Linux, который:

- запускает службы при старте ОС;
- управляет их зависимостями;
- собирает логи через `journald`.

📊 AI stats
model=gpt-5.6-luna · steps=1 · stop=completed
"""


def test_the_answer_from_the_bug_report():
    msgs = check(render(SYSTEMCTL))
    joined = "\n".join(msgs)
    # None of the Markdown source may survive as characters.
    assert "###" not in joined
    assert "**" not in joined
    assert "```" not in joined
    # And the content must all be there, formatted.
    assert "<b>systemd</b>" in joined
    assert "<b>Основные действия</b>" in joined
    assert '<pre><code class="language-bash">' in joined
    assert "<code>journald</code>" in joined
    assert "•" in joined
    assert "AI stats" in joined


INCIDENT = """Разбор инцидента.

Под упал с ошибкой:

```
Error: connect ECONNREFUSED <none>:5432 && retry failed
```

Проверь [дашборд](https://grafana.1ms.my/d/abc?from=now-1h&to=now) и поле `pg_hba.conf`.

| компонент | статус |
|-----------|--------|
| reader | ok |
| writer | fail |
"""


def test_incident_report_with_brackets_url_and_table():
    msgs = check(render(INCIDENT))
    joined = "\n".join(msgs)
    assert "&lt;none&gt;:5432" in joined
    assert "&amp;&amp; retry" in joined
    assert "from=now-1h&amp;to=now" in joined
    assert "<code>pg_hba.conf</code>" in joined
    assert "<pre>" in joined and "writer" in joined
