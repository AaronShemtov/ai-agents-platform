"""Prepare the narrow source patch in CI, where full files are available."""

import ast
import hashlib
from pathlib import Path


def read_checked(path, expected):
    data = Path(path).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Unexpected source revision: {path}: {actual}")
    return data.decode()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one patch anchor: {old!r}")
    return text.replace(old, new, 1)


path = "src/agentcore/loop.py"
text = read_checked(path, "9c8bf0a541c054b671d28efadfd786b572ad817b")
text = replace_once(
    text,
    "    tool_duration_ms: int = 0\n",
    "    tool_duration_ms: int = 0\n    tool_names: list[str] = field(default_factory=list)\n",
)
text = replace_once(
    text,
    "        tool_calls = 0\n        tool_duration_ms = 0\n",
    "        tool_calls = 0\n        tool_duration_ms = 0\n        tool_names: list[str] = []\n",
)
text = replace_once(
    text,
    "                if called:\n                    tool_calls += 1\n                    tool_duration_ms += duration_ms\n",
    "                if called:\n                    tool_calls += 1\n                    tool_duration_ms += duration_ms\n"
    "                    tool_names.append(catalog.resolve(call.name) or call.name)\n",
)
# Locate the two actual call sites, not audit/metrics calls with similar keywords.
lines = text.splitlines(keepends=True)
insertions = []
for name in ("format_usage_footer", "LoopResult"):
    calls = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == name]
    if len(calls) != 1:
        raise RuntimeError(f"Expected exactly one {name} call")
    keywords = [k for k in calls[0].keywords if k.arg == "tool_duration_ms"]
    if len(keywords) != 1:
        raise RuntimeError(f"Missing tool duration keyword in {name}")
    keyword = keywords[0]
    line = lines[keyword.end_lineno - 1]
    indent = line[:len(line) - len(line.lstrip())]
    insertions.append((keyword.end_lineno, indent + "tool_names=tool_names,\n"))
for index, addition in sorted(insertions, reverse=True):
    lines.insert(index, addition)
text = "".join(lines)
ast.parse(text)
Path(path).write_text(text)

for path in ("src/agentcore/usage_display.py", "src/agentcore/ui/usage.py"):
    text = read_checked(path, "948679f42adb2192f3958dea861c5ecff469202a")
    text = replace_once(text, "from __future__ import annotations\n", "from __future__ import annotations\n\nimport json\nfrom collections.abc import Sequence\n")
    text = replace_once(text, '    stopped_because: str = "completed",\n', '    stopped_because: str = "completed",\n    tool_names: Sequence[str] = (),\n')
    text = replace_once(text, '        f"duration={_duration(duration_ms)} · tools={tool_calls} ({_duration(tool_duration_ms)})"\n', '        f"duration={_duration(duration_ms)} · tools={tool_calls} ({_duration(tool_duration_ms)})"\n        f" {json.dumps(list(tool_names), ensure_ascii=False)}"\n')
    ast.parse(text)
    Path(path).write_text(text)

# Inspection only: keep Telegram delivery unchanged, make its relevant code visible.
path = Path("src/agentcore/ui/telegram.py")
text = path.read_text()
lines = text.splitlines()
print("TELEGRAM DELIVERY SOURCE (unchanged):")
for node in ast.walk(ast.parse(text)):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("_send_long", "_split"):
        print("\n".join(lines[node.lineno - 1:node.end_lineno]))
