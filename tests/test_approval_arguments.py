"""The approval request has to show the whole instruction.

`coder__ask` carries the entire request in two long string arguments, and the
old one-line preview — six arguments, sixty characters each — cut the task off
mid-sentence. Approving what you cannot read is not approving it, so these
tests are about nothing being dropped on the way to the message.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agentcore.ui.telegram import MAX_MESSAGE, TelegramUI, _format_arguments


class FakeBot:
    def __init__(self):
        self.messages: list[dict] = []

    async def send_message(self, chat_id, text, parse_mode=None, reply_markup=None):
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
        )


def _ask(arguments, *, tool="coder__ask", reason="причина"):
    bot = FakeBot()
    keyboard = object()
    asyncio.run(
        # No attribute of the instance is touched, so there is nothing to build.
        TelegramUI._ask_approval(
            None,
            SimpleNamespace(bot=bot),
            77,
            tool=tool,
            arguments=arguments,
            reason=reason,
            keyboard=keyboard,
        )
    )
    return bot, keyboard


# -- formatting --------------------------------------------------------------


def test_a_long_value_is_not_truncated():
    task = "Внеси небольшое изменение в default-ветку main пяти репозиториев. " * 20
    out = _format_arguments({"task": task})
    assert task in out
    assert "…" not in out


def test_every_argument_is_shown_not_the_first_six():
    arguments = {f"arg{i}": f"value{i}" for i in range(9)}
    out = _format_arguments(arguments)
    for i in range(9):
        assert f"value{i}" in out


def test_non_string_values_are_shown_as_json():
    out = _format_arguments({"files": ["a.py", "b.py"], "count": 2})
    assert json.dumps(["a.py", "b.py"], ensure_ascii=False, indent=2) in out
    assert "count:\n2" in out


def test_no_arguments_reads_as_a_dash():
    assert _format_arguments({}) == "—"


# -- delivery ----------------------------------------------------------------


def test_a_short_request_is_one_message_carrying_the_buttons():
    bot, keyboard = _ask({"task": "почини кнопку"})
    assert len(bot.messages) == 1
    only = bot.messages[0]
    assert "Подтверди действие" in only["text"]
    assert "coder__ask" in only["text"]
    assert "почини кнопку" in only["text"]
    assert "причина" in only["text"]
    assert only["reply_markup"] is keyboard


def test_a_long_request_is_split_and_arrives_whole():
    # Distinct words so a chunk boundary cannot hide a dropped piece.
    task = "\n".join(f"строка-{i} задачи для кодера" for i in range(600))
    bot, _ = _ask({"task": task})

    assert len(bot.messages) > 1
    delivered = "".join(m["text"] for m in bot.messages)
    for i in (0, 42, 599):
        assert f"строка-{i} задачи" in delivered


def test_only_the_last_message_can_be_approved():
    """Buttons on a chunk still being read would approve an unread task."""
    task = "\n".join(f"строка-{i}" for i in range(600))
    bot, keyboard = _ask({"task": task})

    assert [m["reply_markup"] for m in bot.messages[:-1]] == [None] * (len(bot.messages) - 1)
    assert bot.messages[-1]["reply_markup"] is keyboard


def test_the_header_appears_once_and_the_reason_lands_at_the_end():
    task = "\n".join(f"строка-{i}" for i in range(600))
    bot, _ = _ask({"task": task}, reason="инструмент не относится к известным")

    headers = [m for m in bot.messages if "Подтверди действие" in m["text"]]
    assert len(headers) == 1
    assert headers[0] is bot.messages[0]
    assert "инструмент не относится к известным" in bot.messages[-1]["text"]
    assert "инструмент не относится" not in "".join(m["text"] for m in bot.messages[:-1])


def test_every_chunk_stays_inside_the_telegram_limit():
    task = "\n".join(f"строка-{i} с довольно длинным хвостом текста" for i in range(900))
    bot, _ = _ask({"task": task})
    assert all(len(m["text"]) <= MAX_MESSAGE for m in bot.messages)
