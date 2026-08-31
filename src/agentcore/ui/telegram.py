"""Telegram front door.

Telegram is the UI, not a tool: the agent never "calls Telegram", it talks *through*
it. So this stays a plain python-telegram-bot application rather than another MCP
server.

Two things here are load-bearing rather than stylistic:

  * `concurrent_updates(True)` — without it the Application handles updates one at a
    time, so /cancel would queue up behind the very turn it is meant to cancel, and
    an approval button press could not be delivered while the loop awaits it.
  * edit throttling — Telegram rate-limits message edits to roughly one per second per
    chat. An unthrottled progress indicator gets the bot flood-limited within a few
    tool calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import time
import uuid
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agentcore.audit import AuditLog
from agentcore.config import Settings
from agentcore.loop import AgentLoop
from agentcore.memory import MemoryStore
from agentcore.profiles import AgentProfile

logger = logging.getLogger(__name__)

# Telegram's hard limit is 4096 characters; leave room for the HTML wrapper.
MAX_MESSAGE = 3800
EDIT_INTERVAL = 1.2  # seconds between progress edits
APPROVAL_TIMEOUT = 600.0  # seconds to wait for a button press


class ProgressReporter:
    """Single status message, edited as the turn proceeds."""

    def __init__(self, bot: Any, chat_id: int, message_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id
        self._lines: list[str] = []
        self._last_edit = 0.0

    async def __call__(self, event: str) -> None:
        self._lines.append(event)
        # Keep the message short: the tail is what the user cares about.
        if len(self._lines) > 8:
            self._lines = ["…", *self._lines[-7:]]
        await self._flush()

    async def _flush(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_edit < EDIT_INTERVAL:
            return
        self._last_edit = now
        body = "\n".join(html.escape(line) for line in self._lines)
        with contextlib.suppress(BadRequest):
            # BadRequest is expected when the text has not changed since the last edit.
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=body or "…",
                parse_mode=ParseMode.HTML,
            )

    async def finish(self) -> None:
        with contextlib.suppress(Exception):
            await self._bot.delete_message(chat_id=self._chat_id, message_id=self._message_id)


class TelegramUI:
    def __init__(
        self,
        *,
        settings: Settings,
        profile: AgentProfile,
        agent_loop: AgentLoop,
        memory: MemoryStore,
    ) -> None:
        self._s = settings
        self._profile = profile
        self._loop = agent_loop
        self._memory = memory
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._running: dict[int, asyncio.Event] = {}

    # -- authorisation -------------------------------------------------------

    def _authorised(self, update: Update) -> bool:
        """Fail-closed: an empty allowlist refuses everyone, including the owner."""
        user = update.effective_user
        if user is None:
            return False
        return user.id in self._s.allowed_user_ids()

    async def _refuse(self, update: Update) -> None:
        user = update.effective_user
        logger.warning("refused user id=%s", getattr(user, "id", "?"))
        if update.effective_message:
            await update.effective_message.reply_text(
                f"Нет доступа. Твой Telegram id: {getattr(user, 'id', '?')}"
            )

    # -- commands ------------------------------------------------------------

    async def cmd_start(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        await update.effective_message.reply_text(
            f"Привет. Я <b>{html.escape(self._profile.name)}</b>.\n"
            f"{html.escape(self._profile.description)}\n\n"
            "Просто напиши, что нужно сделать. /help — список команд.",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_help(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        await update.effective_message.reply_text(
            "<b>Команды</b>\n"
            "/new — забыть контекст разговора\n"
            "/model &lt;имя&gt; — сменить модель для этого чата\n"
            "/models — какие модели доступны\n"
            "/tools — какие инструменты подключены\n"
            "/status — режимы записи и лимиты\n"
            "/cancel — прервать текущую задачу",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_new(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        self._memory.reset(update.effective_chat.id)
        await update.effective_message.reply_text("Контекст очищен.")

    async def cmd_models(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        chat = self._memory.get(update.effective_chat.id)
        current = chat.model or self._profile.model or self._s.model_default
        lines = [
            ("• <b>" + html.escape(m) + "</b>" if m == current else "• " + html.escape(m))
            for m in self._s.allowed_models()
        ]
        await update.effective_message.reply_text(
            "Доступные модели (жирным — текущая):\n" + "\n".join(lines),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        if not ctx.args:
            return await self.cmd_models(update, ctx)
        requested = ctx.args[0].strip()
        allowed = self._s.allowed_models()
        if requested not in allowed:
            await update.effective_message.reply_text(
                f"Модель {html.escape(requested)} не в списке разрешённых.\n"
                f"Доступны: {', '.join(html.escape(m) for m in allowed)}\n\n"
                "Чтобы добавить новую — создай deployment в Azure AI Foundry "
                "и допиши его имя в MODEL_ALLOWLIST.",
                parse_mode=ParseMode.HTML,
            )
            return
        self._memory.get(update.effective_chat.id).model = requested
        await update.effective_message.reply_text(f"Модель для этого чата: {requested}")

    async def cmd_tools(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        catalog = await self._loop.catalog(force_refresh=True)
        if not catalog.specs:
            await update.effective_message.reply_text(
                "Ни одного инструмента не доступно — проверь, что MCP-серверы подняты."
            )
            return
        by_server: dict[str, list[str]] = {}
        for spec in catalog.specs:
            name = spec["function"]["name"]
            server, _, bare = name.partition("__")
            by_server.setdefault(server, []).append(bare or name)
        chunks = [
            f"<b>{html.escape(server)}</b> ({len(names)}): "
            + ", ".join(html.escape(n) for n in sorted(names))
            for server, names in sorted(by_server.items())
        ]
        await self._send_long(update, "\n\n".join(chunks))

    async def cmd_status(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        chat = self._memory.get(update.effective_chat.id)
        model = chat.model or self._profile.model or self._s.model_default
        await update.effective_message.reply_text(
            f"Профиль: <b>{html.escape(self._profile.id)}</b>\n"
            f"Модель: <b>{html.escape(model)}</b>\n"
            f"GitHub: <code>{self._s.github_write_mode.value}</code>\n"
            f"Cloudflare: <code>{self._s.cloudflare_write_mode.value}</code>\n"
            f"Защищённые репо: <code>{html.escape(self._s.protected_repos) or '—'}</code>\n"
            f"Лимит шагов: {self._loop.max_steps}\n"
            f"Сообщений в контексте: {len(chat.messages)}",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_cancel(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        event = self._running.get(update.effective_chat.id)
        if event is None:
            await update.effective_message.reply_text("Сейчас ничего не выполняется.")
            return
        event.set()
        await update.effective_message.reply_text("Останавливаюсь после текущего шага…")

    # -- main message handler ------------------------------------------------

    async def on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)

        chat_id = update.effective_chat.id
        if chat_id in self._running:
            await update.effective_message.reply_text(
                "Я ещё занят предыдущей задачей. /cancel — прервать."
            )
            return

        text = (update.effective_message.text or "").strip()
        if not text:
            return

        status = await update.effective_message.reply_text("Думаю…")
        progress = ProgressReporter(ctx.bot, chat_id, status.message_id)
        cancel = asyncio.Event()
        self._running[chat_id] = cancel

        chat = self._memory.get(chat_id)
        model = chat.model or self._profile.model or self._s.model_default
        audit = AuditLog(agent=self._profile.id, chat_id=chat_id)

        try:
            result = await self._loop.run(
                chat=chat,
                user_text=text,
                model=model,
                audit=audit,
                progress=progress,
                approver=self._make_approver(ctx, chat_id),
                cancel=cancel,
            )
            await progress.finish()
            await self._send_long(update, result.text or "(пустой ответ)")
        except Exception as exc:  # noqa: BLE001 - a crash must not kill the poller
            logger.exception("turn failed")
            await progress.finish()
            await update.effective_message.reply_text(
                f"Сломалось: {html.escape(type(exc).__name__)}: {html.escape(str(exc))[:500]}",
                parse_mode=ParseMode.HTML,
            )
        finally:
            self._running.pop(chat_id, None)

    # -- approvals -----------------------------------------------------------

    def _make_approver(self, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
        async def approver(tool: str, arguments: dict[str, Any], reason: str) -> bool:
            token = uuid.uuid4().hex[:8]
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending[token] = future

            preview = ", ".join(f"{k}={_short(v)}" for k, v in list(arguments.items())[:6])
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("✅ Выполнить", callback_data=f"ap:{token}:y"),
                        InlineKeyboardButton("✖ Отклонить", callback_data=f"ap:{token}:n"),
                    ]
                ]
            )
            await ctx.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"<b>Подтверди действие</b>\n"
                    f"Инструмент: <code>{html.escape(tool)}</code>\n"
                    f"Аргументы: <code>{html.escape(preview) or '—'}</code>\n\n"
                    f"{html.escape(reason)}"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            try:
                return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
            except TimeoutError:
                return False
            finally:
                self._pending.pop(token, None)

        return approver

    async def on_callback(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        # Re-check authorisation: anyone who can see the message can press the button.
        if not self._authorised(update):
            await query.answer("Нет доступа", show_alert=True)
            return

        await query.answer()
        _, _, rest = (query.data or "").partition(":")
        token, _, verdict = rest.partition(":")
        future = self._pending.get(token)
        if future is None or future.done():
            await query.edit_message_text("Этот запрос уже неактуален.")
            return

        approved = verdict == "y"
        future.set_result(approved)
        await query.edit_message_text("✅ Подтверждено." if approved else "✖ Отклонено.")

    # -- helpers -------------------------------------------------------------

    async def _send_long(self, update: Update, text: str) -> None:
        for chunk in _split(text, MAX_MESSAGE):
            await update.effective_message.reply_text(
                f"<pre>{html.escape(chunk)}</pre>" if _looks_like_code(chunk) else html.escape(chunk),
                parse_mode=ParseMode.HTML,
            )

    def build_application(self) -> Application:
        app = (
            Application.builder()
            .token(self._s.telegram_bot_token)
            # See module docstring: /cancel and approval buttons must not queue behind
            # the turn they are meant to interrupt.
            .concurrent_updates(True)
            .build()
        )
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("new", self.cmd_new))
        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CommandHandler("models", self.cmd_models))
        app.add_handler(CommandHandler("tools", self.cmd_tools))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("cancel", self.cmd_cancel))
        app.add_handler(CallbackQueryHandler(self.on_callback, pattern=r"^ap:"))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message))
        return app


def _short(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _looks_like_code(text: str) -> bool:
    return text.count("\n") > 3 and any(
        marker in text for marker in ("{", "}", "def ", "apiVersion", "  ")
    )


def _split(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= size:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, size)
        if cut < size // 2:
            cut = size
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    return chunks
