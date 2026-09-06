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
from agentcore.memory import ChatMemory, MemoryStore
from agentcore.profiles import AgentProfile
from agentcore.store import AdbStore

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
        store: AdbStore | None = None,
    ) -> None:
        self._s = settings
        self._profile = profile
        self._loop = agent_loop
        self._memory = memory
        # None means memory lives only in this process, as it did before there
        # was anywhere durable to put it.
        self._store = store
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._running: dict[int, asyncio.Event] = {}

    # -- durable memory ------------------------------------------------------

    async def _hydrate(self, chat: ChatMemory) -> None:
        """Load a chat's history from the database, once per process.

        `hydrated` is set before the read, not after, so a database that is down
        costs one attempt rather than one per message. Losing history is bad;
        adding ten seconds to every reply because of it is worse.
        """
        if chat.hydrated or self._store is None:
            return
        chat.hydrated = True
        history = await self._store.history(
            agent=self._profile.id,
            chat_id=chat.chat_id,
            limit=self._s.memory_history_messages,
        )
        if history:
            chat.adopt(history)
            logger.info(
                "restored %d messages for chat %s", len(history), chat.chat_id
            )

    async def _persist(self, chat: ChatMemory) -> None:
        """Write whatever this turn added.

        Only the new messages: the transcript is rewritten in memory on every
        step (trimming, tool results), and re-storing all of it would duplicate
        the conversation a little more each turn.
        """
        if self._store is None:
            return
        for message in chat.unpersisted():
            await self._store.append(
                agent=self._profile.id, chat_id=chat.chat_id, message=message
            )
        chat.mark_persisted()

    async def _facts_prompt(self) -> str:
        """The facts, as a block appended to the system prompt.

        All of them, unsearched — a few dozen cost about a thousand tokens, and
        an agent that knows things only when a search happens to surface them is
        harder to work with than one that knows nothing.

        The instruction to keep new facts is returned even when there are none
        yet. Gating it on a non-empty list is how the table would have stayed
        empty for good: the model was told it could remember things only inside
        the block that appears once it already has.
        """
        if self._store is None:
            return ""

        keeping = (
            "Если узнаёшь устойчивый факт о нём или его проектах — такой, который "
            "пригодится и в следующих разговорах, — сохрани его сам, вызовом "
            "memory__remember, не спрашивая разрешения и не откладывая на потом. "
            "Узнал точнее — вызови ещё раз с тем же ключом, это заменит старое. "
            "Разовые детали текущей задачи запоминать не надо: они и так в истории "
            "этого разговора."
        )

        facts = await self._store.facts()
        if not facts:
            return keeping

        lines = "\n".join(f"- {f.fact}" for f in facts)
        return (
            "Что ты знаешь о человеке, с которым говоришь (накоплено в прошлых "
            "разговорах, считай это достоверным):\n"
            f"{lines}\n\n" + keeping
        )

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
            "/memories — что я помню о тебе\n"
            "/forget &lt;ключ&gt; — удалить один факт о тебе\n"
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
        chat_id = update.effective_chat.id
        self._memory.reset(chat_id)
        if self._store is not None:
            # Otherwise the next message would restore everything /new was
            # meant to get rid of.
            await self._store.clear(agent=self._profile.id, chat_id=chat_id)
        await update.effective_message.reply_text(
            "Контекст очищен — и в памяти, и в базе. Факты о тебе не тронуты, "
            "их показывает /memories."
        )

    async def cmd_memories(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """What the agent believes about the user, and how to remove any of it.

        The point of putting memory in a database was that it can be read back
        and actually deleted. Without this command that is a claim nobody can
        check.
        """
        if not self._authorised(update):
            return await self._refuse(update)
        if self._store is None:
            await update.effective_message.reply_text(
                "Долговременная память не подключена — я помню только текущий разговор."
            )
            return
        facts = await self._store.facts()
        if not facts:
            await update.effective_message.reply_text(
                "Пока ничего о тебе не записано. Диктовать ничего не надо — "
                "я сам сохраняю то, что пригодится в следующих разговорах."
            )
            return
        lines = [
            f"<code>{html.escape(f.key)}</code> — {html.escape(f.fact)}"
            + (f" <i>({f.updated_at})</i>" if f.updated_at else "")
            for f in facts
        ]
        # Not _send_long: that escapes its input, and these lines are already
        # HTML on purpose — the key has to be copy-pasteable into /forget.
        body = (
            "Что я о тебе помню:\n\n"
            + "\n\n".join(lines)
            + "\n\nУдалить: <code>/forget ключ</code>"
        )
        for chunk in _split(body, MAX_MESSAGE):
            await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML)

    async def cmd_forget(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)
        if self._store is None:
            await update.effective_message.reply_text("Долговременная память не подключена.")
            return
        key = " ".join(ctx.args).strip() if ctx.args else ""
        if not key:
            await update.effective_message.reply_text(
                "Нужен ключ факта: /forget ключ. Список — /memories."
            )
            return
        try:
            removed = await self._store.forget(key)
        # Broad on purpose: the user gets the reason, not a stack trace.
        except Exception as exc:
            logger.exception("forget failed")
            await update.effective_message.reply_text(
                f"Не смог удалить: {html.escape(type(exc).__name__)}"
            )
            return
        await update.effective_message.reply_text(
            f"Удалено: {html.escape(key)}" if removed else f"Такого ключа нет: {html.escape(key)}"
        )

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
        # Belt and braces. The MessageHandler filter is `TEXT & ~COMMAND`, so this
        # should be unreachable — but the audit log once showed four model turns for
        # three messages right after a /new, which is what a command reaching here
        # looks like. Sending "/new" to the model costs a full turn and answers
        # nothing, so drop it rather than trust the filter alone.
        if text.startswith("/"):
            logger.warning("command %r reached the message handler; ignoring", text.split()[0])
            return

        status = await update.effective_message.reply_text("Думаю…")
        progress = ProgressReporter(ctx.bot, chat_id, status.message_id)
        cancel = asyncio.Event()
        self._running[chat_id] = cancel

        chat = self._memory.get(chat_id)
        await self._hydrate(chat)
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
                extra_system=await self._facts_prompt(),
            )
            await progress.finish()
            await self._send_long(update, result.text or "(пустой ответ)")
        except Exception as exc:  # a crash in one turn must not kill the poller
            logger.exception("turn failed")
            await progress.finish()
            await update.effective_message.reply_text(
                f"Сломалось: {html.escape(type(exc).__name__)}: {html.escape(str(exc))[:500]}",
                parse_mode=ParseMode.HTML,
            )
        finally:
            self._running.pop(chat_id, None)
            # In `finally` deliberately. A turn that crashed still asked a
            # question and may have run tools; dropping that half would leave
            # the next turn reading a transcript that jumps from one user
            # message straight to another.
            await self._persist(chat)

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
        app.add_handler(CommandHandler("memories", self.cmd_memories))
        app.add_handler(CommandHandler("forget", self.cmd_forget))
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
