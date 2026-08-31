"""Inline model picker layered over the Telegram UI.

Kept separate from the main UI so the model-selection callback is small and easy to
reason about. Callback data contains an allow-list index rather than a model name:
Telegram caps it at 64 bytes and the index cannot smuggle an arbitrary deployment.
"""

from __future__ import annotations

import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from agentcore.ui.telegram import TelegramUI as BaseTelegramUI

MODEL_CALLBACK_PREFIX = "ap:model:"


class TelegramUI(BaseTelegramUI):
    async def cmd_models(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorised(update):
            return await self._refuse(update)

        chat = self._memory.get(update.effective_chat.id)
        current = chat.model or self._profile.model or self._s.model_default
        allowed = self._s.allowed_models()
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        ("✓ " if model == current else "") + model,
                        callback_data=f"{MODEL_CALLBACK_PREFIX}{index}",
                    )
                ]
                for index, model in enumerate(allowed)
            ]
        )
        await update.effective_message.reply_text(
            "Выбери модель. Текущая: " + html.escape(current),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )

    async def on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data if query is not None else ""
        if not data.startswith(MODEL_CALLBACK_PREFIX):
            return await super().on_callback(update, ctx)

        if not self._authorised(update):
            await query.answer("Нет доступа", show_alert=True)
            return

        raw_index = data.removeprefix(MODEL_CALLBACK_PREFIX)
        allowed = self._s.allowed_models()
        try:
            requested = allowed[int(raw_index)]
        except (ValueError, IndexError):
            await query.answer("Список моделей изменился. Открой /model ещё раз.", show_alert=True)
            return

        chat = update.effective_chat
        if chat is None:
            await query.answer("Не удалось определить чат", show_alert=True)
            return

        self._memory.get(chat.id).model = requested
        await query.answer(f"Выбрана {requested}")
        await query.edit_message_text(
            f"Модель для этого чата: <b>{html.escape(requested)}</b>",
            parse_mode=ParseMode.HTML,
        )
