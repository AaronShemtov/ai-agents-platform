from agentcore.ui.telegram_model_picker import MODEL_CALLBACK_PREFIX


def test_model_callback_prefix_shares_registered_approval_namespace() -> None:
    # Base TelegramUI registers CallbackQueryHandler(pattern=r"^ap:"). Keeping the
    # picker under that namespace makes its callbacks reach the overridden handler.
    assert MODEL_CALLBACK_PREFIX.startswith("ap:")


def test_model_callback_uses_short_index_payload() -> None:
    payload = f"{MODEL_CALLBACK_PREFIX}{999}"
    assert len(payload.encode()) <= 64
