"""The other conversations, as the model is shown them: the note and the index in the
prompt, and what the reading tools print.

What is worth pinning here is everything the model acts on: that the index never names
the conversation it is written for, that it holds nothing that moves with every message
(the prompt's front would stop matching the cache), that every id it is given reads back
as the same conversation, that what it reads says where it came from, and that a secret
pasted somewhere once does not come back through it.
"""

from __future__ import annotations

import re

import pytest

from agentcore import recall
from agentcore.localtools import MEMORY_TOOLS
from agentcore.memory import ConversationKey
from agentcore.store import ConversationSummary, StoredText

GROUP = -1001234567890  # a forum supergroup, as a chat id looks
PRIVATE = 1234567
TODAY = "2026-10-17"
GITHUB_TOKEN = "ghp_" + "A" * 36  # a token's shape, not a token


def conv(
    chat_id: int,
    thread_id: int,
    *,
    messages: int = 4,
    first_day: str = "2026-10-01",
    last_at: str = "2026-10-11 21:42",
    first_user: str = "",
    last_user: str = "",
    first_id: int = 0,
    first_user_cut: bool = False,
    last_user_cut: bool = False,
) -> ConversationSummary:
    return ConversationSummary(
        chat_id=chat_id,
        thread_id=thread_id,
        messages=messages,
        first_day=first_day,
        last_at=last_at,
        first_user=first_user,
        last_user=last_user,
        first_id=first_id,
        first_user_cut=first_user_cut,
        last_user_cut=last_user_cut,
    )


def text(
    id: int,
    *,
    chat_id: int = GROUP,
    thread_id: int = 78,
    role: str = "user",
    body: str = "загадал число 7",
    start: int = 1,
    length: int | None = None,
) -> StoredText:
    return StoredText(
        id=id,
        chat_id=chat_id,
        thread_id=thread_id,
        role=role,
        at="2026-10-11 21:42",
        text=body,
        start=start,
        length=len(body) if length is None else length,
    )


def listing() -> list[ConversationSummary]:
    """As the store returns it: the most recently active first."""
    return [
        conv(GROUP, 939, first_user="какое число я загадал?", last_at="2026-10-11 21:44", first_id=1856),
        conv(GROUP, 78, first_user="В одной теме напиши: «загадал число 7»", first_id=1852),
        conv(GROUP, 0, first_user="Что ты про меня знаешь?", last_at="2026-10-10 20:46", first_id=341),
        conv(PRIVATE, 0, first_user="А хватит qwen?", last_at="2026-10-06 19:25", first_id=21),
    ]


# -- naming a conversation ----------------------------------------------------


@pytest.mark.parametrize(("chat", "thread"), [(GROUP, 78), (GROUP, 0), (PRIVATE, 0)])
def test_a_handle_reads_back_as_the_same_conversation(chat: int, thread: int) -> None:
    assert recall.parse_handle(recall.handle(chat, thread), topic_chat=None) == (chat, thread)


def test_a_bare_topic_number_is_a_topic_of_the_chat_it_is_resolved_against() -> None:
    assert recall.parse_handle("78", topic_chat=GROUP) == (GROUP, 78)
    assert recall.parse_handle(" 78 ", topic_chat=GROUP) == (GROUP, 78)


def test_a_bare_number_names_nothing_without_a_group_to_resolve_it_in() -> None:
    assert recall.parse_handle("78", topic_chat=None) is None


def test_a_topic_link_is_what_the_model_shows_and_may_copy_back() -> None:
    link = recall.topic_link(GROUP, 78)
    assert link == "https://t.me/c/1234567890/78"
    assert recall.parse_handle(link, topic_chat=None) == (GROUP, 78)
    assert recall.parse_handle("t.me/c/1234567890/78", topic_chat=None) == (GROUP, 78)


def test_the_general_topics_own_id_is_where_general_is_stored() -> None:
    """Telegram calls General topic 1; the store calls it 0, like no topic at all."""
    assert recall.parse_handle(f"{GROUP}/1", topic_chat=None) == (GROUP, 0)
    assert recall.parse_handle("1", topic_chat=GROUP) == (GROUP, 0)


@pytest.mark.parametrize(
    "spelled",
    ["", "   ", "тема 78", "General", "78/", "/78", "-100/x", "https://example.com/c/1/2"],
)
def test_anything_else_names_no_conversation(spelled: str) -> None:
    assert recall.parse_handle(spelled, topic_chat=GROUP) is None


def test_only_a_topic_of_a_supergroup_has_a_link() -> None:
    assert recall.topic_link(GROUP, 0) is None  # General
    assert recall.topic_link(PRIVATE, 0) is None
    assert recall.topic_link(-4567, 5) is None  # an ordinary group's id has no -100


def test_a_conversation_is_called_what_a_person_would_call_it() -> None:
    atlas = recall.Atlas([conv(GROUP, 5), conv(GROUP, 0), conv(-4567, 0), conv(PRIVATE, 0)])
    assert atlas.label(PRIVATE, 0) == "личный чат"
    assert atlas.label(GROUP, 78) == "тема 78"
    assert atlas.label(GROUP, 0) == "General"
    assert atlas.label(-4567, 0) == "группа"  # no topics: an ordinary group
    assert atlas.forums == {GROUP}


def test_a_bare_number_from_the_private_chat_means_the_one_forum_there_is() -> None:
    """A private chat has no topics; "тема 78" said there is a topic of the group."""
    atlas = recall.Atlas(listing())
    assert atlas.topic_chat(GROUP) == GROUP
    assert atlas.topic_chat(PRIVATE) == GROUP
    assert atlas.topic_chat(None) == GROUP
    two = recall.Atlas([*listing(), conv(-1009999999999, 5)])
    assert two.topic_chat(PRIVATE) is None  # ambiguous: refuse rather than guess
    assert recall.Atlas([conv(PRIVATE, 0)]).topic_chat(PRIVATE) is None


def test_a_general_message_from_before_the_split_is_marked_as_such() -> None:
    atlas = recall.Atlas(listing())  # the forum's first topic row is 1852
    assert atlas.before_split(GROUP, 0, 1500)
    assert not atlas.before_split(GROUP, 0, 1900)
    assert not atlas.before_split(GROUP, 78, 1500)
    assert not atlas.before_split(PRIVATE, 0, 30)


# -- search terms -------------------------------------------------------------


def test_terms_are_lower_case_words_without_punctuation_or_repeats() -> None:
    assert recall.terms_of("Кластер, «Oracle»? кластер") == ["кластер", "oracle"]


def test_the_words_a_sentence_is_made_of_are_not_searched() -> None:
    """A substring search for «про» finds проект, проверь, просто: most of the table."""
    assert recall.terms_of("что мы решили про миграцию и на чём всё было") == [
        "решили",
        "миграцию",
    ]


def test_yo_is_read_as_ye_because_people_type_both() -> None:
    assert recall.terms_of("Отчёт ЕЩЁ") == ["отчет"]


def test_words_joined_the_way_a_model_joins_them_are_split() -> None:
    assert recall.terms_of("миграц,oracle|кластер/flux;cron") == [
        "миграц",
        "oracle",
        "кластер",
        "flux",
        "cron",
    ]


def test_search_engine_syntax_is_stripped_rather_than_searched_for() -> None:
    assert recall.terms_of("миграц* +oracle -kafka #12") == ["миграц", "oracle", "kafka", "12"]


def test_there_are_at_most_six_terms() -> None:
    assert recall.terms_of("a1 b2 c3 d4 e5 f6 g7 h8") == ["a1", "b2", "c3", "d4", "e5", "f6"]


@pytest.mark.parametrize("query", ["", "   ", "и в на", "7", "«»", "что про"])
def test_a_query_with_nothing_to_search_gives_no_terms(query: str) -> None:
    assert recall.terms_of(query) == []


# -- the system prompt --------------------------------------------------------


def test_the_index_leaves_out_the_conversation_it_is_written_for() -> None:
    block = recall.index_block(listing(), ConversationKey(GROUP, 939))
    assert "тема 939" not in block
    assert "какое число я загадал" not in block
    assert "тема 78 (https://t.me/c/1234567890/78)" in block
    assert "загадал число 7" in block
    assert f"General (id {GROUP}/0)" in block
    assert f"личный чат (id {PRIVATE}/0)" in block


def test_the_index_is_printed_in_the_order_conversations_appeared() -> None:
    """By recency it reshuffled whenever the person moved between two other topics,
    and every reshuffle was a prompt the cache had not seen."""
    block = recall.index_block(listing(), ConversationKey(GROUP, 5))
    order = [block.index(name) for name in ("личный чат", "General", "тема 78", "тема 939")]
    assert order == sorted(order)


def test_moving_between_other_topics_does_not_change_the_index() -> None:
    here = ConversationKey(GROUP, 5)
    a = conv(GROUP, 7, first_id=100, last_at="2026-10-17 10:00")
    b = conv(GROUP, 9, first_id=200, last_at="2026-10-17 10:05")
    assert recall.index_block([b, a], here) == recall.index_block([a, b], here)


def test_the_index_holds_nothing_that_moves_with_every_message() -> None:
    """Counts, times of day and the latest message change on every turn of the
    conversation they describe; in the prompt's front they would break the cache of
    every other conversation's next turn."""
    before = [conv(GROUP, 78, messages=4, last_at="2026-10-11 09:00", last_user="раз")]
    after = [conv(GROUP, 78, messages=40, last_at="2026-10-11 23:59", last_user="два")]
    here = ConversationKey(GROUP, 5)
    assert recall.index_block(before, here) == recall.index_block(after, here)


def test_general_is_described_as_the_shared_part_and_a_topic_by_its_first_message() -> None:
    block = recall.index_block(listing(), ConversationKey(GROUP, 939))
    general = next(line for line in block.splitlines() if line.startswith("- General"))
    assert "до разделения по темам" in general
    assert "Что ты про меня знаешь" not in general  # its first message is months of mixture
    topic = next(line for line in block.splitlines() if line.startswith("- тема 78"))
    assert "первое сообщение: «" in topic
    assert "2026-10-11" in topic


def test_a_topic_with_no_message_from_the_person_is_not_described_as_general() -> None:
    block = recall.index_block([conv(GROUP, 78, first_user="")], ConversationKey(GROUP, 5))
    assert "до разделения по темам" not in block


def test_a_first_message_the_database_cut_short_says_so() -> None:
    block = recall.index_block(
        [conv(GROUP, 78, first_user="вот манифест:", first_user_cut=True)],
        ConversationKey(GROUP, 5),
    )
    assert "«вот манифест:…»" in block


def test_a_long_index_is_cut_and_says_how_much_is_left() -> None:
    many = [conv(GROUP, thread, first_id=thread) for thread in range(2, 2 + recall.INDEX_LIMIT + 3)]
    block = recall.index_block(many, ConversationKey(PRIVATE, 0))
    assert block.count("\n- ") == recall.INDEX_LIMIT
    assert "…и ещё 3: полный список — memory__list_conversations." in block


def test_with_no_other_conversations_the_index_says_so() -> None:
    assert recall.index_block([conv(GROUP, 5)], ConversationKey(GROUP, 5)) == (
        "Других разговоров с ним пока нет."
    )
    assert recall.index_block([], ConversationKey(GROUP, 5)) == "Других разговоров с ним пока нет."


def test_the_note_comes_first_then_the_date_then_the_index() -> None:
    block = recall.prompt_block(listing(), ConversationKey(GROUP, 5), today=TODAY)
    assert block.startswith(recall.RECALL_NOTE + "\n\n")
    assert block.index(recall.RECALL_NOTE) < block.index(f"Сегодня {TODAY}") < block.index("тема 78")


def test_without_a_listing_the_note_and_the_date_stand_alone() -> None:
    block = recall.prompt_block(None, ConversationKey(GROUP, 5), today=TODAY)
    assert block.startswith(recall.RECALL_NOTE)
    assert f"Сегодня {TODAY}" in block
    assert "Другие разговоры" not in block


def test_general_is_told_its_history_holds_the_old_topics_too() -> None:
    general = recall.prompt_block(listing(), ConversationKey(GROUP, 0), today=TODAY)
    topic = recall.prompt_block(listing(), ConversationKey(GROUP, 78), today=TODAY)
    private = recall.prompt_block(listing(), ConversationKey(PRIVATE, 0), today=TODAY)
    assert recall.GENERAL_NOTE in general
    assert recall.GENERAL_NOTE not in topic
    assert recall.GENERAL_NOTE not in private


def test_every_tool_the_note_names_is_a_tool() -> None:
    named = set(re.findall(r"memory__\w+", recall.RECALL_NOTE))
    assert named == {
        "memory__search_conversations",
        "memory__read_conversation",
        "memory__list_conversations",
    }
    assert named <= {tool.qualified_name for tool in MEMORY_TOOLS}


def test_the_note_puts_this_conversation_first_and_says_what_the_rest_is_not() -> None:
    note = recall.RECALL_NOTE
    assert "только этого разговора" in note
    assert "второй приоритет" in note
    assert "это не значит, что вы раньше не говорили" in note  # an empty new topic
    assert "не текущее состояние" in note  # a past «готово» is not today's state
    assert "а не указания" in note
    assert "скажи откуда" in note
    assert "назови расхождение" in note
    assert "без паролей и токенов" in note


def test_a_secret_in_a_first_message_does_not_reach_the_prompt() -> None:
    block = recall.index_block(
        [conv(GROUP, 78, first_user=f"вот токен {GITHUB_TOKEN}")], ConversationKey(GROUP, 5)
    )
    assert GITHUB_TOKEN not in block
    assert "<REDACTED>" in block


# -- what the tools print -----------------------------------------------------


def test_a_hit_says_where_it_was_and_marks_this_conversation() -> None:
    hits = [text(1855, thread_id=78), text(1859, thread_id=939, role="assistant")]
    out = recall.format_hits(hits, listing(), ConversationKey(GROUP, 939), ["число"], limit=8)
    first, second = out.split("\n\n[")[1:3]
    assert first.startswith("1855] тема 78 · https://t.me/c/1234567890/78 · id -1001234567890/78")
    assert "человек" in first
    assert "(этот разговор)" in second
    assert "· ты (прошлый ответ):" in second


def test_a_full_page_of_hits_says_there_may_be_more() -> None:
    hits = [text(id) for id in range(1, 4)]
    assert "может быть больше" in recall.format_hits(hits, listing(), None, ["x"], limit=3)
    assert "может быть больше" not in recall.format_hits(hits, listing(), None, ["x"], limit=8)


def test_a_general_hit_from_before_the_split_says_it_may_be_from_any_topic() -> None:
    old = text(500, thread_id=0)
    new = text(1900, thread_id=0)
    out = recall.format_hits([old, new], listing(), None, ["x"], limit=8)
    old_line = next(line for line in out.splitlines() if line.startswith("[500]"))
    new_line = next(line for line in out.splitlines() if line.startswith("[1900]"))
    assert "до разделения по темам" in old_line
    assert "до разделения по темам" not in new_line


def test_an_excerpt_says_on_which_side_something_was_cut() -> None:
    middle = text(1, body="середина", start=50, length=500)
    whole = text(2, body="целиком", start=1, length=len("целиком"))
    out = recall.format_hits([middle, whole], listing(), None, ["x"], limit=8)
    assert "…середина…" in out
    assert "\nцеликом" in out
    assert "…целиком" not in out
    assert "целиком…" not in out


def test_a_cut_is_counted_the_way_oracle_counts() -> None:
    """dbms_lob.getlength counts UTF-16 code units: an emoji outside the Basic
    Multilingual Plane is two. A Python len() would see a cut in a whole message."""
    body = "готово 🔧"
    whole = text(1, body=body, start=1, length=len(body) + 1)
    assert "готово 🔧…" not in recall.format_hits([whole], [], None, ["x"], limit=8)
    read = recall.format_read(
        [whole], chat_id=GROUP, thread_id=78, conversations=[], current=None, around=None
    )
    assert "дальше ещё" not in read


def test_a_secret_in_what_was_said_does_not_come_back() -> None:
    hit = text(1, body=f"держи {GITHUB_TOKEN} и пароль: hunter22")
    for out in (
        recall.format_hits([hit], listing(), None, ["x"], limit=8),
        recall.format_read(
            [hit], chat_id=GROUP, thread_id=78, conversations=listing(), current=None, around=None
        ),
    ):
        assert GITHUB_TOKEN not in out
        assert "hunter22" not in out
        assert "<REDACTED>" in out


def test_nothing_found_names_the_words_that_were_searched() -> None:
    out = recall.format_hits([], listing(), None, ["миграц", "oracle"], limit=8)
    assert "миграц, oracle" in out
    assert "Ничего не нашлось" in out


def test_reading_marks_the_message_a_search_pointed_at() -> None:
    messages = [text(1860), text(1862, role="assistant"), text(1864)]
    out = recall.format_read(
        messages,
        chat_id=GROUP,
        thread_id=78,
        conversations=listing(),
        current=ConversationKey(GROUP, 939),
        around=1862,
    )
    assert out.startswith("тема 78 · https://t.me/c/1234567890/78")
    assert "\n→ [1862]" in out
    assert "\n[1860]" in out
    assert "→ [1860]" not in out


def test_reading_general_from_before_the_split_says_so() -> None:
    out = recall.format_read(
        [text(500, thread_id=0)],
        chat_id=GROUP,
        thread_id=0,
        conversations=listing(),
        current=None,
        around=None,
    )
    assert "меньше 1852 записаны до разделения по темам" in out


def test_reading_says_how_much_of_a_long_message_is_left() -> None:
    out = recall.format_read(
        [text(10, body="начало", length=1006)],
        chat_id=GROUP,
        thread_id=78,
        conversations=[],
        current=None,
        around=None,
    )
    assert "дальше ещё 1000 симв." in out


def test_reading_says_when_the_message_asked_about_is_not_in_that_conversation() -> None:
    out = recall.format_read(
        [text(10)], chat_id=GROUP, thread_id=78, conversations=[], current=None, around=999
    )
    assert "Сообщения 999 в этом разговоре нет" in out


def test_reading_a_conversation_with_nothing_said_in_it_says_so() -> None:
    out = recall.format_read(
        [], chat_id=GROUP, thread_id=78, conversations=listing(), current=None, around=None
    )
    assert out.endswith("сообщений нет.")


def test_the_list_shows_every_conversation_and_marks_this_one() -> None:
    rows = [
        conv(GROUP, 78, messages=4, first_user="первое", last_user="последнее", first_id=10),
        conv(PRIVATE, 0, messages=505, first_user="одно и то же", last_user="одно и то же"),
    ]
    out = recall.format_list(rows, ConversationKey(GROUP, 78))
    assert out.startswith("Разговоров: 2.")
    assert "тема 78 (этот разговор)" in out
    assert "сообщений: 4" in out and "сообщений: 505" in out
    assert "первое от человека: «первое»" in out
    assert "последнее от человека: «последнее»" in out
    assert out.count("одно и то же") == 1  # one message is not shown as both ends
    assert f"id {PRIVATE}/0" in out


def test_a_long_list_stays_inside_one_tool_result_and_says_what_is_left() -> None:
    """The loop clips a tool result at 12,000 characters, and a clipped list lost the
    oldest conversations without a word."""
    long = "слово " * 60
    rows = [
        conv(GROUP, thread, first_user=long, last_user=long + "!", first_id=thread)
        for thread in range(2, 80)
    ]
    out = recall.format_list(rows, None)
    assert len(out) < 12_000
    assert "…и ещё" in out
    assert out.count("первое от человека") == recall.LIST_DETAILED


def test_an_empty_memory_lists_nothing() -> None:
    assert recall.format_list([], None) == "Разговоров в памяти нет."
