-- Durable memory for the agents: conversation history and facts about the user.
--
-- Captured from the live instance rather than written by hand — the tables were
-- created directly against the database and this file did not exist, so a lost
-- schema could not have been rebuilt from the repository.
--
-- Run as AGENTS, not as ADMIN. The two are deliberately separate: the same
-- Autonomous Database holds the URL shortener under ADMIN, and an agent holding
-- the ADMIN password could drop it by accident. AGENTS has no grants on it —
-- `select from admin.urls` as this user is ORA-00942.
--
-- Two things must exist before this runs, both done once as ADMIN and neither
-- captured here (the privilege grants were not read back off the instance, so
-- writing them down would be a guess):
--
--   * the AGENTS user itself, with a quota on its tablespace;
--   * ORDS_ADMIN.ENABLE_SCHEMA for AGENTS, which is what publishes the
--     /ords/agents/_/sql endpoint the application talks to. Note the path is
--     the schema's own mapping, not ADMIN's.

-- Conversation history. One row per message; replayed into a chat on the first
-- message after a restart, newest MEMORY_HISTORY_MESSAGES rows.
create table conversations (
  id            number generated always as identity,
  agent         varchar2(64)   not null,
  chat_id       number         not null,
  role          varchar2(16)   not null,
  -- CLOB rather than VARCHAR2: a long answer or a large tool result exceeds the
  -- 32k bind limit, and the application already splits its binds on that basis.
  content       clob,
  tool_calls    clob,
  tool_call_id  varchar2(128),
  created_at    timestamp with time zone default systimestamp not null,
  constraint conversations_pk primary key (id)
);

-- The read is always "the newest N for this agent and chat", so the index
-- carries id as its third column and the query is answered from it alone.
create index conversations_by_chat on conversations (agent, chat_id, id);

-- Facts about the user. Small and read whole into the system prompt, so there is
-- nothing to search and no vector column — see the commit that added this.
create table facts (
  fact_key     varchar2(200)  not null,
  fact         varchar2(2000) not null,
  scope        varchar2(64)   default 'user' not null,
  source_chat  number,
  created_at   timestamp with time zone default systimestamp not null,
  updated_at   timestamp with time zone default systimestamp not null,
  constraint facts_pk primary key (fact_key)
);
