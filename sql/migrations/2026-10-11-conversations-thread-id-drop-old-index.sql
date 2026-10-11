-- Step 3 of 2026-10-11-conversations-thread-id.sql, in a file of its own so that nobody
-- running that one runs this with it: a script is split on its semicolons and every
-- statement of it executes.
--
-- Only after the new image is running and has been seen reading and writing through
-- conversations_by_thread. Until then the old image's history query, by agent and chat,
-- is answered by the index dropped here. The new queries are answered by
-- conversations_by_thread, which carries thread_id between the chat and id.

drop index conversations_by_chat;
