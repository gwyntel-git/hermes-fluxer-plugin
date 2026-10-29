"""One turn must use ONE id space: the thread it delivers into, everywhere.

Three live bugs, all the same shape -- an operation that knows the turn's
container, resolving it through the *parent channel* instead:

1. A brand-new thread was named after whatever long-lived session happened to be
   live in the parent channel. On a busy channel that is often a session from
   weeks ago, so a fresh thread appeared with an unrelated title and the rename
   logged ``named from session title '<wrong>'`` as a success.
2. ``/title`` never renamed the thread it was typed in. The auto-thread maps are
   keyed on the PARENT channel (``_record_auto_thread(parent, ...)``) while a
   thread-keyed session stores the THREAD as its ``chat_id``, so joining the two
   on the session row's chat_id always missed. In the live gateway log
   ``grep -c 'named from /title'`` was 0, ever.
3. Typing indicators were posted to the parent channel while the reply landed in
   the thread, because ``send_typing`` used the raw event chat id and the loop
   cached that container for the whole turn -- including a thread created
   mid-turn, which never got an indicator at all.

Fluxer gives a thread the id of the message it was started from, so the id spaces
collapse cleanly: the anchor message id IS the thread id, and a session key names
its container in the ``chat_id`` slot. These tests pin that collapse and run the
*real* session store semantics (via the ``FakeSessionStore`` from the naming
suite's fixture) rather than a stub that would hide the wrong-container lookup.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

# The naming suite owns the store stand-in (and its ``find_session_by_origin``
# semantics, including core's refusal to guess between live participants). Reusing
# it here keeps this module honest about the lookup it exercises, instead of
# pinning behaviour against a stub that always answers.
from tests.test_fluxer_thread_naming import SESSIONS_DDL, FakeSessionStore, _add_session

PARENT = "chan-busy"
ANCHOR = "msg-anchor-fresh"          # Fluxer: the thread's id IS this message's id
USER = "user-rowan"
OTHER = "user-bob"
# The long-lived channel session a parent-id lookup resolves instead of the turn.
CHANNEL_SESSION_TITLE = "Identify current model"
THREAD_TITLE = "Fix the thread id space collapse"

THREAD_KEY = f"agent:main:fluxer:thread:{ANCHOR}:{USER}"
CHANNEL_KEY = f"agent:main:fluxer:channel:{PARENT}:{USER}"


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "require_mention": False,
        "thread_replies": True,
        "delivery_verification": False,
    }
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


@pytest.fixture
def state_db(monkeypatch):
    """A stand-in Hermes session store, reached through HERMES_HOME.

    Same shape as the naming suite's fixture (this module needs the real
    ``find_session_by_origin`` semantics, not a stub that always answers), but
    declared here so the store it opens is the one this test's rows went into.
    """
    home = tempfile.mkdtemp(prefix="fluxer-idspace-home-")
    db = Path(home) / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(SESSIONS_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", home)
    store = FakeSessionStore(db)
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda home=None: store)
    return db


def _channel_session(db, *, title=CHANNEL_SESSION_TITLE, user_id=USER):
    """The two-week-old channel session a parent-id lookup finds."""
    _add_session(
        db, id="sess-stale-channel", session_key=f"agent:main:fluxer:channel:{PARENT}:{user_id}",
        chat_id=PARENT, chat_type="channel", user_id=user_id,
        title=title, title_source="llm", started_at=1.0,
    )


def _thread_session(db, *, title=THREAD_TITLE, user_id=USER):
    """This turn's thread session: chat_id is the THREAD (== the anchor message)."""
    _add_session(
        db, id="sess-fresh-thread", session_key=f"agent:main:fluxer:thread:{ANCHOR}:{user_id}",
        chat_id=ANCHOR, chat_type="thread", user_id=user_id,
        title=title, title_source="llm", started_at=100.0,
    )


# ── bug 1: a new thread is named from its own anchor, never the parent ──────────


@pytest.mark.asyncio
async def test_a_new_thread_ignores_a_live_parent_channel_session(state_db):
    """The live failure: a thread born as 'Identify current model'."""
    _channel_session(state_db)
    adapter = _adapter()
    adapter._inbound_text_by_message[ANCHOR] = "fix the thread id space collapse"
    adapter._inbound_author_by_message[ANCHOR] = USER
    adapter.create_thread = AsyncMock(return_value={"id": ANCHOR, "name": "placeholder"})
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message(PARENT, ANCHOR)

    assert adapter.create_thread.await_args.args[:2] == (
        "chan-busy", "fix the thread id space collapse",
    )


@pytest.mark.asyncio
async def test_a_new_thread_takes_its_own_sessions_title_when_it_has_one(state_db):
    _channel_session(state_db)
    _thread_session(state_db)
    adapter = _adapter()
    adapter._inbound_text_by_message[ANCHOR] = "unrelated placeholder text"
    adapter._inbound_author_by_message[ANCHOR] = USER
    adapter.create_thread = AsyncMock(return_value={"id": ANCHOR, "name": THREAD_TITLE})
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message(PARENT, ANCHOR)

    assert adapter.create_thread.await_args.args[1] == THREAD_TITLE


@pytest.mark.asyncio
async def test_a_new_thread_is_watched_on_its_anchor_not_the_parent(state_db):
    """No title yet: the poller must wait on the anchor's session.

    Watching the parent is what made the stale title reappear a few seconds later
    -- the placeholder was correct at creation time and then overwritten.
    """
    _channel_session(state_db)
    adapter = _adapter()
    adapter._inbound_text_by_message[ANCHOR] = "fix the thread id space collapse"
    adapter._inbound_author_by_message[ANCHOR] = USER
    adapter.create_thread = AsyncMock(return_value={"id": ANCHOR, "name": "fix the thread"})
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message(PARENT, ANCHOR)

    assert watched == [(ANCHOR, ANCHOR, "fix the thread", USER)]


@pytest.mark.asyncio
async def test_the_title_poller_never_waits_for_a_comment_on_the_parent_for_this_turn(state_db):
    """The parent session's title exists, so a parent-keyed poller would rename."""
    _channel_session(state_db)
    adapter = _adapter()
    adapter.rename_thread = AsyncMock(return_value=True)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01)
        patch.setattr(fluxer_adapter, "_TITLE_RENAME_WINDOW_SECONDS", 0.03)
        await adapter._rename_thread_when_titled(ANCHOR, ANCHOR, "fix the thread")

    adapter.rename_thread.assert_not_awaited()


# ── bug 2: /title reaches the thread its session key names ─────────────────────


@pytest.mark.asyncio
async def test_the_title_command_renames_a_thread_keyed_session(state_db):
    """The live failure: '/title' produced no rename at all, ever."""
    _thread_session(state_db)
    adapter = _adapter()
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session(THREAD_KEY, "Renamed by hand")

    adapter.rename_thread.assert_awaited_once_with(
        ANCHOR, "Renamed by hand", only_if_current_name=None,
    )


@pytest.mark.asyncio
async def test_the_title_command_reaches_a_thread_this_process_never_created(state_db):
    """The maps are process-lifetime, so a restart must not silence /title.

    The thread id comes out of the session key, so the rename does not depend on
    an in-memory map entry that a restart has already dropped.
    """
    _thread_session(state_db)
    adapter = _adapter()
    assert adapter.auto_thread_info_for_chat(PARENT) is None  # nothing remembered
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session(THREAD_KEY, "Renamed after restart")

    adapter.rename_thread.assert_awaited_once_with(
        ANCHOR, "Renamed after restart", only_if_current_name=None,
    )


@pytest.mark.asyncio
async def test_the_title_command_guards_on_the_placeholder_this_process_gave(state_db):
    """After a restart the guard cannot be the placeholder: the maps are empty.

    An unguarded rename is the right outcome for an explicit ``/title`` -- it is
    the user's own command -- and ``rename_thread``'s own server-side guard is what
    protects a thread a human renamed by hand in the meantime.
    """
    _thread_session(state_db)
    adapter = _adapter()
    adapter._record_auto_thread(PARENT, ANCHOR, "placeholder here", USER)
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session(THREAD_KEY, "Renamed by hand")

    adapter.rename_thread.assert_awaited_once_with(
        ANCHOR, "Renamed by hand", only_if_current_name="placeholder here",
    )


@pytest.mark.asyncio
async def test_a_channel_keyed_title_still_renames_that_users_thread(state_db):
    """The pre-existing channel behaviour must not regress."""
    adapter = _adapter()
    _add_session(state_db, session_key=CHANNEL_KEY, chat_id=PARENT, user_id=USER)
    adapter._record_auto_thread(PARENT, "th-rowan", "rowan placeholder", USER)
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session(CHANNEL_KEY, "Rowan's title")

    adapter.rename_thread.assert_awaited_once_with(
        "th-rowan", "Rowan's title", only_if_current_name="rowan placeholder",
    )


@pytest.mark.asyncio
async def test_a_channel_keyed_title_still_declines_another_persons_thread(state_db):
    adapter = _adapter()
    _add_session(state_db, session_key=f"agent:main:fluxer:channel:{PARENT}:{OTHER}",
                 chat_id=PARENT, user_id=OTHER)
    adapter._record_auto_thread(PARENT, "th-rowan", "rowan placeholder", USER)
    adapter.rename_thread = AsyncMock()

    await adapter._rename_thread_for_session(
        f"agent:main:fluxer:channel:{PARENT}:{OTHER}", "Bob's title"
    )

    adapter.rename_thread.assert_not_awaited()


# ── bug 3: typing follows the turn into its thread, on every tick ──────────────


async def _drain_loop() -> None:
    """Let a typing task make its first request without waiting its refresh out."""
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_typing_in_a_thread_keyed_turn_refreshes_the_thread(state_db):
    _thread_session(state_db)
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request
    await adapter.send_typing(ANCHOR)
    await _drain_loop()

    assert ("POST", f"/channels/{ANCHOR}/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls
    await adapter.stop_typing(ANCHOR)


@pytest.mark.asyncio
async def test_typing_moves_into_a_thread_created_after_the_loop_started(state_db):
    """The live failure: a thread created mid-turn never got an indicator.

    The loop resolved its container once, at arm time, and kept refreshing the
    parent channel for the rest of the turn -- indicator in one place, reply in
    another. Every tick re-resolves, so the indicator follows the turn into the
    thread the moment the thread exists.
    """
    adapter = _adapter(thread_replies=False)
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fluxer_adapter, "_TYPING_REFRESH_SECONDS", 0.01)
        # Armed before any thread exists: the parent channel is the right answer.
        await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
        await _drain_loop()
        assert ("POST", f"/channels/{PARENT}/typing") in calls

        # The turn opens its thread; the next refresh has to move into it.
        adapter._threads_by_anchor[(PARENT, ANCHOR)] = ANCHOR
        calls.clear()
        for _ in range(80):
            await asyncio.sleep(0.01)
            if ("POST", f"/channels/{ANCHOR}/typing") in calls:
                break
        await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})

    assert ("POST", f"/channels/{ANCHOR}/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls


@pytest.mark.asyncio
async def test_typing_opens_a_pending_thread_before_it_starts_indicating(state_db):
    """A pending thread is created on the first tick, not typed at by id.

    The prospective id names a channel that does not exist yet, so an indicator
    posted there would 404 and the loop would give up for the whole turn.
    """
    adapter = _adapter()
    adapter._threads_pending[ANCHOR] = PARENT
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if path.endswith("/threads"):
            return {"id": ANCHOR, "name": "fix the thread"}
        return {}

    adapter._request = fake_request
    adapter._schedule_title_rename = lambda *args: None

    await adapter.send_typing(ANCHOR)
    await _drain_loop()

    assert ("POST", f"/channels/{PARENT}/messages/{ANCHOR}/threads") in calls
    assert ("POST", f"/channels/{ANCHOR}/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls
    await adapter.stop_typing(ANCHOR)


@pytest.mark.asyncio
async def test_a_typing_tick_never_reverts_to_the_parent_once_the_thread_exists(state_db):
    """A dead-container redirect or a lost anchor must not send it backwards."""
    adapter = _adapter()
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request
    adapter._threads_by_anchor[(PARENT, ANCHOR)] = ANCHOR
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fluxer_adapter, "_TYPING_REFRESH_SECONDS", 0.01)
        await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
        for _ in range(50):
            await asyncio.sleep(0.01)
        await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})

    assert calls and all(path == f"/channels/{ANCHOR}/typing" for _m, path in calls)


# ── bug 4: a skipped rename says so, with ids and a reason ─────────────────────


@pytest.mark.asyncio
async def test_a_thread_that_never_gets_a_title_says_why(state_db, caplog):
    adapter = _adapter()
    caplog.set_level(logging.INFO, logger=fluxer_adapter.logger.name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01)
        patch.setattr(fluxer_adapter, "_TITLE_RENAME_WINDOW_SECONDS", 0.02)
        await adapter._rename_thread_when_titled(ANCHOR, ANCHOR, "no title yet")

    assert ANCHOR in caplog.text
    assert "no session title" in caplog.text


@pytest.mark.asyncio
async def test_a_declined_rename_says_the_thread_was_renamed_by_hand(state_db, caplog):
    _thread_session(state_db)
    adapter = _adapter()
    adapter.rename_thread = AsyncMock(return_value=False)  # server declined the guard
    caplog.set_level(logging.INFO, logger=fluxer_adapter.logger.name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01)
        await adapter._rename_thread_when_titled(ANCHOR, ANCHOR, "placeholder")

    assert ANCHOR in caplog.text
    assert "no longer carries the placeholder" in caplog.text


@pytest.mark.asyncio
async def test_typing_logs_the_container_it_armed_on(caplog):
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={})
    caplog.set_level(logging.INFO, logger=fluxer_adapter.logger.name)

    await adapter.send_typing(PARENT, metadata={"thread_id": "thread-explicit"})
    await _drain_loop()
    await adapter.stop_typing(PARENT, metadata={"thread_id": "thread-explicit"})

    assert "typing indicator armed for container thread-explicit" in caplog.text


@pytest.mark.asyncio
async def test_a_title_rename_on_a_thread_key_logs_the_thread_it_reached(state_db, caplog):
    _thread_session(state_db)
    adapter = _adapter()
    adapter.rename_thread = AsyncMock(return_value=True)
    caplog.set_level(logging.INFO, logger=fluxer_adapter.logger.name)

    await adapter._rename_thread_for_session(THREAD_KEY, "Renamed by hand")

    assert f"Fluxer thread {ANCHOR} named from /title" in caplog.text


# ── the same id space for a parent-channel /stop ───────────────────────────────


def _stop_message(user_id=USER, content="/stop"):
    return {
        "id": "cmd-1",
        "channel_id": PARENT,
        "content": content,
        "author": {"id": user_id},
        "guild_id": "guild-1",
    }


@pytest.mark.asyncio
async def test_a_parent_stop_addresses_the_thread_this_users_turn_runs_in():
    """``/stop`` in the channel must cancel the turn that moved into a thread.

    The turn's session is keyed on the THREAD, so the channel's own key cancels a
    session with no run while the real turn keeps going.
    """
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    adapter._active_auto_threads[(PARENT, USER)] = {ANCHOR: 1}

    chat_id, chat_type, parent, pending = await adapter._resolve_thread_source(
        _stop_message(), PARENT, "channel"
    )

    assert (chat_id, chat_type, parent) == (ANCHOR, "thread", PARENT)
    assert pending is None


@pytest.mark.asyncio
async def test_a_parent_stop_does_not_guess_between_two_active_threads():
    adapter = _adapter()
    adapter._remember_thread_id({"id": "thread-a"}, PARENT)
    adapter._remember_thread_id({"id": "thread-b"}, PARENT)
    adapter._active_auto_threads[(PARENT, USER)] = {"thread-a": 1, "thread-b": 1}

    chat_id, chat_type, _parent, _pending = await adapter._resolve_thread_source(
        _stop_message(), PARENT, "channel"
    )

    assert (chat_id, chat_type) == (PARENT, "channel")


@pytest.mark.asyncio
async def test_another_users_active_thread_does_not_retarget_a_parent_stop():
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    adapter._active_auto_threads[(PARENT, OTHER)] = {ANCHOR: 1}

    chat_id, chat_type, _parent, _pending = await adapter._resolve_thread_source(
        _stop_message(user_id=USER), PARENT, "channel"
    )

    assert (chat_id, chat_type) == (PARENT, "channel")


@pytest.mark.asyncio
async def test_a_thread_a_human_created_never_enters_the_active_index():
    """Only this adapter's own threads are retargetable, even at registration."""
    adapter = _adapter()
    adapter._remember_thread_owner(ANCHOR, False)
    source = SimpleNamespace(
        chat_id=ANCHOR, parent_chat_id=PARENT, chat_type="thread", user_id=USER,
    )

    assert adapter._register_active_auto_thread(source) is None
    assert adapter._active_auto_threads == {}
    chat_id, chat_type, _parent, _pending = await adapter._resolve_thread_source(
        _stop_message(), PARENT, "channel"
    )
    assert (chat_id, chat_type) == (PARENT, "channel")


@pytest.mark.asyncio
async def test_other_parent_commands_keep_the_channel_session():
    """Only /stop addresses a running turn; /new must not land in someone's thread."""
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    adapter._active_auto_threads[(PARENT, USER)] = {ANCHOR: 1}

    chat_id, chat_type, _parent, _pending = await adapter._resolve_thread_source(
        _stop_message(content="/new"), PARENT, "channel"
    )

    assert (chat_id, chat_type) == (PARENT, "channel")


@pytest.mark.asyncio
async def test_the_active_thread_index_is_released_when_the_turn_ends():
    """A finished turn must not keep retargeting later /stop commands.

    ``_register_active_auto_thread`` / ``_unregister_active_auto_thread`` are the
    pair under test: registration happens as the turn starts and the entry has to
    be gone once it ends, or a later ``/stop`` in the channel cancels nothing.
    """
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    source = SimpleNamespace(
        chat_id=ANCHOR, parent_chat_id=PARENT, chat_type="thread", user_id=USER,
    )

    registered = adapter._register_active_auto_thread(source)
    assert registered == (PARENT, USER, ANCHOR)
    assert adapter._active_auto_threads == {(PARENT, USER): {ANCHOR: 1}}

    # A second concurrent turn on the same thread must not be erased by the first
    # one finishing.
    adapter._register_active_auto_thread(source)
    adapter._unregister_active_auto_thread(*registered)
    assert adapter._active_auto_threads == {(PARENT, USER): {ANCHOR: 1}}

    adapter._unregister_active_auto_thread(*registered)
    assert (PARENT, USER) not in adapter._active_auto_threads


@pytest.mark.asyncio
async def test_mention_stripped_parent_stop_resolves_to_active_thread():
    adapter = _adapter()
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    adapter._active_auto_threads[(PARENT, USER)] = {ANCHOR: 1}
    message = _stop_message(content="<@bot-id> /stop")
    message["content"] = "/stop"  # text returned by mention gate
    chat_id, chat_type, parent, pending = await adapter._resolve_thread_source(
        message, PARENT, "channel"
    )
    assert (chat_id, chat_type, parent, pending) == (ANCHOR, "thread", PARENT, None)


@pytest.mark.asyncio
async def test_a_mention_prefixed_command_reaches_the_active_thread():
    """The mention gate's stripped text must reach the container resolver.

    With mentions required, `<@bot> /stop` is stripped to ``/stop`` by the
    admission gate -- but the resolver used to read the RAW ``data['content']``,
    so the leading mention made ``_is_text_command`` fail and the command was
    routed as a brand-new prospective thread instead of the running turn's
    container. The end-to-end handler is exercised here, not the resolver
    directly: the fix is in what the handler passes it.
    """
    adapter = _adapter(require_mention=True)
    adapter.bot_user_id = "bot-9"
    adapter._remember_thread_id({"id": ANCHOR}, PARENT)
    adapter._active_auto_threads[(PARENT, USER)] = {ANCHOR: 1}
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    await adapter._handle_message_create(
        {
            "id": "msg-cmd",
            "channel_id": PARENT,
            "channel_type": "channel",
            "content": "<@bot-9> /stop",
            "author": {"id": USER, "username": "Rowan", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert len(seen) == 1
    # The command keys on the running turn's thread, not a fresh pending thread.
    assert seen[0].source.chat_id == ANCHOR
    assert seen[0].source.chat_type == "thread"
    assert not adapter._threads_pending


def test_pending_auto_thread_is_registered_and_released():
    adapter = _adapter()
    adapter._threads_pending[ANCHOR] = PARENT
    source = SimpleNamespace(chat_id=ANCHOR, parent_chat_id=PARENT, chat_type="thread", user_id=USER)
    registered = adapter._register_active_auto_thread(source)
    assert registered == (PARENT, USER, ANCHOR)
    adapter._unregister_active_auto_thread(*registered)
    assert adapter._active_auto_threads == {}


# ── Greptile 4402981743: typing must honor a failed-creation decision ──────────


@pytest.mark.asyncio
async def test_typing_never_opens_a_thread_after_delivery_gave_up(state_db):
    """A turn that already chose the parent must keep typing in the parent.

    Delivery marks the turn's thread disabled when the first creation fails.
    A later typing refresh re-resolves its container from ``_threads_pending``,
    which still lists the anchor -- so without the guard typing would open the
    thread the delivery had just abandoned and split the turn across two
    containers. The indicator must stay on the parent and never create.
    """
    adapter = _adapter()
    adapter._threads_pending[ANCHOR] = PARENT
    # Delivery already failed to create this turn's thread and disabled it.
    adapter._disable_turn_thread(PARENT, ANCHOR)
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request
    adapter._schedule_title_rename = lambda *args: None

    await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
    await _drain_loop()
    await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})

    assert not any(path.endswith("/threads") for _m, path in calls)
    assert ("POST", f"/channels/{PARENT}/typing") in calls
    assert ("POST", f"/channels/{ANCHOR}/typing") not in calls


# ── Greptile 4402981754: only this adapter's threads earn the parent fallback ──


@pytest.mark.asyncio
async def test_a_human_thread_never_escapes_to_the_parent_channel(monkeypatch):
    """A dead human-created thread must fail in place, not repost to the parent.

    ``_resolve_outbound_target`` used to mark ANY thread (including one a human
    opened) as an auto-chosen container, so a missing-container error moved the
    reply to the parent channel -- where the human thread's audience never sees
    it. Knowing a thread's parent is not this adapter choosing it: the parent
    fallback is only for threads this adapter opened.
    """
    adapter = _adapter()
    adapter._thread_parents[ANCHOR] = PARENT
    adapter._remember_thread_owner(ANCHOR, False)  # a human created it

    _target, _payload, auto_thread_id = await adapter._resolve_outbound_target(
        ANCHOR, None, {}
    )
    assert _target == ANCHOR
    assert auto_thread_id is None

    # And the bot-owned case still earns the fallback (non-regression guard).
    adapter._remember_thread_owner(ANCHOR, True)
    _target, _payload, auto_thread_id = await adapter._resolve_outbound_target(
        ANCHOR, None, {}
    )
    assert auto_thread_id == ANCHOR
