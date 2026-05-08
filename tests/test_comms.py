import threading

import pytest

from talk_box.comms import (
    AgentMessage,
    Mailbox,
    MessageType,
    broadcast,
    reply,
    send,
)


# ---------------------------------------------------------------------------
# MessageType
# ---------------------------------------------------------------------------


class TestMessageType:
    def test_values(self):
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.BROADCAST.value == "broadcast"
        assert MessageType.NOTIFY.value == "notify"

    def test_members(self):
        assert len(MessageType) == 4


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------


class TestAgentMessage:
    def test_frozen(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        with pytest.raises(AttributeError):
            msg.sender = "c"  # type: ignore[misc]

    def test_fields(self):
        msg = AgentMessage(
            sender="analyst",
            recipient="reviewer",
            content="Check this",
            message_type=MessageType.REQUEST,
            metadata={"priority": "high"},
            correlation_id="abc",
        )
        assert msg.sender == "analyst"
        assert msg.recipient == "reviewer"
        assert msg.content == "Check this"
        assert msg.message_type == MessageType.REQUEST
        assert msg.metadata == {"priority": "high"}
        assert msg.correlation_id == "abc"

    def test_auto_message_id(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        assert msg.message_id != ""
        assert len(msg.message_id) == 32  # uuid4 hex

    def test_unique_message_ids(self):
        m1 = AgentMessage(sender="a", recipient="b", content="hi")
        m2 = AgentMessage(sender="a", recipient="b", content="hi")
        assert m1.message_id != m2.message_id

    def test_auto_timestamp(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        assert msg.timestamp > 0

    def test_default_type(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        assert msg.message_type == MessageType.NOTIFY

    def test_default_correlation_id(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        assert msg.correlation_id == ""

    def test_default_metadata(self):
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        assert msg.metadata == {}


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


class TestMailbox:
    def test_empty(self):
        mailbox = Mailbox()
        assert len(mailbox) == 0
        assert mailbox.all == []

    def test_deliver(self):
        mailbox = Mailbox()
        msg = AgentMessage(sender="a", recipient="b", content="hi")
        result = mailbox.deliver(msg)
        assert result is msg
        assert len(mailbox) == 1

    def test_inbox(self):
        mailbox = Mailbox()
        m1 = mailbox.deliver(AgentMessage(sender="a", recipient="b", content="1"))
        mailbox.deliver(AgentMessage(sender="a", recipient="c", content="2"))
        m3 = mailbox.deliver(AgentMessage(sender="c", recipient="b", content="3"))
        inbox = mailbox.inbox("b")
        assert inbox == [m1, m3]

    def test_inbox_includes_broadcasts(self):
        mailbox = Mailbox()
        m1 = mailbox.deliver(AgentMessage(sender="a", recipient="b", content="direct"))
        m2 = mailbox.deliver(AgentMessage(sender="a", recipient="*", content="broadcast"))
        inbox = mailbox.inbox("b")
        assert inbox == [m1, m2]

    def test_outbox(self):
        mailbox = Mailbox()
        m1 = mailbox.deliver(AgentMessage(sender="a", recipient="b", content="1"))
        mailbox.deliver(AgentMessage(sender="b", recipient="a", content="2"))
        m3 = mailbox.deliver(AgentMessage(sender="a", recipient="c", content="3"))
        outbox = mailbox.outbox("a")
        assert outbox == [m1, m3]

    def test_thread(self):
        mailbox = Mailbox()
        req = AgentMessage(
            sender="a",
            recipient="b",
            content="request",
            message_type=MessageType.REQUEST,
        )
        mailbox.deliver(req)
        resp = AgentMessage(
            sender="b",
            recipient="a",
            content="response",
            message_type=MessageType.RESPONSE,
            correlation_id=req.message_id,
        )
        mailbox.deliver(resp)
        # Unrelated message
        mailbox.deliver(AgentMessage(sender="c", recipient="d", content="other"))

        thread = mailbox.thread(req.message_id)
        assert len(thread) == 2
        assert thread[0] is req
        assert thread[1] is resp

    def test_by_type(self):
        mailbox = Mailbox()
        m1 = mailbox.deliver(
            AgentMessage(sender="a", recipient="b", content="req", message_type=MessageType.REQUEST)
        )
        mailbox.deliver(
            AgentMessage(sender="b", recipient="a", content="note", message_type=MessageType.NOTIFY)
        )
        result = mailbox.by_type(MessageType.REQUEST)
        assert result == [m1]

    def test_all_returns_copy(self):
        mailbox = Mailbox()
        mailbox.deliver(AgentMessage(sender="a", recipient="b", content="hi"))
        result = mailbox.all
        result.append(AgentMessage(sender="x", recipient="y", content="extra"))
        assert len(mailbox) == 1


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_delivers(self):
        mailbox = Mailbox()
        msg = send(mailbox, sender="a", recipient="b", content="hello")
        assert msg.sender == "a"
        assert msg.recipient == "b"
        assert msg.content == "hello"
        assert mailbox.inbox("b") == [msg]

    def test_send_default_type(self):
        mailbox = Mailbox()
        msg = send(mailbox, sender="a", recipient="b", content="hi")
        assert msg.message_type == MessageType.NOTIFY

    def test_send_with_type(self):
        mailbox = Mailbox()
        msg = send(
            mailbox,
            sender="a",
            recipient="b",
            content="do this",
            message_type=MessageType.REQUEST,
        )
        assert msg.message_type == MessageType.REQUEST

    def test_send_with_metadata(self):
        mailbox = Mailbox()
        msg = send(
            mailbox,
            sender="a",
            recipient="b",
            content="hi",
            metadata={"urgency": "high"},
        )
        assert msg.metadata == {"urgency": "high"}

    def test_send_with_correlation_id(self):
        mailbox = Mailbox()
        msg = send(
            mailbox,
            sender="a",
            recipient="b",
            content="hi",
            correlation_id="thread-1",
        )
        assert msg.correlation_id == "thread-1"


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_broadcast_delivers_to_star(self):
        mailbox = Mailbox()
        msg = broadcast(mailbox, sender="mgr", content="meeting")
        assert msg.recipient == "*"
        assert msg.message_type == MessageType.BROADCAST

    def test_broadcast_visible_to_all(self):
        mailbox = Mailbox()
        msg = broadcast(mailbox, sender="mgr", content="standup")
        assert mailbox.inbox("dev1") == [msg]
        assert mailbox.inbox("dev2") == [msg]
        assert mailbox.inbox("qa") == [msg]

    def test_broadcast_with_metadata(self):
        mailbox = Mailbox()
        msg = broadcast(mailbox, sender="mgr", content="hi", metadata={"room": "A1"})
        assert msg.metadata == {"room": "A1"}

    def test_broadcast_custom_type(self):
        mailbox = Mailbox()
        msg = broadcast(
            mailbox,
            sender="mgr",
            content="alert",
            message_type=MessageType.NOTIFY,
        )
        assert msg.message_type == MessageType.NOTIFY


# ---------------------------------------------------------------------------
# reply
# ---------------------------------------------------------------------------


class TestReply:
    def test_reply_links_to_original(self):
        mailbox = Mailbox()
        req = send(
            mailbox,
            sender="a",
            recipient="b",
            content="question",
            message_type=MessageType.REQUEST,
        )
        resp = reply(mailbox, req, sender="b", content="answer")
        assert resp.correlation_id == req.message_id
        assert resp.recipient == "a"  # reply goes to original sender
        assert resp.message_type == MessageType.RESPONSE

    def test_reply_in_thread(self):
        mailbox = Mailbox()
        req = send(
            mailbox,
            sender="a",
            recipient="b",
            content="review this",
            message_type=MessageType.REQUEST,
        )
        resp = reply(mailbox, req, sender="b", content="LGTM")
        thread = mailbox.thread(req.message_id)
        assert len(thread) == 2
        assert thread[0] is req
        assert thread[1] is resp

    def test_reply_with_metadata(self):
        mailbox = Mailbox()
        req = send(mailbox, sender="a", recipient="b", content="q")
        resp = reply(mailbox, req, sender="b", content="a", metadata={"score": 9})
        assert resp.metadata == {"score": 9}

    def test_multiple_replies_to_one_request(self):
        mailbox = Mailbox()
        req = send(
            mailbox,
            sender="mgr",
            recipient="dev",
            content="status?",
            message_type=MessageType.REQUEST,
        )
        r1 = reply(mailbox, req, sender="dev", content="50% done")
        r2 = reply(mailbox, req, sender="dev", content="75% done")
        thread = mailbox.thread(req.message_id)
        assert len(thread) == 3
        assert thread[1] is r1
        assert thread[2] is r2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_sends(self):
        mailbox = Mailbox()
        errors = []

        def sender(agent_id: int):
            try:
                for i in range(50):
                    send(
                        mailbox,
                        sender=f"agent_{agent_id}",
                        recipient="collector",
                        content=f"msg_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sender, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mailbox) == 500  # 10 threads × 50 messages
        assert len(mailbox.inbox("collector")) == 500

    def test_concurrent_read_write(self):
        mailbox = Mailbox()
        errors = []

        def writer():
            try:
                for i in range(100):
                    send(mailbox, sender="w", recipient="r", content=f"m_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    mailbox.inbox("r")
                    mailbox.outbox("w")
                    mailbox.all
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mailbox) == 500


# ---------------------------------------------------------------------------
# Integration scenario
# ---------------------------------------------------------------------------


class TestIntegrationScenario:
    def test_multi_agent_workflow(self):
        mailbox = Mailbox()

        # Manager sends requests to two agents
        req1 = send(
            mailbox,
            sender="manager",
            recipient="analyst",
            content="Analyze Q1 data",
            message_type=MessageType.REQUEST,
        )
        req2 = send(
            mailbox,
            sender="manager",
            recipient="writer",
            content="Draft summary",
            message_type=MessageType.REQUEST,
        )

        # Agents reply
        resp1 = reply(mailbox, req1, sender="analyst", content="Revenue up 15%")
        resp2 = reply(mailbox, req2, sender="writer", content="Q1 was strong...")

        # Manager broadcasts result
        bcast = broadcast(mailbox, sender="manager", content="Report complete!")

        # Verify inboxes
        assert len(mailbox.inbox("analyst")) == 2  # req1 + broadcast
        assert len(mailbox.inbox("writer")) == 2  # req2 + broadcast
        assert len(mailbox.inbox("manager")) == 3  # resp1 + resp2 + broadcast

        # Verify threads
        assert len(mailbox.thread(req1.message_id)) == 2  # req1 + resp1
        assert len(mailbox.thread(req2.message_id)) == 2  # req2 + resp2

        # Verify type filtering
        assert len(mailbox.by_type(MessageType.REQUEST)) == 2
        assert len(mailbox.by_type(MessageType.RESPONSE)) == 2
        assert len(mailbox.by_type(MessageType.BROADCAST)) == 1

        # Total messages
        assert len(mailbox) == 5


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "AgentMessage",
            "MessageType",
            "Mailbox",
            "send",
            "broadcast",
            "reply",
        ]:
            assert hasattr(talk_box, name)

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "AgentMessage",
            "MessageType",
            "Mailbox",
            "send",
            "broadcast",
            "reply",
        ]:
            assert name in talk_box.__all__
