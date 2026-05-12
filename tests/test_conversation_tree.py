"""Tests for talk_box.conversation_tree."""

from __future__ import annotations

import pytest

from talk_box.conversation import Conversation, Message
from talk_box.conversation_tree import ConversationTree, TreeNode


class TestTreeNode:
    """Tests for TreeNode dataclass."""

    def test_create_node(self):
        msg = Message(content="Hello", role="user")
        node = TreeNode(message=msg)
        assert node.message is msg
        assert node.parent_id is None
        assert node.children == []
        assert node.node_id  # UUID generated

    def test_to_dict_from_dict(self):
        msg = Message(content="Hello", role="user")
        node = TreeNode(message=msg, parent_id="parent-123", children=["c1", "c2"])
        d = node.to_dict()
        restored = TreeNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.parent_id == "parent-123"
        assert restored.children == ["c1", "c2"]
        assert restored.message.content == "Hello"
        assert restored.message.role == "user"


class TestConversationTreeAddMessage:
    """Tests for adding messages to the tree."""

    def test_add_first_message(self):
        tree = ConversationTree()
        node = tree.add_message("Hi", "user")
        assert node.message.content == "Hi"
        assert node.parent_id is None
        assert tree._active_leaf_id == node.node_id
        assert node.node_id in tree._root_ids

    def test_add_two_messages_linear(self):
        tree = ConversationTree()
        n1 = tree.add_message("Hi", "user")
        n2 = tree.add_message("Hello!", "assistant")
        assert n2.parent_id == n1.node_id
        assert n1.children == [n2.node_id]
        assert tree._active_leaf_id == n2.node_id

    def test_messages_returns_active_path(self):
        tree = ConversationTree()
        tree.add_message("A", "user")
        tree.add_message("B", "assistant")
        tree.add_message("C", "user")
        msgs = tree.messages
        assert [m.content for m in msgs] == ["A", "B", "C"]

    def test_len(self):
        tree = ConversationTree()
        assert len(tree) == 0
        tree.add_message("x", "user")
        tree.add_message("y", "assistant")
        assert len(tree) == 2

    def test_node_count(self):
        tree = ConversationTree()
        tree.add_message("x", "user")
        tree.add_message("y", "assistant")
        assert tree.node_count == 2


class TestConversationTreeBranching:
    """Tests for fork_at, sibling_ids, switch_to_branch."""

    def _build_tree_with_branch(self):
        """Build a tree: A -> B -> C, then fork at C and add D."""
        tree = ConversationTree()
        a = tree.add_message("A", "user")
        b = tree.add_message("B", "assistant")
        c = tree.add_message("C", "user")
        # Fork at C — active tip goes to C's parent (B)
        tree.fork_at(c.node_id)
        d = tree.add_message("D", "user")
        return tree, a, b, c, d

    def test_fork_at_sets_parent(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        # D should be a sibling of C (both children of B)
        assert d.parent_id == b.node_id
        assert set(tree._nodes[b.node_id].children) == {c.node_id, d.node_id}

    def test_active_path_after_fork(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        # Active path should be A -> B -> D
        assert [n.message.content for n in tree.active_path] == ["A", "B", "D"]

    def test_sibling_ids(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        siblings = tree.sibling_ids(c.node_id)
        assert c.node_id in siblings
        assert d.node_id in siblings
        assert len(siblings) == 2

    def test_switch_to_branch(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        # Active path is A -> B -> D.  Switch to C's branch.
        tree.switch_to_branch(c.node_id)
        assert [n.message.content for n in tree.active_path] == ["A", "B", "C"]

    def test_switch_back(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        tree.switch_to_branch(c.node_id)
        tree.switch_to_branch(d.node_id)
        assert [n.message.content for n in tree.active_path] == ["A", "B", "D"]

    def test_branches_at(self):
        tree, a, b, c, d = self._build_tree_with_branch()
        branches = tree.branches_at(c.node_id)
        assert len(branches) == 2
        # Each branch is a list of nodes from the branch point down
        contents = sorted([br[0].message.content for br in branches])
        assert contents == ["C", "D"]

    def test_fork_at_root(self):
        """Fork at the first message to create an alternative opening."""
        tree = ConversationTree()
        a = tree.add_message("A", "user")
        tree.fork_at(a.node_id)
        b = tree.add_message("B", "user")
        assert b.parent_id is None
        assert tree._root_ids == [a.node_id, b.node_id]
        assert [n.message.content for n in tree.active_path] == ["B"]

    def test_fork_at_unknown_raises(self):
        tree = ConversationTree()
        with pytest.raises(KeyError):
            tree.fork_at("nonexistent")

    def test_deep_branch_navigation(self):
        """Three branches at the same point, each with follow-ups."""
        tree = ConversationTree()
        a = tree.add_message("A", "user")
        b = tree.add_message("B", "assistant")

        # Branch 1: C1 -> D1
        c1 = tree.add_message("C1", "user")
        tree.add_message("D1", "assistant")

        # Branch 2: fork at C1, add C2 -> D2
        tree.fork_at(c1.node_id)
        c2 = tree.add_message("C2", "user")
        tree.add_message("D2", "assistant")

        # Branch 3: fork at C2, add C3 -> D3
        tree.fork_at(c2.node_id)
        c3 = tree.add_message("C3", "user")
        tree.add_message("D3", "assistant")

        # Active should be C3 branch
        assert [n.message.content for n in tree.active_path] == ["A", "B", "C3", "D3"]

        # Switch to C1 branch
        tree.switch_to_branch(c1.node_id)
        assert [n.message.content for n in tree.active_path] == ["A", "B", "C1", "D1"]

        # Siblings at branch point
        siblings = tree.sibling_ids(c2.node_id)
        assert len(siblings) == 3


class TestConversationTreeConversion:
    """Tests for to_linear and from_conversation."""

    def test_to_linear(self):
        tree = ConversationTree(conversation_id="test-conv")
        tree.add_message("Hi", "user")
        tree.add_message("Hello", "assistant")
        convo = tree.to_linear()
        assert convo.conversation_id == "test-conv"
        assert len(convo.messages) == 2
        assert convo.messages[0].content == "Hi"
        assert convo.messages[1].content == "Hello"

    def test_from_conversation(self):
        convo = Conversation(conversation_id="orig-id")
        convo.add_message("A", "user")
        convo.add_message("B", "assistant")
        convo.add_message("C", "user")
        tree = ConversationTree.from_conversation(convo)
        assert tree.conversation_id == "orig-id"
        assert len(tree) == 3
        assert [m.content for m in tree.messages] == ["A", "B", "C"]

    def test_from_conversation_roundtrip(self):
        """from_conversation -> to_linear preserves messages."""
        convo = Conversation()
        convo.add_message("X", "user")
        convo.add_message("Y", "assistant")
        tree = ConversationTree.from_conversation(convo)
        restored = tree.to_linear()
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "X"
        assert restored.messages[1].content == "Y"


class TestConversationTreeSerialization:
    """Tests for to_dict and from_dict."""

    def test_roundtrip_linear(self):
        tree = ConversationTree(conversation_id="ser-1")
        tree.add_message("A", "user")
        tree.add_message("B", "assistant")
        d = tree.to_dict()
        restored = ConversationTree.from_dict(d)
        assert restored.conversation_id == "ser-1"
        assert [m.content for m in restored.messages] == ["A", "B"]

    def test_roundtrip_with_branches(self):
        tree = ConversationTree()
        a = tree.add_message("A", "user")
        tree.add_message("B", "assistant")
        c = tree.add_message("C", "user")
        tree.fork_at(c.node_id)
        tree.add_message("D", "user")

        d = tree.to_dict()
        restored = ConversationTree.from_dict(d)
        # Active path should be A -> B -> D
        assert [m.content for m in restored.messages] == ["A", "B", "D"]
        # C's branch should still be navigable
        siblings = restored.sibling_ids(c.node_id)
        assert len(siblings) == 2

    def test_empty_tree_roundtrip(self):
        tree = ConversationTree()
        d = tree.to_dict()
        restored = ConversationTree.from_dict(d)
        assert len(restored) == 0
        assert restored.active_path == []

    def test_to_dict_keys(self):
        tree = ConversationTree(conversation_id="k")
        d = tree.to_dict()
        assert set(d.keys()) == {"conversation_id", "nodes", "root_ids", "active_leaf_id"}
