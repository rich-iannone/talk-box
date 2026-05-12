"""Conversation tree: branching conversation history.

Wraps the linear ``Conversation`` model with a tree structure that supports
forking at any user message and navigating between branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from talk_box.conversation import Conversation, Message


@dataclass
class TreeNode:
    """A node in the conversation tree.

    Each node holds a single message and links to its parent and children.
    """

    message: Message
    node_id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "message": self.message.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        return cls(
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            children=list(data.get("children", [])),
            message=Message.from_dict(data["message"]),
        )


class ConversationTree:
    """A tree-structured conversation that supports branching.

    Messages form a tree where each user message can have multiple
    alternative follow-ups (branches).  The tree maintains a *current
    path* from the root to the active leaf.  Forking creates a new
    branch at any previous user message.

    Parameters
    ----------
    conversation_id
        Unique identifier for this conversation tree.
    """

    def __init__(self, conversation_id: str | None = None) -> None:
        self.conversation_id = conversation_id or str(uuid4())
        self._nodes: dict[str, TreeNode] = {}  # node_id -> TreeNode
        self._root_ids: list[str] = []  # top-level node IDs (in order)
        self._active_leaf_id: str | None = None  # tip of current branch

    # -- Properties ----------------------------------------------------------

    @property
    def active_path(self) -> list[TreeNode]:
        """Return the current branch as a list of nodes from root to leaf."""
        if self._active_leaf_id is None:
            return []
        # Walk up from leaf to root, then reverse
        path: list[TreeNode] = []
        nid: str | None = self._active_leaf_id
        while nid is not None:
            node = self._nodes[nid]
            path.append(node)
            nid = node.parent_id
        path.reverse()
        return path

    @property
    def messages(self) -> list[Message]:
        """Messages along the current branch (linear view)."""
        return [n.message for n in self.active_path]

    @property
    def node_count(self) -> int:
        """Total number of nodes across all branches."""
        return len(self._nodes)

    # -- Adding messages -----------------------------------------------------

    def add_message(self, content: str, role: str) -> TreeNode:
        """Append a message to the current branch tip.

        Parameters
        ----------
        content
            Message text.
        role
            Message role (``"user"``, ``"assistant"``).

        Returns
        -------
        TreeNode
            The newly created node.
        """
        msg = Message(content=content, role=role)
        node = TreeNode(message=msg, parent_id=self._active_leaf_id)

        self._nodes[node.node_id] = node

        if self._active_leaf_id is not None:
            self._nodes[self._active_leaf_id].children.append(node.node_id)
        else:
            self._root_ids.append(node.node_id)

        self._active_leaf_id = node.node_id
        return node

    # -- Branching -----------------------------------------------------------

    def fork_at(self, node_id: str) -> None:
        """Set the active branch tip to the *parent* of ``node_id``.

        After forking, the next ``add_message`` will create a sibling
        branch alongside ``node_id``.

        Parameters
        ----------
        node_id
            The node to fork at.  The new branch diverges from this
            node's parent — i.e. this node and its subtree become one
            sibling, and the next message starts a new sibling.

        Raises
        ------
        KeyError
            If ``node_id`` is not in the tree.
        """
        node = self._nodes[node_id]
        self._active_leaf_id = node.parent_id

    def branches_at(self, node_id: str) -> list[list[TreeNode]]:
        """Return all sibling branches that share the same parent.

        Parameters
        ----------
        node_id
            A node whose siblings to find.

        Returns
        -------
        list[list[TreeNode]]
            Each element is a branch (list of nodes from the branch point
            down to the leaf).  The branch containing ``node_id`` is
            included.
        """
        node = self._nodes[node_id]
        parent_id = node.parent_id

        if parent_id is not None:
            sibling_ids = self._nodes[parent_id].children
        else:
            sibling_ids = self._root_ids

        branches: list[list[TreeNode]] = []
        for sid in sibling_ids:
            branch = self._walk_to_leaf(sid)
            branches.append(branch)
        return branches

    def sibling_ids(self, node_id: str) -> list[str]:
        """Return the IDs of all siblings of ``node_id`` (including itself).

        Siblings share the same parent.
        """
        node = self._nodes[node_id]
        if node.parent_id is not None:
            return list(self._nodes[node.parent_id].children)
        return list(self._root_ids)

    def switch_to_branch(self, node_id: str) -> None:
        """Switch the active branch to pass through ``node_id``.

        Follows ``node_id`` down to its deepest *first-child* descendant
        and sets that as the active leaf.

        Parameters
        ----------
        node_id
            A node to switch to.  The active path will include this node.
        """
        leaf = self._walk_to_leaf(node_id)
        if leaf:
            self._active_leaf_id = leaf[-1].node_id

    # -- Conversion ----------------------------------------------------------

    def to_linear(self) -> Conversation:
        """Export the current branch as a plain ``Conversation``.

        Returns
        -------
        Conversation
            A linear conversation containing only the active branch's messages.
        """
        convo = Conversation(conversation_id=self.conversation_id)
        for node in self.active_path:
            convo.messages.append(node.message)
        return convo

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationTree:
        """Create a tree from an existing linear conversation.

        Parameters
        ----------
        conversation
            A linear conversation to import.

        Returns
        -------
        ConversationTree
            A tree with a single branch containing all messages.
        """
        tree = cls(conversation_id=conversation.conversation_id)
        for msg in conversation.messages:
            node = TreeNode(message=msg, parent_id=tree._active_leaf_id)
            tree._nodes[node.node_id] = node
            if tree._active_leaf_id is not None:
                tree._nodes[tree._active_leaf_id].children.append(node.node_id)
            else:
                tree._root_ids.append(node.node_id)
            tree._active_leaf_id = node.node_id
        return tree

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tree to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "root_ids": list(self._root_ids),
            "active_leaf_id": self._active_leaf_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTree:
        """Deserialize a tree from a dictionary."""
        tree = cls(conversation_id=data["conversation_id"])
        for nid, ndata in data["nodes"].items():
            tree._nodes[nid] = TreeNode.from_dict(ndata)
        tree._root_ids = list(data.get("root_ids", []))
        tree._active_leaf_id = data.get("active_leaf_id")
        return tree

    # -- Internals -----------------------------------------------------------

    def _walk_to_leaf(self, node_id: str) -> list[TreeNode]:
        """Walk from ``node_id`` to the deepest first-child leaf."""
        path: list[TreeNode] = []
        nid: str | None = node_id
        while nid is not None:
            node = self._nodes[nid]
            path.append(node)
            nid = node.children[0] if node.children else None
        return path

    def __len__(self) -> int:
        """Number of messages on the current branch."""
        return len(self.active_path)
