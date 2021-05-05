"""
Tree Node
"""
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Iterable, List, Union


@dataclass
class TreeNode:
    name: str
    sup_node: "TreeNode" = field(default=None, init=False, repr=False)
    sub_nodes: List["TreeNode"] = field(default_factory=list,
                                        init=False, repr=False)

    @property
    def is_root(self) -> bool:
        return self.sup_node is None

    @property
    def is_leaf(self) -> bool:
        return len(self.sub_nodes) == 0

    @property
    def full_name(self) -> str:
        if self.is_root:
            return self.name
        return f"{self.sup_node.full_name}/{self.name}"

    @property
    def root(self) -> "TreeNode":
        if self.is_root:
            return self
        return self.sup_node.root

    @property
    def leafs(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return list(chain.from_iterable(
            [node.leafs for node in self.sub_nodes]
        ))

    @property
    def all_sub_nodes(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return [self] + list(chain.from_iterable(
            [node.all_sub_nodes for node in self.sub_nodes]
        ))

    @property
    def all_nodes(self) -> List["TreeNode"]:
        return self.root.all_sub_nodes

    def append(self, node: Union[str, "TreeNode"]):
        node = self.as_treenode(node)
        if not node.is_root:
            raise ValueError(f"Cannot append {node}, "
                             f"already have parent {node.sup_node}.")
        sub_node_names = [sub_node.name for sub_node in self.sub_nodes]
        if node.name in sub_node_names:
            index = sub_node_names.index(node.name)
            while len(node.sub_nodes) > 0:
                self.sub_nodes[index].append(node.pop())
        else:
            node.sup_node = self
            self.sub_nodes.append(node)

    def extend(self, nodes: Iterable[Union[str, "TreeNode"]]):
        for node in nodes:
            self.append(node)

    def merge(self, node: Union[str, "TreeNode"]):
        if node.name != self.name:
            raise ValueError("Cannot merge nodes with different names.")
        while len(node.sub_nodes) > 0:
            self.append(node.pop())

    def pop(self, *args) -> "TreeNode":
        node = self.sub_nodes.pop(*args)
        node.sup_node = None
        return node

    def remove(self, node: "TreeNode"):
        if not isinstance(node, TreeNode):
            raise TypeError("Can only remove TreeNode.")
        if not node.is_root:
            if node.sup_node is self:
                self.pop(self.sub_nodes.index(node))
            elif not self.is_leaf:
                for sub_node in self.sub_nodes:
                    sub_node.remove(node)

    @property
    def rank(self) -> int:
        if self.is_root:
            return 0
        return self.sup_node.rank + 1

    @classmethod
    def as_treenode(cls, obj: Any) -> "TreeNode":
        if isinstance(obj, cls):
            return obj
        return TreeNode(str(obj))

    def __len__(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(len(node) for node in self.sub_nodes)

    def __add__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.merge(node)
        return self

    def __sub__(self, node: "TreeNode") -> "TreeNode":
        self.remove(node)
        return self

    def __truediv__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.append(node)
        return self.sub_nodes[-1]

    def __contains__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only contain TreeNode.")
        if node == self:
            return True
        return any(node in sub_node for sub_node in self.sub_nodes)

    # TODO: add eq function
