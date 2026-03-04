from typing import List


class Node:
    """
    一般化変換のLatticeを構築するNodeクラス
    height: Nodeの一般化レベルの高さ
    generalization: 一般化の定義を保持する辞書
    from_nodes: このノードに至るエッジを持つノードのリスト
    to_nodes: このノードから出るエッジを持つノードのリスト

    generalization should be:
    [
        ("dim1", level1),
        ("dim2", level2),
        ...
        ("dimN", levelN)
    ]
    """

    def __init__(self, generalization: List[tuple], **kwargs):
        self.height: int
        self.generalization: List[tuple] = generalization  # [(column, level), ...]
        self.from_nodes: list["Node"] = []
        self.to_nodes: list["Node"] = []
        self.marked: bool = False
        self.deleted: bool = False
        self.graph_gen_parents: list["Node"] = []

        # init height
        self.height = sum([tup[1] for tup in self.generalization])

    def is_root(self) -> bool:
        """
        このノードがルートノードかどうかを確認する
        return: True if this node is root, False otherwise
        """
        return len(self.from_nodes) == 0

    def is_marked(self) -> bool:
        """
        k匿名性を満たすNodeとしてマークされているかを確認
        """
        return self.marked

    def mark(self) -> None:
        """
        k匿名性を満たすNodeとしてマークする
        """
        self.marked = True

    def add_dst_node(self, dst: "Node") -> None:
        """
        遷移先のノードを追加する
        param dst: toノード
        """
        self.to_nodes.append(dst)

    def add_src_node(self, src: "Node") -> None:
        """
        遷移元のノードを追加する
        param src: fromノード
        """
        self.from_nodes.append(src)

    def add_inclement_parent(self, parent: List["Node"]) -> None:
        """
        一般化レベルが上がる親ノードを追加する
        param parent: 親ノード
        """
        self.graph_gen_parents.extend(parent)

    def __lt__(self, other):
        """PriorityQueueでの比較用, heightで比較"""
        return self.height < other.height

    def delete(self) -> None:
        """
        このノードを削除する
        """
        self.deleted = True
        for dst_node in self.to_nodes:
            dst_node.from_nodes.remove(self)

    def __hash__(self):
        """
        Nodeオブジェクトのハッシュ値をgeneralization辞書の内容に基づいて定義
        辞書はハッシュ不可能なので、タプルに変換してハッシュ値を計算
        """
        # 辞書のitemsをソートしてタプルに変換することで、キーの順序が異なっても同じハッシュ値になるようにする
        return hash(frozenset(self.generalization.items()))
