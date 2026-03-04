import pandas as pd
import itertools
from typing import List

from .node import Node
from .utils import vprint


class Lattice:
    def __init__(self, hierarchy: pd.DataFrame) -> None:
        """
        Q: 準識別子のリスト
        """
        self.nodes: List["Node"] = []
        self.Q: List[str] = hierarchy["column"].unique().tolist()
        self.hierarchy: pd.DataFrame = hierarchy
        self.attributes: int = 0

    def _single_attribute_initialization(self) -> None:
        """
        単一属性の一般化について初期化、Incognitoの初期条件
        """
        # 単一属性について一般化変換を構築
        for q in self.Q:
            tmp_nodes = []
            # 一般化の定義を取得
            generalizations = self.hierarchy[self.hierarchy["column"] == q]
            max_generalization_level = generalizations["parent_level"].max()
            for generalization_level in range(max_generalization_level + 1):
                # ノードを生成
                node = Node([(q, generalization_level)])
                tmp_nodes.append(node)
            # 親子関係を構築
            for i in range(
                1, len(tmp_nodes)
            ):  # rootはそこに至るノードがないのでスキップ
                # それ以外は前のノードをfromにする
                tmp_nodes[i].add_src_node(tmp_nodes[i - 1])
                tmp_nodes[i - 1].add_dst_node(tmp_nodes[i])

            self.nodes.extend(tmp_nodes)

    def _node_generation(self) -> None:
        """
        属性数+1のノードを生成する
        """
        # i個の属性があるとき、i-1個目までの属性とレベルが同じ かつ i個目の属性が左<右
        active_nodes = []
        new_nodes_tmp = []  # edgeを張る前の一時的なnode
        for node in self.nodes:
            if not node.deleted:
                active_nodes.append(node)

        for p, q in itertools.permutations(active_nodes, 2):
            # 属性名でソート
            p.generalization = sorted(p.generalization, key=lambda x: x[0])
            q.generalization = sorted(q.generalization, key=lambda x: x[0])
            # i-1個目までの属性とレベルが同じ かつ i個目の属性が左<右のとき，generate
            if (p.generalization[:-1] == q.generalization[:-1]) and (
                p.generalization[-1][0] < q.generalization[-1][0]
            ):
                new_generalization = set(p.generalization) | set(q.generalization)
                append_node = Node(new_generalization)
                append_node.add_inclement_parent([p, q])
                new_nodes_tmp.append(append_node)

        vprint(len(new_nodes_tmp), "nodes generated.")
        self.nodes = new_nodes_tmp

    def _edge_generation(self) -> None:
        """
        ノード間のエッジを生成する
        """
        for p, q in itertools.combinations(self.nodes, 2):
            # node上で親子関係があるのは、p,qについて
            # 1: 親1が同じで、親2同士に一般化関係がある場合
            # 2: 親1同士に一般化関係があり、親2が同じ場合
            # 3: 親1同士、親2同士の両方に一般化関係がある場合
            # p -> q のエッジを生成することを考える、permutationsなので逆のパターンも知覚できる

            # 親1: from_nodes[0]
            # 親2: from_nodes[1]

            # 1: 親1が同じ and 親2に一般化関係がある場合
            cond_1 = (p.graph_gen_parents[0] == q.graph_gen_parents[0]) and (
                (  # p -> q
                    q.graph_gen_parents[1] in p.graph_gen_parents[1].to_nodes
                    and p.graph_gen_parents[1] in q.graph_gen_parents[1].from_nodes
                )
                and (  # q -> p
                    p.graph_gen_parents[1] in q.graph_gen_parents[1].to_nodes
                    and q.graph_gen_parents[1] in p.graph_gen_parents[1].from_nodes
                )
            )

            # 2: 親1にはp -> qの一般化関係がある and 親2が同じ場合
            cond_2 = (p.graph_gen_parents[1] == q.graph_gen_parents[1]) and (
                # p -> q
                (
                    q.graph_gen_parents[0] in p.graph_gen_parents[0].to_nodes
                    and p.graph_gen_parents[0] in q.graph_gen_parents[0].from_nodes
                )
                # q -> p
                and (
                    p.graph_gen_parents[0] in q.graph_gen_parents[0].to_nodes
                    and q.graph_gen_parents[0] in p.graph_gen_parents[0].from_nodes
                )
            )

            # 3: 親1同士に一般化関係がある and 親2同士に一般化関係がある場合
            cond_3 = (
                q.graph_gen_parents[0] in p.graph_gen_parents[0].to_nodes
                and p.graph_gen_parents[0] in q.graph_gen_parents[0].from_nodes
            ) and (
                q.graph_gen_parents[1] in p.graph_gen_parents[1].to_nodes
                and p.graph_gen_parents[1] in q.graph_gen_parents[1].from_nodes
            )

            if cond_1 or cond_2 or cond_3:
                # p -> q のエッジを追加したnodeのリストを生成
                if p.height > q.height:
                    # q -> p
                    q.add_dst_node(p)
                    p.add_src_node(q)
                # elif p.height == q.height:
                #     continue
                else:
                    # p -> q
                    p.add_dst_node(q)
                    q.add_src_node(p)

    def graph_generation(self) -> None:
        """
        属性を+1したLatticeを生成する
        """
        vprint("node_generation: ", end="")
        self._node_generation()
        vprint("edge_generation")
        self._edge_generation()
        self.attributes += 1

    def increment_attributes(self) -> None:
        """
        属性の数を1増やす
        """
        if self.attributes == 0:
            self._single_attribute_initialization()
        else:
            self.graph_generation()
        self.attributes += 1
