from typing import List
import pandas as pd
import queue
import time

from . import df_operations
from .lattice import Lattice
from .utils import vprint


class Incognito:
    def __init__(self, T: pd.DataFrame, hierarchy: pd.DataFrame, k: int) -> None:
        self.T: pd.DataFrame = T  # 対象のテーブル
        self.Q: List[str] = hierarchy["column"].unique().tolist()  # 準識別子のリスト
        self.hierarchy: pd.DataFrame = hierarchy  # 一般化階層の定義df
        self.k: int = k  # k-匿名性のk値
        self.lattice: Lattice  # 構築済みのLattice
        self.execution_time: float = None  # 実行時間

    def run(self) -> List[List[tuple]]:
        """
        Incognitoの実行
        return: 一般化されたDataFrame
        """
        start_time = time.perf_counter()

        self.lattice = Lattice(self.hierarchy)
        # self.lattice.increment_attributes()  # initialization of the lattice
        priority_queue = queue.PriorityQueue()

        # 属性の組み合わせ数をボトムアップしていく
        for attributes in range(len(self.Q)):
            vprint(f"Processing attributes: {attributes + 1} / {len(self.Q)}")
            # n-1 attributes の Lattice から n attributes のものに更新
            self.lattice.increment_attributes()
            vprint(
                "Current lattice nodes:",
                len([node for node in self.lattice.nodes if not node.deleted]),
            )
            # nodeの高さによる優先度付きqueue
            for node in self.lattice.nodes:
                ## rootを流し込んで初期化
                if node.is_root() and not node.deleted:
                    priority_queue.put(node)

            vprint("pruning... ", end="")
            pruning_count = 0
            while not priority_queue.empty():
                node = priority_queue.get()

                # k匿名を満たすとしてマークされていたら、スキップ
                if node.is_marked() or node.deleted:
                    continue
                else:
                    # nodeに定義された一般化変換に従い、一般化を実施、k匿名性を検証
                    # ノードの一般化変換を取得: level-0 -> level-n
                    def row_match(row):
                        # すべての(dim, level)条件を満たすか
                        result = any(
                            (row["column"] == dim)
                            and (row["child_level"] == 0)
                            and (row["parent_level"] == level)
                            for (dim, level) in node.generalization
                        )
                        return result

                    eval_generalization = self.hierarchy[
                        self.hierarchy.apply(row_match, axis=1)
                    ]
                    # 一般化を適用
                    generalized_df = df_operations.generalize(
                        self.T, eval_generalization
                    )

                    # k匿名性の確認
                    cols = [tup[0] for tup in node.generalization]
                    k_anonymous = df_operations.is_k_anonymous(
                        generalized_df, cols, self.k
                    )

                    # k匿名性を満たすなら、ノードとその直親をマーク
                    if k_anonymous:
                        node.mark()
                        for dst_node in node.to_nodes:
                            dst_node.mark()

                    # k匿名でないとき、一段上のノードを優先度付きqueueに追加
                    else:
                        for dst_node in node.to_nodes:
                            priority_queue.put(dst_node)
                        node.delete()
                        pruning_count += 1
            vprint(f"{pruning_count} nodes are pruned.")

        result_generalizations = [
            sorted(node.generalization, key=lambda x: x[0])
            for node in self.lattice.nodes
            if not node.deleted
        ]

        self.execution_time = time.perf_counter() - start_time

        return result_generalizations

    def get_result(self) -> dict:
        """
        Incognitoの結果を取得
        return: 一般化変換の(key, level)のペアのリスト: [(workclass, 1), (sex, 1), ...]
            {
                List[tuple]: pd.Dataframe,
                List[tuple]: pd.Dataframe,
                ...
            }
        """
        generalizations = [
            node.generalization for node in self.lattice.nodes if not node.deleted
        ]
        result = {}

        for generalization in generalizations:

            def row_match(row):
                return any(
                    (row["column"] == dim)
                    and (row["child_level"] == 0)
                    and (row["parent_level"] == level)
                    for (dim, level) in generalization
                )

            generalize_hierarchy = self.hierarchy[
                self.hierarchy.apply(row_match, axis=1)
            ]

            # 一般化変換
            generalized_df = df_operations.generalize(self.T, generalize_hierarchy)

            result[tuple(sorted(generalization, key=lambda x: x[0]))] = generalized_df

        return result

    def print_result(self) -> None:
        """
        結果を表示する
        """
        print(f"\nIncognito result:")
        lattice_result = [
            node.generalization for node in self.lattice.nodes if not node.deleted
        ]
        print(
            f"There are {len(lattice_result)} combinations of generalization levels satisfying k-anonymity (k={self.k}):"
        )
        for i, node in enumerate(lattice_result):
            conditions = sorted(node, key=lambda x: x[0])
            print(i + 1, conditions)
        print()

    def verify_result(self) -> bool:
        """
        処理後の結果の検証を行う
        return: 検証結果 (True: 正常, False: 異常)
        """
        print("Verifying Incognito result...")
        result = [node for node in self.lattice.nodes if not node.deleted]
        for node in result:
            # conditions = self._node_to_generalization_tuples(node, self.hierarchies)

            # ノードの一般化変換を取得
            def row_match(row):
                return any(
                    (row["column"] == dim)
                    and (row["child_level"] == 0)
                    and (row["parent_level"] == level)
                    for (dim, level) in node.generalization
                )

            generalize_hierarchy = self.hierarchy[
                self.hierarchy.apply(row_match, axis=1)
            ]

            # 一般化変換
            generalized_df = df_operations.generalize(self.T, generalize_hierarchy)

            # k匿名性の確認
            # conditions_tup = [
            #     f"{getattr(node, f'dim{i}')}={getattr(node, f'level{i}')}"
            #     for i in range(1, num_dims + 1)
            # ]
            # print(f"node: {', '.join(conditions_tup)}")
            vprint(
                generalized_df.groupby(
                    [tup[0] for tup in node.generalization], dropna=False
                ).size()
            )
            if not df_operations.is_k_anonymous(
                generalized_df, [tup[0] for tup in node.generalization], self.k
            ):
                print(
                    f"{node.generalization} -does not satisfy k-anonymity (k={self.k})."
                )
                return False

        print(f"All {len(result)} nodes satisfies k-anonymity (k={self.k}).")
        return True

    def save_result(self, output_dir: str) -> None:
        """
        Incognito実行結果を保存

        出力構造:
        output_dir/
        ├── generalizations/
        │   ├── sex0_workclass2.csv
        │   ├── sex1_workclass0.csv
        │   └── ...
        └── metadata.json
        """
        from pathlib import Path
        from datetime import datetime
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # generalizations ディレクトリを作成
        gen_dir = output_path / "generalizations"
        gen_dir.mkdir(exist_ok=True)

        result = self.get_result()

        if not result:
            print("Warning: No valid generalizations found. Saving metadata only.")
            valid_generalizations = []
            generalizations_metadata = []
        else:
            valid_generalizations = list(result.keys())
            generalizations_metadata = []

            # 各一般化を保存
            for gen_tuple in valid_generalizations:
                # ファイル名生成: sex0_workclass2.csv のような形式
                filename_parts = [f"{col}{level}" for col, level in sorted(gen_tuple, key=lambda x: x[0])]
                filename = "_".join(filename_parts) + ".csv"
                csv_path = gen_dir / filename

                # データ保存
                gen_df = result[gen_tuple]
                gen_df.to_csv(csv_path, index=False)

                # メタデータに記録
                height = sum(level for _, level in gen_tuple)
                generalizations_metadata.append({
                    "filename": filename,
                    "generalization": dict(gen_tuple),
                    "height": height
                })

            print(f"{len(valid_generalizations)} generalizations saved to {gen_dir}")

        # メタデータ作成
        metadata = {
            "algorithm": "Incognito",
            "k": self.k,
            "quasi_identifiers": self.Q,
            "num_valid_generalizations": len(valid_generalizations),
            "generalizations": sorted(generalizations_metadata, key=lambda x: x["height"]),
            "execution_time": self.execution_time,
            "num_records": len(self.T),
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to: {metadata_path}")
        print(f"\nAll results saved to: {output_path.absolute()}")
