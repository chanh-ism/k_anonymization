# Incognito - Python Implementation

k-匿名化アルゴリズム Incognito のPython実装です。

## 概要
Incognitoは、データセットのk匿名性を担保する一般化変換を高速に探索するアルゴリズムです。準識別子の少ない組み合わせから順に検証を行い、k匿名性を満たし得ないパターンを枝刈りすることで探索空間を絞り込み、高速な探索を行います。k-匿名性を満たすすべての一般化変換が得られるため、アプリケーションの目的に合わせて最適な変換を柔軟に選択できます。

### 参考文献
> [LeFevre, K., *et al.*, "Incognito: efficient full-domain K-anonymity." *In Proc. ACM SIGMOD*, 49–60, 2005.](https://dl.acm.org/doi/10.1145/1066157.1066164)

## セットアップ

### 1. データセットの配置

プロジェクトルートに `Data/` ディレクトリを作成し、データセットと階層定義ファイルを配置します：

```bash
Data/
├── adult/
│   ├── adult.csv              # データセット
│   └── hierarchies/           # 階層定義ファイル
│       ├── age.csv
│       ├── education.csv
│       ├── marital-status.csv
│       ├── native-country.csv
│       ├── occupation.csv
│       ├── race.csv
│       ├── salary-class.csv
│       ├── sex.csv
│       └── workclass.csv
├── atus/
├── cup/
└── ...
```

### 2. Python環境の構築

#### uvを使う場合（推奨）
```bash
# uvのインストール（https://docs.astral.sh/uv/getting-started/installation/）
$ uv sync
```

#### condaを使う場合
```bash
# condaのインストール（https://docs.conda.io/projects/conda/en/latest/user-guide/install/）
$ conda env create -f conda_env.yaml
```

### 3. プログラムの実行

#### 基本的な使い方
```bash
$ uv run python main.py --k 10 --q_cols sex workclass marital-status
```

#### 実行結果
実行すると、`((準識別子名, 変換レベル), ...)`をkey、対応する一般化変換済みのDataFrameをvalueとするdictが返されます。

#### コマンドラインオプション
```
usage: main.py [-h] [--dataset DATASET] [--k K] [--q_cols Q_COLS [Q_COLS ...]] [--verbose] [--dropna] [--output OUTPUT]

options:
  -h, --help            ヘルプを表示
  --dataset DATASET     使用するデータセット（デフォルト: 'adult'）
  --k K                 k-匿名性のパラメータ（デフォルト: 10）
  --q_cols Q_COLS [Q_COLS ...]
                        一般化する準識別子のリスト（例: 'workclass', 'education'）
  --verbose             詳細な出力を有効化
  --dropna              NaNを含むレコードを削除
  --output OUTPUT       結果の出力ディレクトリ（未指定の場合は自動生成）
```

### プログラムからの利用

```python
from src import Incognito, utils

# データセット読み込み
dataset = utils.read_dataset("adult")
dataset = utils.dropna(dataset)  # 欠損値削除（オプション）

# 階層定義読み込み
q_cols = ["sex", "workclass", "marital-status"]
hierarchies_dir = "Data/adult/hierarchies"
hierarchy = utils.read_hierarchies_by_col_names(q_cols, hierarchies_dir)
hierarchy = hierarchy[hierarchy["child_level"] == 0]

# Incognitoアルゴリズム実行
incognito = Incognito(dataset, hierarchy, k=10)
incognito.run()

# 結果取得・表示
incognito.print_result()

# 結果保存
incognito.save_result("result/my_experiment")
```

## Result

実行結果は指定したディレクトリ（または自動生成されたディレクトリ）に保存されます：

```
result/adult_sex_workclass_k10_20260106_154731/
├── generalizations/
│   ├── sex0_workclass2.csv
│   ├── sex1_workclass0.csv
│   ├── sex1_workclass1.csv
│   └── sex1_workclass2.csv
└── metadata.json
```

### generalizations/

Incognitoの結果（k-匿名性を満たす一般化）を適用したデータセット。ファイル名から各属性の一般化レベルが分かります：

- `sex0_workclass2.csv`: sexレベル0、workclassレベル2で一般化
- `sex1_workclass0.csv`: sexレベル1、workclassレベル0で一般化
