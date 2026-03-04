This branch provides the scripts for reproducing the experimental results of the work ___"Evaluation of k-Anonymization Algorithms on Data Utility, Classification Performance, and Vulnerability on Multiple Datasets"___, presented at ___[CSEC 112](https://www.ipsj.or.jp/kenkyukai/event/dps206csec112.html)___.

In this work, the considered algorithms were evaluated on 2 datasets.
To reproduce the results, run the provided ipython notebooks for each dataset in the following order:
1. `csec112_#DATASET#_1_create_sample.ipynb`
2. `csec112_#DATASET#_2_evaluate.ipynb`
3. `csec112_#DATASET#_3_plots.ipynb`
4. (`csec112_london2002_4_plots_compare_datasets.ipynb`)

After that, the results are stored in `./results/#DATASET#`.