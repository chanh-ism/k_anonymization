# Python (re)implementation of several *k*-anonymization algorithms.

Currently, the module is under active development by the researchers of the [SDC4Society project](https://www.sdc4society.org/) for research and fact-checking purposes.

The following algorithms have been implemented and tested:

- Datafly [[1]](#references)
- Classic Mondrian [[2]](#references)
- k-Member [[3]](#references)
- One-pass K-Means (OKA) [[4]](#references)
- Perturbation [[5-6]](#references)

Before using/developing the module, run the following command to install dependencies and the module itself.

```shell
pip install -e .
```

## References
[1] Sweeney, Latanya. "Datafly: A system for providing anonymity in medical data." Database Security XI. IFIP Advances in Information and Communication Technology. Springer, 1998. https://doi.org/10.1007/978-0-387-35285-5_22

[2] LeFevre, Kristen, David J. DeWitt, and Raghu Ramakrishnan. "Mondrian multidimensional k-anonymity." Proceedings of the 22nd International conference on data engineering (ICDE'06). IEEE, 2006. https://doi.org/10.1109/ICDE.2006.101.

[3] Byun, JW., Kamra, A., Bertino, E., Li, N. "Efficient k-Anonymization Using Clustering Techniques." Proceedings of the International conference on database systems for advanced applications (DASFAA 2007). Lecture Notes in Computer Science, vol 4443. Springer, 2007. https://doi.org/10.1007/978-3-540-71703-4_18.

[4] Lin, Jun-Lin, and Meng-Cheng Wei. "An efficient clustering method for k-anonymization." Proceedings of the 2008 international workshop on Privacy and anonymity in information society. ACM, 2008. https://doi.org/10.1145/1379287.1379297.

[5] I. Dai, C. Koji, and T. Katsumi. "k-匿名性の確率的指標への拡張とその適用例." In コンピュータセキュリティシンポジウム2009 (CSS2009) 論文集, vol. 2009, pp. 1–6. 情報処理学会, 2011. https://ipsj.ixsq.nii.ac.jp/records/74904.

[6] I. Dai, C. Koji, and T. Katsumi. “数値属性における, k-匿名性を満たすランダム化手法.” In コンピュータセキュリティシンポジウム2011 (CSS2011) 論文集,vol. 2011, pp. 450–455. 情報処理学会, 2011. https://ipsj.ixsq.nii.ac.jp/records/77972.