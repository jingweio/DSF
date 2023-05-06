## Graph Neural Networks with Diverse Spectral Filtering
Code for WWW 2023 paper ["Graph Neural Networks with Diverse Spectral Filtering"](https://dl.acm.org/doi/pdf/10.1145/3543507.3583324)

Video: [short-version](https://slideslive.com/39000277) & [long-version]()

## Interpretable Diverse Filters
<p align = "center">
<img src = "https://github.com/jingweio/DSF/blob/main/figures/Intep.png">
</p>
<p align = "left">
Figure 1: (a)-(c) Diverse filters learned from real-world networks, where five representative curves are plotted for illustration. On each graph, these filters display similar overall shapes but different local details in function curves, showing the capability of our DSF in capturing both the global graph structure and locally varied linking patterns. (d) Visualization of node-specific filter weights on Cornell dataset, where alike color indicates similar filter weights between nodes. Overall, nodes can be differentiated based on their disjoint underlying regions as circled by the blue and green dashed lines, and far-reaching nodes can still learn similar filter weights due to their akin local structures. E.g., vertices on the graph border are mostly ingrained in a line subgraph such as • − • − •, and some unusual cases can be handled (see details in Section 5.4). These results justify the enhanced model interpretability by learning diverse spectral filters on the micro level.
</p>

## Further analysis on Local Graph Frequency (not included in our published version)
Upate (as of Feb 10, 2023): To quantify the diversity degree of our Local Graph Frequency across the graph, we propose a new metric called Diversity of Local Graph Frequency, and denote it as $\tau_n$ w.r.t. the $n^{\text{th}}$ eigenvector. The definition is provided below. Further details about the Local Graph Frequency refer to our paper.

<p align = "center">
<img src=https://github.com/jingweio/DSF/blob/main/figures/divLGF.png width=70% />
</p>
  
For each network, we decompose its laplacian matrix, compute the Diversity of Local Graph Frequency, and visualize the distribution in Figure 2. To further demonstrate the most representative distribution of our Local Graph Frequency, we visualize the group of Local Graph Frequency with the closet diversity degree to the mean. The histgrams on multiple networks are drawed in Figure 3.

<p align = "center">
<img src = https://github.com/jingweio/DSF/blob/main/figures/staDis_distrib.png>
</p>
<p align = "left">
Figure 2: Diversity of Local Graph Frequency on networks with different diversity degrees. Each number beside data name represents the averaged Diversity of Local Graph Frequency as computed in Definition 1. (a) Distributions of the diversity degrees of Local Graph Frequency, i.e., $\{\tau_n\}_{n=1}^{N}$, on different networks. Each column represents one graph. (b) Diversity degree of Local Graph Frequency, i.e., $\{\tau_n\}_{n=1}^{N}$, on different graphs. Each curve denotes one graph. The values are sorted in ascending order for better visualization.
</p>

<p align = "center">
<img src = "https://github.com/jingweio/DSF/blob/main/figures/main__LocUfreq_histgram_typical.png">
</p>

<p align = "left">
Figure 3: Distribution of Local Graph Frequency on various real graphs. Each number beside data name represents the averaged Diversity of Local Graph Frequency.
</p>




## Datasets
We use the following 11 benchmark datasets in our experiments.
**Chameleon** and **Squirrel** are two wikipedia networks where web pages are connected by mutual links. Each web page has some keywords as features and is classified into five categories. 
**Wisconsin**, **Cornell**, and **Texas** are three webpage datasets collected by Carnegie Mellon University, where nodes are web pages classified into five classes, and edges correspond to hyperlinks. The bag-of-word representations of web pages are taken as node features.
**Twtich-DE** is a social network where nodes, edges, and labels respectively represent twitch users, mutual friendship, and whether a streamer uses explicit language or not. Node features encode users’ information in streaming habits, game preference, and location.
**Cora**, **Citeseer**, and **Pubmed** are three widely used citation networks with strong homophily, where nodes are scientific papers, edges denote undirected citations, and each node is assigned with one topic as well as bag-of-word features.
**Computers** and **Photo** are two Amazon co-purchase graphs. Nodes are goods connected by an edge if they are frequently bought together. The product reviews are encoded into the bag-of-words to be node features, and the product category corresponds to the class label.


## Requirements
- pytorch==1.8.0
- pytorch_geometric==2.0.1
- dgl==0.7.1
- networkx==2.5.1
- sklearn==0.24.2
- scipy==1.5.3
- numpy==1.19.5
- optuna==2.9.1


## Hyper-parameters Setting
As extensive experiments with different base models over various datasets need be conducted, we tune our hyper-parameters using [Optuna](https://github.com/optuna/optuna) for 200 trails with a broad searching space defined as
- learning rate $\sim$ [1e-4, 1e-1]
- weight decay $\sim$ [5e-8, 1e-2]
- dropout $\sim$ {0, 0.1, ..., 0.8} by 0.1
- iterative optimization coefficients $\eta_1, \eta_2 \sim$ {0.1, 0.2, ..., 1.0} by 0.1
- orthogonal regularization parameter $\lambda_\text{orth} \sim$ [1e-2, 1]
- the number of raw positional features $f_p \sim$ {2, 4, ..., 32} by 2
- the initializing methods for node positional embeddings $\sim$ {LapPE, RWPE}.


## Citation
```
@inproceedings{guo2023graph,
  title={Graph Neural Networks with Diverse Spectral Filtering},
  author={Guo, Jingwei and Huang, Kaizhu and Yi, Xinping and Zhang, Rui},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={306--316},
  year={2023}
}
```
