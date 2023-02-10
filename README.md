## Graph Neural Networks with Diverse Spectral Filtering
Code for WWW 2023 paper "Graph Neural Networks with Diverse Spectral Filtering"

## Abstract
Spectral Graph Neural Networks (GNNs) have achieved tremendous success in graph machine learning, with polynomial filters applied for graph convolutions, where all nodes share the ***identical*** filter weights to mine their local contexts. Despite the success, existing spectral GNNs usually fail to deal with complex networks (e.g., WWW) due to such ***homogeneous*** spectral filtering setting that ignores the regional ***heterogeneity*** as typically seen in real-world networks. To tackle this issue, we propose a novel ***diverse*** spectral filtering (DSF) framework, which automatically learns node-specific filter weights to exploit the varying local structure properly. Particularly, the diverse filter weights consist of two components --- A global one shared among all nodes, and a local one that varies along network edges to reflect node difference arising from distinct graph parts --- to balance between local and global information. As such, not only can the global graph characteristics be captured, but also the diverse local patterns can be mined with awareness of different node positions. Interestingly, we formulate a novel optimization problem to assist in learning diverse filters, which also enables us to enhance any spectral GNNs with our DSF framework. We showcase the proposed framework on three state-of-the-arts including GPR-GNN, BernNet, and JacobiConv. Extensive experiments over 10 benchmark datasets demonstrate that our framework can consistently boost model performance by up to 4.92% in node classification tasks, producing diverse filters with enhanced interpretability.

## Interpretable Diverse Filters
<img src="https://github.com/jingweio/DSF/blob/main/figures/intep.png"/>


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


## UPDATE (as of Feb 10, 2023): Further analysis on measuring

<p align="center">
      <img src="https://github.com/jingweio/DSF/blob/main/figures/staDis_distribBox.png" align="left">
      <img src="https://github.com/jingweio/DSF/blob/main/figures/staDis_distribLine.png" align="right">
</p>

<img src="https://github.com/jingweio/DSF/blob/main/figures/main__LocUfreq_histgram_typical.png"/>

