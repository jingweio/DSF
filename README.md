## Graph Neural Networks with Diverse Spectral Filtering
Code for WWW 2023 paper "Graph Neural Networks with Diverse Spectral Filtering"

## Abstract
Spectral Graph Neural Networks (GNNs) have achieved tremendous success in graph machine learning, with polynomial filters applied for graph convolutions, where all nodes share the \textit{identical} filter weights to mine their local contexts. Despite the success, existing spectral GNNs usually fail to deal with complex networks (e.g., WWW) due to such ***homogeneous*** spectral filtering setting that ignores the regional ***heterogeneity*** as typically seen in real-world networks. To tackle this issue, we propose a novel ***diverse*** spectral filtering~(DSF) framework, which automatically learns node-specific filter weights to exploit the varying local structure properly. Particularly, the diverse filter weights consist of two components --- A global one shared among all nodes, and a local one that varies along network edges to reflect node difference arising from distinct graph parts --- to balance between local and global information. As such, not only can the global graph characteristics be captured, but also the diverse local patterns can be mined with awareness of different node positions. Interestingly, we formulate a novel optimization problem to assist in learning diverse filters, which also enables us to enhance any spectral GNNs with our DSF framework. We showcase the proposed framework on three state-of-the-arts including GPR-GNN, BernNet, and JacobiConv. Extensive experiments over 10 benchmark datasets demonstrate that our framework can consistently boost model performance by up to 4.92% in node classification tasks, producing diverse filters with enhanced interpretability.
\end{abstract}

## Requirement
- pytorch==
- torch_geometric==
- dgl==
- networkx==
- sklearn==
- scipy==
- numpy==
- optuna==


## Datasets
We use the following 10 benchmark datasets in our experiments. **Chameleon** and **Squirrel** are two wikipedia networks where web pages are connected by mutual links. Each web page has some keywords as features and is classified into five categories. 
**Wisconsin**, **Cornell**, and **Texas**~ are three webpage datasets collected by Carnegie Mellon University, where nodes are web pages classified into five classes, and edges correspond to hyperlinks. The bag-of-word representations of web pages are taken as node features. **Twtich-DE** is a social network where nodes, edges, and labels respectively represent twitch users, mutual friendship, and whether a streamer uses explicit language or not. Node features encode users’ information in streaming habits, game preference, and location. **Cora** and **Citeseer** are three widely used citation networks with strong homophily, where nodes are scientific papers, edges denote undirected citations, and each node is assigned with one topic as well as bag-of-word features. **Computers** and **Photo** are two Amazon co-purchase graphs. Nodes are goods connected by an edge if they are frequently bought together. The product reviews are encoded into the bag-of-words to be node features, and the product category corresponds to the class label.

## Hyper-parameters Setting
We implement

## Interpretable Experimental Results
<img src="https://github.com/jingweio/DSF/blob/main/intep.png"/>
