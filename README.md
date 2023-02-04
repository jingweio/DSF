## Graph Neural Networks with Diverse Spectral Filtering
Code for WWW 2023 paper "Graph Neural Networks with Diverse Spectral Filtering"

## Abstract
Spectral Graph Neural Networks (GNNs) have achieved tremendous success in graph machine learning, with polynomial filters applied for graph convolutions, where all nodes share the \textit{identical} filter weights to mine their local contexts. Despite the success, existing spectral GNNs usually fail to deal with complex networks (e.g., WWW) due to such ***homogeneous*** spectral filtering setting that ignores the regional ***heterogeneity*** as typically seen in real-world networks. To tackle this issue, we propose a novel ***diverse*** spectral filtering~(DSF) framework, which automatically learns node-specific filter weights to exploit the varying local structure properly. Particularly, the diverse filter weights consist of two components --- A global one shared among all nodes, and a local one that varies along network edges to reflect node difference arising from distinct graph parts --- to balance between local and global information. As such, not only can the global graph characteristics be captured, but also the diverse local patterns can be mined with awareness of different node positions. Interestingly, we formulate a novel optimization problem to assist in learning diverse filters, which also enables us to enhance any spectral GNNs with our DSF framework. We showcase the proposed framework on three state-of-the-arts including GPR-GNN, BernNet, and JacobiConv. Extensive experiments over 10 benchmark datasets demonstrate that our framework can consistently boost model performance by up to 4.92% in node classification tasks, producing diverse filters with enhanced interpretability.
\end{abstract}

## Requirement
torch==
dgl==
torch_geometric==
networkx==
sklearn==
scipy==
numpy==
pickle==
optuna==


## Datasets
We use the following 10 benchmark datasets in our experiments. **Chameleon** and **Squirrel**[1] are two wikipedia networks where web pages are connected by mutual links. Each web page has some keywords as features and is classified into five categories. 
**Wisconsin**, **Cornell**, and **Texas**~[2] are three webpage datasets collected by Carnegie Mellon University, where nodes are web pages classified into five classes, and edges correspond to hyperlinks. The bag-of-word representations of web pages are taken as node features. **Twtich-DE**[1,3] is a social network where nodes, edges, and labels respectively represent twitch users, mutual friendship, and whether a streamer uses explicit language or not. Node features encode users’ information in streaming habits, game preference, and location. **Cora** and **Citeseer**[4] are three widely used citation networks with strong homophily, where nodes are scientific papers, edges denote undirected citations, and each node is assigned with one topic as well as bag-of-word features. **Computers** and **Photo**[5,6] are two Amazon co-purchase graphs. Nodes are goods connected by an edge if they are frequently bought together. The product reviews are encoded into the bag-of-words to be node features, and the product category corresponds to the class label.

## Hyper-parameters Setting
We implement

## Interpretable Experimental Results
<img src="https://github.com/jingweio/DSF/blob/main/intep.png"/>


## References
[1] Benedek Rozemberczki, Carl Allen, and Rik Sarkar. 2021. Multi-scale attributed node embedding. Journal of Complex Networks 9, 2 (2021), cnab014. 
[2] Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. 2020. Geom-gcn: Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287 (2020).
[3] Derek Lim, Xiuyu Li, Felix Hohne, and Ser-Nam Lim. 2021. New benchmarks for
learning on non-homophilous graphs. arXiv preprint arXiv:2104.01404 (2021).
[4] PrithvirajSen,GalileoNamata,MustafaBilgic,LiseGetoor,BrianGalligher,and Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29, 3 (2008), 93–93.
[5] JulianMcAuley,ChristopherTargett,QinfengShi,andAntonVanDenHengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 43–52.
[6] Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Günnemann. 2018. Pitfalls of graph neural network evaluation. Relational Representation Learning Workshop, NeurIPS 2018 (2018).
[7] Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. 2021. Adaptive univer- sal generalized pagerank graph neural network. In International Conference on LearningRepresentations.
[8] Mingguo He, Zhewei Wei, Hongteng Xu, et al. 2021. Bernnet: Learning arbitrary
graph spectral filters via bernstein approximation. Advances in Neural Information
Processing Systems 34 (2021), 14239–14251.
[9] Xiyuan Wang and Muhan Zhang. 2022. How powerful are spectral graph neural networks. In ICML.
