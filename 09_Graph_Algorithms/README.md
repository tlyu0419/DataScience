# Graph Algorithms

> Graphs are mathematical structures used to model pairwise relations between objects. 

## Introduction

What Are Graphs? 

What Are Graph Analytics and Algorithms? 

Graph Processing, Databases, Queries, and Algorithms 

- OLTP and OLAP

Why Should We Care About Graph Algorithms? 

Graph Analytics Use Cases

Conclusion

## Graph Theory and Concepts

Terminology

Graph Types and Structures

- Random, Small-World, Scale-Free Structures 

Flavors of Graphs

- Connected Versus Disconnected Graphs
- Unweighted Graphs Versus Weighted Graphs 
- Undirected Graphs Versus Directed Graphs 
- Acyclic Graphs Versus Cyclic Graphs 
- Sparse Graphs Versus Dense Graphs
- Monopartite, Bipartite, and k-Partite Graphs

Types of Graph Algorithms

- Pathfinding
- Centrality
- Community Detection

Summary

## Graph Platforms and Processing

Graph Platform and Processing Considerations

- Platform Considerations
- Processing Considerations

Representative Platforms

- Selecting Our Platform
- Apache Spark
- Neo4j Graph Platform

Summary

## Pathnding and Graph Search Algorithms

Example Data: The Transport Graph

- Importing the Data into Apache Spark

- Importing the Data into Neo4j

Breadth First Search

- Breadth First Search with Apache Spark

Depth First Search

Shortest Path

- When Should I Use Shortest Path?

- Shortest Path with Neo4j

- Shortest Path (Weighted) with Neo4j

- Shortest Path (Weighted) with Apache Spark

- Shortest Path Variation: A*

- Shortest Path Variation: Yen’s k-Shortest Paths

All Pairs Shortest Path

- A Closer Look at All Pairs Shortest Path

- When Should I Use All Pairs Shortest Path?

- All Pairs Shortest Path with Apache Spark

- All Pairs Shortest Path with Neo4j

Single Source Shortest Path

- When Should I Use Single Source Shortest Path?

- Single Source Shortest Path with Apache Spark

- Single Source Shortest Path with Neo4j

Minimum Spanning Tree

- When Should I Use Minimum Spanning Tree?

- Minimum Spanning Tree with Neo4j

Random Walk

- When Should I Use Random Walk?

- Random Walk with Neo4j

Summary

## Centrality Algorithms

Example Graph Data: The Social Graph

- Importing the Data into Apache Spark

- Importing the Data into Neo4j

Degree Centrality

- Reach

- When Should I Use Degree Centrality?

- Degree Centrality with Apache Spark

Closeness Centrality

- When Should I Use Closeness Centrality?

- Closeness Centrality with Apache Spark

- Closeness Centrality with Neo4j

- Closeness Centrality Variation: Wasserman and Faust

- Closeness Centrality Variation: Harmonic Centrality

Betweenness Centrality

- When Should I Use Betweenness Centrality?

- Betweenness Centrality with Neo4j

- Betweenness Centrality Variation: Randomized-Approximate Brandes

PageRank

- Influence

- The PageRank Formula

- Iteration, Random Surfers, and Rank Sinks

- When Should I Use PageRank?

- PageRank with Apache Spark

- PageRank with Neo4j

- PageRank Variation: Personalized PageRank

Summary



## Community Detection Algorithms

Example Graph Data: The Software Dependency Graph

- Importing the Data into Apache Spark 

- Importing the Data into Neo4j

Triangle Count and Clustering Coefficient

- Local Clustering Coefficient

- Global Clustering Coefficient

- When Should I Use Triangle Count and Clustering Coefficient?

- Triangle Count with Apache Spark

- Triangles with Neo4j

- Local Clustering Coefficient with Neo4j

Strongly Connected Components

- When Should I Use Strongly Connected Components?

- Strongly Connected Components with Apache Spark

- Strongly Connected Components with Neo4j 

Connected Components

- When Should I Use Connected Components?

- Connected Components with Apache Spark

- Connected Components with Neo4j

Label Propagation

- Semi-Supervised Learning and Seed Labels 

- When Should I Use Label Propagation?

- Label Propagation with Apache Spark

- Label Propagation with Neo4j

Louvain Modularity

- When Should I Use Louvain?

- Louvain with Neo4j

Validating Communities

Summary



## Graph Algorithms in Practice

Analyzing Yelp Data with Neo4j

- Yelp Social Network

- Data Import

- Graph Model

- A Quick Overview of the Yelp Data

- Trip Planning App

- Travel Business Consulting

- Finding Similar Categories

Analyzing Airline Flight Data with Apache Spark

- Exploratory Analysis

- Popular Airports

- Delays from ORD

- Bad Day at SFO

- Interconnected Airports by Airline

Summary

## Using Graph Algorithms to Enhance Machine Learning.

Machine Learning and the Importance of Context

- Graphs, Context, and Accuracy

Connected Feature Engineering

- Graphy Features

- Graph Algorithm Features

Graphs and Machine Learning in Practice: Link Prediction

- Tools and Data

- Importing the Data into Neo4j

- The Coauthorship Graph

- Creating Balanced Training and Testing Datasets

- How We Predict Missing Links

- Creating a Machine Learning Pipeline

- Predicting Links: Basic Graph Features

- Predicting Links: Triangles and the Clustering Coefficient

- Predicting Links: Community Detection

Summary

Wrapping Things Up















### Graphs

- Graphs

  - Verticles / Nodes

    Vertex is a fundamental unit of graphs, usually labeled and denoted by a circle

  - Edge

    Edge is the connecting link between two vertces

  - Degree

    Degree is the number of edges connected to that vertex

  - Graph Order 

    Graph Order is the number of vertices

  - Graph Size

    Graph Size is the number of edges

  - Degree Sequence is the degree sequence written in an increasing order

- Subgraphs

  - Vertex Set

    Vertex Set is the set of all vertices

  - Edge Set

    Edge Set is the set of all edges

  - Subgraph G of graph P, is a graph that is constructed from the subset of the vertices and edges

- Graph Isomorphism

- Graphs Isomorphism occurs when two graphs contain the same amount of vertices connected in the same way(being alike)

- Complement Graph

  - The Complement of Graph G is Graph P such that two vertices that weren't adjacent in G are adjacent in P.
  - adjacent: Two verices are adjacent if they are connected by an edge
  - Self Complementary Graphs are graphs that are isomorphic to their complemrnt graphs

- Multigraphs

  - Multigraph is a graph with vertices connected with more than one edge.
  - ex. More than one way connected two cities.

- Matrix Representation

  - Two vertices are adjacent if they are connected by an edge
  - Two edges are incident if they share a vertex
  - Matrix Type
    - Adjacency Matrix (Edge table)
    - Incidence Matrix ( node and edge relation in matrix，but it not intuition! )

- Walks, Trails and Paths

  - Walk is a way of getting from one vertex to another and it consists of an alternating sequence of vertices and edges.
  - A walk is said to be open if the starting and ending vertices are different, and is called closed if the starting and ending vertices are the same
  - Trail is a walk such that all the edges are distinct
  - a trail that is closed is called a tour
  - Path is a walk such that all the edges and vertices are distinct
  - We can have Path such that it starts and ends at the same vertices and it is called a cycle
  - Self Avoiding Walk (SAW) is a sequence of moves that doesn't visit the same point twice
  - Length of a Walk, Trail or a Path is the number of edges of the Walk, Trail or the Path.

- Distance

  - Distance between one vertex and another is the number of edges in the shortest path.
  - Diameter is the greatest distance(path) between any pair of vertices.

- Connectedness

  - A graph is said to be connected if there is a path between every pair of vertices, otherwise, it is disconnected.
  - Cut is partition of the vertices into two disjoint sets.
    - Cut-Edge is the edge(s) whose removal results in disconnection
    - Cut-vertex is the vertex(s) whose removal results in disconnection
  - For any two vertices U&V, U-V Separating Set is the set of verices, whose removal results in disconnection between U & V.
  - Minimum U-V Separating Sets is a separating set with minimum cardinality.

- Menger's theorem

  - the minimum separating set is a subset of the separating set of the separating set
  - Internally Disjoint Paths are that have the same starting and ending vertices, and don't share any vertices in between
  - Menger'sTheorem the minimum number of vertices in a u-v separating set quuals the maximum number of internally disjoint u-v paths in G.
  - Basically says the minimum of vertices in a separate thing set equals the maximum number of internally distrain pass. And it can be used as a tool in order for us to find maximum number of intelli disjoint paths for a minimum separating set or to find a minimum separating set from maximum number of internally disjoint path for grads that are coplicated.

- Sum of Degrees of Vertices Theorem

  - Sum of Degrees of Vertices Theorem states that for any graph, the sum of the degrees of the vertices equals twice the number of edges.
  - the degree of a vertex is how many edges connected to that particular vertex.

### Graph Types

- Trivial, Null and Simple Graphs
  - Null Graph is a graph with no edges.
  - Trivial Graph is a graph with only one vertex.
  - Simple Graph is a graph with no loops and no parallel edges.
    - loop is a an edge that connects a vertex to itself.
    - Parallel Edges are two or more edges that connect two vertices.
    - In a Simple Graph with n vertices, the degree of every vertex is at most n-1

- Regular, Complete and Weighted Graphs
  - Regular Graph is a graph in which each vertex have the same degree.
  - Complete Graph is a graph in whichvertices have edges with all the other vertices(except themselves)
  - Weighted Graph is a graph such that its edges and vertices are given numerical value

- Directed, Undirected and Mixed Graphs
  - Directed Graph(digraph) is a graph with edges having directions.
    - Indegree(deg-) is the number of head(s) adjacent to a vertex
    - Outdegree(deg+) is the number of tails adjacent to a vertex
    - A vertex with deg- = 0 is called a source
    - A vertex with deg+ = 0 is called a Sink
    - If for every vertex deg+ = deg- then the graph is called a balanced graph
  - Undirected Graph is a graph with edges having no directions.
  - Mixed Graph is a graph with directed and undirected edges.

- Cycle, Path, Wheel and Lolipop Graphs
  - Cycle Graph($C_n$) is a graph such that each vertex is degree two.
  - Cycle in a Graph is a closed path
  - Girth of a graph is the length of a shortest cycle
  - Path Graph is a graph with two vertices having drgree 1, and the rest having degree 2.
  - Wheel Graph is a graph formed by connecting a single vertex to all the vertices of a cycle graph.
  - The number of the cycles in a wheel graphs is given by $n^2 - 3n + 3$
  - Lollipop Graph is graph constructed form joining a complete graph with a path graph
- Planar, Cubic and Random Graphs
  - 

- Ladder and Prism Graphs

- Web and Signed Graphs

- Peterson Graphs

- Bipartile Graphs

- Platonic Graphs

### Graph Operations

Introduction

Vertex Addition and Deletion

Edge Addittion and Deletion

Vertex Contraction and Edge Contraction

Graph Minor and Graph Transpose

Line Graphs

Dual Graphs

Graph Power

Y - $\Delta$Transform

Graph Join and Graph Product

Hajos Construction

Graph Union and Graph Intersection

Series - Parallel Composition

### Graph Coloring

Introduction

Vertex Coloring

Edge Coloring

Chromatic Polynomial

Total and List Coloring

Rainbow Coloring

Vizing's Theorem

Four Color Theorem



### Paths

Introduction

The Konigsberg Bridge Problem

Euler Paths and Circuits

Flwury's Algorithm

Hamiltoniann Paths and Circuits

Hamitonian Decomposition

Ore's Theorem

Dirac's Theorem

Shortest Path Problem

Five Room Puzzle

Knight's Tour

### Trees

Introduction

Trees

Tree Types

Rooted Trees

Tree Structures

Binary Trees

Spanning Trees

Binary Expression Trees

Tree Traversal

Forests

### Graph Match

Introduction

Graph Match

Hosoya Index

Berge's Lemma

Vertex and Edge Cover

Konig Theorem

### Networks

### Centrality

A large volume of research on networks has been devoted to the concept of **centrality**. This research addresses the question: *Which are the most important or central vertices in a network?* There are of course many possible definitions of importance, and correspondingly many centrality measures for nodes in a network.

Once the centrality of nodes have been determined, it is possible to make a ranking of the nodes according to their centrality scores. For instance, one might want to obtain the most important Web pages about a certain topic or the most important academic papers covering a given issue. Moreover, one might be interested to know which are the nodes whose removal from the networks would have the most important consequences in the network structure. A network property that is directly influenced by the removal of nodes is connectivity. For instance, which are the Internet routers whose failure would mostly damage the network connectivity?

#### Degree

- Degree is a simple centrality measure that counts how many neighbors a node has. If the network is directed, we have two versions of the measure: in-degree is the number of in-coming links, or the number of predecessor nodes; out-degree is the number of out-going links, or the number of successor nodes. 

- Typically, we are interested in in-degree, since in-links are given by other nodes in the network, while out-links are determined by the node itself. Degree centrality thesis reads as follows:

  > *A node is important if it has many neighbors, or, in the directed case, if there are many other nodes that link to it, or if it links to many other nodes.*

#### Eigenvector

- A natural extension of degree centrality is **eigenvector centrality**. In-degree centrality awards one centrality point for every link a node receives. But not all vertices are equivalent: some are more relevant than others, and, reasonably, endorsements from important nodes count more. The eigenvector centrality thesis reads:

  > *A node is important if it is linked to by other important nodes.*

- Eigenvector centrality differs from in-degree centrality: a node receiving many links does not necessarily have a high eigenvector centrality (it might be that all linkers have low or null eigenvector centrality). Moreover, a node with high eigenvector centrality is not necessarily highly linked (the node might have few but important linkers).

- Eigenvector centrality, regarded as a ranking measure, is a remarkably old method. Early pioneers of this technique are Wassily W. Leontief (The Structure of American Economy, 1919-1929. Harvard University Press, 1941) and John R. Seeley (The net of reciprocal influence: A problem in treating sociometric data. The Canadian Journal of Psychology, 1949).

- **Methodology**

  - A top down approach that seeks to maximize modularity. It concerns decomposing a modularity matrix.

- **Evaluation**

  - More accurate than fast greedy
  - Slower than fast greedy
  - Limitation: not stable on degenerated graphs (might not work!)

#### Katz

- A practical problem with eigenvector centrality is that it works well only if the graph is (strongly) connected. Real undirected networks typically have a large connected component, of size proportional to the network size. However, real directed networks do not. If a directed network is not strongly connected, only vertices that are in strongly connected components or in the out-component of such components can have non-zero eigenvector centrality. The other vertices, such as those in the in-components of strongly connected components, all have, with little justification, null centrality. This happens because nodes with no incoming edges have, by definition, a null eigenvector centrality score, and so have nodes that are pointed to by only nodes with a null centrality score.

- A way to work around this problem is to give each node a small amount of centrality for free, regardless of the position of the vertex in the network. Hence, each node has a minimum, positive amount of centrality that it can transfer to other nodes by referring to them. In particular, the centrality of nodes that are never referred to is exactly this minimum positive amount, while linked nodes have higher centrality. It follows that highly linked nodes have high centrality, regardless of the centrality of the linkers. However, nodes that receive few links may still have high centrality if the linkers have large centrality. This method has been proposed by Leo Katz (A new status index derived from sociometric analysis. Psychometrika, 1953) and later refined by Charles H. Hubbell (An input-output approach to clique identification. Sociometry, 1965). The Katz centrality thesis is then:

  > *A node is important if it is linked from other important nodes or if it is highly linked.*

#### PageRank

- A potential problem with Katz centrality is the following: if a node with high centrality links many others then all those others get high centrality. In many cases, however, it means less if a node is only one among many to be linked. The centrality gained by virtue of receiving a link from an important node should be diluted if the important vertex is very magnanimous with endorsements.

- PageRank is an adjustment of Katz centrality that takes into consideration this issue. There are three distinct factors that determine the PageRank of a node: 

- (i) the number of links it receives, 

  - (ii) the link propensity of the linkers, and 
  - (iii) the centrality of the linkers. 

- The first factor is not surprising: the more links a node attracts, the more important it is perceived. Reasonably, the value of the endorsement depreciates proportionally to the number of links given out by the endorsing node: links coming from parsimonious nodes are worthier than those emanated by spendthrift ones. 

- Finally, not all nodes are created equal: links from important vertices are more valuable than those from obscure ones. This method has been coined (and patented) by Sergey Brin and Larry Page (The anatomy of a large-scale hypertextual web search engine. Computer networks and ISDN systems, 1998). The PageRank thesis might be summarized as follows:

  > *A node is important if it linked from other important and link parsimonious nodes or if it is highly linked.*

#### Kleinberg

- So far, a node is important if it contains valuable content and hence receives many links from other important sources. Nodes with no incoming links cumulate, in the best case, only a minimum amount of centrality, regardless of how many other useful information sources they reference. One can argue that a node is important also because it links to other important vertices. For instance, a review paper may refer to other authoritative sources: it is important because it tells us where to find trustworthy information. Thus, there are now two types of central nodes: authorities, that contain reliable information on the topic of interest, and hubs, that tell us where to find authoritative information. A node may be both an authority and a hub: for instance, a review paper may be highly cited because it contains useful content and it may as well cite other useful sources. This method has been conceived by Jon M. Kleinberg (Authoritative sources in a hyperlinked environment. In ACM-SIAM Symposium on Discrete Algorithms, 1998). The Kleinberg centrality thesis reads:

  > *A node is an authority if it is linked to by hubs; it is a hub if it links to authorities.*

- Kleinberg centrality is an elegant way to avoid the problem of ordinary eigenvector centrality on directed networks, that nodes outside strongly connected components or their out-components get null centrality. However, we can still add to Kleinberg centrality an exogenous, possibly personalized, factor (as in the Katz method) or normalize vertex centralities by the out-degrees of vertices that point to them (as in the PageRank method).

#### Closeness

- The average distance from a given starting node to all other nodes in the network.

- Closeness centrality differs from either degree or eigenvector centrality. For instance, consider a node A connected to a single other node B. Imagine that node B is very close to the other nodes in the graph, hence it has a large closeness score. It follows that node A has a relatively large closeness as well, since A can reach all the nodes that B reaches in only one additional step with respect to B. However, A has degree only 1, and its eigenvector score might not be impressive.

- The largest possible value for the mean geodesic distance occurs for a root of a **chain graph**, a graph where the $n$ nodes form a linear sequence or chain, where the roots are the two ends of the chain. Using $n-1$ at the denominator of the mean, the mean distance from one root to the other vertices is

  > $\displaystyle{\frac{1}{n-1} \sum_{i=1}^{n-1} i = \frac{(n-1) n}{2 (n-1)} = \frac{n}{2}}$

- The minimum possible value for the mean geodesic distance occurs for the central node of a **star graph**, a network composed of a vertex attached to $n-1$ other vertices, whose only connection is with the central node. The central node reaches all other nodes in one step and it reaches itself in zero steps, hence the mean distance is $(n-1)/ (n-1) = 1$.

- There are some issue that deserve to be discussed about closeness centrality. One issue is that its values tend to span a rather **small dynamic range** from smallest to largest. Indeed, geodesic distances between vertices in most networks tend to be small, the typical and largest geodesic distances increasing only **logarithmically** with the size of the network. Hence the dynamic range of values of $l_i$ (and hence of $C_i$) is small.

- This means that in practice it is difficult to distinguish between central and less central vertices using this measure. Moreover, even small fluctuations in the structure of the network can change the order of the values substantially. For instance, for the actor network, the network of who has appeared in films with who else, constructed from the [Internet Movie Database](http://www.imdb.com/), we find that in the largest component of the network, which includes 98% of all actors, the smallest mean distance $l_i$ of any actor is 2.4138 for the actor Christopher Lee (*The Lord of the Rings*), while the largest is 8.6681 for an Iranian actress named Leia Zanganeh. The ratio of the two is just 3.6 and about half a million actors lie in between. The second best centrality score belongs to Donald Pleasence, who scores 2.4164, very close to the winner. Other centrality measures typically don't suffer from this problem because they have a wider dynamic range and the centrality values, in particular those of the leaders, are widely separated.

- A second issue about closeness centrality is that geodesic distance between two vertices is infinite if the vertices are not reachable through a path, hence $l_i$ is **infinite** and $C_i$ is zero for all vertices $i$ that do not reach all other vertices in the graph. The most common solution is to average the distance over only the reachable nodes. This gives us a finite measure, but distances tend to be smaller, hence closeness centrality scores tend to be larger, for nodes in small components. This is usually undesirable, since in most cases vertices in small components are considered less important than those in large components.

- A second solution is to assign a distance equal to the number of nodes of the graph to pairs of vertices that are not connected by a path. This again will give a finite value for the closeness formula. Notice that the longest path in a graph with $n$ nodes has length $n-1$ edges, hence the assigned distance to unreachable nodes is one plus the maximum possible distance in the network.

- A third solution is to compute closeness centrality only for those vertices in the largest connected component of the network. This is a good solution when the largest component is a giant one that covers the majority of the nodes.

- A fourth (and last) solution is to redefine closeness centrality in terms of the **harmonic mean distance** between vertices, that is the average of the inverse distances:

  > $\displaystyle{C_i = \frac{1}{n-1} \sum_{j \neq i} \frac{1}{d_{i,j}}}$

- Notice that we excluded from the sum the case $j = i$ since $d_{i,i} = 0$ and this would make the sum infinite. Moreover, if $d_{i,j} = \infty$ because $i$ and $j$ are not reachable, then the corresponding term of the sum is simply zero and drops out.
  $$
  C(x) = -\frac{N}{\sum_y d(x,y)}
  $$

- 定義：該節點到所有節點之最短距離總和之倒數，值越大代表越容易觸及其他節點

  - 強調點在網絡的價值，值越大表示越位在中心

- 意涵：該節點觸及網絡內所有節點的程度

  - 如人際關係，接近中心性越高，個性越合群

  - 病毒傳播，接近中心性越高，越容易感染

#### Betweeness

- Measure how often a node appears on shortest paths between nodes in the network.

- 在節點對之間的最短途徑通過單一節點的頻率，即多數節點須通過該節點以觸及其他節點

- $$
  C_B(i) = \sum_{j<k}g_{jk}(i)/g_{jk}
  $$

  - $g_{jk}(i)$ = jk兩節點之間通過$i$的最短捷徑次數
  - $g_{jk}$ = 連結 j k兩節點的最短捷徑個數

- 意涵：節點的中心程度或在網絡的重要性+該節點與其他節點的關係

  - 掌握較多的資源
  - 掌握較大的權力，可以過濾資訊
  - 控制網絡流動，影響節點彼此的連結，或阻礙網絡運作
  - 表明这个节点在网络中的重要程度，他是不是一个交流枢纽，例如紧密中心性里面。7和8就具有很高的介数中心性。如果把7或者8从图中拿走，会造成图的分裂。
  - 強調點在其他點之間的調節能力，控制能力指數

#### Harmonic Centrality

- 整體概念與接近中心性相同
- 節點不相連，距離無限大->接近中心性=0
- 故先取倒數再相加

#### Prestige

- measure the direction is important property of the relation

- for directed network only

- prestigious actor is the object of extensive ties as a recipient(has input degrees)

- 如很多人會想跟女神成為朋友，但女神不一定想和每個人都成為朋友

- 凸顯actor本身的重要性而非中心性

  - Indegree prestige

    - number of directly connected neighbours considering indegrees

    - $$
      P_{ID}(i) = \frac{d_I(i)}{n-1}
      $$

    - 其他節點指向i節點的次數 

  - Domain Prestige

    - Share of total nodes which can reach a node.

    - $$
      P_D(i)=\frac{ \mid I_i \mid}{(n-1)}
      $$

      網絡內哪一些節點可以直接或間接地指到v節點

  - Proximity Prestige

    - Considers directly and indirectly linked nodes and path-lengths
    - 路徑長度用來加權，越短路徑越有價值

  - Rank Prestige

    - Considers specified prominesce value from in-degree nodes.



#### Average Cluster Coefficient(AvgCC)

- the tendency for who share connection in a social network to become connected
- How can we measure the prevalence of triadic closure in a network?
  - Local Clustering coefficient of a node
  - Global Cluster Coefficient

#### Eccentricity

- The distance from a given starting node to the farthest node from it in the network.
- 離網絡中最遠的點的距離，數字越小表示越處在網絡中心

### Similarity

### Communties

- For directed graph: go with **Info Map**. Else, pls continue to read.
- If compuational resources is not a big problem, and the graph is < 700 vertices & 3500 edges, go with **Edge Betweenness**; it yields the best result.
- If cares about modularity, any of the remaining algorithms will apply;
  - If the graph is particularly small: < 100 vertices, then go with **optimal modularity**;
  - If you want a first try-on algorithm, go with **fast greedy** or **walktrap**
  - If the graph is bigger than 100 vertices and not a de-generated graph, and you want something more accurate than fast greedy or walktrap, go with **leading eigenvectors**
  - If you are looking for a solution that is similar to K-means clustering, then go for **Spinglass**

#### Edge betweenness

- **Definition of edge betweenness**:

  > Number of shortest path that passes the edge.

  - It’s not difficult to imagin that, if there is an edge that connects two different groups, then that edge will has to be passed through multiple times when we count the shortest path. Therefore, by removing the edge that contains with the highest number of shortest path, we are disconnecting two groups.

- **Methodology**

  - Top down hierarchical decomposition process.

- **Evalution**

  - Generally this approach gives the most satisfying results from my experience.
  - Pretty slow method. The computation for edge betweenness is pretty complex, and it will have to be computed again after removing each edge. Suitable for graph with less than 700 vertices and 3500 edges.
  - It produces a dendrogram with no reminder to choose the appropriate number of communities. (But for IGraph it does a function that output the optimal count for a dendrogram).

- 是一种层次分解过程，其中边缘以其边缘中介度分数（即通过给定边缘的最短路径的数量）的递减顺序被移除。这是因为连接不同组的边缘更可能包含在多个最短路径中，这仅仅是因为在许多情况下它们是从一个组到另一个组的唯一选择。这种方法产生了良好的结果，但由于边缘中介性计算的计算复杂性，并且因为在每次去除边缘之后必须重新计算中间性得分，所以该方法非常慢。具有~700个顶点和~3500个边缘的图形大约是可以使用此方法分析的图形的上限大小。另一个缺点是`Edge.betweenness.community`构建了一个完整的树形图，并没有给出任何关于在何处削减树形图以获得最终组的指导，因此您将不得不使用其他一些措施来决定（例如，分区的模块化分数）在树形图的每个级别）

- The edge betweenness score of an edge measures the number of shortest paths through it, see edge_betweenness for details. The idea of the edge betweenness based community structure detection is that it is likely that edges connecting separate modules have high edge betweenness as all the shortest paths from one module to another must traverse through them. So if we gradually remove the edge with the highest edge betweenness score we will get a hierarchical map, a rooted tree, called a dendrogram of the graph. The leafs of the tree are the individual vertices and the root of the tree represents the whole graph. 

- cluster_edge_betweenness performs this algorithm by calculating the edge betweenness of the graph, removing the edge with the highest edge betweenness score, then recalculating edge betweenness of the edges and again removing the one with the highest score, etc. 

- edge.betweeness.community returns various information collected throught the run of the algorithm. See the return value down here.

#### Fastgreedy

- **Methodology**:
  - Bottom up hierarchical decomposition process. It will merge two current communities iteratively, with the goal to achieve the maximum modularity gain at local optimal.
- **Evaluation**:
  - Pretty fast, can merge a sparse graph at linear time.
  - Resolution limit: when the network is large enough, small communities tend to be combined even if they are well-shaped.
- This function tries to find dense subgraph, also called communities in graphs via directly optimizing a modularity score.
- This function implements the fast greedy modularity optimization algorithm for finding community structure, see A Clauset, MEJ Newman, C Moore: Finding community structure in very large networks.

是另一种分层方法，但它是自下而上而不是自上而下。它试图以贪婪的方式优化称为模块化的质量功能。最初，每个顶点属于一个单独的社区，并且迭代地合并社区，使得每个合并是局部最优的（即，产生模块化的当前值的最大增加）。当不可能再增加模块性时，算法停止，因此它为您提供分组和树形图。该方法很快，并且它通常作为第一近似尝试的方法，因为它没有要调整的参数。然而，已知具有分辨率限制，即低于给定大小阈值的社区（取决于节点和边缘的数量，如果我没记错的话）将始终与邻近社区合并。

#### Infomap

- **Methodology**:

  > It is based on information theoretic principles; it tries to build a grouping which provides the shortest description length for a random walk on the graph, where the description length is measured by the expected number of bits per vertex required to encode the path of a random walk.

- **Evaluation**:

  - Used for directed graph analytics

- Find community structure that minimizes the expected description length of a random walker trajectory

- The original paper: M. Rosvall and C. T. Bergstrom, Maps of information flow reveal community structure in complex networks, PNAS 105, 1118 (2008) http://dx.doi.org/10.1073/pnas. 0706851105, http://arxiv.org/abs/0707.0609 A more detailed paper: M. Rosvall, D. Axelsson, and C. T. Bergstrom, The map equation, Eur. Phys. J. Special Topics 178, 13 (2009). http://dx.doi.org/10.1140/epjst/e2010-01179-1, http://arxiv.org/abs/0906.1405.

#### Label propagation

- **Methodology**
  - A bit like k-clustering, with initialization k different points. It uses an iterative method (again just like k-means): the target label will be assigned with the most “vote” of the lables from its neighbors; until the current label is the most frequent label.
- **Evaluation**
  - Very fast
  - Like K-Means, random initialization yields different results. Therefore have to run multiple times (suggested 1000+) to achieve a consensus clustering.
- This is a fast, nearly linear time algorithm for detecting community structure in networks. In works by labeling the vertices with unique labels and then updating the labels by majority voting in the neighborhood of the vertex.
- This function implements the community detection method described in: Raghavan, U.N. and Albert, R. and Kumara, S.: Near linear time algorithm to detect community structures in large-scale networks. Phys Rev E 76, 036106. (2007). This version extends the original method by the ability to take edge weights into consideration and also by allowing some labels to be fixed. 
- From the abstract of the paper: “In our algorithm every node is initialized with a unique label and at every step each node adopts the label that most of its neighbors currently have. In this iterative process densely connected groups of nodes form a consensus on a unique label to form communities.”
- 是一种简单的方法，其中每个节点都分配了一个 *k* labels。该方法然后迭代地进行并且以每个节点以同步方式获取其邻居的最频繁标签的方式将标签重新分配给节点。当每个节点的标签是其邻域中最频繁的标签之一时，该方法停止。它非常快，但根据初始配置（随机决定）产生不同的结果，因此应该多次运行该方法（例如，图表的1000次），然后建立共识标签，这可能是乏味。

#### Leading eigenvector

- This function tries to find densely connected subgraphs in a graph by calculating the leading nonnegative eigenvector of the modularity matrix of the graph.
- The function documented in these section implements the ‘leading eigenvector’ method developed by Mark Newman, see the reference below. The heart of the method is the definition of the modularity matrix, B, which is B=A-P, A being the adjacency matrix of the (undirected) network, and P contains the probability that certain edges are present according to the ‘configuration model’. In other words, a P[i,j] element of P is the probability that there is an edge between vertices i and j in a random network in which the degrees of all vertices are the same as in the input graph. 
- The leading eigenvector method works by calculating the eigenvector of the modularity matrix for the largest positive eigenvalue and then separating vertices into two community based on the sign of the corresponding element in the eigenvector. If all elements in the eigenvector are of the same sign that means that the network has no underlying comuunity structure. Check Newman’s paper to understand why this is a good method for detecting community structure.

#### Multi-level(Louvain)

- **Methodology**
  - Similar to fast greedy, just that nodes are not combined, they move around communities to make dicision if they will contribute to the modularity score if they stay.
- This function implements the multi-level modularity optimization algorithm for finding community structure, see references below. It is based on the modularity measure and a hierarchial approach.
- This function implements the multi-level modularity optimization algorithm for finding community structure, see VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of community hierarchies in large networks, http://arxiv.org/abs/arXiv:0803.0476 for the details. 
- It is based on the modularity measure and a hierarchial approach. Initially, each vertex is assigned to a community on its own. In every step, vertices are re-assigned to communities in a local, greedy way: each vertex is moved to the community with which it achieves the highest contribution to modularity. When no vertices can be reassigned, each community is considered a vertex on its own, and the process starts again with the merged communities. The process stops when there is only a single vertex left or when the modularity cannot be increased any more in a step.

#### Optimal

- **Definition of modularity**
  - Modularity compares the number of edges inside a cluster with the expected number of edges that one would find in the cluster if the network were a random network with the same number of nodes and where each node keeps its degree, but edges are otherwise randomly attached.
  - Modularity is a measure of the segmentation of a network into partitions. The higher the modularity, the denser in-group connections are and the sparser the inter-group connections are.
- **Methodology**
  - GNU linear programming kit.
- **Evaluation**:
  - Better for smaller communities with less than 100 vertices for the reasons of implementation choice.
  - Resolution limit: when the network is large enough, small communities tend to be combined even if they are well-shaped.
- This function calculates the optimal community structure of a graph, by maximizing the modularity measure over all possible partitions.
- This function calculates the optimal community structure for a graph, in terms of maximal modularity score. 
- The calculation is done by transforming the modularity maximization into an integer programming problem, and then calling the GLPK library to solve that. Please the reference below for details. Note that modularity optimization is an NP-complete problem, and all known algorithms for it have exponential time complexity. This means that you probably don’t want to run this function on larger graphs. Graphs with up to fifty vertices should be fine, graphs with a couple of hundred vertices might be possible.

#### Spinglass

- This function tries to find communities in graphs via a spin-glass model and simulated annealing.
- This function tries to find communities in a graph. A community is a set of nodes with many edges inside the community and few edges between outside it (i.e. between the community itself and the rest of the graph.) 
- This idea is reversed for edges having a negative weight, ie. few negative edges inside a community and many negative edges between communities. Note that only the ‘neg’ implementation supports negative edge weights. 
- The spinglass.cummunity function can solve two problems related to community detection. If the vertex argument is not given (or it is NULL), then the regular community detection problem is solved (approximately), i.e. partitioning the vertices into communities, by optimizing the an energy function. 
- If the vertex argument is given and it is not NULL, then it must be a vertex id, and the same energy function is used to find the community of the the given vertex. 



#### Walktrap

- **Methodology**
  - Similar to fast greedy. It is believed that when we walk some random steps, it is large likely that we are still in the same community as where we were before. This method firstly performs a random walk 3-4-5, and merge using modularity with methods similar to fast greedy.
- **Evaluation**:
  - A bit slower than fast greedy;
  - A bit more accurate than fast greedy.
- This function tries to find densely connected subgraphs, also called communities in a graph via random walks. The idea is that short random walks tend to stay in the same community.
- This function is the implementation of the Walktrap community finding algorithm, see Pascal Pons, Matthieu Latapy: Computing communities in large networks using random walks, http://arxiv.org/abs/physics/0512106

是一种基于随机游走的方法。一般的想法是，如果您在图表上执行随机游走，那么步行更有可能保持在同一社区内，因为在给定社区之外只有少数边缘。 Walktrap运行3-4-5步骤的短随机游走（取决于其中一个参数），并使用这些随机游走的结果以自下而上的方式合并单独的社区，如`fastgreedy.community`。同样，您可以使用模块化分数来选择剪切树形图的位置。它比快速贪婪方法慢一点，但也更准确一些（根据原始出版物）。



### Layout

#### bipartite

- Minimize edge-crossings in a simple two-row (or column) layout for bipartite graphs
- The layout is created by first placing the vertices in two rows, according to their types. Then the positions within the rows are optimized to minimize edge crossings, using the Sugiyama algorithm (see layout_with_sugiyama).

#### star

- A simple layout generator, that places one vertex in the center of a circle and the rest of the vertices equidistantly on the perimeter.
- It is possible to choose the vertex that will be in the center, and the order of the vertices can be also given.

#### tree

- A tree-like layout, it is perfect for trees, acceptable for graphs with not too many cycles.
- Arranges the nodes in a tree where the given node is used as the root. The tree is directed downwards and the parents are centered above its children. For the exact algorithm, the refernce below. If the given graph is not a tree, a breadth-first search is executed first to obtain a possible spanning tree.

#### circle

- Place vertices on a circle, in the order of their vertex ids.
- If you want to order the vertices differently, then permute them using the permute function.

#### nicely

- This function tries to choose an appropriate graph layout algorithm for the graph, automatically, based on a simple algorithm. See details below.
- layout_nicely tries to choose an appropriate layout function for the supplied graph, and uses that to generate the layout. The current implementation works like this: 
  1. If the graph has a graph attribute called ‘layout’, then this is used. If this attribute is an R function, then it is called, with the graph and any other extra arguments. 
  2. Otherwise, if the graph has vertex attributes called ‘x’ and ‘y’, then these are used as coordinates. If the graph has an additional ‘z’ vertex attribute, that is also used. 
  3. Otherwise, if the graph is connected and has less than 1000 vertices, the Fruchterman-Reingold layout is used, by calling layout_with_fr. 
  4. Otherwise the DrL layout is used, layout_with_drl is called.

#### grid

- This layout places vertices on a rectangulat grid, in two or three dimensions.
- The function places the vertices on a simple rectangular grid, one after the other. If you want to change the order of the vertices, then see the permute function.

#### sphere

- Place vertices on a sphere, approximately uniformly, in the order of their vertex id.
- layout_on_sphere places the vertices (approximately) uniformly on the surface of a sphere, this is thus a 3d layout. It is not clear however what “uniformly on a sphere” means. If you want to order the vertices differently, then permute them using the permute function

#### randomly

- This function uniformly randomly places the vertices of the graph in two or three dimensions.
- Randomly places vertices on a [-1,1] square (in 2d) or in a cube (in 3d). It is probably a useless layout, but it can use as a starting point for other layout generators.

#### Davidson-Harel layout

- Place vertices of a graph on the plane, according to the simulated annealing algorithm by Davidson and Harel.

#### DrL

- DrL is a force-directed graph layout toolbox focused on real-world large-scale graphs, developed by Shawn Martin and colleagues at Sandia National Laboratories.

#### Fruchterman-Reingold layout

- Place vertices on the plane using the force-directed layout algorithm by Fruchterman and Reingold.
- See the referenced paper below for the details of the algorithm. 
- Ref
  - Fruchterman, T.M.J. and Reingold, E.M. (1991). Graph Drawing by Force-directed Placement. Software - Practice and Experience, 21(11):1129-1164.

#### GEM

- Place vertices on the plane using the GEM force-directed layout algorithm
- Ref
  - Arne Frick, Andreas Ludwig, Heiko Mehldau: A Fast Adaptive Layout Algorithm for Undirected Graphs, Proc. Graph Drawing 1994, LNCS 894, pp. 388-403, 1995.

#### graphopt

- A force-directed layout algorithm, that scales relatively well to large graphs.
- layout_with_graphopt is a port of the graphopt layout algorithm by Michael Schmuhl. graphopt version 0.4.1 was rewritten in C and the support for layers was removed (might be added later) and a code was a bit reorganized to avoid some unneccessary steps is the node charge (see below) is zero. 
- graphopt uses physical analogies for defining attracting and repelling forces among the vertices and then the physical system is simulated until it reaches an equilibrium. (There is no simulated annealing or anything like that, so a stable fixed point is not guaranteed.) 
- See also http://www.schmuhl.org/graphopt/ for the original graphopt

#### Kamada-Kawai

- Place the vertices on the plane, or in the 3d space, based on a phyisical model of springs.
- Ref
  - Kamada, T. and Kawai, S.: An Algorithm for Drawing General Undirected Graphs. Information Processing Letters, 31/1, 7–15, 1989.

#### Large Graph

- A layout generator for larger graphs.

#### multidimensional scaling

- Multidimensional scaling of some distance matrix defined on the vertices of a graph.
- layout_with_mds uses metric multidimensional scaling for generating the coordinates. Multidimensional scaling aims to place points from a higher dimensional space in a (typically) 2 dimensional plane, so that the distance between the points are kept as much as this is possible. 
- By default igraph uses the shortest path matrix as the distances between the nodes, but the user can override this via the dist argument. 
- This function generates the layout separately for each graph component and then merges them via merge_coords.

#### Sugiyama graph

- Sugiyama layout algorithm for layered directed acyclic graphs. The algorithm minimized edge crossings.
- This layout algorithm is designed for directed acyclic graphs where each vertex is assigned to a layer. Layers are indexed from zero, and vertices of the same layer will be placed on the same horizontal line. The X coordinates of vertices within each layer are decided by the heuristic proposed by Sugiyama et al. to minimize edge crossings. 
- You can also try to lay out undirected graphs, graphs containing cycles, or graphs without an a priori layered assignment with this algorithm. igraph will try to eliminate cycles and assign vertices to layers, but there is no guarantee on the quality of the layout in such cases. 
- The Sugiyama layout may introduce “bends” on the edges in order to obtain a visually more pleasing layout. This is achieved by adding dummy nodes to edges spanning more than one layer. The resulting layout assigns coordinates not only to the nodes of the original graph but also to the dummy nodes. The layout algorithm will also return the extended graph with the dummy nodes.

- Ref
  - K. Sugiyama, S. Tagawa and M. Toda, "Methods for Visual Understanding of Hierarchical Systems". IEEE Transactions on Systems, Man and Cybernetics 11(2):109-125, 1981.



## Packages

- NetworkX
- igraph
- [graph-tool](https://graph-tool.skewed.de/performance)

- Ref
  - [用python分析《三国演义》中的社交网络.ipynb](https://github.com/blmoistawinde/hello_world/blob/master/sanguo_network/用python分析《三国演义》中的社交网络.ipynb)
  - [PageRank 簡介](http://jpndbs.lib.ntu.edu.tw/DB/PageRank.pdf)
  - [python与图论的桥梁——igraph](https://zhuanlan.zhihu.com/p/97135627)
  - [社区网络分析学习笔记 —— 算法实现及 igraph 介绍](https://zhuanlan.zhihu.com/p/40227203)
  - [PageRank算法原理与实现](https://zhuanlan.zhihu.com/p/86004363)
  - [基于社交网络分析算法（SNA）的反欺诈（一）](https://zhuanlan.zhihu.com/p/34405766)
  - [基于社交网络分析算法（SNA）的反欺诈（二）](https://zhuanlan.zhihu.com/p/34433105)
  - [基于社交网络分析算法（SNA）的反欺诈（三）](https://zhuanlan.zhihu.com/p/34436303)
  - [扒完社交网络关系才明白，《权力的游戏》凭什么是神作](https://zhuanlan.zhihu.com/p/28880958)
  - [谁是社会网络中最重要的人？](https://zhuanlan.zhihu.com/p/31198752)
  - [如何简单地理解中心度，什么是closeness、betweenness和degree？](https://www.zhihu.com/question/22610633/answer/493452601)
  - [Network Science](https://www.sci.unich.it/~francesc/teaching/network/)
  - [Why Social Network Analysis Is Important](https://blog.f-secure.com/why-social-network-analysis-is-important/)

In SNA， We focus on relations and structures.

- 緣起：從圖形理論(Graph Theory)發展的量化技巧

  - 量化演算法
  - 圖形視覺化

- SNA與傳統研究方法的比較

  - 傳統：收集解釋變數對於目標變數是否有影響與影響程度
  - SNA：研究節點與節點之間的關係，找出位在網絡核心的人。

- 應用

  - 了解整體社會網絡的全貌
    - 班級內的社交網絡，可以看到核心與邊陲的人物。
  - 節點與節點彼此的連結，來定義節點屬性。
    - 透過學術研究者彼此間的合作狀況、引用情形找出權威人物、著作
  - 依據節點屬性分群其子群代表的意義
    EX學術領域分析
  - 子群之間彼此的關係
    EX交流密切與否
  - 子群內節點的情形分析
    EX子群凝聚力，各子群內使用者特性
  - 子群結構的對應
    EX使用者相關推薦

- 基本概念

  - 社會網絡 = 節點(node) + 連結(edge)
  - node
    - 或稱 Actor/ Vertex，可以是個人、產品、事件或國家
  - edge
    - 或稱 Link / Arc，可以是喜歡、訊問、著作、父母
    - 可以依據有無方向性分為directed / unidirected
    - 可以依據連結的強度設定 Weigtht
  - 圖像基本定義
    - Graph G
      - a finite set of nodes， N(each node is unique)
      - a finite set of edges， E(each edge is unique)
    - Each edge is a pair (n1, n2) where n1, n2 $\in$ N
    - 無向圖
      - All edges are two-way.
      - Edges are unordered pairs.
      - 在matrix中的資料會對稱
    - 有向圖
      - All edges are one-way as indicated by the arrows.
      - Edges are orderes pairs.
      - 在matrix中的資料不會對稱
  - 名詞解釋
    - walk
      - 節點與連結皆可無限計算，如金錢流向
    - path
      - 節點與連結僅能計算一次，如病毒感染(感染過後就會產生抗體不會再次感染)
    - Trial
      - 每一條連結僅能計算一次(節點可重複計算)，如八卦訊息
    - Geodesics Distance
      - 任兩節點之間最短的 path，如包裹快遞
    - Main Component
      - The largest pair of nodes
    - Isolate
      - single node
    - Socail Network One Mode
      - 圖上的節點屬性單一
    - Social Network Two Mode
      - 圖上節點有多個屬性，如同時有老師、文章等等
    - Ego-Centered Network egonet
      - 以單一節點為中心去看周遭的節點

- 視覺化

  - 節點：大小/顏色
  - 邊：顏色深淺
  - Layout

- 常見問題

  - For a set N with n elements, how many possible edges there?
    - For undirected graphs, the number of pairs in N = $N = n \times (n-1)/2$ 
    - For directed graphs, the number of ordered pairs in N: $N = n^2 - n = n \times (n-1)$

- 網絡概觀

  - Average degree = edges / nodes

    - degree = indegree + outdegree
    - weugthed degree： edge*weight

  - - - 

    - Radius 半徑

      - minimum of eccentricity
  - eccentricity = radius -> central vertex

- Diameter 直徑

      - maximum of eccentricity
      - eccentricity = diameter -> periphery vertex

  - weigthed graph diameter the hops of a vertex to the farthest vertex 
    - if the network is disconnected-> the diameter is infinite(無限大)

- Average path length

  - the average number of steps along the shortest paths for all possible pairs of nodes.

        網絡内所有可能連結最短路徑加總的平均步數(連結最短路徑長度總和/連結數)

  It is a measure of the efficiency of information or mass transport on a network.

- Graph Density

  the total number of edges present in graph/the total numberof edges possible the graph

    

  

- 網絡與分群

  - Modularity(Louvain Modularity Detection)
    - Measure the fraction of the links that connect codes of the same type(with the same community) minus the excepted value of the same quantity with the same community divisions but random connection between nodes.
    - 坐落與group1或group2的連結數量減掉在隨機狀況坐落於group1或group2的期望連結數
    - 用來測量網絡的結構，分成幾群
    - A value of Q = 0 indicates that the community structure is no stronger than would be excepted by random chance and value other than zero represent deviations from randomness.
    - hagh modularity 代表同社群內節點有強烈關係，最大值為1，通常介於0.3-0.7之間
    - 限制：採用最佳化的方法來偵測網絡架構，當網絡大到一定程度時，無法偵測相對小的社群(即一般社群)
  - Girvan-Newman Clustering
    - 算法流程
      1. Compute betweenness centrality for each edge
      2. Remove edge with highest score
      3. Recalculate betweennesses for all edges affected by the removal
      4. Repeat from step 2 until no edges remain
  - Multi-Dimensional-Scaling
  - Lineage
    - 在分群裡面找出節點的關係

- filter

  透過篩選功能將整體網絡區分成子群或個人網絡，進一步觀察小網絡的特性

  - Attribute
    - Equal
    - Inter Edges：著重在分群後單一子群的內部鏈接
    - Intra Edges：用來找出各分群之間跨群的連接路徑
    - Non-null：隱藏圖像的缺值的節點與連結
    - Partion：選擇節點屬性以利觀察網絡內的子集
    - Partion Count：利用節點的次數來篩選要分析的子群體
    - Range：選擇連結的屬性範圍以利觀察網絡特定節點群
  - Edge
    - Edge Type:用來篩選不同種類的連結
    - Edge Weigth：利用連結權重篩選需要的連結
    - Mutual Edge：篩選單向連結，僅保留雙向連結
    - Self-Loop：移除節點的自我連結
  - operator
    - Intersection:篩選交叉節點
    - Mask(Edges)：篩選特定範圍連結
    - Not(Edges)：不顯示特定連結
    - Not(Nodes)不顯示特定節點
    - UNION：聯合設定多個篩選條件
  - Topology
    - Degree Range
    - In Degree Range
    - Out Degree Range
    - Ego Network:了解單一網絡與其他不同階層節點相連的可能途徑
      可以看1度、2度連結關係
    - Giant Component：顯示網絡內的最大的連通圖
    - Has Self-loop：找出有自換循環的節點和保留有相關的節點
    - K-brace(Only for undirected)
      - repreatedly deleting all edges of embeddedness less than k
      - deleting all single-node connected components
    - K-core
      - 子群體內每一個節點直接連結至少K個節點
      - K-core=1，消除網絡圖內的單一節點
      - K-core=2， 篩選後的網絡圖，任意節點至少與2個節點相連結
    - Mutual Degree Range
    - Neighbor Network：功能類似於Ego network
  - Dynamic& Saved queries
    - 針對動態圖做篩選

- Layout

  - Divisions_分類

    - Open Ord
      - 適用：無向+權重
      - 用途：用於分群
      - 運算方式：力引導+模擬退火法
      - 使用連結權重：否

  - Complementarities_互補

    - ForceAtlas

      - 適用：

        - 解讀真實資料，以最少偏誤地角度明確解讀網絡圖像
        - Random network models
          - a fixed number of nodes
          - an edge between any pair of vertices with a uniform probability p
        - Real networks are not exactly like thes
          - Nodes added to the network with time, it's evolving, instead of static
          - Tend to have a relatively few nodes of high connectivity(the "Hub" nodes)
          - Thanks to the hub nodes, the scale-free model has a shorter average path length than a random
          - No single node can be seen as "typical", no intrinsic "scale"

      - 用途：

        - 小世界網絡(Small-World)

        - 無尺度網絡圖(Scale-free networks)

          小的程度中心性的節點很多，大的程度中心性的節點很少，冪次法則

        - 使用連結權重：是

    - YiFanHu

      - 與 Force Atlas Layout 概念相同，透過優化整體網絡內交叉節點的互斥程度來運算網絡新圖像
      - 兩者差異
        - YiFan Hu Layout：只計算與節點相鄰的節點對(pair)
        - Force Atlas Layout：計算網絡內所有的節點對(pair)來運算
      - 使用連結權重：否
      - Barnes-Hut演算法：將距離夠近的物體整合在一起，一棵四叉樹是通過將平面不斷遞歸地劃分成4個小區域來構建的
      - YiFan Hu Proportional 的圖形節點的配置是以比例置換(proportional displacement)的方式進行，可以處理較大的網絡圖

    - Fruntchman-Reingold

      - 運算方式：彈簧模型+靜電力
        - 兩節點受彈力作用，過近的點會被彈開而過遠的點被拉近；通過不斷的反復運算，整個佈局達到動態平衡，趨於穩定。
        - 兩點之間加入靜電力的無力模型，通過計算系統的總能量並使得能量最小化，從而達到佈局的目的
      - 可運算節點範圍：1-1000
      - 使用連結權重：否

  - Ranking_排序

    - Circular
      - 將資料根據設定的屬性(ID，Degree)排列成圓形，觀察節點與節點之間的分佈
    - Dual Circle
      - 兩層的Circle，指定的節點顯示在外圈
    - Circle Pack
      - 可以指定用分群結果的變數來視覺化
    - Radial Axis
      - 計算方式：將節點依屬性分組，相似的節點分為一組並將其繪製成向外放射的軸線，以此類推
      - 適用目的：用於觀察各組之間的節點與連結的分佈，進而比較各組之間的同質性(homophily)

  - Geo Layout 地理佈局

    - 使用經緯度來定位節點

  - Multi-Dimensional Scaling 

    - 計算圖像節點的路徑距離來表示相似度結構分析，二維向量重疊程度來代表相似程度

  - Isometric ayout

    - 等軸測投影
    - 在二維平面呈現三維物體的方法，三條坐標軸的投影縮放比例相同，並且任意兩條坐標軸投影之間的角度都是120度

  - Network Splitter3D

    - 先用其他的layout方式佈局，再用 Newwork Splitter3D的layout視覺化

  - 

  - Others

- 場景

  - 犯罪嫌疑人資料梳理
  - 銀行異常金流
  - 擔保品分析
  - 上網行為異常
  - 商品推薦
  - 社交關係
  - 廣告推播
  - MOMO把折價券當廣告再發，aroung

- note: 拿到資料本身就是結構化的資料了

- 233萬的node，是請人

- 底層資料是文本還是用NER的技術從文本抽取

https://snap.stanford.edu/data/egonets-Facebook.html

http://www.csie.ntnu.edu.tw/~u91029/Graph.html

https://kknews.cc/news/eokvl6n.html

http://alrightchiu.github.io/SecondRound/graph-introjian-jie.html

[【Graph Embedding】LINE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56478167)

[【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56733145)

- Louvain 算法流程

  1. 將每個節點視為一個獨立群集

  2. 將單個節點重新分配最改善模組性的集群中，否則維持原樣，直到所有節點都不再需要重新分配
  3. 將第一階段的群集以獨立節點呈現，並將前一個群集的邊合併，以成為連結新節點的加權邊
  4. 重複2跟3，直到不再出現重新分配或合併的需求

  - 缺點是重要但小的群集可能被吸收了(如意見領袖、有錢人?)

