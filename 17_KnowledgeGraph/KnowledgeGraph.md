# Knowledge Graph

## 處理什麼問題(優點)

1. 高維數據
2. Embed knowledges to a vector space representation
3. 雖然 one hot encoding 也能夠表示知識，但當實體和關係太多時會導致維度災難 



## 方法

- 希望用三元組(head, relation and tail)來表達知識 
  - 如 sky tree, location, Tokyo
  - 可以做置換，產生負向的資料

[一文理解Ranking Loss/Margin Loss/Triplet Loss - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/158853633)



## Core idea

Embed head/tail entities and relation in the same vector space, such that
$$
head\_entity + relation = tail\_entity\\
$$


## 應用

- 連結預測
  - (?, r, t): 給導演(r)和驚魂(t)，希望知道導演(h)是誰
  - (h, r, ?): 給驚魂(h)和導演(r)，希望知道被哪個人導演
- 推薦系統
  - 協同過濾在用戶-物品的交互稀少時表現會不理想，可以透過混合的推薦系統來優化模型效度
- 問答系統



### 使用時機

## Loss Function

- [【译】理解 Ranking Loss，Contrastive Loss，Margin Loss，Triplet Loss，Hinge Loss 等易混淆的概念 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/355870299)
- [Loss function及regulation总结-2 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40394834)



## Data

### Input

- input.txt

- entity2id.txt
- relation2id.txt



### output

- embedded_entity.txt

- embedded_relation.txt



## Python Code



## 效度



## Trans家族

[知识表示学习Trans系列梳理(论文+代码) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/144412694)

### Trans-E(2013)

- 只適合處理一對一的關係，無法處理一對多，多對一或多對多的關係
- 有两个知识，(skytree, location, tokyo)和(gundam, location, tokyo)。经过训练，“sky tree”实体向量将非常接近“gundam”实体向量。但实际上它们没有这样的相似性。



### Trans-H(2014)

- TransH的目标是处理一对多/多对一/多对多关系，并且不增加模式的复杂性和训练难度。
- 其基本思想是将**关系**解释为超平面上的转换操作。每个关系都有两个向量，超平面的范数向量Wr和超平面上的平移向量(dr)。



### Trans-R(2015)

- Learning Entity and Relation Embeddings for Knowledge Graph Completion（2015）

- TransE和TransH模型都假设实体和关系是语义空间中的向量，因此相似的实体在同一实体空间中会非常接近。

  然而，每个实体可以有许多方面，不同的关系关注实体的不同方面。例如，`(location, contains, location)`的关系是'contains'，`(person, born, date)`的关系是'born'。这两种关系非常不同。

  为了解决这个问题，我们让TransR在两个不同的空间，即**实体空间**和**多个关系空间**(关系特定的实体空间)中建模实体和关系，并在对应的关系空间中进行转换，因此命名为TrandR。



### Trans-D(2015)

- Knowledge Graph Embedding via **D**ynamic Mapping Matrix（2015）
- TransR也有其不足之处。
  - 首先，head和tail使用相同的转换矩阵将自己投射到超平面上，但是head和tail通常是一个不同的实体，例如，`(Bill Gates, founder, Microsoft)`。'Bill Gate'是一个人，'Microsoft'是一个公司，这是两个不同的类别。所以他们应该以不同的方式进行转换。
  - 第二，这个投影与实体和关系有关，但投影矩阵仅由关系决定。
  - 最后，TransR的参数数大于TransE和TransH。由于其复杂性，TransR/CTransR难以应用于大规模知识图谱。ssh

## QA

- 抽出關係之後怎麼放入模型?
- 在基金模型中
- 如何挑選負樣本?
  - 太簡單的樣本對於訓練沒有幫助



## Ref

- [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
  - [TransE论文：多元关系数据嵌入_NULL-CSDN博客](https://blog.csdn.net/FFFNULL/article/details/51130028?spm=1001.2014.3001.5501)
  - [TransE论文第2节：翻译模型_NULL-CSDN博客](https://blog.csdn.net/FFFNULL/article/details/51139927?spm=1001.2014.3001.5501)
  - [TransE论文第3节：相关工作_NULL-CSDN博客](https://blog.csdn.net/FFFNULL/article/details/51150389?spm=1001.2014.3001.5501)
  - [TransE论文第4节：实验_NULL-CSDN博客_transe实验](https://blog.csdn.net/FFFNULL/article/details/51158519?spm=1001.2014.3001.5501)
  - [TransE论文剩余部分_NULL-CSDN博客](https://blog.csdn.net/FFFNULL/article/details/51163035?spm=1001.2014.3001.5501)
  - 

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- [刘知远大神《知识表示学习研究进展》的学习小结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/356147538)
- [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://persagen.com/files/misc/Wang2017Knowledge.pdf)
- [知识图谱嵌入的Translate模型汇总（TransE，TransH，TransR，TransD） (qq.com)](https://mp.weixin.qq.com/s/2YbfL_1_SyM4wNozyaj4lw)
- [知识表示学习Trans系列梳理(论文+代码) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/144412694)

[[數學分析\] 淺談各種基本範數 (Norm) (ch-hsieh.blogspot.com)](https://ch-hsieh.blogspot.com/2010/04/norm.html)