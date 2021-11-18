# Natural Language Precessing



- 分类、生成、序列标注、句子对标注

- [Python 文本数据分析初学指南](https://datartisan.gitbooks.io/begining-text-mining-with-python/content/)

- Often when performing analysis, lot of data is numerical, such as sales numbers, physical measuremets, quantifiable categories.Computers are very good at handling direct numerical information.
- As humans we can tell there is a plethora of information inside of text documents.But a computer needs specialized processing techniques in order to "understand" raw text data.
  - email / message / website...
- Text data is highly unstructured and can be in multiple languages!
- Use Cases
  - Calssifying Emails as Spam vs Legitimate
  - Sentiment Analysis of text Movie Reviews: Positive/Negitive
  - Analyzing Trends from written customer feedback forms.
  - Understanding text commands, 'Hey Google, play this song'

> https://www.wisers.ai/zh-hk/browse/event-detection-tracking/specifications/
>
> https://zhuanlan.zhihu.com/p/36736328

https://zhuanlan.zhihu.com/p/25928551

[【干货】文本分类算法集锦，从小白到大牛，附代码注释和训练语料](https://zhuanlan.zhihu.com/p/64602471)

## 應用

- 偵測詐騙郵件
- 情感分析
- 搜尋建議更正
- 詞類標識
- 機器翻譯
- 語音辨識
- 人名識別辨識
- 文本摘要
- 文本生成
- 句法分析
- [https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/671246/](https://codertw.com/程式語言/671246/)
- https://oosga.com/nlp/
- https://zhuanlan.zhihu.com/p/100985227
- https://zhuanlan.zhihu.com/p/38318074

## 流程

1. 獲取語料
   - 語料，NLP任務研究的內容
   - 通常用一個文本集合作為語料庫(Corpus)
   - 來源，如小說、文本、或可以自行開發爬蟲程式在網路上抓取資料
2. 預料預處理
   - 語料清洗
     - 留下有用的，刪除噪音數據
     - 常見的數據清洗方式有人工去除重複、對齊、刪除和標註等、或者規則提取內容、正則表達式匹配、根據詞性或命名實體識別提取、邊寫腳本或者代碼批處理等。
   - 分詞
     - 將文本分成詞語
     - 常見的分詞方法有基於字符串匹配的分詞方法、基於理解的分詞方法、基於統計分詞方法和基於規則的分詞方法
   - 詞性標註
     - 給詞語打詞類標籤，如動詞、形容詞、名詞等等。通常用在情感分析、知識推理等任務中
     - 常見的詞性標註方法有基於規則、基於統計等方式，如基於最大熵的詞性標註、基於統計最大機率輸出詞性和基於HMM的詞性標註。
   - 去除停用詞
     - 去掉對文本特徵沒有任何貢獻作用的詞性，比如標點符號、語氣、人稱等等。
3. 特徵工程
   - 把分詞表示成電腦能夠計算的類型，一般為向量
   - 常用的表示模型有詞袋模型(Bag of Word)、TF-IDF；詞向量則是wordvec
4. 特征選擇
   - 選擇合適、表達能力強的特徵
   - 常見的特徵選擇方式有DF，MI，IG，CHI，WLLR， WFO
5. 模型訓練
   - 機器學習模型有KNN、SVM、Naive Bayes、決策樹、GBDT、K-means等
   - 深度學習模型有CNN、RNN、LSTM、SeqSeq、FastText、TextCNN
   - 需注意欠擬合、過擬合的問題
     - 過擬合為在訓練資料表現很好，但是在測試集上表現很差
       - 解決方式有增大數據的訓練量
       - 增加正則化想，如L1正則和L2正則
       - 特徵選取不合理，人工篩選特徵和使用特徵選擇算法
       - 採用DropOut方式
     - 欠擬合是模型不能夠很好的擬合數據，解決方式有
       - 增加其他特徵項
       - 增加模型複雜度，比如神經網絡加更多層、線性模型通過添加多項式使模型的泛化能力更強
       - 減少正則化參數，正則化的目的是用來防止過擬合的，但是當模型出現欠擬合的情形時，可以適度減少正則化參數
6. 評價指標
   - 錯誤率、精度、準確度、精確度、召回率、F1-Score
   - ROC曲線、AUC曲線
7. 模型上線應用
   - 第一就是線下訓練模型，然後將模型做線上部署
   - 第二種就是在線上訓練，在線訓練完成後把模型pickle持久化

## Libraries

### re

[Regular Expressions Cheat Sheet by DaveChild - Download free from Cheatography - Cheatography.com: Cheat Sheets For Every Occasion](https://cheatography.com/davechild/cheat-sheets/regular-expressions/)

[regex101: build, test, and debug regex](https://regex101.com/)

### NLTK

- NLTK- Natural Language Toolkit is a very popular open source.
- Initially released in 2001, it is much older than Spacy(released 2015).
- It also provides many functionalities, but includes less efficient implementations.

### Spacy 

- Open Source Natural Language Processing Library

- Designed to effectively handle NLP tasks with the most efficient implementation of common algorithms.

- For many common NLP tasks, Spacy is much faster and more efficient, at the cost of the user not being able to choose a specific algorithmic implementations.

- However, Spacy does not include pre-created models for some applications, such as sentiment analysis, which is typically easier to perform with NLTK.

- install

  ```python
  conda install -c conda-forge spacy 
  
  python -m spacy download en
  ```

- Ref

  - [Spacy-Basics.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/00-Spacy-Basics.ipynb)

  - [Chinese_models_for_SpaCy](https://github.com/howl-anderson/Chinese_models_for_SpaCy)

## Process

### Tokenizations

- Tokenization is the process of breaking up the original text into component pieces(tokens).

  - Prefix: Character(s) at the beginning
    - ex: $ / ( / "
  - Suffix: Character(s) at the end
    - ex: km / ) / . / , / !
  - Infix：Character(s) in between
    - ex: - / -- / ...
  - Exception: Special-case rule to split a string into serveral tokens or prevent a token from being split when punctuation rules are applied
    - ex: let's / U.S.

  ![](https://lh3.googleusercontent.com/pw/ACtC-3fah2pChHTiJvetIOpadWl7DPyn6QdjGvbnyJJ4xMvYnE2X52Vjr4cOzSwmfukGJmqCIXCqkujLF9J4kJ0_f9S3Ff9Su3TFhUIzSDpTUKU838gRVoNIGkrlkUJ_6efFG_J7GnTD1ZK6JJvqYcQFBRe3=w979-h585-no?authuser=1)

- Named Entities

  - The language model recognizes that certain words are organizational names while others are locations, and still other combinations relate to money, dates, etc. Named entities are accessible through the ents property of a Doc object.

- Ref

  - [Tokenization.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/01-Tokenization.ipynb)

### Stemming

- Often when we searching text for a certain keyword, it help if the search returns variations of the word.
- For instance, searching for "boat" might also return "boats" and "boating". Here, "boat" would be the stem for [boat, boater, boating, boats].
- Stemming is a somewhat crude method for cataloging related words; it essentially chops off letters from the end until the stem is reach.
- This works fairly well in most cases, but unfortunately English has many exceptions where a more sophisticated process is required.
- Ref
  - [Stemming.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/02-Stemming.ipynb)



### Lemmatizations

- In contrast stemming, lemmatization looks beyong word reduction, and considers a language's full vocabulary to apply a morphological analysis to words.
- The lemma of 'was' is 'be' and the lemma of 'mice' is 'mouse'. Further, the lemma of 'meeting' might be 'meet' or 'meeting' depending on its use in a sentence.
- Lemmatization is typically seen as much more informative than simple stemming, which is whySpacy has opted to only have Lemmatization avaliable instead of stemming.

- Lemmatization looks at surrounding text to determine a given word's part of speech, it does not categorize phrases.
- Ref
  - [Lemmatization.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/03-Lemmatization.ipynb)

### StopWords

- Words like 'a' and 'the' appear so frequently that they don't require tagging as thoroughly as nouns, verbs and modifiers.
- We call these stop words and they can be filtered from the text to be processed.
- Spacy holds a built-in list of smoe 305 English stop words.
- Sample Code
  - [Stop-Words.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/04-Stop-Words.ipynb)

### Vocubulary and Phrase Matching

Identify and label specific phrases that match patterns we can define ourselves.

We can think of this as a powerful version of Regular Expression where we actually take parts of speech into account for our pattern search.

- Rule-based Matching

- PhraseMatcher

- Ref

- [Vocabulary-and-Matching.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/01-NLP-Python-Basics/05-Vocabulary-and-Matching.ipynb)

  - https://spacy.io/usage/linguistic-features#section-rule-based-matching

### Speech Tagging

- Parts-of-Speech(POS)
  - Most words are rare, and it's common for words that look completely different to mean almost the same thing. 
  - The same words in a different order can mean something completely different. 
  - Even splitting text into useful word-like units can be difficult in many languages.
  - While it's possible to solve some problems starting from only the raw characters, it's usually better to use linguistic knowledge to add useful information.
  - That's exactly what spaCy is designed to do: you put in raw text, and get back a **Doc** object, that comes with a variety of annotations.
  - In this lecture we'll take a closer look at coarse POS tags (noun, verb, adjective) and fine-grained tags (plural noun, past-tense verb, superlative adjective).
- Ref
  - [POS-Basics.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/02-Parts-of-Speech-Tagging/00-POS-Basics.ipynb)

### Named Entity Recognition

- Named-entity recognition (NER) seeks to locate and classify named entity mentions in unstructured text into pre-defined categories such as the person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.
- what if we have several terms to add as possible NERs?
- In this continued lecture, we will go over how to add in multiple phrases as NERs.
- For example, if we are working with a vacuum company, we might want to add both **vacuum cleaner** and **vacuum-cleaner** as PROD (product) NERs.
- 無需提前設定關鍵字，利用人工智能技術自動識別文本中出現的公司名/機構名、人名、職位、時間、地點、品牌、產品及各種自定義實體資訊。
- 應用場景：信息提取，關係抽取，句法分析，機器翻譯，語義理解，知識圖譜，輿情分析，問答系統Chatbot等。
- Ref
  - [Named-Entity-Recognition.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/02-Parts-of-Speech-Tagging/02-NER-Named-Entity-Recognition.ipynb)

### Sentence Segmentation

- In Spacy Basics we saw briefly how Doc objects are divided into sentences.
- In this lecture we'll learn how sentence segmentation works, and how to set our own segmentation rules to break up docs into sentences based on our own rules.

- Ref
  - [Sentence-Segmentation.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/02-Parts-of-Speech-Tagging/04-Sentence-Segmentation.ipynb)
  - [POS-Assessment-Solutions.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/02-Parts-of-Speech-Tagging/06-POS-Assessment-Solutions.ipynb)
  - [POS-Challenge.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/02-Parts-of-Speech-Tagging/07-POS-Challenge%20(optional).ipynbb)

## Text Classification

- Ref:
  - [An Introduction to Statistical Learning](https://faculty.marshall.usc.edu/gareth-james/ISL/ISLR Seventh Printing.pdf)

- Modeling process and evaluation method are equal to the Normal Machine Learning  project.
- The difference is the Feature Extraction method.

### Text Feature Extraction

- Most classic machine learning algorithms can’t take in raw text. 
- Instead we need to perform a feature “extraction” from the raw text in order to pass numerical features to the machine learning algorithm.

- Count Vectorization
  - count the occurence of each word to map text to a number.
- TF-IDF Vectorizations
  - calculates term frequency-inverse document frequency value for each word(TF-IDF). 
  - Term frequency **tf(t,d)**: is the raw count of a term in a document, i.e. the number of times that term t occurs in document d.
  - However, Term Frequency alone isn’t enough for a thorough feature analysis of the text!Let’s imagine very common terms, like “a” or “the”...
  - Because the term "the" is so common, term frequency will tend to incorrectly emphasize documents which happen to use the word "the" more frequently, without giving enough weight to the more meaningful terms "red" and "dogs". 
  - An inverse document frequency factor is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.
  - It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient)
  - TF-IDF allows us to understand the context of words across an entire corpus of documents, instead of just its relative importance in a single document.

- Ref
  - [SciKit-Learn-Primer.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/03-Text-Classification/00-SciKit-Learn-Primer.ipynb)
  - [Feature-Extraction-from-Text.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/03-Text-Classification/01-Feature-Extraction-from-Text.ipynb)
  - [Text-Classification-Assessment-Solution.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/03-Text-Classification/04-Text-Classification-Assessment-Solution.ipynb)

## Sentiment Analysis

- In order to use Spacy’s embedded word vectors, we must download the **larger** spacy english models.
- Full details can be found at:**https://spacy.io/usage/models**
- At the command line download the medium or large spacy english models:
  - **python -m spacy download en_core_web_md**
  - **python -m spacy download en_core_web_lg**
- Now that you have the larger models that contain the word vectors, let’s discuss how word vectors are created.
- Many of the reviews had positive things to say about a movie reserving final judgment to just the very last sentence. So even a negative review can highlight positive things.
- Maybe saying oh the actors were really good in this movie but the script was horrible leading to a bad movie.
- That sort of dichotomy within a single review can be really hard for something like Vader to the text.
- And sometimes it takes something more robust like TFA IDF in order to create your own sort of classification

- 通過機器學習、深度學習、和自然語言處理等技術自動識別文本所表達的情緒，幫助用戶即時獲取預警情報和洞悉重要資訊。
- 應用場景：
  - 口碑分析、營銷評估與市場調查：自動分析海量媒體數據中用戶所表達的情感傾向，洞察客戶相關品牌、產品或競品的運營效果與口碑優劣，了解用戶偏好及行業趨勢，為市場推廣、產品研發、商業分析和決策提供指導。

  - 全媒體監測與危機公關服務：可結合慧科的熱點話題及事件檢測與追蹤技術檢測負面情緒與事件，進行近乎實時的情報監測、突發事件預警、品牌危機跟蹤，幫助用戶及時發現並了解所關注話題/事件的發展動態，為下一步決策提供有效的指引。

  - 金融市場洞察與風險管控：自動偵探全媒體負面財經報導，結合社交媒體分析大眾投資心理，第一時間掌握金融市場動態、發掘投資良機、或偵測信貸風險。

### word Vectors

- Word2vec is a two-layer neural net that processes text. 

- Its input is a text corpus and its output is a set of vectors: feature vectors for words in that corpus.

- The purpose and usefulness of Word2vec is to group the vectors of similar words together in vectorspace. 

- That is, it detects similarities mathematically. 

- Word2vec creates vectors that are distributed numerical representations of word features, features such as the context of individual words. 

- It does so without human intervention.

- Given enough data, usage and contexts, Word2vec can make highly accurate guesses about a word’s meaning based on past appearances. 

- Those guesses can be used to establish a word’s association with other words (e.g. “man” is to “boy” what “woman” is to “girl”)

- Word2vec trains words against other words that neighbor them in the input corpus.

- It does so in one of two ways, either using context to predict a target word (a method known as continuous bag of words, or CBOW), or using a word to predict a target context, which is called skip-gram

  ![](https://lh3.googleusercontent.com/pw/ACtC-3eeadPlSIldCNAgA1ZE1dmwf-WHYU6A9UxFEt7NSZpkX6OuYfed8DxVxQzYX-beqq75SDlYN6m3VBXDGltBr0qIGPydAuaJYjgkr3ib4xaKKvbPrrYTD9QNZkQFJ-ZH0fGQCQUE0uIZ4GcBci8ITCgx=w833-h485-no?authuser=1)

- Recall that each word is now represented by a **vector.**

- In spacy each of these vectors has 300 dimensions. 

- This means we can use Cosine Similarity to measure how similar word vectors are to each other.

- This means we can also perform vector arithmetic with the word vectors.

  - **new_vector = king - man + woman**

- This creates new vectors (not directly associated with a word) that we can then attempt to find most similar vectors to.

- **new_vector closest to vector for queen**

- Interesting relationships can also be established between the word vectors

- Ref

  - [Semantics-and-Word-Vectors.ipynb](https://github.com/TLYu0419/NLP_Natural_Language_Processing_with_Python/blob/master/04-Semantics-and-Sentiment-Analysis/00-Semantics-and-Word-Vectors.ipynb)

### Sentiment Analysis

- We’ve already explored text classification and using it to predict sentiment labels on pre-labeled movie reviews.
- But what if we don’t already have those labels?
- Are there methods of attempting to discern sentiment on raw unlabeled text?
- VADER (Valence Aware Dictionary for sEntiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion.
- It is available in the NLTK package and can be applied directly to unlabeled text data.
- Primarily, VADER sentiment analysis relies on a dictionary which maps lexical features to emotion intensities called sentiment scores. 
- The sentiment score of a text can be obtained by summing up the intensity of each word in the text.
- For example, words like “love”, “like”, “enjoy”, “happy” all convey a **positive** sentiment.
- VADER is intelligent enough to understand basic context of these words, such as “**did not love**” as a negative sentiment.
- It also understands capitalization and punctuation, such as “**LOVE!!!!**”
- Sentiment Analysis on raw text is always challenging however, due to a variety of possible factors:
  - Positive and Negative sentiment in the same text data.
  - Sarcasm using positive words in a negative way.

- Let’s explore using VADER sentiment analysis with NLTK and Python!

- Ref
  - [Sentiment-Analysis.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/04-Semantics-and-Sentiment-Analysis/01-Sentiment-Analysis.ipynb)
  - [Sentiment-Analysis-Project.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/04-Semantics-and-Sentiment-Analysis/02-Sentiment-Analysis-Project.ipynb)
  - [Sentiment-Analysis-Assessment-Solutions.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/04-Semantics-and-Sentiment-Analysis/04-Sentiment-Analysis-Assessment-Solutions.ipynb)

## Topic Modeling

- Understand Topic Modeling
- Learn Latent Dirichlet Allocation
- Implement LDA
- Understand Non-Negative Matrix Factorization
- Implement NMF
- Apply LDA and NMF with a project

### Overview

- Topic Modeling allows for us to efficiently analyze large volumes of text by clustering documents into topics.
- A large amount of text data is **unlabeled** meaning we won’t be able to apply our previous supervised learning approaches to create machine learning models for the data!
- If we have **unlabeled** data, then we can attempt to “discover” labels.
- In the case of text data, this means attempting to discover clusters of documents, grouped together by topic.
- A very important idea to keep in mind here is that we don’t know the “correct” topic or “right answer”!
- All we know is that the documents clustered together share similar topic ideas.
- It is up to the user to identify what these topics represent.
- We will begin by examining how Latent Dirichlet Allocation can attempt to discover topics for a corpus of documents!

### Latent Dirichlet Allocation

- Johann Peter Gustav Lejeune Dirichlet was a German mathmatician in the 1800s who contributed widely to the field of modern mathematics.
- There is a probability distribution named after him “Dirichlet Distribution”.
- Latent Dirichlet Allocation is based off this probability distribution.
- In 2003 LDA was first published as a graphical model for topic discovery in *Journal of Machine Learning Research* by David Blei, Andrew Ng and Michael I. Jordan.
  - [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- Assumptions of LDA for Topic Modeling
  - Documents with similar topics use similar groups of words
  - Latent topics can then be found by searching for groups of words that frequently occur together in documents across the corpus.
  - Documents are probability distributions over latent topics.
  - Topics themselves are probability distributions over words.
- LDA represents documents as mixtures of topics that split out words with certain probabilities. 
- It assumes that documents are produced in the following fashion:
  - Decide on the number of words N the document will have.
  - Choose a topic mixture for the document (according to a Dirichlet distribution over a fixed set of K topics). 
  - e.g. 60% business, 20% politics, 10% food
- Generate each word in the document by:
  - First picking a topic according to the multinomial distribution that you sampled previously (60% business, 20% politics, 10% food)
  - Using the topic to generate the word itself (according to the topic’s multinomial distribution). 
  - For example, if we selected the food topic, we might generate the word “apple” with 60% probability, “home” with 30% probability, and so on.
- Assuming this generative model for a collection of documents, LDA then tries to backtrack from the documents to find a set of topics that are likely to have generated the collection.
- Now imagine we have a set of documents. 
- We’ve chosen some fixed number of K topics to discover, and want to use LDA to learn the topic representation of each document and the words associated to each topic. 
- Go through each document, and randomly assign each word in the document to one of the K topics.
- This random assignment already gives you both topic representations of all the documents and word distributions of all the topics (note, these initial random topics won’t make sense).
- Now we iterate over every word in every document to improve these topics.
- For every word in every document and for each topic **t** we calculate:
  - p(topic **t** | document **d**) = the proportion of words in document **d** that are currently assigned to topic **t**
  - p(word **w** | topic **t**) = the proportion of assignments to topic **t** over all documents that come from this word **w**
- Reassign w a new topic, where we choose topic t with probability **p(topic t | document d)** * **p(word w | topic t)** 
- This is essentially the probability that topic t generated word w
- After repeating the previous step a large number of times, we eventually reach a roughly steady state where the assignments are acceptable.
- At the end we have each document assigned to a topic.
- We also can search for the words that have the highest probability of being assigned to a topic
- We end up with an output such as:
  - Document assigned to Topic #4
  - Most common words (highest probability) for Topic #4:
    - [‘cat’,’vet’,’birds’,’dog’,...,’food’,’home’]
  - It is up to the user to interpret these topics.
- Two important notes:
  - The user must decide on the amount of topics present in the document.
  - The user must interpret what the topics are.

- Ref
  - [Latent-Dirichlet-Allocation.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/05-Topic-Modeling/00-Latent-Dirichlet-Allocation.ipynb)
  - [怎么确定LDA的topic个数？](https://www.zhihu.com/question/32286630)
  - [一文详解LDA主题模型](https://zhuanlan.zhihu.com/p/31470216)
  - [LDA主题模型详解（面试的问题都在里面）](https://zhuanlan.zhihu.com/p/105937136)
  - [文本挖掘从小白到精通（五）---主题模型的主题数确定和可视化](https://zhuanlan.zhihu.com/p/75484791)
  - [LDA(Latent Dirichlet Allocation)主题模型](https://zhuanlan.zhihu.com/p/36394491)
  - [20181012 lda explanation-allen-lee](https://www.slideshare.net/ssuser7414b2/20181012-lda-explanationallenlee)
  - [LDA in Python – How to grid search best topic models?](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/)

### Non-negative Matrix Factorization

- Non-negative Matrix Factorization is an unsupervised algorithm that simultaneously performs dimensionality reduction and clustering.
- We can use it in conjunction with TF-IDF to model topics across documents.
- Given a non-negative matrix A, find k-dimension approximation in terms of non-negative factors W and H 
- Approximate each object (i.e. column of A) by a linear combination of k reduced dimensions or “basis vectors” in W.
- Each basis vector can be interpreted as a cluster. The memberships of objects in these clusters encoded by H.
- Input: Non-negative data matrix (**A**), number of basis vectors (**k**), initial values for factors W and H (e.g. random matrices).
- Objective Function: Some measure of reconstruction error between A and the approximation WH
- Expectation–maximization optimisation to refine W and H in order to minimise the objective function. Common approach is to iterate between two multiplicative update rules until convergence
- Step
  1. Construct vector space model for documents (after stopword filtering), resulting in a term-document matrix A.
  2. Apply TF-IDF term weight normalisation to A
  3. Normalize TF-IDF vectors to unit length.
  4. Initialise factors using NNDSVD on A.
  5. Apply Projected Gradient NMF to A.
- Basis vectors: the topics (clusters) in the data.Coefficient matrix: the membership weights for documents relative to each topic (cluster).
- Just like LDA, we will need to select the number of expected topics beforehand (the value of **k**)!
- Also just like with LDA, we will have to interpret the topics based off the coefficient values of the words per topic.
- 應用
  - 新聞文本聚類
    - 背景
      - 單一事件的裁罰造成業者全年獲利縮水20%以上(美國裁罰兆豐)
      - 雖然有KYC的甄審流程，但即時性不如外部的新聞資訊
      - 雖然新聞能夠更及時偵測風險，但是資料量太大，沒辦法逐條新聞瀏覽，因此需要針對新聞做文本聚類
    - 目標
      - 透過新聞文本聚類，降低甄審人員的工作量
      - 協助做姓名檢核，降低67%新聞閱讀量
    - Package
      - spacy：處理英文
      - ckiptagger：處理繁中
      - nltk：詞幹提取
    - 由平均側影係數找出最佳分群組數，中信的專案約 0.62
    - 將新聞依照風險類型做文本聚類，讓甄審人員進一步瀏覽新聞確認是不是自己的客戶
      - 還可以在系統中將關鍵字標註顏色，方便找出風險詞彙
    - 中信有投稿IJCAI，獲得許多獎項
- Ref
  - [Non-Negative-Matrix-Factorization.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/05-Topic-Modeling/01-Non-Negative-Matrix-Factorization.ipynb)
  - [LDA-NMF-Assessment-Project-Solutions.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/05-Topic-Modeling/03-LDA-NMF-Assessment-Project-Solutions.ipynb)

## Deep Learning for NLP

- Understand basic overview of Deep Learning
- Understand basics of LSTM and RNN
- Use LSTM to generate text from source corpus
- Create QA Chat Bots with Python

### Introduction to Neural Networks

- We’ve seen how a single perceptron behaves, now let’s expand this concept to the idea of a neural network!
- Let’s see how to connect many perceptrons together and then how to represent this mathematically!

- Multiple Perceptrons Network
  - Input Layers
    - Real values from the data
  - Hidden Layers
    - Layers in between input and output3 or more layers is “deep network”
  - Output Layer
    - Final estimate of the output
- As you go forwards through more layers, the level of abstraction increases.
- Let’s now discuss the activation function in a little more detail!
  - sigmoid 
  - tanh
  - ReLU: ReLu tends to have the best performance in many situations.
- Deep Learning libraries have these built in for us, so we don’t need to worry about having to implement them manually!
- Now that we understand the basics of neural network theory, let’s move on to implementing and building our own neural network models with Keras!

### Keras Basics

- Let’s learn how to create a very simple neural network for classifying the famous Iris data set!
- The iris data set contains measurements of flower petals and sepals and has corresponding labels to one of three classes (3 flower species).
- Ref
  - [Keras-Basics.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/06-Deep-Learning/00-Keras-Basics.ipynb)

### Recurrent Neural Networks Theory

- Examples of Sequences

  - Time Series Data (Sales)
  - Sentences
  - Audio
  - Car Trajectories
  - Music

- Cells that are a function of inputs from previous time steps are also known as *memory cells.*

- RNN are also flexible in their inputs and outputs, for both sequences and single vector values.

  ![](https://lh3.googleusercontent.com/pw/ACtC-3dlW8z6aZRkm3X-XPOP7NjKJczeuZE-Y5d5_MNQ-dlRdpNSrar0dH8PojCpmcC4m8HH1KhD9zqNJ0fu277x3GMYh_Iyo9wU7bfbcBMAMeTO2o0a2kQzu44Iv3TzxR9Yl4SLLTpsj37kNzeyT7icgh_P=w1196-h510-no?authuser=1)

- We can also create entire layers of Recurrent Neurons...

- RNN are also very flexible in their inputs and outputs.

  - Sequence to Sequence
  - Sequence to Vector
  - Vector to Sequence

- Now that we understand basic RNNs we’ll move on to understanding a particular cell structure known as LSTM (Long Short Term Memory Units).

### LSTM and GRU

- An issue RNN face is that after a while the network will begin to “forget” the first inputs, as information is lost at each step going through the RNN.

- We need some sort of “long-term memory” for our networks.

- The LSTM (Long Short-Term Memory) cell was created to help address these RNN issues.

- Let’s go through how an LSTM cell works!

- Keep in mind, there will be a lot of Math here! Check out the resource link for a full breakdown!

- Fortunately Keras has a really nice API that makes LSTM and RNN easy to work with.

- Coming up next, we’ll learn how to format data for RNNs and then how to use LSTM for text generation.

  

### Text Generation With Python and Keras

- Process Text
- Clean Text
- Tokenize the Text and create Sequences with Keras
- Create the LSTM Based Model
- Split the Data into Features and Labels
  - X Features (First n words of Sequence)
  - Y Label (Next Word after the sequence)
- Fit the Model
- Generate New Text Based off a Seed
- Ref：
  - [Text-Generation-with-Neural-Networks.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/06-Deep-Learning/01-Text-Generation-with-Neural-Networks.ipynb)

### QA Bot

- We will be implementing a chat bot that can answer questions based on a “story” given to the bot.

- We will use the BaBi dataset released by Facebook research.

- https://research.fb.com/downloads/babi/

- Story

  - Jane went to the store. Mike ran to the bedroom.

- Question

  - Is Mike in the store?

- Answer

  - No

- End-to-End Memory Networks

  - Sainbayar Sukhbaatar
  - Arthur Szlam
  - Jason Weston
  - Rob Fergus

- **You must read the paper to understand this network!**

- How the QA Bot Network Works

  - Model takes a discrete set of inputs **x1, ..., xn** that are to be stored in the memory, a query **q**, and outputs an answer **a**
  - Each of the **x** , **q**, and **a** contains symbols coming from a dictionary with **V** words.
  - The model writes all **x** to the memory up to a fixed buffer size, and then finds a continuous representation for the **x** and **q**.
  - End to End Network:
    - Input Memory Representation
    - Output Memory Representation
    - Generating Final Prediction
  - Create a full model with RNN and Multiple Layers
    - Input Memory Representation of stories.
    - Use Keras for Embedding to convert sentences **x**
    - **Encoders C and M**
    - **Question Encoder**
    - Output Memory Representation
    - Each **x** has a corresponding output vector **c**
    - In the single layer case, the sum of the output vector **o** and the input embedding u is then passed through a final weight matrix W (of size V × d) and a softmax to produce the predicted label:

- Understand the steps on how to Vectorize the DataCreate a function that can vectorize data for us.

- Make sure to read the paper before continuing to this lecture!

- Build the Neural Network

  - Input Encoder M
  - Input Encoder C
  - Question Encoder

- Complete the Network

- Fit/Train the network

  - (We will load a pre-trained network)

- Plot out Training History

- Evaluate on Test Set

- Create our Own Stories and Questions

- Ref

  - [End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895.pdf)
  - [Chat-Bots.ipynb](https://github.com/TLYu0419/DataScience/blob/master/Natural_Language_Processing/Natural_Language_Processing_with_Python/06-Deep-Learning/02-Chat-Bots.ipynb)



### 關係抽取

> 利用自然語言處理與機器學習技術，自動抽取中文文本中實體間的關係信息。

#### **特色與優勢**

- 同時支持繁簡中文關係抽取
- 海量數據輔助挖掘算法與規則學習
- 開放式關係抽取，不限定關係種類
- 可精確判斷兩實體間的關係方向

#### **應用場景**

- 客戶背景調查 (Know Your Customer)：通常的客戶背景調查解決方案缺乏從海量非結構化全媒體數據中提取目標客戶資料構建關係圖的手段。通過開放式關係抽取技術可以自動分析媒體報道，提取目標客戶的相關資訊，探尋關聯關係。
- 知識圖譜構建與擴充：透過開放式關係抽取可從各式非結構化文本來源（如新聞、社群、年報等）找出更多結構化數據庫中缺乏的實體關係三元組，以最快的速度及豐富的資料更新知識圖譜。



## 多篇文本處理

> 針對海量全媒體數據流及時發掘與識別最相關、最負面、最重要及最有價值的資訊情報

### 評論歸納

評論歸納 Review Summary 基於信息抽取及語義分析對用戶評論數據進行自動意見抽取與聚類，並從聚類結果中歸納出最具代表性的評論。

#### **特色與優勢**

- 同時支持繁簡體中文分析
- 海量數據輔助規則學習與優化
- 精準抽取、靈活定制
- 基於語義的評論歸納聚類

#### **應用場景**

- 產品口碑分析：分析旗下產品之用戶反饋與評論，全面歸納了解用戶關注的主要維度及相關意見。
- 競爭者分析：分析競爭者相關產品之用戶評論，快速了解對手優缺點，以進一步做出應對策略。
- 潛在用戶開發：根據用戶反饋評論對用戶進行歸類畫像，並針對潛在用戶製定應對策略。

### 相似文章聚類

> 相似文章聚類 (Document Clustering) 自動把標題或內容相似的文章分群組合，方便用戶對海量文章進行有效的組織、瀏覽和摘要。

#### 特色與優勢

- 基於大數據流處理架構。單日處理能力近千萬級，單篇處理速度數毫秒以內，處理能力可靈活擴展。
- 支持基於標題、內容或兩者結合的多維度相似度計算；支持自定義相似度條件。
- 自主研發的增量式聚類算法，可持續將新文章與長時間跨度的歷史文章進行聚類。
- 提供快速和靈活的聚類結果查詢API，根據需求返回不同維度的聚類結果。
- 聚類準確率達到行業內領先水平。

#### 應用場景

相似文章聚類是一個應對信息量爆炸的底層大數據技術，通過自動聚合重複或相似內容，提供更便捷的信息閱讀與檢索。除了可用於文章抄襲檢測、新聞轉載侵權檢測等直接應用外，對提升後續自然語言處理或機器學習分析的準確性與處理速度也有重要意義。

### 熱點話題發現

> 熱門話題發現 (Hot Topic Discovery) 旨在從目標媒體數據中自動識別、快速聚合出在固定時間範圍內熱度較高的話題，幫助用戶迅速了解熱點議題。

#### **特色與優勢**

- 算法全自動無監督: 無需人工投入提供標註語料。
- 數據多樣性: 可對微博、論壇、新聞等不同類型的數據統一進行處理，實現全媒體熱點話題發現。
- 話題表示更形象化: 每個話題由標題、關鍵詞圖、相關文章列表、趨勢分析圖四部分組成，話題內容展示全面。
- 話題準確性高: 採用語義特征，運用圖算法的優勢，相比其它話題發現算法準確性更高。
- 優於主題模型算法Topic Modelling: 無需提前設定熱門話題個數，算法自動聚合。

#### **應用場景**

熱點話題與追踪屬於數據挖掘的範疇，可應用於信息安全、輿情監測、突發預警等領域。

- 企業決策: 有效挖掘海量資訊，幫助企業及時了解所關注話題發展動態，為未來決策提供有效的指引。
- 輿情監控: 幫助政府或企業發現并了解大眾關注議題，特別是重大負面話題的影響，為做出正確輿論引導，提供分析依據。
- 金融動態: 金融專業人士或理財用戶可以根據金融相關議題了解金融投資市場變化，儘早發現有差異信息，以獲得更好的投資良機。

### 事件檢測與追蹤

> 事件檢測與追蹤 (Event Detection and Tracking) 旨在幫助用戶應對日益嚴重的互聯網信息爆炸問題，近乎實時地從新聞媒體、社交平台等數據流中自動挖掘及持續追蹤具體事件，並可根據用戶自定義條件，在事件檢測到的第一時間發送預警。

#### **特色與優勢**

- 基於大數據流處理框架，近乎實時的事件檢測技術
- 跨媒體、跨平台、跨語言多樣性數據統一通用化處理技術
- 多維度事件聚類，基於語義相關事件追蹤
- 多角度事件評測指標（影響力、滲透度、負面性、可信度、綜合指數）
- 自定義事件實時監測及預警
- 借助知識圖譜，深入挖掘事件關聯情報

#### **應用場景**

事件檢測與追蹤主要針對流式互聯網大數據進行實時處理與挖掘，提供數據情報服務：

- 市場監測與危機公關：進行實時情報監測、品牌危機跟蹤，幫助企業用戶實時了解所關注事件的發展動態，為企業下一步決策提供有效指引。

- 輿情監控與危機管理：幫助政府或企業等用戶第一時間發現事件，特別是重大負面事件，以便及時決策，進行危機應對管理。

- 金融情報分析：幫助金融從業人士或投資者第一時間了解金融相關事件，洞察金融投資市場變化，及時發掘投資良機，快速決策。

- 口碑分析與營銷評估：結合話題分類與情感分析，深入了解相關品牌、產品的運營效果，口碑優劣，為進一步分析決策提供指導。

應用：

- 分析社交媒体中的大众情感
- 鉴别垃圾邮件和非垃圾邮件
- 自动标注客户问询
- 将新闻文章按主题分类



### 文本摘要

- [【NLP】文本生成评价指标的进化与推翻](https://cloud.tencent.com/developer/article/1650473)

### 文本生成

- [直觀理解 GPT-2 語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)

- [NLP民工的乐园: 几乎最全的中文NLP资源库](https://github.com/fighting41love/funNLP)
- [Text Mining with R](https://www.tidytextmining.com/)

- [自然语言处理中句向量获取方式的简要综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/350957155)

- [ownthink/Jiagu: Jiagu深度学习自然语言处理工具 知识图谱关系抽取 中文分词 词性标注 命名实体识别 情感分析 新词发现 关键词 文本摘要 文本聚类 (github.com)](https://github.com/ownthink/Jiagu)
