[TOC]

# 資料科學概論

## 機器學習簡介

- 大數據的特性分成資料量（Volume）、多樣性（Variety）、傳輸速度（Velocity）與真實性（Veracity）
  - 資料量：無論是天文學、生物醫療、金融、聯網物間連線、社群互動…每分每秒都正在生成龐大的數據量，如同上述所說的 TB、PB、EB 規模單位。
  - 多樣性：分析多樣化的資料──從文字、位置、語音、影像、圖片、交易數據、類比訊號… 等結構化與非結構化包羅萬象的資料，彼此間能進行交互分析、尋找數據間的關聯性。
  - 傳輸速度：隨著使用者每秒都在產生大量的數據回饋，過去三五年的資料已毫無用處。一旦資料串流到運算伺服器，企業便須立即進行分析、即時得到結果並立即做出反應修正，才能發揮資料的最大價值。
  - 真實性：資料收集的時候是不是有資料造假、即使是真實資料，是否能夠準確的紀錄、資料中有沒有異常值、有異常值的話該怎麼處理… 等等。
- 機器學習是甚麼? 
  - 讓機器從資料中找尋規律與趨勢⽽不需要給定特殊規則
  - 給定⽬標函數與訓練資料，學習出能讓⽬標函數最佳的模型參數
  - ⼀個機器學習模型中會有許多參數(parameters)，例如線性回歸中的 w (weights)跟 b (bias) 就是線性回歸模型的參數
    - 當我們輸入⼀個 x 進到模型中，不同參數的模型就會產⽣不同的 ŷ 
    - 希望模型產⽣的 ŷ 跟真實答案的 y 越接近越好
    - 找出⼀組參數，讓模型產⽣的 ŷ 與真正的 y 很接近，這個步驟就有點像學習的概念
- 機器學習的組成及應用
  - 監督式學習 (Supervised Learning)
    - 會有⼀組成對的 (x, y) 資料，且 x 與 y 之間具有某種關係，如圖像分類，每⼀張圖都有對應到的標記 (y)，讓模型學習到 x 與 y 之間的對應關係
    - ⽬前主流且有⾼準確率的機器學習應⽤多以此類型為主，但缺點是必須要蒐集標註資料。應用在圖像分類、詐騙偵測領域。

  - 非監督式學習(Unsupervised Learning) 
    - 僅有 x 資料⽽沒有標註的 y，例如僅有圖像資料但沒有標記。
      非監督式學習通常透過降維 (Dimension Reduction)、分群 (Clustering) 的⽅式實現
    - 非監督式的準確率通常都低於監督式學習，但如果資料收集非常困難時，可應⽤此⽅法。應用在維度縮減、分群、壓縮等。

  - 強化學習
    - 增強式學習是透過定義環境(Environment)、代理機器⼈ (Agent)及獎勵 (Reward)，讓機器⼈透過與環境的互動學習如何獲取最⾼的獎勵。
    - Alpha GO 就是透過增強式學習的⽅式訓練，增強式學習近幾年在棋類、遊戲類都取得巨⼤的進展，是⽬前非常熱⾨的研究領域。應用在下圍棋、打電玩。
- 跟統計分析的差別
  - 統計講因果，機器學習重視預測的準確度
  - 不重視因果關係很容易得到奇怪的結論，如
    - 基地台的數量與癌症人數正相關
      - 所以我應該搬到沒有基地台的地方嗎?
    - 員工人數和公司營收正相關
      - 所以我應該建議公司盡量多招募些員工嗎?
- Ref
  - [「机器学习」到底需要多少数据？](https://zhuanlan.zhihu.com/p/34523880)





## 角色與職務分工

以下是我自己的觀察和想法，但絕對不會也沒有標準答案

### 數據/商業分析師

- What is Data Analysis
  - A process of inspecting, cleansing, transforming and modeling data with the goal of discovering useful information, informing conclusion and supporting decision-making.
  -  

- 核心信念

  - AI 只是技術，技術的背後要解決的問題才是重點
  - 

- 職責

  - Analyze to help the company's decisions. 
  - 要擅長定義商業問題與設計解決方案
  - 要清楚哪些問題適合用資料科學解決，哪些不是，以及資料科學的限制
  - 要能依據情境判斷需要解釋性或準確度(通常ML工程師都喜歡準確度忽略解釋)
  - 要有能力使用各種演算法(但還沒到優化演算法的地步)
  - 要很擅長簡報和說故事的技巧，變得更有演講的魅力與發揮影響力
  - 實驗是用來回答問題，問題是用來決策的
    - Don't: 接受別人沒有經過思考就說你都做做看然後跟我們說結果
  - 在公司裡面還是要重視公司重視的東西，也就是獲利
    - 站在老闆的角度幫她思考
  - Model-Market Fit
    - 即使你的模型真的很準確，但真的符合User的需要嗎?
    - 以買被子為例，模型的推薦最後可能還不如從柔軟度、溫暖、...等等來的實用
  - Case Report: 有沒有跟User聊過?
  - Case Series: 經歷一系列的User訪談
  - 不只可解釋，還要理解資料與模型(有辦法解釋才有辦法報告)
    → 統計學、研究⽅法論中有豐富的⽅法
  - ⼀個⼈時間有限(要掌握公司中的資源，人、設備，召集大家一起解決問題)
    → 定位⾓⾊、協作技能（專案管理、產品管理）、真⼼喜歡資料領域 
    ⽽不是不停鑽研演算法；變化迅速的科技業中不變的真理
  - 要有溝通的能力，告訴對方我需要什麼，為什麼需要這樣

    - 如果資料很髒、很亂，與其把時間拿去建模，不如思考怎麼建立雙向溝通的管道

- 工具(用星星呈現)

  - PowerPoint

- 技能(用星星呈現)

  - SQL
  - ML
  - DL
  - 視覺化

- 職涯規劃(學習路徑)

  - 

- 提醒

  - 要更有產品思維，因為分析的洞察其實很抽象(不要淪為只會說空話)

  - 要比機器學習工程師更懂業務需求，掌握核心 Domian

  - 可以多與使用者談解決的什麼問題，不要只是講模型準確度怎樣

  - 會需要建立簡單的模型LR，NN，DL，但要知道演算法的假設

    - 要符合這些假設的預測跟解釋才有意義，不是直接就拿來用!殘差獨立...etc

    [近期大规模裁员, 为什么那么多公司先裁做data的？](https://mp.weixin.qq.com/s?__biz=MzA4NzM3MTkzNw==&mid=2652388050&idx=1&sn=8d37dfd2b9b5a4a67f70d592f5f8e18b&chksm=8bd6e31cbca16a0afeea9871baf3668f9fadafaa21a9c8a06c582afb695481237be83d8de13b#rd)

    > 資料驅動產⽣的額外 impact，減去團隊薪資、資料收集的開銷，才是 return

- 給新鮮人的一句話
  - 我們可以不用



### 資料工程師

- 核心信念
  - 把核心關心的問題解決的話就是好的工具
  - 找到錯誤debug，其實是非常浪漫的事情(?)
- 職責
  - Prepare the data infra to enable others to work with. 
  - 幫助公司處理資料的流程、維運資料庫
  - 從APP收集回來的資料沒有比汙水乾淨多少
  - 最常遇到的問題是，客戶的需求定義不清楚，或者做完又說要改(所以需要商業分析師幫忙診斷)
  - data 容量夠不夠是很常被忽略的問題
    - 一天20G，一年下來就很可觀
- 工具(用星星呈現)
- 技能(用星星呈現)
  - 網路爬蟲>>開發一時爽，維護火葬場
- 職涯規劃(學習路徑)
- 提醒
- 給新鮮人的一句話

### 資料科學家

- 核心信念

  - 想玩技術

- 職責

  - 依情境選擇合適的演算法
  - 當模型的表現不理想時要有能力調整參數甚至優化演算法
  - 要會模型的輸出結果診斷問題
  - 先跑便宜的實驗，降低不確定性
    - 察覺不確定性
    - 減少一些不必要的心理壓力
    - 有時候我們以為不會錯的事情，犯錯的機會比我們預期的大很多
  - 
  - Create software to optimize the company's operations
  - Attenion & Money is all you need
    - 有錢有設備很重要XD
    - 很容易忽略實務面，落地還需資料工程師

- 工具(用星星呈現)

- 技能(用星星呈現)

  - 

- 職涯規劃(學習路徑)

- 提醒

  - 開發時間冗長，要想辦法加速

  - 市場上越來越多AutoML，或者No-Code的產品，要找出自己的價值

  - 建模過程中會需要花許多時間處理瑣碎的資料清理過程

  - 建模和調參充滿著不確定性，不容易給長官、使用者保證開發時程

  - 有時候統計指標看似很好(R^2= 0.9?)，但當犯錯成本很高時會很不符合業務需求

  - 當模型使用的資料越多，對模型就是越大的風險->不是模型的變數越多越好

  - 建置模型不是直接就上最複雜的模型，可以先嘗試簡單的模型，如果已經達到業務需求就不需要繼續花這麼多時間建新模型

  - robustness, explanability, out of distribution inference

  - 在分析的時候，數據理解的過程常常被忽略掉(但不應該這樣)

  - 到底什麼時候要retrain模型，要有監測的機制，查為什麼不准了

  - 不能讓目的達到，但過程卻歪掉

  - 99%的Precision, Recall的準確度還是不夠高

    - Youtube下架1%的影片，其中半數重新審查後還重新上架

  - [我以為在建模結果大部分在 Debug](https://buzzorange.com/techorange/2020/11/19/machine_learning_why_so_boring/)

    > 「品質檢查、除錯、修復，⾄少要花 65% 的時間。

  - [AI影片錯殺率太高，YouTube重設審查人員](https://www.ithome.com.tw/news/140122)

    > 下架的 1,100 多萬則中，有 32 萬則接獲申訴，其中近半審查後重新上架。

  - [Nature 論⽂遭受嚴重質疑：實驗⽅法有根本缺陷](https://zhuanlan.zhihu.com/p/71129104)

    > 「演算法在測試集上的表現，遠遠超過了訓練集，這不是有資料洩漏嗎？」

  - [Science：有调查有真相！某些AI领域多年无实际进展](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247502265&idx=1&sn=a173d8b0348e742c4e8f348ba7eff0db&chksm=9094c02aa7e3493c7f209257423ccad044a1cccb3f709610b9961ae5e0ef23b1eb76f448da15#rd)

    > ⼀篇論⽂聲稱獲得了巨⼤的性能提升，⽽實際上是對比對象精度較低。」

  - Overfit: 研究者很容易忍不住就相信這種幻想

  - 統計學通常不太會Overfitting, ML很容易Ovverfitting, 那兩個預測的結果不一樣的時候，你要相信哪個?-->統計學

  - Data Leakage

    - 沒有Domain的時候，很容易不小心把不該放進去的變數放進去
    - 有些變數是for特定的族群才會有資料，做假設檢定的時候即使非常顯著，也不能放進模型中

  - Spurious Relationship

    - 虛假關係
    - 辨識到是郎還是哈士奇，99%準確度，但實際上模型辨識的是背後是不是雪景
    - 模型上的檢視需要Domain的解釋才能確保上線
    - 公司的營收跟員工人數都是正成長
      - 但要知道是真實的因果關係是什麼
    - Pizza跟50嵐的價格都是正成長

- 給新鮮人的一句話

### How to choose the right position?

- It may change in different periods of your life.
- 薪水很重要，但不應該是唯一考量的指標
- Ref
  - [Data Scientist vs Data Engineer](https://www.datacamp.com/community/blog/data-scientist-vs-data-engineer)
  - [Data Scientist、Data Analyst、Data Engineer 的区别是什么?](https://www.zhihu.com/question/23946233)
  - [Why Data Scientists Must Focus on Developing Product Sense](https://www.kdnuggets.com/2018/04/data-scientists-product-sense.html)
  - [Why so many data scientists are leaving their jobs](https://www.kdnuggets.com/2018/04/why-data-scientists-leaving-jobs.html)
  - [真．資料團隊與分工](https://blog.v123582.tw/2020/10/31/%E7%9C%9F%E3%83%BB%E8%B3%87%E6%96%99%E5%9C%98%E9%9A%8A%E8%88%87%E5%88%86%E5%B7%A5/)



## 產業應用

- 即使alphago已經打敗人類，但離電影中的強人工智慧還有一大段距離

### 金融

- 有越來越多的公司嘗試導入 Chatbot 期望降低客服人員的工作量，而 Chatbot 成功的關鍵就在於設計良好的服務旅程讓客戶能快速並滿意的完成服務。在這場專題演講中，講者將與大家分享國泰是如何透過社群網絡分析 (SNA) 來設計與優化 Chatbot 中的服務旅程。
- 

### 醫療

- 醫療一定會有小數據的問題
- 影像辨識技術，偵測心跳速度(健康)
- Prevention預防
  - train 模型判斷有沒有戴口罩
  - 蒐集網站上的數據預測感染的機率，掃描大量的文章把關鍵字找出來
- Precision精準
  - 在醫院偵測病人有沒有乖乖躺好、想下床，床的護手有沒有脫落
  - 用VR幫醫生跟病人解說過程
- prevalence(普及)
  - 透過聊天機器人提供病人有用的資訊(幫醫生節省很多工作)
    - 但醫生會需要認真幫忙建立醫學文檔
- 穿戴裝置偵測健康

### 電信

### 電商

- 拍衣服，自動到商場搜尋款式

### 犯罪

- 透過照片的背景找到地點(對抗性剝削的施暴者)

### 科技

- 視訊的時候置換背景，後來可以幫忙穿襯衫、打領帶XD
- 用手拍的瀑布會晃，但可以改用AI生成永遠的不停止的瀑布
- Gmail 的 smart reply
- 偵測空汙，用照相的方式
- Grammarly 的 情感評分

### 交通

- 以前物件偵測的運算量大，還要上雲端，導致很難落地，現在由於技術的革新，更過去的問題就迎刃而解
- yoloV4的提升，帶來巨大的改變，因為運算量小，快速讓我們的在交通上可以更好
- 偵測違規停車
- 專心駕駛的偵測
- 車流分析

### 零售

- 貨架的偵測，看有沒有擺東西，這樣賣場人員就不用一直去巡邏(而且還可以知道是哪個產品缺乏)

## 技術

### NLP的工具

- Only NLU
  - 網頁查詢
    詐騙郵件偵測
    情緒分析
    輿情分析
    判例分析
    閱讀理解
    病例分析
    公司內部資料分析
- Only NLG
  - 故事生成
    新聞寫作
    廣告生成
    專利生成
- NLU & NLG
  - 機器翻譯
    聊天機器人

## 機器學習開發流程

1. 商業主題，要解決什麼問題
   - 定義⽬標與評估準則：要預測的⽬標是甚麼? 
     - 回歸問題?
     - 分類問題?
   - 要⽤什麼資料來進⾏預測? (predictor 或 x)
   
2. 蒐集資料：針對這個主題，我們要定義出具體的指標，以及在這個主題之下相關的解釋變數。

3. 資料處理：
   - 缺失值填補
   - 離群值
   - 常態化
   - 特徵工程
   
4. 建置模型：
   1. 將資料分為
      - 訓練集, training set
      - 驗證集, validation set
      - 測試集, test set
   2. 設定評估準則
      - 回歸問題 (預測值為實數)
        - RMSE, Root Mean Square Error
        - Mean Absolute Error
        - R-Square
      - 分類問題 (預測值為類別)
        - Accuracy
        - F1-score
        - AUC, Area Under Curve
   3. 建立模型並調整參數
      - 根據設定⽬標建立機器學習模型
        - Regression, 回歸模型
        - Tree-based model, 樹模型
        - Neural network, 神經網路
      - 各模型都有其超參數需調整，根據經驗與對模型了解、訓練情形等進⾏調參
      - 如果有資料不平衡的問題，可以用 Upsampling, DownSampling, SMOTE 等方式建立新資料集再建模
   4. 透過CV調整最佳參數
   
5. 專案佈署：檢視模型實施後是否能有良好的預測、解釋效果。並且也可以在這個階段找出進一步優化模型的客能。

   - 導入	
     - 建立資料搜集、前處理等流程
     - 送進模型進⾏預測
     - 輸出預測結果
     - 視專案需求整合前後端
     - 建議統⼀資料格式，⽅便讀寫 (.json, .csv)

   - 如何確立⼀個機器學習模型的可⽤性？

     當我們訓練好⼀個機器學習模型，為了驗證其可⾏性，多半會讓模型正式上線，觀察其在實際資料進來時的結果；有時也會讓模型跟專家進⾏PK，挑⼀些真實資料讓模型與專家分別測試，評估其準確率。

- Ref
  - [The 7 Steps of Machine Learning (AI Adventures)](https://www.youtube.com/watch?v=nKW8Ndu7Mjw)
  - [ML Lecture 0-1: Introduction of Machine Learning](https://www.youtube.com/watch?v=CXgbekl66jc)
  - [機器學習的機器是怎麼從資料中「學」到東西的？](https://kopu.chat/2017/07/28/機器是怎麼從資料中「學」到東西的呢/)
  - [我們如何教導電腦看懂圖像](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-tw)



## 人工智慧開發在實務上的挑戰

- AI is more than a model: Four steps to complete workflow success
- Modelins is an important step in the workflow, bit the model is note the end og the journey.

### Date preparation

- Data cleansing and preparation
- Human insight
- simulation-generated data



### AI modeling

- Model design and selection
- model tuning(transfer learning)
- hardware accelerated training(ex: CPU, GPU, TPU)
- interoperability

### Simulation and Test

- Integration with complex system
- System simulation
- System verification and validation

### Deployment

- Embedded devices
- enterprise systems
- edge, cloud, desktop

## AI 應用成功的三大要素

### Data

- 在Paperwithcode 的網站列出的開源資料庫高達4539個，和影像相關的達到1444個，這些是否符合現實實際應用的狀況? No!
- AI迷思(對於AI的期許會太高)
  - 人看的出來的東西，數據有類似的資料AI就應該可以辨識的出來?
    - AI看的是數字，不是看圖
      - 把原本是熊貓的照片加上雜訊後反而變成是長臂猿
    - Google 的影像辨識把黑人辨識成大猩猩，因為涉及種族的議題，導致Google被罰錢
  - 收集的數據不符合現實實際狀況會造成模型誤判的問題
    - 把obama的照片降低解析度再透過AI模型還原，結果黑人變成白人
  - 測試透過模型自動捕捉足球比賽的鏡頭，結果模型一直不小心把裁判的光頭誤判成足球而追蹤到錯誤的位置
- 多少資料量才夠進行AI應用?資料量越多越好，越多AI越準?
  - 數據需要強調的是質，並非量
  - 問題在於資料的母體是什麼?
  - 202109 Gartner 針對 AI 的 Hypr Cycle 提出的四個趨勢
    - 負責任的AI(Responsible AI)
    - **小而廣的數據方法(Small and Wide Data)**
      - 問題在於資料的母體是什麼? 
      - 要盡量確保抽樣後的資料能接近現實世界的真實分佈
      - 有沒有辦法讓AI專案盡量收集到近似於真實世界中的data pool?
        - 沒有辦法，因為母體是未知的，只能盡量去更新資料集&模型，盡量滿足實務上的應用
    - AI平台的操作性(Operationliaztion of AI Platforms)
    - 有效利用資源(Efficient Use of Resources)
- Edge AI 要成功應用
  - 輕量化
  - 偵測準確
    - Data is a new oil(electricity) and ML is a way to produce it.
  - 在如何收集和有效的取得Data
    - Definition
    - Collection
    - Labeling
    - Selection
    - Augmentation
  - AI專案進行最強調的是資料
    - 場景收集定義、有效標註工具與方法、有效的資料選擇演算法
      - 需要分成不同的情境抽樣與標註
  - 專案流程
    - 資料庫的資料訓練AI模型 > 部署到邊緣裝置 >  應用 > 從邊緣裝置收集新&未標註的資料 > 重新人工標註資料 > 放回資料庫供下次訓練模型
  - 如何有小的資料選擇演算法
    - 怎麼選資料來標註
      - Active Learning: Measure the uncertainty/disagreement between the models
      - 把不確定性高的樣本拿去訓練模型反而會讓模型不容易學習，可以先透過模型過濾掉不確定性高的樣本，接著再進行模型訓練
    - 什麼才是應該要拿出來標註的資料
    - 傳統機器學習採用 Activate Learning

### Computing Resource

- Cloud
  - 高效能計算
  - 非同步(高延遲)結果推論
  - 巨量資料傳輸(影像傳輸封包過大)
  - 成本過高
- Edge
  - 低延遲
  - 高隱私
  - 較低成本

- 雲端計算後會將模型部署給邊緣裝置，而邊緣裝置則會上傳偵測結果回雲端運算環境
- 以 30FPS(1920*1920)的相機，要監控車流的即時情況為例
  - Cloud: 上傳原始影響需要100MB的傳輸，一分鐘需要5.8GB，一天8.2TB，即時降低解析傳輸仍需約0.5TB
  - Edge: 僅需要上傳每分鐘車流數據
- 到底要Cloud或Edge還是取決於你的應用方式
  - Computation Limitation: 單位時間計算能力
  - Memory Limitation: 模型參數量、精度
  - Power Consumption Limition: 主晶片、周邊元件功耗
  - 開發框架限制
    - OS: Windows, Linux, RTOS
    - AI engine: Pytorch, Tensorflow, Caffe, Mxnet,...
    - Inference NN Accelerator: CMSIS-NN, TensorFlow Lite, Openvino, TensorRT, Onnxruntime
  - 價格限制
- 評估指標
  - 偵測的準確度
  - 偵測的即時性
  - AI影響判讀: 準確度高，但即時性要求低
  - 交通、車流: 準確度和即時性都高
  - 人臉識別

- Edge 要做到 Realtime 需考慮的面向(Barrier)
  - Model Inference
    - Edge AI Model: Mobilenet, CspNet, or TinyML
    - Inference Platform
      - GPU
      - NPU
      - AI Accelerator
  - Post-Processing
    - Data bandwidth
    - Non-Maximum Suppression
    - Tracking(Algorithm)
    - Database寫入和上傳
    - Display/Log-Upload
  - 

### Learning Algorithm

- Which AI Algorithm?

  - ML
    - Linear Classifier
    - SVM
    - XGBoost
  - DL
    - VGG
    - ResNet
    - Inception
    - DenseNet
    - MobileNet

  

  - [Machine Learning Models Explained - AI Wiki (paperspace.com)](https://docs.paperspace.com/machine-learning/wiki/machine-learning-models-explained)

- 到底要怎麼挑&挑哪個?

  - 其實都不重要，重要的是要用AI來做什麼事情，絕對不是事事都深度學習
  - Classification
  - Semantic Segmentation
  - Object detection
  - Instance Segmentation

- AI開源模型或自建模型

- Deep learning model structure 建置就像在組積木

  - 自己用積木建立一個模型
  - 採用文獻的模型
  - 依據不同文獻來建立自己的模型

## 分析工具

- Anaconda
  - 適合電腦空間大，預先安裝 720 多個 Python 套件，所占容量約 3.0 Gb
- miniconda
- 適合電腦空有限，可自由控制需要安裝的 Python 套件，所占容量約 600 Mb
  - Cheatsheet
    - `conda --version` 檢視 conda 版本
    - `conda update PACKAGE_NAME`更新指定套件
    - `conda --help` 檢視 conda 指令說明文件
    - `conda list --ENVIRONMENT` 檢視指定工作環境安裝的套件清單
    - `conda install PACAKGE_NAME=MAJOR.MINOR.PATCH` 在目前的工作環境安裝指定套件
    - `conda remove PACKAGE_NAME` 在目前的工作環境移除指定套件
    - `conda create --name ENVIRONMENT python=MAIN.MINOR.PATCH` 建立新的工作環境且安裝指定 Python 版本
    - `conda activate ENVIRONMENT` 切換至指定工作環境
    - `conda deactivate` 回到 base 工作環境
    - `conda env export --name ENVIRONMENT --file ENVIRONMENT.yml` 將指定工作環境之設定匯出為 .yml 檔藉此複製且重現工作環境
    - `conda remove --name ENVIRONMENT --all` 移除指定工作環境
- IDE
  - VSCODE
  - Jupyter
    - [十大至简规则，用Jupyter Notebook写代码应该这样来](https://zhuanlan.zhihu.com/p/75547694)
    - [deshaw/jupyterlab-execute-time: Execute Time Plugin for Jupyter Lab (github.com)](https://github.com/deshaw/jupyterlab-execute-time)
  - PyCharm
    - 
- Ref
  - [Miniconda 手把手安裝教學 輕量化 Anaconda 客製化自由選擇](https://www.1989wolfe.com/2019/07/miniCONDAwithPython.html)
  - [15个好用到爆炸的Jupyter Lab插件](https://zhuanlan.zhihu.com/p/101070029)
  - [輕鬆學習 Python：conda 的核心功能](https://medium.com/datainpoint/python-essentials-conda-quickstart-1f1e9ecd1025)



## 建立虛擬環境

在開發 Python 專案時，很常遇見的問題是不同專案會需要不同的 Python 版本與不同的 package ，因此這時候就會需要建立出不同的環境進行開發，避免彼此在使用時收到影響。

- conda

  ```python
  # 確認目前有哪些虛擬環境
  conda env list
  
  # 建立虛擬環境 以建置 python 3.5版本的環境，並將環境命名為myenv為例
  conda create --name myenv python=3.5
  
  # 啟動虛擬環境
  activate myenv
  
  # 離開虛擬環境
  deactivate
  
  # 刪除虛擬環境
  conda env remove --name myenv
  ```



- 常用

  ```python
  conda create --name xxxxx
  conda activate xxxxx
  conda install -c conda-forge jupyterlab
  pip install pandas
  ```

- pip

  ```linux
  python3 -m venv dyu_tm_workshop
  source dyu_tm_workshop/bin/activate
  pip install jupyterlab
  jupyter lab
  pip install -r requirements.txt
  ```
  
  
  
  - [12. 虛擬環境與套件 — Python 3.10.0 說明文件](https://docs.python.org/zh-tw/3/tutorial/venv.html)


## Jupyter 插件

- Table of Content
- Autopep8: 自動排版程式碼
- [variable inspector](https://github.com/lckr/jupyterlab-variableInspector)
- ExecuteTime
- [jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time)
  - [Automatic cell execution timing in jupyter lab](https://stackoverflow.com/questions/56843745/automatic-cell-execution-timing-in-jupyter-lab)
- https://towardsdatascience.com/5-extensions-that-will-make-you-switch-to-jupyter-lab-32c6b66ac755
- [Auto-completion](https://github.com/kiteco/jupyterlab-kite)

## Python basic

- [Data Types in Python](https://www.analyticsvidhya.com/blog/2021/05/data-types-in-python/)

### Work with txt file

- read mode
  - r: read only

```python
myfile = open('test.txt')
myfile.read() 
>>> 'Hello, this is a quick test file\nThis is the second line of the file'

myfile.close()

# readlines
myfile = open('test.txt')
myfile.readlines()
>>> ['Hello, this is a quick test file\n', 'This is the second line of the file']
```

- write

  ```python
  myfile = open('test.txt','w+')myfile.read()>>> ''myfile.write('MY BRAND NEW TEXT')myfile.seek(0)myfile.read()>>> 'MY BRAND NEW TEXT'myfile.close
  ```

  - 將 list 寫成txt

    若遇到中文在encoding的部分很容易出錯，需要在open時加上encoding='utf-8'的參數!

    ```python
    with open("stop_words.txt", "w", encoding="utf-8") as outfile:    outfile.write("\n".join(stopwords))
    ```

    

- append

  ```python
  myfile = open('whoops.txt','a+')
  myfile.write('MY FIRST LINE IN A+ OPENING')
  myfile.close()
  newfile = open('whoops.txt')
  newfile.read()
  >>> 'MY FIRST LINE IN A+ OPENING'
  newfile.close()
  
  myfile = open('whoops.txt', mode='a+')
  myfile.write('This is an added line, because i used a+ mode')
  myfile.seek(0)
  myfile.read()
  >>> 'MY FIRST LINE IN A+ OPENINGThis is an added line, because i used a+ mode'
  
  myfile.write('\nThis is a read new line, on the next line')
  myfile.seek(0)
  myfile.read()
  >>> 'MY FIRST LINE IN A+ OPENINGThis is an added line, because i used a+ mode\nThis is a read new line, on the next line'
  myfile.close()
  ```

- with

  ```python
  with open('whoops.txt', 'r') as mynewfile:
      myvariable = mynewfile.readlines()
  myvariable
  >>> 'MY FIRST LINE IN A+ OPENINGThis is an added line, because i used a+ mode\nThis is a read new line, on the next line'
  ```

  

### Loops

- for
- while



### Pandas

提供了高性能、易用的資料結構及資料分析工具


  - Pandas exercises

    - [Getting and knowing](https://github.com/guipsamora/pandas_exercises#getting-and-knowing)
    
    - [Filtering and Sorting](https://github.com/guipsamora/pandas_exercises#filtering-and-sorting)
    
    - [Grouping](https://github.com/guipsamora/pandas_exercises#grouping)
    
    - [Apply](https://github.com/guipsamora/pandas_exercises#apply)
    
    - [Merge](https://github.com/guipsamora/pandas_exercises#merge)
    
    - [Stats](https://github.com/guipsamora/pandas_exercises#stats)
    
    - [Visualization](https://github.com/guipsamora/pandas_exercises#visualization)
    
    - [Creating Series and DataFrames](https://github.com/guipsamora/pandas_exercises#creating-series-and-dataframes)
    
    - [Time Series](https://github.com/guipsamora/pandas_exercises#time-series)
    
    - [Deleting](https://github.com/guipsamora/pandas_exercises#deleting)
    
    - melt
    
    - pivot
    
    - ```
      pandas.set_option('display.max_rows', None)
      ```
    
- spread data: pd.pivot

- gather data: pd.melt

- explode: Pandas expand rows from list data available in column

  - https://stackoverflow.com/questions/39011511/pandas-expand-rows-from-list-data-available-in-column

  

  


  - Save you data to different excel sheets

    ```python
    import pandas as pd
    import xlsxwriter
    df = pd.DataFrame({'Fruits': ["Apple","Orange","Mango","Kiwi"],
                         'City' : ["Shimla","Sydney","Lucknow","Wellington"]
                      })
    print(df)
    excel_writer = pd.ExcelWriter('pandas_df.xlsx', engine='xlsxwriter')
    df.to_excel(excel_writer, sheet_name='first_sheet')
    df.to_excel(excel_writer, sheet_name='second_sheet')
    df.to_excel(excel_writer, sheet_name='third_sheet')
    excel_writer.save()
    ```

    

  - Ref

    - [pandas_exercises](https://github.com/guipsamora/pandas_exercises)
    - [資料科學家的 pandas 實戰手冊：掌握 40 個實用數據技巧](https://leemeng.tw/practical-pandas-tutorial-for-aspiring-data-scientists.html)
    - [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
    - [Pandas 魔法筆記(1)-常用招式總覽](https://www.finlab.tw/pandas-%e9%ad%94%e6%b3%95%e7%ad%86%e8%a8%981-%e5%b8%b8%e7%94%a8%e6%8b%9b%e5%bc%8f%e7%b8%bd%e8%a6%bd/)


### Numpy

必用的科學計算基礎包，底層由C實現，計算速度快。

- Functions
  - np.select

### datetime

```python
from datetime import datetime
today = datetime(year = 2019, month=2, day=28)
print(f'{today: %B %d, %Y}')
>>> February 28, 2019
```

- 參考資料

  - [Python's `strftime` directives](



### Matplotlib

- [matplotlib](https://matplotlib.org/gallery/index.html)
- [Python Gallery](https://python-graph-gallery.com/)
- [R-Gallery](https://www.r-graph-gallery.com/)
- [R-Gallery (Interactive plot)](https://bl.ocks.org/mbostock)
- [样式美化matplotlib.pyplot.style.use定制画布风格](https://zhuanlan.zhihu.com/p/37891729)

### Seaborn

- [seaborn](https://seaborn.pydata.org/examples/index.html)

### Regular Expressions

- Regular expressions allow for pattern searching in a text document.

- The syntax for regular expressions can be very intimidating at first.

- For special code, you need to use backslash allow python to understand that it is a special code.

- 正規表達式是使⽤⼀段語法，來描述符合該語法規則的⼀系列⽂本。常⽤簡稱：regex, regexp。 

- 正規表達式常⽤來處理⽂本資料。例如搜尋、過濾、新增、移除、隔離等功能。

- 正規表達式運作及基本語法

  - 正規表達式(regex)由兩種字元所組成：
    - 詮釋字元(metacharacters)。擁有特殊意義的字元。
    - 字⾯⽂字(literal)，或稱為⼀般⽂字。 
  - 可以把 regex 比喻成⼀段句⼦。詮釋字元是語法，字⾯⽂字是單字。 單字+語法=>句⼦，這個句⼦就是⼀種表達式。此表達式可⽤來尋找 匹配(matching)此規則的⼀系列⽂字。 
  - Regex 檢驗的對象是⽂本中的「⾏」，⽽不是單詞。

- special code

  - ^:以...開頭
  - $：以...結尾

- Sample Code

  - [Regular Expressions](https://github.com/TLYu0419/NLP_Natural_Language_Processing_with_Python/blob/master/00-Python-Text-Basics/02-Regular-Expressions.ipynb)

- functions

  - re.search:在一個字符串中搜索匹配正則表達式的第一個位置，返回match對象
  - re.match：從一個字符串的開始位置起匹配正則表達式，返回match對象
  - re.findall：搜索字符串，以列表類型返回全部能匹配的字串
  - re.split：將一個字符串按照正則表達式匹配結果進行分割，返回列表類型
  - re.finditer：搜索字符串，返回一個匹配結果的迭代類型，每個迭代元素是match對象
  - re.sub：在一個字符串中替換所有匹配正則表達式的字串，返回替換後的字符串

- 用法

  - 函數式用法(一次性操作)

    ```python
    rst = re.search(r'[1-9]\d{5}', 'BIT 100081')
    ```

  - 物件式用法(多次操作)

    將正則表達式的字符形式編譯成正則表達式對象

    ```python
    pat = re.compile(r'[1-9]\d{5}')
    rst = pat.search('BIT 100081')
    ```

  - 只保留中文

    ```python
    import re
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") # 匹配不是中文、大小寫、數字的其他字符
    string1 = '@ad&*jfad張132（www）。。。'
    string1 = cop.sub('', string1)
    string1
    >>> adjfad張132www
    ```

  - 用空白串接list中的資料

    ```python
    ' '.join(['Hello', 'word'])
    ```

    

- 參考資料

  - [Regexone](https://regexone.com/)
    - 這是⼀個互動式學習網站，裡⾯有15道題⽬，讓學員們練習 regex。匹配的結果會即時顯⽰，相當適合練習建構regex。 
  - [Pythex](https://pythex.org/)
    - 線上建構regex，並測試結果是否能匹配⽂本。 
  - [常⽤ Regular Expression 範例](https://note.artchiu.org/2009/09/24/%E5%B8%B8%E7%94%A8-regular-expression-%E7%AF%84%E4%BE%8B/)
    - 常⽤的regex patterns參考。

  - [Regular expression operations](https://docs.python.org/3.9/library/re.html)



### PyPDF2

- The PyPDF2 library is made to extract text from PDF files directly from a word processor, but keep in mind: NOT ALL PDFS HAVE TEXT THAT CAN BE EXTRACTED!

  - Some PDFs are created through scanning, these scanned PDFs are more like image files.

  ```python
  pip install PyPDF2
  
  import PyPDf2
  myfile = open('US_Declaration.pdf', mode='rb')
  pdf_reader = PyPDF2.PdfFileReader(myfile)
  pdf_reader.numPages
  >>> 5
  
  # read pdf
  page_one = pdf_reader.getPage(0)
  page_one.extractText()
  >>> 'Declaration of Independence ...'
  myfile.close()
  
  # write pdf
  f = open('US_Declaration.pdf', 'rb')
  pdf_reader = PyPDF2.PdfFileReader(f)
  first_page = pdf_reader.getPage(0)
  pdf_writer = PyPDF2.PdfFileWriter()
  pdf_writer.addPage(first_page)
  pdf_output = open('MY_BRAND_NEW.pdf', 'wb')
  pdf_writer.write(pdf_output)
  pdf_output.close()
  f.close()
  ```

### pyinstaller

- [Python打包後的執行檔檔案太大?Pyinstaller與Numpy的那些事](https://medium.com/@rick.huang1609/python打包成執行檔後檔案太大-pyinstaller與numpy的那些事-dcc75ff9d42c)

- [python實戰筆記之（10）：使用pyinstaller打包python程式](https://www.itread01.com/content/1547118036.html)

### PySimpleGUI

- [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/)

