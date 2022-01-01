[TOC]

# 資料收集

- 資料蒐集的方式可以分為以下幾種，說明如下

## 調查問卷

- 傳統的資料蒐集方式，由領域知識的專家設計具有信、效度的問卷，並用有代表性的方式發放、回收問卷，最後將資料登入至電腦中
- 將問卷內容登入至網路問卷平台，並藉由連結的方式發放問卷，請受訪者在線上填答問卷
  - [Google表單](https://www.google.com.tw/intl/zh-TW/forms/about/)
  - [Survey Monkey](https://www.surveymonkey.com/)
  - [Survey Cake](https://www.surveycake.com/)

## 資料庫

- 中/大型的公司通常會建置資料庫，藉此統一存放公司各項業務的資料表

- MySQL

  - [MySQL Insert](https://www.mysqltutorial.org/mysql-insert-statement.aspx)

### SQL

[SQL_Tutorial](https://github.com/TLYu0419/DataScience/blob/master/02_Data_Collenction/SQL_Tutorial.ipynb)

- create table


- drop table

- ALTER

- INSERT

- UPDATE

- DELETE

- SELECT

- AS

- ORDER BY

- WHERE

  - =：等於
  - \>:大於
  - \<:小於
  - <>: 不等於
  - \>=: 大於等於
  - <=：小於等於

  - AND/OR

  - IN/NOT IN


- BETWEEN

- LIKE

- LIKE ANY

- GROUP BY

- COUNT

- DISTINCT

- SUM

- AVG

- HAVING

- UNION

- UNION ALL

- CAST

  - 文字形態轉數字

  - 數字形態轉文字

  - 文字轉時間

  - 文字轉日期

- 日期運算

  - 將時間轉為timestamp format

  - 計算兩個timeformat的時間差距

- LPAD/RPAD

  - 補0

- CASE WHEN

- QUANTILE

- JOIN

  - INNER JOIN
  - LEFT JOIN

- REPLACE

- 宣告變數

- 萃取時間

  - 當天日期

  - 當年

  - 當月

  - 當號

- STRTOK

  - 用底線切割字串，並取第一個值

- QUALIFY

  - 可以直接將 row over 的排序結果放在篩選條件，不用多做一次子查詢






### Hive

- 可以透過SQL語言作資料查詢

- Hadoop 生態圈

- 透過Hive幫忙SQL語法把操作複雜的MapReduce過程

- 常用指令

  - 查看欄位及欄位型別

  ```HIVE
  DESCRIBE tables_name
  DESC table_name
  ```

  - 查看欄位細項內容

  ```HIVE
  DESCRIBE FORMATTED table_name
  ```

  - 查看 CREATE SCHEMA

  ```HIVE
  SHOW CREATE TABLE table_name
  ```

  - Creat tale

  ```HIVE
  CREATE TABLE [IF NOT EXISTS] [db_name.]table_name 
  (col_name data_type [COMMENT 'col_comment'],...) 
  [PARTITIONED BY (col_name data_type [COMMENT 'col_comment'], ...)] 
  [ROW FORMAT DELIMITED]
    [FIELDS TERMINATED BY char [ESCAPED BY char]
  [STORED AS file_format]
  ```

  - Insert Data

    Append Data

  ```HIVE
  insert into table
  ```

  - 先清空Data，再insert Data

  ```HIVE
  insert overwrite table
  ```

  - 刪除資料表

  ```HIVE
  DROP TABLE table_name
  ```

  - 清空資料表

  ```HIVE
  TRUNCATE TABLE table_name
  ```

- Partition vs. Index

  在構建數據模型時，可以根據數據的大小和預期的使用模式確定對索引和/或分區的最佳使用。

  Partition：目的是做數據區隔，縮小讀取/查詢的數據量。

  - 底層其實就是利用目錄存放，來達到縮小資料範圍。

  - 每個分區會創建子目錄
  - 使用where子句，來指定分區列
  - PARTITION 欄位必須放在最後面

  Index：目的是加快對Table中的搜索速度。

  - 僅限於單表
    - 建立時會建一個單獨的表
  - 可以進行分區
    - 但不能與外部表一起使用

- rlike: 相當於 sql 中的 like any



### SPARK

SPARK都是在記憶體裡面執行，所以速度會比較快

RDD、YARN、

Shuffle

- Row-oriented

  - inset/ update/delete都很快和輕量

  - 有做index的dataframe
  - 建index會犧牲建立資料表的時間，換取insert/update/delete的效率
- Column-Oriend
  - 用在大數據的資料處理
  - 方便做資料壓縮，儲存

[Tutorial](https://github.com/TLYu0419/DataScience/blob/master/02_Data_Collenction/Pyspark_Tutorial.ipynb)

- Installation

  ```python
  !pip install pyspark
  ```

- Load Data

- Preprocessing

  - select column
  - add  column
  - drop column
  - rename column
  - drop na

- Feature engineer

  - Fill na
    - specific value
    - statistic value
  - Filter operation
    - single rule
    - and
    - or
    - not

- groupBy

  - sum
  - mean
  - count

- agg

- Modeling

- Evaluation

#### FAQ

- [Pyspark: Exception: Java gateway process exited before sending the driver its port number — SparkByExamples](https://sparkbyexamples.com/pyspark/pyspark-exception-java-gateway-process-exited-before-sending-the-driver-its-port-number/)
- [Apache Spark Tutorial | Spark tutorial | Python Spark - YouTube](https://www.youtube.com/watch?v=IQfG0faDrzE)

## 公開資料集

- 政府/企業釋出的資料集，供民眾或資料科學家進行分析、解決問題
  - [openml](https://www.openml.org/search?type=data)
  - [UCI MLdataset](https://archive.ics.uci.edu/ml/datasets.php)
  - [台灣社會變遷基本調查](https://www2.ios.sinica.edu.tw/sc/cht/home.php)
  - [經濟地理資訊系統](https://egis.moea.gov.tw/EgisWeb/)
  - [捷運各站進出量統計](https://www.metro.taipei/cp.aspx?n=FF31501BEBDD0136)



## 網路爬蟲

### 基礎知識

#### 資料來源與檔案存取

- 資料釋出的三個來源
  1. 檔案: 資料會包成檔案提供下載，格式可能包含常⽤的標準格式，例如「CSV」、「JSON」等等通⽤的格式
  2. 開放接口（API）: 提供程式化的連接的接⼝，讓⼯程師/分析師可以選擇資料中要讀取的特定部分，⽽不需要把整批資料事先完整下載回來 
  3. 網⾴爬蟲: 資料沒有以檔案或 API 提供，但出現在網⾴上。可以利⽤網路爬蟲程式，將網⾴的資料解析並保存所需的部分
- 資料的來源⽅式很多，檔案 & API 是由資料擁有者主動釋出，爬蟲則是資料擁有者被動公開的。所以需要取得資料時，通常會先考慮前兩者⽅法，真的無法才使⽤網路爬蟲。

#### 資料格式

- csv

  - CSV（Comma Seperated Values）逗號分隔值，是⼀種常⾒的資料格式，使用逗號將不同欄位做為分隔。可以使⽤⼀般的⽂字編輯器以原始格式開啟，也可以使⽤ excel 或 number 等試算表軟體以表格⽅式開啟。

  - 優點

    - 結構單純
    - ⼈機皆可讀
    - 檔案⼩

  - 缺點

    - 未限定編碼(big5, utf-8 … )
    - 值內有逗點或換⾏可能造成欄位判斷錯誤
    - 第⼀⾏不⼀定是欄位名稱

    ```python
    import csv
    spamReader = csv.reader(open('egg.csv'), delimiter='',quotechar='|')
    for row in spamReader:
        print(','.join(row))
    ```

    

- json

  - JSON（JSON stands for JavaScript Object Notation）JavaScript 物件格式， 是⼀種延伸⾃ JavaScript 物件來儲存和交換簡單結構的輕量級純⽂字資料交換格式。⼀般格式如下，每⼀筆資料都會⽤屬性 + 數值的格式紀錄，也可以是巢狀資料。
  - 優點
    - 可以存放結構較複雜的資料
    - ⼤部份瀏覽器都⽀援 
  - 缺點
    - 檔案較⼤（不過比XML⼩）
    - 不⼀定適合轉換成表格型式

- XML

  - XML（eXtensible Markup Language）可延伸標記式語⾔，是⼀種標記式語⾔，利⽤標籤紀錄資料的屬性與數值，常⽤來處理包含各種資訊的資料等。

  - 優點

    - 可以存放結構較複雜的資料
    - ⼤多瀏覽器可幫忙排版成較易讀格式 

  - 缺點

    - 檔案較⼤
    - 不⼀定適合轉換成表格型式

  - XML 檔案格式會利⽤ ... 標籤的⽅式記錄資料：

    ```html
    <標籤名稱 屬性='值'> 內文 </標籤名稱>
    ```

  - XML⽂件的字元分為標記（Markup）與內容（content）兩類。標記通常以<開頭，以>結尾；每⼀個標籤代表⼀個元素，元素當中有屬性與內容兩種設定。

  - 解析工具

    - xml.dom

      將 XML 資料在記憶體中解析成⼀個樹狀結構，通過對樹的操作來操作。

    - xml.etree

      輕量級的 DOM，具有⽅便友 好的API。程式碼可⽤性好， 速度快，消耗記憶體少。

    - xmltodict(推薦)

      將 XML 轉成 Dict ，可以 利⽤ Dict 的⽅式做操作。

#### 下載檔案

- 在 Python 可以使⽤第三⽅套件「urllib」中的「urlretrieve」⽅法來下載檔案。 

- urllib 是⼀個⽤於網路資源（URL）操作的函式庫（library），例如：檔案下 載就是這⼀類。

  ```python
  from urllib.request import urlretrieve
  urlretrieve ("http://www.example.com/songs/mp3.mp3", "mp3.mp3") 
  ```

#### 路徑的⽤法

- 在 Python 程式當中，有兩種表⽰路徑的⽅法

  - 相對路徑： 
    - 「 ./data/sample.csv」=> 與程式相同⽬錄下 data 資料夾中的 sample.csv 檔案
    - 「 ../data/sample.csv」=> 程式的前⼀層⽬錄下 data 資料夾中的 sample.csv 檔案 
  - 絕對路徑：
    - 「C:\Users\cupoy\Desktop\sample.csv」=> windows 環境中，桌⾯的 sample.csv 檔案
    - 「/Users/cupoy/Desktop/sample.csv」=> mac 環境中，桌⾯的 sample.csv 檔案

- 路徑的寫法有可能因為不同的作業系統⽽產⽣歧義，因此我們通常可以搭配 OS 這個內建套件來幫我們處理路徑字串。

  - 例如：windows 環境⽤ 「\」，mac 環境⽤ 「/」

    ```python
    import os
    os.path.join('/hello/','good/boy/','doiido')
    ```

#### Python File I/O

- File I/O 全名叫 File Input and Output，意思是如何在程式當中存取⼀個外部 的檔案。在⼤部分的程式語⾔當中，這是⼀個基本功能，不⽤使⽤額外的套件。

- 在檔案存取時候，依照權限可以分為以下幾種：

  - ‘r’ read，讀取檔案 

  - ‘w’ write，寫入檔案（會覆蓋原本的內容）

  - 'a' append，寫入檔案（會對已存在的的部分增加內容）

    ```python
    # 讀取檔案
    fh = open("example.txt", "r")
    fh.read()
    fh.close()
    
    # 寫入檔案
    fh = open("example.txt", "w")
    fh.write("To write or not to write\nthat is the question!\n")
    fh.close()
    ```

- 讀取檔案

  - read()：⼀次將整個⽂件讀成⼀個字串

  - readline()：⼀次讀取⽂件中的⼀⾏資料成字串

  - readlines()：將整個⽂件逐⾏存成⼀個列表

    ```python
    f = open("test.txt", "w")
    print(f.read())
    print(f.readline())
    for line in f.readlines():
     print(line)
    f.close()
    ```

- 寫入檔案

  - write()：將⼀個字串寫進檔案

  - writelines()：將⼀個字串列表寫進檔案

    ```python
    f = open("test.txt", "w")
    s = ‘Hello World’
    f.write( seq )
    seq = ["Hello", “ ”,"World"]
    f.writelines( seq )
    f.close() 
    ```

- 從檔案中整理資料

  - ⼀個 File 在 Open 之後，如果沒有 Close，則會將檔案的狀態顯⽰為被佔 ⽤，因此可能造成資源的浪費或是其他⼈無法使⽤該檔案。 

  - Python 提出了⼀個 With 的語法，稱為資源管理器，⽤於 File 存取時能夠⾃ 動在使⽤完畢後直接關閉佔⽤。

    ```python
    with open("example.txt", "w") as fh:
     fh.write("To write or not to write\nthat is the question!\n")
    with open("example.txt", "r") as fh:
     fh.read() 
    ```

#### 使⽤API 存取網路資料

- HTTP 協定

  「HyperText Transfer Protocol(HTTP) 是⼀種⽤⼾端瀏覽器和伺服端伺 服器之間溝通的標準協定，規範了「客⼾端」與「伺服器端」兩者間如 何傳輸資料。」

  - get: 請求獲取url位置的資源

  - head：請求獲取URL位置資源的響應消息報告，即獲取資源的頭部信息

    > 網站很大，或只想快速取得重要訊息時使用

  - post：請求向URL位置的資源後附加新的數據

  - put：請求向URL位置存儲一個資源，覆蓋原URL位置的資源

  - patch：請求局部更新URL位置的資源，及改變該處資源的部分內容

  - delete：請求刪除URL位置存儲的資源

- Request & Response

  HTTP 協定簡單來說就是：使⽤者端發送⼀個「Request 請求」，伺服器端根據請求回傳⼀個「Response 回覆」

- Request 可以分成幾種⽅法

  HTTP 將請求（Request）依據使⽤情境分成兩種⽅法：GET & POST。GET ⽅法將要傳送的資料以 Query String 的⽅式接在網址後⾯。Query String 是⼀種 Key-Vaule 的組成。POST ⽅法則是將傳送的資料寫在傳送的封包當中，從網址上看不到。

  - GET：將資料帶在網址上，利⽤ ?Key=Value 接在網址之後
  - POST：⽤ form 的⽅式帶資料

- ⼀般網⾴網⾴會利⽤ HTML 表單來發送請求，可以再傳資料的時候 設置⽅法。

- Request 分為 Headers 與 Content

  - Header ⽤來帶「發送⽅」相關的資訊

  - Body 帶需要傳送給伺服器的資料

    ![](C:/Users/TLYu0419/Documents/Github/DataScience/images/request.jpg)

    

- Response 有幾種⽅式

  HTTP 的回應（Respone）常⾒有兩種格式，API 與 HTML View。兩者的差別 是回傳的 HTTP 內容不同。

  - API 通常會以 JSON 格式⽽成的字串，是提供給程式存取。
  - HTML View 是包含 JS、CSS、HTML 三種語⾔所組成的網⾴原始碼，會交由瀏覽器進⾏處理。

- Response 分為 Headers 與 Content

  - Header ⽤來帶「發送⽅」相關的資訊

  - Body 則會帶回傳給瀏覽器的資料

    ![](C:/Users/TLYu0419/Documents/Github/DataScience/images/response.jpg)

- Response 狀態代碼

  - 200 OK 代表請求已成功被伺服器接收、理解、並接受 
  - 3XX Redirection ⽤來重新導向，後續的請求位址（重新導向⽬標） 
  - 4XX Error ⽤⼾端看起來可能發⽣了錯誤，妨礙了伺服器的處理。 
  - 5XX Server Error 伺服器在處理請求的過程中有錯誤或者異常狀態發⽣

- 什麼是 API ？

  - 資料的來源⽅式很多，檔案 & API 是由資料擁有者主動釋出，爬蟲則是資料擁 有者被動公開的。所以需要取得資料的時，通常會先考慮前兩者⽅法，真的無 法才使⽤網⾴爬蟲。

  - 第⼆種資料的發布⽅式是 API 的存取接⼝，API 的全名叫應⽤程式介⾯ （Application Programming Interface），表⽰程式與程式間的溝通⽅式或介 ⾯。換句話說，就是資料⽅寫好⼀組程式，提供你可以⽤程式的⽅式與其串接 資料。

  ```python
  import requests
  # 引入函式庫
  r = requests.get('https://github.com/timeline.json')
  # 想要爬資料的⽬標網址
  response = r.text
  # 模擬發送請求的動作
  ```

- 加入 headers

  - Headers 是指從這個 API 的哪個部分來來獲知請求中的內容是使⽤用何種編碼⽅方式，需輸入的欄欄位包含了 Key 和 Value 兩部分。通常會包含發送資料⽅方的資訊，例如「時間」、「權限」、「瀏覽器資訊」等等的。
  - 所以 Headers 可以視為是最基本的檢查機器，可以用來判斷發出 Request 的那⼀⽅是否為⼀個正常的來源。因此我們在這邊可以加上⼀些資訊，讓我們利⽤Python 發出的 Request 更像是⼀個正常的使⽤者行為。
  - 從瀏覽器的開發者工具中可以在這裡看出瀏覽器帶了⼀堆 Headers 在請求中，我們可以從這裡找到要加入哪些 headers 的參數。在一開始不確定的情況下，會建議把所有參數都加上去！

  ```python
  import requests
  # 定義標頭檔內容
  headers = {'user-agent': 'my-app/0.0.1'}
  r  = requests.get('https://www.zhihu.com/api/v4/questions/55493026/answers',
                    headers=headers)
  response = r.text
  ```


#### parse html code

```python
string = '/pycontw/?refid=18&_ft_=top_level_post_id.10161302929498438%3Acontent_owner_id_new.160712400714277%3Apage_id.160712400714277%3Astory_location.6%3Astory_attachment_style.question%3Atds_flgs.3%3Aott.AX_CdJrLpsZIe7E7%3Apage_insights.%7B%22160712400714277%22%3A%7B%22page_id%22%3A160712400714277%2C%22page_id_type%22%3A%22page%22%2C%22actor_id%22%3A160712400714277%2C%22dm%22%3A%7B%22isShare%22%3A0%2C%22originalPostOwnerID%22%3A0%7D%2C%22psn%22%3A%22EntGroupQuestionCreationStory%22%2C%22post_context%22%3A%7B%22object_fbtype%22%3A657%2C%22publish_time%22%3A1622258227%2C%22story_name%22%3A%22EntGroupQuestionCreationStory%22%2C%22story_fbid%22%3A%5B10161302929498438%5D%7D%2C%22role%22%3A1%2C%22sl%22%3A6%2C%22targets%22%3A%5B%7B%22actor_id%22%3A160712400714277%2C%22page_id%22%3A160712400714277%2C%22post_id%22%3A10161302929498438%2C%22role%22%3A1%2C%22share_id%22%3A0%7D%5D%7D%2C%22197223143437%22%3A%7B%22page_id%22%3A197223143437%2C%22page_id_type%22%3A%22group%22%2C%22actor_id%22%3A160712400714277%2C%22dm%22%3A%7B%22isShare%22%3A0%2C%22originalPostOwnerID%22%3A0%7D%2C%22psn%22%3A%22EntGroupQuestionCreationStory%22%2C%22post_context%22%3A%7B%22object_fbtype%22%3A657%2C%22publish_time%22%3A1622258227%2C%22story_name%22%3A%22EntGroupQuestionCreationStory%22%2C%22story_fbid%22%3A%5B10161302929498438%5D%7D%2C%22role%22%3A1%2C%22sl%22%3A6%7D%7D&__tn__=%7E%7E-R'

import urllib
urllib.parse.unquote(string)
>'/pycontw/?refid=18&_ft_=top_level_post_id.10161302929498438:content_owner_id_new.160712400714277:page_id.160712400714277:story_location.6:story_attachment_style.question:tds_flgs.3:ott.AX_CdJrLpsZIe7E7:page_insights.{"160712400714277":{"page_id":160712400714277,"page_id_type":"page","actor_id":160712400714277,"dm":{"isShare":0,"originalPostOwnerID":0},"psn":"EntGroupQuestionCreationStory","post_context":{"object_fbtype":657,"publish_time":1622258227,"story_name":"EntGroupQuestionCreationStory","story_fbid":[10161302929498438]},"role":1,"sl":6,"targets":[{"actor_id":160712400714277,"page_id":160712400714277,"post_id":10161302929498438,"role":1,"share_id":0}]},"197223143437":{"page_id":197223143437,"page_id_type":"group","actor_id":160712400714277,"dm":{"isShare":0,"originalPostOwnerID":0},"psn":"EntGroupQuestionCreationStory","post_context":{"object_fbtype":657,"publish_time":1622258227,"story_name":"EntGroupQuestionCreationStory","story_fbid":[10161302929498438]},"role":1,"sl":6}}&__tn__=~~-R'
```



#### 參考資料

- 資料來源與檔案存取
  - [Reading and Writing CSV Files in Python](https://realpython.com/python-csv/)
- 資料格式
  - [Difference Between XML and HTML](https://techdifferences.com/difference-between-xml-and-html.html)
  - [淺談 HTTP Method：表單中的 GET 與 POST 有什麼差別？](https://blog.toright.com/posts/1203/%E6%B7%BA%E8%AB%87-http-method%EF%BC%9A%E8%A1%A8%E5%96%AE%E4%B8%AD%E7%9A%84-get-%E8%88%87-post-%E6%9C%89%E4%BB%80%E9%BA%BC%E5%B7%AE%E5%88%A5%EF%BC%9F.html)
  - [[不是工程師] 休息(REST)式架構? 寧靜式(RESTful)的Web API是現在的潮流？](https://progressbar.tw/posts/53)
  - 介紹常見的五種HTTP Method
- [HTML URL Encoding Reference](https://www.w3schools.com/tags/ref_urlencode.ASP)

### 靜態網頁爬蟲

#### HTTP 網⾴架構-靜態網⾴

- HTTP 協定

  - HyperText Transfer Protocol(HTTP) 是⼀種⽤⼾端瀏覽器和伺服端伺服器之間溝通的標準協定，規範了「客⼾端」與「伺服器端」兩者間如何傳輸資料。

- Request & Response

  - HTTP 協定簡單來說就是：使⽤者端發送⼀個「Request 請求」，伺服器端根據請求回傳⼀個「Response 回覆」
  - HTTP Status
    - 200 伺服器回應Data成功。
    - 206 取得片段資料，Http Request 中有的 Range 屬性，可以指定要取得那一段Bytes數。
    - 301 目標網頁移到新網址(永久轉址)。
    - 302 暫時轉址
    - 304 已讀取過的圖片或網頁，由瀏覽器緩存 (cache) 中讀取。
    - 401 需身分驗證，如 SSL key or htaccess pasword。
    - 403 沒有權限讀取，可能是 IP 被阻檔或是伺服器限制。
    - 404 伺服器未找到目標網址，檔案不存在。
    - 408 Client Request timeout
    - 411 沒有指定 content-length，使用 POST 傳送參數時，必須指定參數的總長度
    - 414 URL 太長導致伺服器拒絕處理。
    - 429 Requests 太多
    - 500 伺服器發生錯誤 : 可能是 htaccess 有錯
    - 503 伺服器當掉 : maybe is code dump
    - 505 不支此 HTTP 版本

- 靜態網⾴運作原理

  - 所謂的靜態網⾴，表⽰網⾴是在 Server-side 就已經產⽣回來的，所以你看的網⾴上的資料是固定的（除非重新要求 Server-side）。這樣時候，我們可以來解析⼀下那資料，網⾴，瀏覽器，是怎麼被串起來的呢？⼀般來說流程是這樣：

    1. 使⽤者（Client-side）發出請求，稱為是 Request。 

    2. 伺服器（Server-side）收到請求，根據請求處理後回應，稱為是 Response。 
    3. 產⽣的回應如果是純資料的話，屬於 API 的⼀種；如果是網⾴的話，就會回傳⼀個包含 HTML 標籤的網⾴格式。 
    4. 瀏覽器接收包含 HTML 標籤的網⾴格式，呈現網⾴給使⽤者。

- 網⾴的組成

- 每⼀個 HTTP Request 的回傳形式有兩種，⼀種是 API、⼀種是 HTML 。所謂的 HTML，其實就是現在⼤家所看到「網⾴」的原始碼。在真實的使⽤下，網⾴除了 HTML ，其中還包含了 CSS 與 JavaScript 兩種程式碼。

- HTML

  - HTML（超⽂本標記語⾔，HyperText Markup Language）⽤於結構化網⾴內容。DOM 指的是HTML的分層結構。每個尖括號中的標籤稱為⼀個元素（元素），網⾴瀏覽器通 過解析 DOM 來理解⾴⾯的內容。

    - 三個常⽤來定位的 HTML 屬性：

      - id
      - class
      - name

    - HTML 標籤與屬性

      - 對於爬蟲程式⽽⾔，我們不需要非常熟悉 HTML 語法也沒關係，但是我們⾄少要知道幾個屬性：
        -  id 是 HTML 元素的唯⼀識別碼，⼀個網⾴中是不可重複的。主要是定位與辨識使⽤。 
        -  name 主要是⽤於獲取提交表單的某表單域信息，同⼀表單內不可重複。 
        -  class 是設置標籤的類，⽤於指定元素屬於何種樣式的類，允許同⼀樣式元素使⽤重複的類名稱。

      ![](C:/Users/TLYu0419/Documents/Github/DataScience/images/html_format.jpg)

- CSS

- 層疊樣式表（Cascading Style Sheets，CSS）⽤於為HTML⾴⾯的圖形表達設 計樣式.CSS樣式包括選擇器（selectors）和規則（rules），規則由屬性跟值組 成

- JavaScript

  - JavaScript的是⼀種動態腳本語⾔，它可以向瀏覽器發送命令，在網⾴加載完 成之後再去修改網⾴內容。腳本可以直接放在HTML中的兩個腳本標籤之間。
    - HTML DOM/CSS 的操作
    - HTML 事件函数/監聽
    - JavaScript 特效與動畫
    - AJAX 非同步的網⾴溝通

- ⼀般的網⾴通常會包含三種程式碼，負責不同⽤途

  - JavaScript：Behavior
  - CSS：Presentation
  - HTML：Content & Structure

- 網⾴運作的流程

  - 從瀏覽器取得網⾴回傳之後，會先載入 HTML 的內容，再把 CSS 的樣式加上 去，最後才會運⾏ JavaScript 的語法。

#### 靜態網⾴的資料爬蟲策略

- 網路爬蟲，簡單來說，就是模擬使⽤者的⾏為，把資料做⼀個攔截的動作。基本上可以簡化為：

  - 模擬 Request 
  - 攔截 Response 
  - 從 Response 整理資料

- 靜態爬蟲的處理

  利⽤「request」和「BeautifulSoup」這兩個函式庫進⾏

  1. 模擬 Request & 攔截 Response 
  2. 從 Response 整理資料

- Requests Library

  - Requests 是⼀個 Python HTTP 庫，該項⽬的⽬標是使 HTTP 請求更簡單，更⼈性化。 
  - 其主要⼯作作為負責網⾴爬蟲中的 HTTP Request & Respone 的部分。

- BeautifulSoup Library

  - Beautiful Soup 是⼀個 Python 包，功能包括解析HTML、XML⽂件、修復含 有未閉合標籤等錯誤的⽂件。這個擴充包為待解析的⾴⾯建立⼀棵樹，以便提 取其中的資料。 
  - 其主要⼯作作為負責網⾴爬蟲中的 解析資料 的部分。

  ```python
  soup = BeautifulSoup(html_doc) # => 將原始的 HTML 字串轉成物件
  # 利⽤標籤
  soup.title # <dom> => 取出第⼀個 title 標籤的物件
  soup.title.name # title => 取出第⼀個 title 標籤的物件的標籤名稱
  soup.title.text # The story => 取出第⼀個 title 標籤的物件的⽂字
  # 取出屬性
  soup.p['class'] # [title-class] => 取出第⼀個 p 標籤的物件中的 class 屬性
  soup.a[‘id’] # title-id => 取出第⼀個 p 標籤的物件中的 id 屬性
  # 利⽤ find ⽅法
  soup.find(id='titile-id')#  => 取出第⼀個 id = title-id 的物件
  soup.find(‘p’, class_=”content”)# => 取出第⼀個 class = content 的 p 標籤物件
  soup.find_all(‘a’, class_=”link”)# => 取出所有 class = link 的 a 標籤物件
  ```

#### 圖片下載

- 圖片爬蟲流程

  - 與⼀般爬蟲⽬標是⽂字的過程，圖片爬蟲其實只是要多送⼀次請求
    - BEAUTIFULSOUP - PARSE + 定位
    - 透過 SRC 屬性取得圖片位置
    - 送出請求

- 圖片副檔名會造成的問題

  - 副檔名是讓電腦決定要⽤甚麼軟體開啟檔案的提⽰
  - 更改副檔名不等於轉檔
  - 副檔名錯誤就無法正確開啟檔案

- 網路上顯⽰的圖片格式不⼀定是正確的

  - 在這個第三⽅服務⽤不同副檔名都可以檢視到同樣的圖片
  - http://i.imgur.com/Cgb5oo1.jpg
  - http://i.imgur.com/Cgb5oo1.png
  - http://i.imgur.com/Cgb5oo1.gif

- 下載圖片並以正確副檔名儲存

  - 為了要⽤正確的副檔名存檔，我們必須下載下來之後先判斷圖片格式這邊可以藉由 PIL.Image 來判斷格式

  ```python
  from PIL import	Image
  resp = requests.get(image_url,	stream=True)
  image = Image.open(resp.raw)
  print(image.format)	#	e.g.	JPEG
  # 假設我們重新組合圖片檔名與副檔名 logo.jpeg	之後
  # 可以⽤ requests 的⽅式也可以⽤ PIL 儲存圖片
  image.save('logo.jpeg')
  ```



### 動態網頁爬蟲

- 動態網⾴有別於靜態網⾴產⽣資料的⽅式。靜態網⾴是透過每⼀次使⽤者請求，後端會產⽣⼀次網⾴回傳，所以請求與回傳是⼀對⼀的， 有些⼈把他們稱為同步。
- 在動態網⾴的話，是透過 Ajax 的技術，來完成非同步的資料傳輸。換句話說，就是在網⾴上，任何時間點都可以發送請求給後端，後端只回傳資料，⽽不是回傳整個網⾴

#### HTTP 動態網頁架構說明與非同步取得資料

- AJAX（Asynchronous JavaScript and XML）
  - AJAX（Asynchronous JavaScript and XML）是⼀種在瀏覽器中讓⾴ ⾯不會整個重載的情況下發送 HTTP 請求的技術。使⽤ AJAX 來與伺 服器溝通的情況下，不會重新載入整個⾴⾯，⽽只是傳遞最⼩的必要 資料。原⽣的老舊 AJAX 實現標準為 XHR，設計得⼗分粗糙不易使⽤，⽽ jQuery 其中的 AJAX 功能是前端早期相當普及的 AJAX 封裝， 使得 AJAX 使⽤起來容易許多。
- 非同步載入的優點
  - 提升使⽤者體驗
  - 節省流量

- 動態網⾴爬蟲如何進⾏
  - 動態網⾴與靜態網⾴最⼤的不同是資料是在什麼時間點取得的。動態網⾴是在瀏覽器已經取得 HTML 後，才透過 JavaScript 在需要時動態地取得資料。因此，爬蟲程式也必須要考慮動態取得資料這件事情，才有辦法正確地找到想要的資料。

- 先思考⼀下動態網⾴的運作流程
  1. 使⽤者（Client-side）發出請求，稱為是 Request。 
  2. 伺服器（Server-side）收到請求，根據請求處理後回應，稱為是 Response。 
  3. 產⽣的回應如果是純資料的話，屬於 API 的⼀種；如果是網⾴的話，就會回傳 ⼀個包含 HTML 標籤的網⾴格式。
  4. 瀏覽器接收包含 HTML 標籤的網⾴格式，呈現網⾴給使⽤者。 
     => 此時是還沒有資料的！ 
  5. 當瀏覽器解析 HTML 後，開始運⾏ JavaScript 時會動態的呼叫 API Request 取得資料。
  6. 瀏覽器中的 JavaScript 會將資料更新到現有的 HTML 上，呈現網⾴給使⽤ 者。

- 動態網⾴的爬蟲問題是什麼？
  - 動態網⾴必須藉由 JavaScript 在第⼀次 Request 後，再動態載入更多的資料。因此，依據過去傳統的 HTTP ⼀來⼀回的機制，會找不到時機點執⾏動態的 Request 更新。另外單純靠 Python 程式，也無法執⾏ JavaScript。

- 兩種策略
  - 模擬使⽤者打開瀏覽器
  - 模擬 JavaScript 取得新資料

- 法⼀：模擬使⽤者打開瀏覽器
  - 原本靜態爬蟲的策略是模擬 Request，那我們現在可以模擬更多⼀ 點，改為模擬使⽤者從「發出 Request」到「JavaScript 動態載入資料」的過程。也就是說，這邊的做法是從模擬使⽤者打開瀏覽器的⾏為，到模擬器執⾏JavaScript 動態載入之後。

- 法⼆：模擬 JavaScript 取得新資料
  - 另外⼀種⽅法是我們知道 Python 無法直接執⾏ JavaScript 的。但本質上 JavaScript 也是透過呼叫 API 的⽅式拉資料，因此我們只要模 仿 JavaScript 呼叫 API 這個動作，改由 Python 執⾏即可。

#### 瀏覽器開發者工具介紹

- ⼤部分的瀏覽器都會提供開發者⼯具，開發者⼯具主要會包含網⾴的 結構與網⾴資料的溝通兩⼤主要功能。
- 包含哪些資訊
  - 打開Chrome 開發者⼯具 在Chrome菜單中選擇 更多⼯具 > 開發者⼯具 在⾴⾯元素上右鍵點擊，選擇 “檢查” 使⽤ 快捷鍵 Ctrl+Shift+I (Windows) 或 Cmd+Opt+I (Mac)
  - Elements
    - 元素⾯板（ Elements ）：顯⽰網⾴中的 HTML 元素與佈局，可以在 這裡做簡單的修改。另外注意右⽅會顯⽰選取的 HTML DOM 的 CSS 樣式。
  - Console
    - Console 可以⽤來執⾏ JavaScript ，通常我們會在上⾯做簡單的測 試。
  - Source
    - Source 會顯⽰這次載入的網⾴所需要所有檔案，像是 CSS、JS 或是 圖片等等
  - Network
    - Network 會記錄所有網⾴存取過程中所有的 HTTP 傳輸，⽽每個 HTTP 都會包含 Request 與 Response。
  - Application
    - Application 主要會存放的是網⾴在瀏覽器的暫存資料，包含 Cookie 或是 Storate 。
- 開發者⼯具如何幫助爬蟲？
  - Elements：觀察需要取得的 DOM 位置與 Selector 定位。 
  - Network：⽤來觀察存取資料的 API Request（尤其是在動態網⾴ 爬蟲）。 
  - Application：有⼀種登入機制會把權限狀態存在 Cookie ，可以在 這裡查看。



#### 使用Selenium + BeautifulSoup 模擬瀏覽器執行

- 關於這種利⽤到 JavaScript 的非同步特性載入更多資料的網⾴稱為動 態網⾴。⽽爬蟲程式也會因為沒有執⾏到 JavaScript 導致資料不完全 的現象。 

- 第⼀種解法會採⽤ selenium 這樣的瀏覽器模擬⼯作，從模擬使⽤者打 開瀏覽器的⾏為，到模擬器執⾏JavaScript 動態載資料之後

- 利⽤ Selenium 模擬操作瀏覽器

  - Selenium 是⼀個瀏覽器⾃動化（Browser Automation）⼯具，讓程式 可以直接驅動瀏覽器進⾏各種網站操作。最早的⽬的是⽤來進⾏網⾴ 測試使⽤，這邊我們藉由特性來運⾏ JavaScript 作為爬蟲⽤。

  - 準備 Selenium 環境

    - 安裝 selenium 套件

      ```python
      $ pip install selenium
      ```

    - 下載 Chrome 驅動程式

      上⾯第⼀步驟只是安裝 Selenium 模組⽽已，必須要下載對應的瀏覽器 Chrome 的驅動程式（建議放在程式相同⽬錄下）：http://chromedriver.chromium.org/downloads

  - 範例：使⽤ Selenium 進⾏爬蟲

    ```python
    from selenium import webdriver
    browser = webdriver.Chrome(executable_path='./chromedriver')
    browser.get("http://www.google.com")
    browser.close()
    browser.page_source
    ```

    - 執⾏後會真的看到電腦打開⼀個新個瀏覽器，⽽且跳轉到設定的網址上！ 透過 browser.page_source 可以取出，⽬前網⾴上當下的 HTML，不過這是⼀個 HTML 格式的字串，此時就可以再利⽤ BeautifulSoup 進⾏解析。



#### 利用開發者工具，觀察模擬 API 存取

- 關於這種利⽤到 JavaScript 的非同步特性載入更多資料的網⾴稱為動 態網⾴。⽽爬蟲程式也會因為沒有執⾏到 JavaScript 導致資料不完全 的現象。 
- 第⼆種解法透過利⽤ Python 模仿 JavaScript 呼叫 API 這個動作，也 可以達到動態取得資料的⽬的。
- 利⽤ 開發者⼯具 觀察 JavaScript ⾏為
  - 可以從瀏覽器的開發中⼯具中提供的「Network」功能，去查看所有 網⾴中的傳輸⾏為。這種透過 JavaScript 動態載入的請求，就可以從 中觀察到。接著就可以利⽤我們前⾯談到的 API 爬蟲⽅式進⾏。
    1. 在網⾴上叫出 Console 切換到 Network Tab，中間選 XHR，這裡 會記載所有網⾴中的 API 呼叫
    2. 此時點選重新整理，會發現網址沒有 動（表⽰沒有發送新的 HTML 網⾴請 求），但畫⾯有新的內容出現，左下 ⾓也多了⼀次新的 API 呼叫。
    3. 點開就可以得到完整的 API 呼叫 內容，包含網址，Headers 和資 料。
  - 簡單來說，在⼀個動態網⾴中，我們可以簡單地透過開發者的⼯具的 觀察，知道 JavaScript 發送了哪些請求。換句話說，我們就可以模仿 這部分，改⽤ Python 來發出 API，將原問題簡化為前⾯講過的 API 存取。

### Scrapy爬蟲框架

- Scrapy 是為了持續運行設計的專業爬蟲框架

- 使用時機
  - 簡單&一次性的任務：requests
  - 大量&重複性的任務：Scrapy

#### 多網頁爬蟲實作策略介紹

- 多網⾴爬蟲概念

  - 多網⾴爬蟲基本上就是逐⼀對網址清單上的網址爬蟲 ⽽根據網址清單型式的不同會有額外的策略
    - ⾃訂清單列表：透過⽂件或是 List 紀錄⽬標網⾴網址 
    - HTML 清單列表：\<div>\<li>等 tag 紀錄⽬標網⾴網址

- 單⼀網站多網⾴爬蟲概念

  - 如果是要爬取單⼀網站下的多個網⾴，我們不太可能拿到所有⽬標網⾴的清 單，此時比較適合的策略是階層式搜尋網⾴並爬取
    1. 了解網站⽂件結構 
    2. 從網站⾸⾴逐⼀根據超連結 (e.g.  tag) 找到其他網⾴網址
    3. 爬取網⾴內容並檢查是否有超連結連到其他網⾴

- 跨網站多網⾴爬蟲概念

  - 概念上跟多網⾴爬蟲⼀樣可以列出多個⽬標網站的網址清單 再根據單⼀網站多網⾴爬蟲的概念逐⼀爬取 過程中有可能超連結會連到其他網站上的網⾴ 
  - 隨著需要搜索的次數愈多，不確定性愈⾼ 需要更謹慎的檢查每次⽬標網址的合法性

- 階層式搜尋的注意事項

  - 紀錄已搜尋過的網址，避免重複爬蟲與無窮回圈 

  - 合法超連結格式為絕對路徑

    - 相對路徑建議可以透過 urllib.parse.urljoin 轉換

  - 建議在處理網址問題時要先了解每個片段的意義 scheme://netloc/path;params?query#fragment	 

    其中比較重要的是判斷網域的 netloc 與路徑的 path

- 超連結的注意事項

  - 超連結可以不是網址格式
    - \<a>可以是其他非網址格式的型式

- 網址網域的注意事項

  - 超連結網址可以是任何網路位置
    - 網域建議可以透過 urllib.parse.urlparse 判斷
    - ⼦網域建議可以透過 tldextract.extract 判斷

- 爬蟲的禮貌運動

  - 網站擁有者有時候會限制爬蟲⾏為 (e.g. 搜尋引擎的爬蟲，可以爬全網站網 ⾴；⼀般爬蟲，只能爬⾸⾴的內容) 
    - 這些規則通常會放在⾸⾴底下的 robots.txt 
    - e.g. https://www.facebook.com/robots.txt
  - 建議開發者根據這些不允許存取的路徑， 讓爬蟲直接忽略

- 參考資料

  - [URL wkik](https://zh.wikipedia.org/wiki/%E7%BB%9F%E4%B8%80%E8%B5%84%E6%BA%90%E5%AE%9A%E4%BD%8D%E7%AC%A6)
    - 中⽂版的 wiki 詳細講解網址 URL 的⽂法與其意義 
  - [W3C 超連結屬性](https://www.w3schools.com/tags/att_a_href.asp)
    - 本篇有提到  tag 超連結的注意事項，並非所有值都適合 送請求做爬蟲，W3C ⽂件中有定義及範例可以幫助理解 
  - [關於 robots.txt](https://support.google.com/webmasters/answer/6062608?hl=zh-Hant)
    - Google 對於 robots.txt 的解釋，包含⽤途與限制 
  - [google/robotstxt](https://github.com/google)
    - Google 在 2019.07 開放 robots.txt 的解析器，該程式⼀直被 ⽤於 Google engine 服務，後來甚⾄發展成 Robots Exclusion Protocol (REP) 標準

#### Scrapy常用命令

- startproject：創建一個新工程
- genspider：創建一個爬蟲
- settings：獲得爬蟲配置信息
- crawl：運行一個爬蟲
- list：列出工程中所有爬蟲
- shell：啟動URL調試命令行

#### 建立流程 + 送出請求

- Scrapy 框架與套件的差異

  - 前⾯我們⼤多使⽤ requests + BeautifulSoup 套件的組合來實作 但是你必須要⾃⼰額外考慮很多爬蟲的細節（甚⾄你沒想過）
    - 先對誰送請求 (Scheduling) 
    - 資料儲存的 pipeline (ETL) 
    - etc …
  - 框架基於這些細節都有對應的程式碼，我們可以根據他們定義好的規則去設計 撰寫，細節則可以全部交給框架實作

- 建立 Scrapy 專案

  - Scrapy 並不僅僅是⼀個 Python 套件，⽽是⼀個框架 因此在建立 Scrapy 專案時還要遵循他的專案結構 

  - 除了⼀般 import scrpay 的⽤法以外，也提供了命令列的功能 我們可以透過 scrapy startproject [專案名稱] ⾃動產⽣⼀個符合框架的檔案結構

  - 舉例來說，當我們執⾏ scrapy startproject myproject 會看到 Scrapy ⾃動建立符合框架的檔案結構

    ![](https://lh3.googleusercontent.com/9rmqg6Ui0eCtaExRhUjjGyEYsj9u2qwfVugg0Ni4CEprcnZjPD6WnIToALjRiuQgD5ZASlmsh0jGwc21qYDPggjxCPDZE3IzalmkFh9p8_nV22xUG4p7G80EjCazCwC4n1FA57cufYDtORkiLkEa7MjcVixLIr88_ns6_EWqNMsbtVS0EZKxBUtHB56CaWmKV6zyLLNxV9gB1J7LiDjKaRvK4jQ11DUDsqYAlDJAZ57wOEEI-FmDqwU1I8NIxQmHMlWLWgsf85EqnrMxOuW_PKL-xrXp-Fo5R_QNcOsg7mdn5usNBR8Z6XTtaNHsk_zd0X_tPrzWKotoP_Fi4Irbxtpx6ZMMgCNxTYxx8ikFck22o4NHw6omM7Xk50xixHo6JsQDxyeuStQum0SHWUxrhGHoCtFg58EDCyInT4PQLrFiJM_k6aRojP1k7Qr43BiarXqpPNX8Kt2SUWQNiXdHC-cTWe4Y73yoIpyLqA5tZb73kdeyj5kNqvxcCRzYJh103CYQrr6NQPV69v813qThlL9kT774oqS_ysE6YLsvIF3ISF_VHurK6ti5gbEYsbPM4YU-zlaJ5W1lTKOnDMhQZmppzAN0PM52A0P8LdubuHFfPG3bxf4zWpFltMlyB8RPvqFw-uRVGKOqhrcYJPwrnnY9ygkNLagh4PXtsEyRSjB2mBBXb1CMnZrbDJnJscAfUSbCzVduSos6-lcZtX81JzJFBbSvnJvIrklY95F5P8YE_AKM5WcGat4=w553-h272-no)

- Scrapy 的第⼀隻爬蟲

  - 同樣地，我們也可以透過命令列來產⽣出⼀個符合框架的爬蟲類別 scrapy genspider [爬蟲名稱] [爬蟲⽬標網址]

  - 舉例來說，當我們執⾏ 
    scrapy genspider PTTCrawler www.ptt.cc

    ![](https://lh3.googleusercontent.com/kRJ0IAFM3RpR12xa7mpzwSsxjc1fhIu5pQHZpzUdXYEiwPv19RB6TI2YXwFkXgo3G_U-EC61SSB8aUNQX39OmdLJfNXncTFqsnOi6z98LwBH6Q7Bj24ixFOpiYo83F2yzLfC36TNjeKb_B4GwyARIYzWQRn3mKJlQta7qB4O3j3D5Jbgv0NmAVgflsaivJtL-Um-ND8udD_Vo6bArLUlLXoBLncB5FVaRa5Y0iHjZU8qunyKLgw4vR11HjFD_F0zGQDJxmuqIj8GTtF_o4ePTU3r_IFSXbZ8GqRoEEIuOkSjDiiLxVe42M8r6IHOqZP3hBNuP-D1y52CJqOBBILuoNvNjFQ1r-pR_dN9hvCDByoQ6GQfix_UMmPBix7Z7SGpp2ugFdc5N6ARajsWzQZgFl-zOUhxW5_GhcsSExyiS7xhNJwL8YCLMASSmBscQjTS6n5I32-Jn2HNUf4ROK09p7zTQwNpYAyRL0oCERAw80bCyMz1ytm5MWxtWR3YNn22t3LxsxATxtDPC8en73OGOjX-EnohEWl0LcgPyfW7JRJm8pw8cSbG77b6CEePYCW9slG-ell_hYCoxQoZarDahvuVb-LAAlRTMAeNeRkTkrhOqRj1iSyVv_JVlOOfvYsSPZrj7rRVgdFPKtYXCy5-_oF9c1tQxSE-Tl5cjcsue4k-qE2P1WrBJMuJmFyV6XrMlByrOa5ch_VkwS4fzrj_SxAJIQFGO44vYRnWKJjKpBV3dm5nJK4RkkY=w1681-h601-no)

- Scrapy 請求過程

  - 此時可透過指令開始爬蟲 scrapy crawl PTTCrawler 參考前⼀⾴的程式，這邊其實是因為 scrapy 在背後會呼叫 start_requests 進 ⾏爬蟲，為了⽅便理解我們將其內容寫下來
    1. 準備傳送請求
    2. 實際傳送請求
    3. 實際收到回應
    4. 傳送回應
    5. 解析回應的邏輯

#### XPath + Item Pipeline

- Scrapy 元素定位

  - 爬蟲最重要的兩個部份就是送請求跟定位元素 在介紹 Scrapy 之前我們使⽤的分別是 requests 與 BeautifulSoup 
  - 上個⼩節我們介紹了 Scrapy.Request 來取代 requests，為了符合框架設定所 以不適合混⽤，但定位元素並不限定 
  - 這邊為了程式碼的⼀致性，介紹 Scrapy 定位元素的⽅式
  - Scrapy 定位元素的⽅式分別有 XPath 與 CSS selector 兩種 與之前課程中介紹的概念差不多，只是呼叫的 function 不同
    - 送出 Scrapy.Request 後取得 Response 物件 
    - 透過 Response.selector.xpath() 或 Response.selector.css() 定位元素
  - 如同 BeautifulSoup ⼀樣，搜尋符合條件的元素可以選擇要回傳⼀個或是多個
    - BeautifulSoup
      - soup.find()/soup.find_all()
    - Scrapy.Selector
      - .get() / .getall()

- 定義資料格式

  - 通常我們爬完資料都會以字典型式傳遞與儲存 當爬蟲愈來愈多，資料的格式也愈來愈多種，將會變得難以管理 

  - 由於字典格式 (dict) 本⾝容易新增欄位與覆蓋 容易在沒有注意到的地⽅改變資料格式與型態

  - Scrapy 透過明確定義資料格式來解決這個問題

    ```python
    class Product(scrapy.Item):
        name = scrapy.Field()
        price = scrapy.Field()
        last_update = scrapy.Field()
    ```

    - scrapy.Field() 定義了每個資料屬性的型態 
    - Product() 定義了資料格式

- 定義資料格式

- 當爬完資料要儲存時，需要決定儲存成哪⼀種格式 (e.g. Product) Scrapy 框架會幫你檢查儲存格式的正確性

- 處理資料流程

  - 前⾯我們定義了資料格式 (Item) 
  - 框架會接著會進入資料處理的流程 (Itme Pipeline) 
  - 主要處理資料的⽬的包含 
    - 檢查爬到的數據是否正確 
    - 檢查是否重複，是否需要丟棄資料 
    - 將爬完的資料存到資料庫或是⽂檔

- 處理資料流程範例

  - Scrapy 框架主要處理資料的幾個時機點
    - process_item
      - 每個 Item Pipeline 都需要實作，⽤來檢查資料數據與是否丟棄等決定 
    - open_spider
      - 當爬蟲開啟時需要處理的流程 (e.g. 檢查資料庫是否可⽤) 
    - close_spider
      - 當爬蟲關閉時需要處理的流程 (e.g. 關閉資料庫連線)

- 資料處理的檔案位置

  ![](C:/Users/TLYu0419/Documents/Github/DataScience/images/scrapy.jpg)

- 參考資料

  - [Srcapy Selector 官⽅⽂件](https://docs.scrapy.org/en/latest/topics/selectors.html)

  - [進階功能：Item Loader](https://docs.scrapy.org/en/latest/topics/loaders.html)
  - 在 Scrapy 中定義好如何定位元素，以及該元素應該如何存到 Item 格 式中，爬蟲過程框架可以幫你⾃動爬玩送到 pipeline 處理的功能 

  - [進階功能：Scrapy Feed exports](https://docs.scrapy.org/en/latest/topics/feed-exports.html#topics-feed-exports)
  - 我們在 pipline 中寫入 JSON 只是熟悉 pipeline 的操作，實際上如果要 把資料存成某種格式應該參考 feed exports 的⽅式

#### API

- 我們在前⾯介紹如何執⾏ Scrapy 的爬蟲都是透過命令列
- 但其實框架本⾝有提供 API 讓我們可以從外部去呼叫並執⾏爬蟲甚⾄是其他元件，這樣可以⽅便我們串聯 其他非框架本⾝或是沒有提供的功能

#### 多網頁爬蟲

- 我們⽬前的爬蟲功能是對「所有給予的 PTT ⽂章網址」進⾏爬蟲 實作 PTT 多網⾴爬蟲的實作有兩個⽅向
- 外部決定網址 + 框架對給予網址進⾏爬蟲
  - 在外部 (e.g. main.py) 對⽂章列表進⾏爬蟲取得所有⽂章網址
  - 把所有⽂章網址傳入 scrapy 爬蟲 
- 框架爬⽂章列表 + ⽂章內容
- 這兩種⽅式都可以，但是先從外部取得網址的⽅式會比較慢 這邊我們可以更深入了解框架送請求的過程為什麼會比較快

- 原本 requests 的⽅式，程式會送出第⼀個請求後會等到第⼀個 response 傳回 來才會送第⼆個請求
- ⽽框架內的請求⽅式 yield scrapy.Request
- 在送出第⼀個請求後會直接送第⼆個請求，並不會卡著等第⼀個 response，⽽ 是等第⼀個 response 送回來的時候再處理
- 這種⽅式可以縮短因為網路延遲造成的等待，加速整個爬蟲過程

- Ref
  - [【知乎】Scrapy中的scrapy.Spider.parse()如何被調⽤?](https://www.zhihu.com/question/30201428) 
    - 參考 Scrapy 的架構圖，再透過該篇⽂章可以更加了解 parse 的時 候 yield request 跟 yield item 的差別



#### 流程

1. 啟動專案

   ```prompt
   scrapy stratproject project_name
   # scrapy stratproject und
   ```


教學

1. [[Scrapy 爬蟲] 什麼是Scrapy以及為什麼要用Scrapy 爬取網頁?](https://www.youtube.com/watch?v=0pWJHy_fNWA)

   1. conda install scrapy

   2. cd 到工作的資料夾

   3. scrapy startproject projectname

      ex scrapy startproject undnews

2. [[Scrapy 爬蟲] 如何撰寫第一支Scrapy 爬蟲以抓取蘋果即時新聞?](https://www.youtube.com/watch?v=fnwvYAtCFko)

3. [[Scrapy 爬蟲] 如何從蘋果新聞的清單聯結抓取下一層的內容頁面?](https://www.youtube.com/watch?v=w4PPlkJFzCo&t=42s)

4. [[Scrapy 爬蟲] 如何使用items.py整理Scrapy 爬取下來的資料並輸出成JSON檔?](https://www.youtube.com/watch?v=Me9SpR0SE08)

5. [[Scrapy 爬蟲] 如何使用pipelines.py將Scrapy 爬取下來的資料儲存置資料庫之中?](https://www.youtube.com/watch?v=Xq4yRuePSdk)

6. [[Scrapy 爬蟲] 如何使用Scrapy 的CrawlSpider 實現多網頁爬取?](https://www.youtube.com/watch?v=KSA12AKDr_o&t=3s)

   1. from scrapy.spiders import CrawlSpider, Rule
   2. from scrapy.linkextractors import LinkExtractor

7. [[Scrapy 爬蟲] 如何設置 Job 以分段爬蟲任務?](https://www.youtube.com/watch?v=2xjAArPnOH8)

   1. 當網頁很大，可以將工作分段暫停
   2. 分段
      1. scrapy crawl udnnews -s JOBDIR=job1
      2. 按 Ctrl + C 是暫停
      3. 再輸入一次一下命令即可繼續
      4. scrapy crawl undnews -s JOBDIR=job1



### 進階爬蟲技術

- 在前⾯的課程中，我們討論了⼀個網⾴從該如何思考和撰寫。接下來我們要討論的是「爬蟲可以順利拉到資料，然後呢？」我們針對這三個⽅向來做優化：
  - 反爬
  - 加速
  - 自動化更新

- 反爬是什麼？常見的反爬蟲機制有哪些？
  - 許多網站為了保護資料，避免網頁上的公開資訊被網頁爬蟲給抓取，因此有了「反爬蟲」的機制出現。爬蟲工程師也發展了出⼀系列「反反爬蟲」的策略！
    - 檢查 HTTP 標頭檔
    - 驗證碼機制
    - 登入權限機制
    - IP 黑名單

- 如何為爬蟲程式加速？
  - 第⼆種實務爬蟲需要考慮的問題是加速，當資料量龐⼤或是更新速度較為頻繁的狀況下。依照正常的爬蟲程式，可以會因此受到應⽤上的限制。所以必須⽤程式的⽅法，來思考如何加速爬蟲的處理速度。
    - 多線程爬蟲加速
    - 非同步爬蟲

- 利⽤排程⾃動化更新
  - 真實世界中的資料是瞬息萬變的，也代表資料會有更新的需求。但爬蟲爬的資料只是⼀個片刻，所以必須要思考如何與資料源上的資料做同步或是更新，確保拿到的資料是最新的。常⾒的做法可以利⽤⼀個排程機制，週期性地重新抓取資料。
    - 在迴圈中加上 Sleep
    - 利⽤ threading 的 Timer
    - 第三⽅套件 schedule
  - 參考資料
    - [Python爬蟲系统學習⼗⼀：常⾒反爬蟲機制與應對⽅法](https://blog.csdn.net/guangyinglanshan/article/details/79043612)
    - [Python爬蟲筆記（六）— 應對反爬策略](https://blog.csdn.net/dhaiuda/article/details/81410535)

#### 編碼

```python
import requests
resp = requests.get('http://www.baidu.com')
resp.status_code
>>>200
resp.text
>>># 這裡會出現許多亂碼看不懂
# 修正編碼
r.encoding
>>> 'ISO-8859-1'
r.apparent_encoding
>>>'utf-8'
r.encoding = 'uf-8'
r.text
>>> # 正常顯示內容
```



### 反爬

- 瀏覽器標頭與基本資訊

  - 檢查 HTTP 的發送請求⽅是否合法
    - 前⾯我們在提到網⾴的傳輸有講到 HTTP 協定，HTTP 會將網路的傳輸分為 「Request」和「Response」兩種⾓⾊。
    - 其中 Request ⼜可以分為幾個部分：
      - Header：瀏覽器⾃動產⽣，包含跟發送⽅有關的資訊。 
      - Body：網⾴服務真正要傳送的資料
    - Header 包含發送⽅的資訊
      - ⼀般來說，Header 可能會包含： 
        - 發送⽅的位址（Host）
        - 發送⽅的瀏覽器版本（User-Agent） 
        - 發送⽅的語⾔/格式 … 等等

  - 讓爬蟲程式也加上 Header
    - 因為 Header 是由瀏覽器⾃動產⽣，因此如果透過程式發出的請求預設是沒有 Header 的。透過檢查 Header 是最基本的反爬機制。
    - 解法：在爬蟲程式的 Request 加上 Header！

  ```python
  import requests
  headers = {'user-agent': 'my-app/0.0.1'}
  r = requests.get('https://www.zhihu.com/api/v4/questions/55493026/
  answers',headers=headers)
  response = r.text 
  ```

  - 怎麼檢查 Request 要帶哪些 Header？

    1. 右鍵點選檢查
    2. 下方點選 Network
    3. 找到網址對應的請求
    4. 切換到 Headers 項目
    5. 找到 Request 的 Headers

  - 在 Request 上加上 Headers

    - 實際上的 Headers 應該參考瀏覽器的。但範例為了⽅便，我們這邊是先⾃⼰定 義⼀的比較基本的。但不是每⼀個網站都可以通過，比較保險的⽅式建議模仿 瀏覽器所帶出的標頭且整理成 dict 的型態（如下）

    ```pythn
    headers = {
     'accept': '...',
     'accept-encoding': '...',
     'accept-language': '...',
     ...
     'user-agent': '...'
    } 
    ```

- Robots協議

  - Https://www.jd.com/robots.txt
  - 網頁允許/不允許的爬蟲權限與內容

- 驗證碼處理

  - 驗證碼機制是許多網站再傳送資料的檢查機制，對於非⼈類操作與⼤量頻繁操 作都有不錯的防範機制。

  - 驗證碼是⼀種圖靈測試

  - CAPTCHA 的全名是「Completely Automated Public Turing test to tell Computers and Humans Apart」，或「全⾃動區分電腦與⼈類的圖靈測試」， 實作的⽅式很簡單，就是問⼀個電腦答不出來，但⼈類答得出來的問題。

  - 爬蟲該怎麼辦？

  - 爬蟲在實作上遇到驗證碼的做法會是這樣，先把圖抓回來， 再搭配圖形識別⼯具找出圖中的內容。

  - 環境⼯具準備

    - Tesseract

      - Tesseract 是⼀個OCR庫(OCR是英⽂Optical Character Recognition的縮寫)，它⽤來對⽂ 字資料進⾏掃描，然後對影像檔案進⾏分析處理，獲取⽂字及版⾯資訊的過程 
      - 安裝⽅式：https://github.com/tesseract-ocr/tesseract/wiki

    - pytesseract

      - 在 Python 中呼叫 Tesseract 的套件

      - 安裝⽅式（利⽤ pip）：https://pypi.org/project/pytesseract/

      ```python
      import requests
      import pytesseract
      from io import BytesIO
      response = requests.get('https://i0.wp.com/www.embhack.com/wp-content/uploads/
      2018/06/hello-world.png')
      img = Image.open(BytesIO(response.content))
      code = pytesseract.image_to_string(img)
      print(code)
      ```

- 參考資料

  - [python識別驗證碼](https://www.cnblogs.com/benpao1314/p/9999283.html)
  - [Python 實現識別弱圖片驗證碼](https://cloud.tencent.com/developer/article/1187805)


#### 登入授權模擬

- 權限管理機制

  - ⼤部分網站都有權限管理機制，使⽤上也會有登入/登出的機制。但由於爬蟲多 半是基於 HTTP Request Response ⼀來⼀回的⽅式取資料。接下來我們將討 論在爬蟲中要如何加上登入的做法。

- 登入有兩種實作⽅法

  在開始講爬蟲登入之前，我們必須要知道現⾏的網站是如何做到登入這件事 的。主要有兩種做法：

  - cookie/ session

    cookie 是⼀種存放於瀏覽器的暫存空間，傳統的登入機制⽽會將驗證登入後的 結果存在這裡，後續透過瀏覽器資料將 cookie 跟著 request ⼀起傳出去。所 以 server 只要檢查 request 帶來的 cookie 是否存放正確的登入資訊，即可以 判斷是否已登入過。

  - tokenbased

    另外⼀種登入⽅式，是登入之後會得到⼀個 Token（令牌），由使⽤者⾃⾏保 管，之後再發 Request 的時候帶在 Header 當中。這個⽅法其實就是我們之前 講 FB API 的⽤法，這裡就不⽰範了。

- 利⽤ cookie/session 做登入

  - 第⼀種做法，可以先模仿⼀個「登入」的請求，把這個請求的狀態保存，再接 著發送第⼆次「取資料」的請求。

  ```python
  import requests
  rs = requests.session()
  payload={
   'from':'/bbs/Gossiping/index.html',
   'yes':'yes'
  }
  res = rs.post('https://www.ptt.cc/ask/over18',verify = False, data = payload)
  res = rs.get('https://www.ptt.cc/bbs/Gossiping/index.html',verify = False)
  soup = BeautifulSoup(res.text,'html.parser')
  print(soup.text) 
  ```

  - 第⼆種做法，直接觀察瀏覽器記錄的資訊是什麼，將 cookie 帶在請求當中。

  ```python
  import requests
  res = requests.get('https://www.ptt.cc/bbs/Gossiping/index.html',verify = False,
  cookies={'over18': '1'})
  soup = BeautifulSoup(res.text,'html.parser')
  print(soup.text) 
  ```

#### 代理 IP

- 當我們在對特定網站進行網路爬蟲的任務時，經常會遇到 鎖定IP 的反爬蟲機制，這時候透過代理伺服器來向網站請求資料就是對應的解決方式!

- 代理伺服器
  - 這邊的解法我們會採⽤「代理伺服器（Proxy）」的概念來處理，所謂的代理 伺服器即是透過⼀個第三⽅主機代為發送請求，因此對於網站⽅⽽⾔，他收到 的請求是來⾃於第三⽅的。

- 在 Python 中加上 proxy 參數

```python
proxy_ips = [...]
resp = requests.get('http://ip.filefab.com/index.php',
 proxies={'http': 'http://' + ip}) 
```

- Ref

  - [USProxy.ipynb](https://github.com/TLYu0419/DataScience/blob/master/WebCrawler/USProxy/USProxy.ipynb)
- 哪裡有第三⽅的代理伺服器可以⽤？

  - 國外：http://spys.one/en/ 、https://free-proxy-list.net/ 、https://www.us-proxy.org/
  - 中國：http://cn-proxy.com/

#### 滑動圖片

1. 利用python+opencv拆解缺塊位置
   1. 利用cv2將圖片讀取到程式中
   2. 判斷圖片中的物體邊緣輪廓
   3. 取出缺塊所在的位置
2. 利用Python+selenium模擬滑動行爲
   1. 打開瀏覽器前往網頁下載原始圖片
   2. 找出按鈕元素
   3. 模擬使用者拖拉方塊行爲





#### 加速

- 多線程爬蟲

  - 當資料量龐⼤或是更新速度較為頻繁的狀況下。依照正常的爬蟲程式，可以會因此受到應⽤上的限制。所以必須⽤程式的⽅法，來思考如何加速爬蟲的處理速度。

  - 簡單來說就是時間可貴!

  - 第⼀種加速的⽅法是「多線程爬蟲」，多線程爬蟲的意思是⼀次可以多個程式 重複執⾏，因此也可以稱為平⾏處理。

  ```python
  import _thread
  import time
  def print_time( threadName, data):
      for d in data:
          time.sleep(2)
          print(threadName, ' => ', d)
  _thread.start_new_thread( print_time, ("Thread-1", range(0, 5, 2), ) )
  _thread.start_new_thread( print_time, ("Thread-2", range(1, 5, 2), ) ) 
  ```

  - 簡單來說，可以想像成 _thread.start_new_thread 會開⼀個分⽀ 執⾏，不⽤等到結束就繼續執⾏下⼀⾏程式。

  - Ref
    - [Multi-threading vs. asyncio](https://www.reddit.com/r/learnpython/comments/5uc4us/multithreading_vs_asyncio/)

  

- 非同步爬蟲

  - 當資料量龐⼤或是更新速度較為頻繁的狀況下。依照正常的爬蟲程式，可以會因此受到應⽤上的限制。所以必須⽤程式的⽅法，來思考如何加速爬蟲的處理速度。

  - 第⼆種加速的⽅法是「非同步爬蟲」，⼀般程式都需要等前⼀⾏執⾏完畢之後 才會執⾏下⼀⾏，⽽非同步爬蟲的作法則是當某⼀⾏程式開始執⾏時（不⽤等 到結束）就繼續執⾏下⼀⾏。

  - Python 中實現非同步

  ```python
  import aiohttp
  import asyncio
  async def fetch(session, url):
      async with session.get(url) as response:
          return await response.text()
  async def main():
      async with aiohttp.ClientSession() as session:
          html = await fetch(session, 'http://python.org')
          print(html)
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main()) 
  ```

- Ref

  - [加速爬蟲: 異步加載 Asyncio](https://morvanzhou.github.io/tutorials/data-manipulation/scraping/4-02-asyncio/)

#### 搜尋引擎

- 當我們在搜尋資料時，最常想到的就是Google，但是 Google 提供的API卻有時間限制。如果不想一直花時間等待，可以考慮使用其他的搜尋引擎，例如Yahoo，Bing...

  - Google
  - Yahoo
  - Bing

#### 自動化更新機制(排程)

- 真實世界中的資料是瞬息萬變的，也代表資料會有更新的需求。但爬蟲爬的資料只是⼀個片刻，所以必須要思考如何與資料源上的資料做同步或是更新，確保拿到的資料不會是錯誤或是假的。

- ⾃動化更新的做法

  - 在迴圈中加上 Sleep

  - 利⽤ threading 的 Timer
  - 第三⽅套件 schedule

```python
def timer(n):
    '''
  	每n秒執⾏⼀次
  	'''
  	while True:
  	print time.strftime('%Y-%m-%d %X',time.localtime())
  	yourTask() # 此處為要執⾏的任務
  	time.sleep(n) 
```

- 利⽤ threading 的 Timer

  ```python
  def printHello():
  	print "Hello World"
  	t = Timer(2, printHello)
  	t.start()
  	if __name__ == "__main__":
  		printHello() 
  ```

- 第三⽅套件 schedule

  ```python
  import schedule
  import time
  def job():
      print("I'm working...")
  	schedule.every(10).minutes.do(job)
  	schedule.every().hour.do(job)
  	schedule.every().day.at("10:30").do(job)
  	schedule.every(5).to(10).minutes.do(job)
  	schedule.every().monday.do(job)
  	schedule.every().wednesday.at("13:15").do(job)
  	schedule.every().minute.at(":17").do(job)
  	while True:
   		schedule.run_pending()
   		time.sleep(1) 
  ```

- Ref
  - [Network - Analyze Requests](https://ithelp.ithome.com.tw/articles/10247206)

## 網路爬蟲案例

- ChinaTimes
- Dcard
- everylettermatters
- Facebook
- GooglePlay
- GoogleTrends
- 104人力銀行
- 1111人力銀行
- MoMo
- 台灣博碩士論文網
- 聯合新聞網
- PChome
- 松果購物
- ProxyPool
- 露天拍賣
- USProxy

