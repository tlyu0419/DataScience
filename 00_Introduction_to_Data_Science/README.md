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

- Ref
  - [「机器学习」到底需要多少数据？](https://zhuanlan.zhihu.com/p/34523880)



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
    - 當我們訓練好⼀個機器學習模型，為了驗證其可⾏性，多半會讓模型正式上線，觀察其在實際資料進來時的結果；有時也會讓模型跟專家進⾏PK，挑⼀些真實資料讓模型與專家分別測試，評估其準確率。

- Ref
  - [The 7 Steps of Machine Learning (AI Adventures)](https://www.youtube.com/watch?v=nKW8Ndu7Mjw)
  - [ML Lecture 0-1: Introduction of Machine Learning](https://www.youtube.com/watch?v=CXgbekl66jc)
  - [機器學習的機器是怎麼從資料中「學」到東西的？](https://kopu.chat/2017/07/28/機器是怎麼從資料中「學」到東西的呢/)
  - [我們如何教導電腦看懂圖像](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-tw)



## 角色與職務分工

- 資料科學家
- 數據分析師
- 資料工程師

- Ref
  - [Data Scientist vs Data Engineer](https://www.datacamp.com/community/blog/data-scientist-vs-data-engineer)
  - [Data Scientist、Data Analyst、Data Engineer 的区别是什么?](https://www.zhihu.com/question/23946233)
  - [Why Data Scientists Must Focus on Developing Product Sense](https://www.kdnuggets.com/2018/04/data-scientists-product-sense.html)
  - [Why so many data scientists are leaving their jobs](https://www.kdnuggets.com/2018/04/why-data-scientists-leaving-jobs.html)
  - [真．資料團隊與分工](https://blog.v123582.tw/2020/10/31/%E7%9C%9F%E3%83%BB%E8%B3%87%E6%96%99%E5%9C%98%E9%9A%8A%E8%88%87%E5%88%86%E5%B7%A5/)



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
- Jupyter extensions
  - Table of Content
  - Autopep8: 自動排版程式碼
  - variable inspector
  - ExecuteTime
  - [jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time)
  
- Ref
  - [Miniconda 手把手安裝教學 輕量化 Anaconda 客製化自由選擇](https://www.1989wolfe.com/2019/07/miniCONDAwithPython.html)
  - [15个好用到爆炸的Jupyter Lab插件](https://zhuanlan.zhihu.com/p/101070029)
  - [輕鬆學習 Python：conda 的核心功能](https://medium.com/datainpoint/python-essentials-conda-quickstart-1f1e9ecd1025)



## 建立虛擬環境

在開發 Python 專案時，很常遇見的問題是不同專案會需要不同的 Python 版本與不同的 package ，因此這時候就會需要建立出不同的環境進行開發，避免彼此在使用時收到影響。

- 確認目前有哪些虛擬環境

  ```python
  conda env list
  ```

- 建立虛擬環境

  以建置 python 3.5版本的環境，並將環境命名為myenv為例

  ```python
  conda create --name myenv python=3.5
  ```

- 啟動虛擬環境

  ```python
  activate myenv
  ```

  

- 離開虛擬環境

  ```python
  deactivate
  ```

  

- 刪除虛擬環境

  ```python
  conda env remove --name myenv
  ```

## Jupyter 插件

- 
- 

## Python basic

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
- 
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

### sqlite3

```python
import sqlite3
mydb = sqlite3.connect('mydb')
creat_table='creat table tableA (columnA varchar(512), columnB varchar(128))'
mydb.execute(creat_table)
```

```python
# 使用Python 連結資料庫
import sqlite3 as lite
con = lite.connect('test.sqlite')
cur = con.cursor()
cur.execute('SELECT SQLITE_VERSION()')
data = cur.fetchone()
print(data)
con.close()
```

```python
# 透過SQLite 做資料新增、查詢
import sqlite3 as lite
with lite.connect("test.sqlite") as con:
cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS PhoneAddress")
cur.execute("CREATE TABLE PhoneAddress(phone CHAR(10) PRIMARY KEY, address TEXT, name TEXT unique, age INT
NOT NULL)")
cur.execute("INSERT INTO PhoneAddress VALUES('0912173381','United State','Jhon Doe',53)")
cur.execute("INSERT INTO PhoneAddress VALUES('0928375018','Tokyo Japan','MuMu Cat',6)")
cur.execute("INSERT INTO PhoneAddress VALUES('0957209108','Taipei','Richard',29)")
cur.execute("SELECT phone,address FROM PhoneAddress")
data = cur.fetchall()
for rec in data:
print(rec[0], rec[1])
```

```python
# 使用Pandas 儲存資料
# 建立DataFrame
import sqlite3 as lite
import pandas
employee = [{'name':'Mary',
             'age':23 ,
             'gender': 'F'},
            {'name':'John',
             'age':33 ,
             'gender': 'M'}]
df = pandas.DataFrame(employee)
# 使用Pandas 儲存資料
with lite.connect('test.sqlite') as db:
df.to_sql(name='employee', index=False, con=db,
if_exists='replace')
```

```python
# 存儲資料到資料庫
import sqlite3 as lite
import pandas
with lite.connect('house.sqlite') as db:
df.to_sql('rent_591', con = db, if_exists='replace', index=None)
```

### pyinstaller

- [Python打包後的執行檔檔案太大?Pyinstaller與Numpy的那些事](https://medium.com/@rick.huang1609/python打包成執行檔後檔案太大-pyinstaller與numpy的那些事-dcc75ff9d42c)

- [python實戰筆記之（10）：使用pyinstaller打包python程式](https://www.itread01.com/content/1547118036.html)

### PySimpleGUI

- [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/)

