# Teradata SQL

[TOC]

# 簡介

```SQL
SELECT TXN_DATE,
       SUM(TXN_AMT) AS “TOT_AMT” 
FROM   EVENT_CC_TXN
WHERE  TXN_DATE BETWEEN ‘2015-11-01‘
       AND ‘2015-11-30’ 
ORDER BY TXN_DATE DESC 
GROUP BY 1;
```

# 建立資料表

```SQL
CREATE TABLE TMP.LEO_XXXXXXXXXXXXX_01 AS
(
SELECT
   TXN_DATE AS "日期"
  ,MERCHANT_NAME AS "商店"
  ,COUNT(DISTINCT CUSTOMER_ID) AS "客戶數"
  ,SUM(TXN_AMT) AS "總金額"
  ,AVG(TXN_AMT) AS "平均金額"
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE '%百貨%'
HAVING AVG(TXN_AMT) > 1000
GROUP BY 1,2
) WITH DATA;

```

# 刪除資料表

```SQL
DROP TABLE 
TMP.LEO_XXXXXXXXXXXXX_01;
```

# 常用功能

|   表格名稱   |    欄位名稱    |   欄位說明   |
| :----------: | :------------: | :----------: |
| EVENT_CC_TXN |  CUSTOMER_ID   |   持卡人ID   |
| EVENT_CC_TXN | MERCHANT_NAME  |   特店名稱   |
| EVENT_CC_TXN |    TXN_DATE    |   交易日期   |
| EVENT_CC_TXN |    TXN_AMT     | 台幣交易金額 |
| EVENT_CC_TXN |    TXN_CODE    |   交易代碼   |
| EVENT_CC_TXN | CARD_TYPE_CODE |   卡別代碼   |
| EVENT_CC_TXN |    CARD_NBR    |   信用卡號   |



## SELECT

```SQL
SELECT 
  CUSTOMER_ID
  ,TXN_DATE 
  ,MERCHANT_NAME
  ,TXN_AMT 
FROM EVENT_CC_TXN;
```



## AS

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN

```



## ORDER BY

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
ORDER BY TXN_DATE DESC
         TXN_AMT ASC;
```



## WHERE

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE TXN_DATE = ‘2015-11-01’; 
```



## 判斷式

- =：等於
- \>:大於
- \<:小於
- <>: 不等於
- \>=: 大於等於
- <=：小於等於

- Between
- In
- Not in
- Like
- And
- Or
- +
- -
- *
- /

## AND/OR

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE TXN_DATE = ‘2015-11-01’
      AND TXN_AMT > 10000; 
```



## IN / NOT IN

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE TXN_DATE IN (‘2015-11-01’)
      AND TXN_AMT > 10000; 

```



## BETWEEN

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE TXN_DATE 
      BETWEEN ‘2015-11-01’
      AND ‘2015-11-30’
      AND TXN_AMT > 10000; 

```



## LIKE

```SQL
SELECT 
  CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’; 
```



## GROUP BY

```SQL
SELECT 
   CUSTOMER_ID AS “客戶”
  ,TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,TXN_AMT AS “金額”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2,3,4 
```



## COUNT

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(CUSTOMER_ID) AS “客戶數”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2;
```



## DISTINCT

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(DISTINCT CUSTOMER_ID) AS “客戶數”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2;
```



## SUM

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(DISTINCT CUSTOMER_ID) AS “客戶數”
  ,SUM(TXN_AMT) AS “總金額”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2;
```



## AVG

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(DISTINCT CUSTOMER_ID) AS “客戶數”
  ,SUM(TXN_AMT) AS “總金額”
  ,AVG(TXN_AMT) AS “平均金額”
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2;
```



## 四則運算

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(DISTINCT CUSTOMER_ID) AS “客戶數”
  ,SUM(TXN_AMT) AS “總金額”
  ,AVG(TXN_AMT) AS “平均金額”
  ,"總金額"/"客戶數" AS "總金額/客戶數"
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
GROUP BY 1,2;

```

## 函式條件

```SQL
SELECT
   TXN_DATE AS “日期”
  ,MERCHANT_NAME AS “商店”
  ,COUNT(DISTINCT CUSTOMER_ID) AS “客戶數”
  ,SUM(TXN_AMT) AS “總金額”
  ,AVG(TXN_AMT) AS “平均金額”
  ,"總金額"/"客戶數" AS "總金額/客戶數"
FROM EVENT_CC_TXN
WHERE MERCHANT_NAME LIKE ‘%百貨%’
HAVING AVG(TXN_AMT) > 1000
GROUP BY 1,2;
```

## UNION / UNION ALL

- UNION: 去除重複的串接
- UNIONALL： 不去除重複的串接

```SQL
SELECT * FROM TABLE1
UNION
SELECT * FROM TABLE1
UNION
SELECT * FROM TABLE3
```



## CAST

- 文字形態轉數字

  ```SQL
  SELECT CAST('1   ' AS DEC(4,0))
  ```

- 數字形態轉文字

  ```SQL
  SELECT CAST(1 AS VARCHAR(4))
  ```

- 文字轉時間

  ```SQL
  SELECT CAST('153330' AS TIME(0) FORMAT 'HHMISS');
  ```

- 文字轉日期

  ```python
  SELECT CAST ('20191201' as date format 'yyyymmdd') as yyyymmdd
  ```

## 日期運算

```teradata
SELECT CAST ('20191201' as date format 'yyyymmdd') as yyyymmdd -- 轉為日期格式
		, add_months(yyyymmdd, 1) as yyyymmdd2 -- 加1個月
		, extract(year from yyyymmdd2) -- 萃取年的資訊
		, extract(month from yyyymmdd2) -- 萃取月的資訊
		, extract(day from yyyymmdd2) -- 萃取日的資訊
```

- 將時間轉為timestamp format

  ```teradata
  SELECT CAST(KEYIN_DATE||' '||KEYIN_TIME AS TIMESTAMP(0) FORMAT 'YYYY-MM-DDbHH:MI:SS')
  FROM TABLEA
  ```

- 計算兩個timeformat的時間差距

  ```TERADATA
  SELECT (MAX_TIME - MIN_TIME) HOUR(4) TO SECOND AS DIFF_SEC AS DIFF_SEC -- 計算兩個時間差了幾個小時、分鐘與秒
  		, EXTRACT(HOUR FROM DIFF_SEC) * 3600 + EXTRACT(MINUTE FROM DIFF_SEC) * 60 + EXTRACT(SECOND FROM DIFF_SEC) AS USAGE_TIME
  ```

  





- 在 UNION 不同時期的表格時會有欄位不同而無法串接的狀況，這時需要幫舊表格加上欄位與形態才能串接 

```SQL
SELECT CUSTOMER_ID, KEYIN_DATE, CAST(NULL AS VARCHAR(100)) AS COOKIE_ID FROM EVENT_ICS_FLOW_LOG201908
UNION
SELECT CUSTOMER_ID, KEYIN_DATE, COOKIE_ID FROM EVENT_ICS_FLOW_LOG201909
```

## LPAD/RPAD

補0

```python
SELECT LPAD('12345', 8, '0')
```





## CASE WHEN

```SQL
SELECT CUSTOMER_ID
		, AGE
        , CASE WHEN AGE < 20 THEN '20-'
         		WHEN AGE < 30 THEN '30-'
         		WHEN AGE < 40 THEN '40-'
        		ELSE '40+' END AS AGE2
FROM PARTY_DRV_DEMO
```



## QUANTILE

```SQL
SELECT CUSTOMER_ID, PURCHASE_AMT, QUANTILE(100, PURCHASE_AMT) AS nPURCHASE_AMT
FROM bacc_temp.NT86000_CHATBOT_CUSTATR
```



## JOIN

```SQL
SELECT * FROM EVENT_CTI_CALL_TYPE_TXN A
LEFT JOIN (SELECT * FROM EVENT_CTI_INBOND_TXN202001) B
ON A.CALL_NBR=B.CALL_NBR
WHERE B.CALL_NBR <> ''
```



## REPLACE

```teradata
SELECT REPLACE (REGION_NAME, 'AST', 'ASTERN')
FROM GEOGRAPHY
```



## 宣告變數

```sql
WITH VARIABLES AS
(
	SELECT '2019-05-01' AS MINDATE,
    	   '2019-05-02' AS MAXDATE
)
SELECT *
FROM EVENT_ICS_QRY_FLOW_LOG201905, VARIABLES
WHERE KEYIN_DATE >= VARIABLES.MINDATE AND KEYIN_DATE <= VARIABLES.MINDATE
```



## 萃取時間

- 當天日期

  ```SQL
  SELECT DATE
  ```

- 當年

  ```SQL
  SELECT EXTRACT (YEAR FROM DATE)
  ```

- 當月

  ```SQL
  SELECT EXTRACT (MONTH FROM DATE)
  ```

- 當號

  ```SQL
  SELECT EXTRACT (DAY FROM DATE)
  SELECT EXTRACT (HOUR FROM '12:34:56')
  SELECT EXTRACT (MINUTE FROM '12:34:56')
  SELECT EXTRACT (SECOND FROM '12:34:56')
  ```



## 多行註解

ctrl + /

```SQL
/*
SELECT * FROM table
*/
```



## STRTOK

```teradata
-- 用底線切割字串，並取第一個值
select strtok('HiHi_Tony', '_', 1)
```



# 表格變更



UPDATE

INSERT

DELETE

ADD(DROP)新增或取消欄位

# 常用設定

- 查詢新資料時不關閉舊有查詢的結果

  Tools -> Options -> Query -> Close Answerset windows before submitting a new query