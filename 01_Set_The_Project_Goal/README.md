[TOC]

# 設定專案目標

> ML是好工具，是達成目標的方法之一，但不要為了ML而ML

## ML的使用時機

### 適合

- 數據量大到領域專家無法解決
- 偵測異常、惡意檔案缺陷、異常行為分析、生產線產品
- 大數據分類與篩選、推薦系統、客戶分群
- 協助處理大量複雜的感官數據：自動駕駛

### 不適合

- 高階主管不支持時
- 可以用簡單的規則解決問題時
- 沒有辦法取得ML需要的資料品質和數據量時
- 要求精準到不能容許錯誤時
- 產品需要資訊透明，AI常常是黑盒子



## 為什麼這個問題重要？

> 對於投稿 Proposal 來說，如果能說明為什麼要在這個場合發表會是很加分的事情

- What problem you want to solve?

- What's the customer pain point?

- What value you want to deliver to your customers?

- What customer need?

- 

- 好玩：預測⽣存 (吃雞) 遊戲誰可以活得久, [PUBG](https://www.kaggle.com/c/pubg-finish-placement-prediction)

- 企業核⼼問題：⽤⼾廣告投放, [ADPC](https://www.kaggle.com/c/avito-demand-prediction)

- 公眾利益 / 影響政策⽅向：[停⾞⽅針](https://www.kaggle.com/new-york-city/nyc-parking-tickets/home), [計程⾞載客優化](https://www.kaggle.com/c/nyc-taxi-trip-duration)

- 對世界很有貢獻：[肺炎偵測](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

  

## 資料從何⽽來？

- 來源與品質息息相關， 根據不同資料源，我們可以合理的推測/懷疑異常資料異常的理由與頻率
- 資料來源如：網站流量、購物⾞紀錄、網路爬蟲、格式化表單、[Crowdsourcing](https://en.wikipedia.org/wiki/Crowdsourcing)、紙本轉電⼦檔



## 資料的型態是什麼?

- 結構化資料：檢視欄位意義以及名稱，如數值, 表格,...etc 
- 非結構化資料：需要思考資料轉換與標準化⽅式（如圖像、影片、⽂字、⾳訊…etc）



## 要回答什麼問題?

- 分析案
  - 藉由資料分析來回答商業/社會問題，並嘗試找到解決問題的方式
  - 透過**視覺化**的技巧，協助長官理解想傳遞的想法
  - 要有清楚、具體的建議，與後續的工作項目
  - 需留意分析的主題需符合研討會主題、公司目標
- 模型案
  - 建立**預測模型**精準的找到目標客戶或訂定目標
  - 要清楚的告訴業務單位改變跟影響是什麼
  - 需要設計追蹤的機制定期檢視模型成效
  - 每個問題都應該要可以被評估、被驗證，常⾒的[衡量指標](https://blog.csdn.net/aws3217150/article/details/50479457)如：
    - 分類：Accuracy, AUC, MAP, F1-score, ...etc

    - 迴歸：MAE, RMSE, R-Square, ...etc
  - **範例：我們應該要/可以回答什麼問題？**
    - ⽣存 (吃雞) 遊戲
      - 玩家排名：平均絕對誤差 (Mean Absolute Error, MAE)
      - 怎麼樣的⼈通常活得久/不久 (如加入遊戲的時間、開始地點、單位時間內取得的資源量, ...) → 玩家在⼀場遊戲中的存活時間：迴歸 (Mean Squared Error, MSE)
    - 廣告投放
      - 不同時間點的客群樣貌如何 → 廣告點擊預測 → 預測哪些受眾會點擊或⾏動：Accuracy / Receiver Operating Curve, ROC
      - 哪些素材很好/不好 → 廣告點擊預測 → 預測在版⾯上的哪個廣告會被點擊：ROC / MAP@N (eg. MAP@5, MAP@12)

## 需求訪談技巧

### 一個好的訪談包含三個步驟

準備工作、執行訪談、後續追蹤

- 準備工作
  - 闡明目標
  - 確認目標對象
  - 準備訪談大綱

> 訪談時程和訪談指引

- 執行訪談 
  - 善用三步驟
    - 介紹
    - 核心
    - 結論

> 快速筆記、粗略的資料和下一步

- 後續追蹤
  - 撰寫詳細筆記
  - 跟進文件或事項 ( 若有承諾 )
  - 澄清不清楚的論點 ( 若有 )

> 整合後的資訊供未來分析或呈現

### 準備工作要點

- 事由:闡明專案背景 (background) 及情境 (context) 
- 定位:釐清 “訪談者” 訪談原因，及 “受訪者” 受邀原因
- 原則:以 “交朋友” 為出發點建立長期合作關係
- 方法:“Give and Take” 互惠方式，創造雙方皆有收穫的對話
- 結果:設想各種可能的情境

- Do's / Don'ts
  - Do’s
    - 在安排面談時先溝通邏輯 / 目的
    - 先提供會議大綱和問題清單給訪談對象，讓對方有足夠的準備時間 (若可以)
    - 尊重訪談對象，因為他們在日常工作中挪出額外時間協助你的工作 (而不是反過來)
    - 時時保持專業、禮貌、尊重
  - Don't
    - 要求安排面談，卻未解釋安排面談目的或這場訪談對整個專案的重要性
    - 提供過度細節的 “問卷” 給受訪者
    - 利用訪談機會做進度報告
    - 未做好準備並期待受訪者主導會議
    - 在訪談時不能接受不同的看法

### 執行訪談要點

- 介紹
  - 重點事項
    - 讓受訪者放鬆並建立融洽關係
    - 確認受訪者知道被訪談的原因
    - 確認訪談重點並鼓勵開放討論
  - 訪談技巧
    - 問候: 自我介紹、感謝和簡短聊天
    - 目的: 描述專案目標、專案成員角色和面談目標
    - 議程: 概述訪談架構、確認細節及徵求額外問題
- 核心
  - 重點事項
    - 提問、聆聽、總結並確定認知
    - “兩耳一口” 準則，在回應之前做到聽見、理解和評估訊息
    - 建立互信關係的同時記下重要細節
  - 訪談技巧
    - 利用 “由粗到細” 的提問模式，例：
      - 你能不能跟我解釋授信核准的流程 ?
      - 拖慢整體流程的主要原因為何 ?
      - 主管簽核是否嚴重拖慢整體流程 ?
    - “暖身題”；逐漸提出有爭議的議題
- 結論
  - 重點事項
    - 重述討論重點和議題
    - 確認雙方後續需提供資料
    - 討論 “下一步”
  - 訪談技巧
    - 分開強調 “已經同意的事項” 及”有待進一步討論的問題”
    - 確認對方額外想討論的主題
    - 確認顧問日後的追蹤活動
    - 感謝受訪者的時間和貢獻

- Do's / Don'ts
  - Do's
    - ▪ 將訪談指引當作參考，聚焦在維持一個流暢的討論，確保 “自然地” 提出面談問題
    - 對受訪者表達同理心，雙向溝通並跟進有趣的進展
    - 大部分的時間應讓受訪者說話，並視情形展現業務知識
    - 在討論敏感 “政治” 議題時，可考慮闔上筆記本並提議不將回應納入紀錄
  - Don'ts
    - 將訪談指引當作手冊，嘗試回答所有的問題或證明一個小細節
    - 根據自己的問題順序提問，而非順著受訪者回答的方向
    - 僅想像對方會怎麼說，而不是聽他們親自說
    - 花更多時間反駁 / 辯證，而不是聆聽對方

### 後續追蹤要點及注意事項

- 訪談筆記技巧
  - 著重於建構一個有邏輯的大綱，包含討論到的議題並加入有用資訊
  - 將筆記組織圍繞幾個大主題、問題種類 (按照訪談指引) 或其他重要要素
  - 養成一個好的習慣，在 24 小時之內整理你的筆記，確保你抓到所有討論的重點

> 一般來說，一輪的訪談可能會分別與不同部門進行相同的提問，因此好的訪談筆記也會幫助修改未來的訪談指引 !

- Do's / Don'ts
  - Do's
    - 另以信件或電話表達感謝，尤其是希望同時提醒後續可能的活動 (如有)
    - 整理訪談結論及重點，建構有邏輯的大綱
    - 確實跟進後續追蹤事項
  - Don'ts
    - 逐字記錄訪談
    - “上對下” 地要求對方完成代辦事項
    - 後續的專案工作及結果未讓受訪者知悉

