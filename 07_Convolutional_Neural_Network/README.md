# 深度學習應用卷積神經網路

卷積神經網路(CNN)常用於影像辨識的各種應用，譬如醫療影像與晶片瑕疵檢測

## 卷積神經網路 (Convolution Neural Network, CNN) 簡介

> - 了解甚麼是卷積神經網路 (Convolution Neural Network)
> - 了解甚麼是卷積 (Convolution) 的原理

### 卷積神經網路的強大

CNN 模型能夠輕易辨別這兩隻狗的品種

![](https://lh3.googleusercontent.com/PB1LpVR3fz5zoyhhvN0Sb5Y8-z6VvNiQonj54UpwnHE3_NQAFzB4X-t_BpbNL_ArUb8n6MYKVT2TB9aTfi3vAukqYtmt1I_y6GT02bb-SN9SV-eENIkD8zXdIuof6JRmc2_bn9Swg95gM-EkvPZ0-jQ8nY_vobZQ4uaYYjWpVIGL9yycEcT9iVbs3mpz1sbYD4eCooMIZeTS8m1SyslqtvNRWC4dj74rzN4ftG61fYhTRC6Au7iKfwfVyxqkO3NDMFhngdR0GMe6qkg6_Efu9NwoRTnEJ9VdHUOIGGgSC34iRr3BEwN7sa9LZyobqXelrZ9RZPGkETsdHPlxA8mtodJQCCZAGQWU3JFzbHgDq1F6Nj86zdhkkgBCPAmp_Wvz6LQDhZtMUv-_4P_fm9ng8v6mlcKP9F3l8vLLQCRDnSAnB7jg0qcG_kVVMiz_qTyZeuJcSEkRLKvEYtpqES9w741O1yr6PKFyL6qk05pZifAMIenxQBiTRaX2nhaNa48iGqxAwVuR8JYZZ-yBaEDl_ZTX--pi-7hvp2TP_iLtHPYaWrbLKVPBcr0VZD9Ve0d0v_G8MujXWsInya8ZXbq0_UVSlMNBzj70LbN2dtTNraldHs1x3Gz0_XIf-OStRysIcVjSmzeq6m80eOcz2QpiyzQtZjeTzl7bjwQ5M4vutRvmhcFya6p_d_LF7-MheJgBy6l4XGgrWDD-GKqOY6Y2wkhK=w995-h432-no)



### CNN 在圖像辨識競賽中超越⼈類表現

ImageNet Challenge 是電腦視覺的競賽，需要對影像進⾏ 1000 個類別的預測，在 CNN 出現後⾸次有超越⼈類準確率的模型

![](https://lh3.googleusercontent.com/AKEXo44HiPQSSegThAi-zlP8IY9ZtLFWnJ34ta1KN8h2KvKy0E7MsnNlUTBBo4I0kVdaAVONSD6-pBNRZ8fwF3T6Vto1ReObC0tA1tV_CPphZFhj8MViSEHDcTEJ_7s8th6t22_jNwGiOTsDw1igzojJpvVWh3U3zS8fIQVR7vI_OqnZJ5IxWjKlswuSTTLzYSg_MgqP1bnpsBvJxHD_Y6mSQU3pq4SKz2fbnfQsMlTdPvosXQm_xju-SDUAuNeoTMQCw8gUlxu8cPR-oI09iNy-M4-fNtO8SR8LHF2A5jBaBmbNO4uqDs1gx5E-lj-evjDJ7IEN9Pk-pXB7XNMMAmkj712N56Oqt4HScwoB8f4yEs16CeO9KrpI0kLKQBTN5NqH1omLBb09fBwyr12ANcBoPizXAcwi5D28N3-YGaysW-A8HT_caxWBAvvPR9eHRfQ9JGBN27qAGwcCbx3Yv5wPdwh6vRYyP8VXCDjtumyfcc93DpZ1x6V9RYpXz4Pur18k91cHCmjJrfLTikQloy6luHDcYcrqUOyIP-wEehePa3DTBWI9wk9h-3rQjUn81zo9kvtmvnrkWJZTH8IDfFZBQEwbF3UJETkK2J8fVBZmuiwKTVp3iWM97FIe8hEBQ4AhlIqxnhNjzcGHbM91un7DSjwXb_8fkZUpuZqdO3cm0UEdo1E9ig6hB0rx6-_MbB2gO_lhmpdatU3TLfc4wN4j=w543-h343-no)

### 卷積是甚麼？

- The primary propose of Convolution is to find features in image using the feature detector put them into a feature map.
- 卷積其實只是簡單的數學乘法與加法
- 利⽤濾波器 (filter) 對圖像做卷積來找尋規則
- 下圖的濾波器是⼀個斜直線，可⽤來搜尋圖像上具有斜直線的區域

![](https://lh3.googleusercontent.com/7hV42k3RMTEUNyZSC1CSfXpo6hAz6lBds2X38nCnSxgJHnWZUhedUkg82mkbgtcArguG9S9iVEtnK2tJmDXb7rO25G4ibyvw_XfKAdqQMQCosGbQYS1B0UlDSLN8eIhCSrV6pY3TvnqmkTVNbiwSKa6L4XHUKZOY-BdWkqpHtydi09NIwtfTu8CFNcvKr6FWCYm46D-P4NcizrPgd8wtTAAJtldG2RiY7FZsPP9Tay5H6yHqqPhzkWy3Vz2Y7hyOBwRgAvxTS68fsnRfR53fI0F18m1AfnyacUFqNQQxWkdLK_oiLqkwg_fl5ELoWViE1xNyD1vG47qUnkgNCJQz5KKSIHwnhMsqvkZaS67T98xuIsGjKG0YvCgVeZ6HEPihioYspvRkyFQezcAEF4pR5-veWBQGplwlBfvm8yr73PttpZA51DIzNMB6572cLBiYt8OtDOOzESLTsrjwtbAg0Ba2r3Kn4oTZYFcMEl31Klhkkc-YfjqOWCH8r_1QT-mBJZ0VaiEIodprSjDvIErW8mQLsKiHByNvxe3eGVVuaUjGlzVHGE15Gb9l2aBxoaTdg4WFWABFfciYZDAy_yegRLxOwNctBvIRW2bh3hhuW4lFVUY-60w7_iz12_NIO5lPIi-SVARiXUsqqICnuS2G7FFng_K2QT9ChIOGeZoQqYoBwy9_OEEl7QitZkimhemyDirK9arXqmHkeyHBq7fqUjZp=w896-h389-no)

- 紅⾊的數字就是濾波器 (filter)，可以看到是⼀個 2x2 的矩陣(值為 [[0, 1], [1, 2]])

- 卷積是將影像與 filter 的值相乘後再進⾏加總，即可得到特徵圖(Feature Map)

  ![](https://lh3.googleusercontent.com/wDvPQf4hn-vQumTV0R03llmPObn3vRswzuMJiFKr-lQgaW8os1mJuF-HQY4OSwTB0Bx3qbk0GswsRdGASaEGA75XWHYcs_7xdBXv0g5i_O67tRD9gpEoJT8ctxhGGakfuuT3EPFgyiLXskNTs1I3dLRI_iamqfIVi_Okam97qBgYeY_aQT1D7mGHkDnbKI7PbFxoRYYq5LIcew8USRLI5SzATrAKhha77RoUj5vBK2QykouQjhbC0ktxthihhwjnr_hfazGoH8N8ISfqfTlXVluQPSy8XkXrQFDoTu0vifZsehNCBv4wGvJoOgROxWnNe_kRZnkRU4Va1p88MBUFpSelaVJlsdFD56ciItLCYMYhR0asX8HSQcSEw74rSTxFuukKQ95y2uWS59a2SoPB1xYCVGcWxHswi_DyPuQ-zxjAdnaELLf1WLqSTFA4uKi_3fQPh-rYKThmc1j3p7QmLYHbuxkCUo20_5jRWenJjIG9w-E4J1md7doB64WDmtq8B4_lG46lpi19lv6Fkl3evx3NpuZIIEbOGygOyaao7BucASVKaa_jIaJ7Lx96H-xSEKfyj3ZfaM-8IgxyLKwZPbyn6cb523C3oBiTqCGpvdLbT7Sn63Zoa7qMANfKgAeBOIhYw20luGR1-QhuC3VscNpT1j7zJXpIbWOQHOjnQ6uE-3c8fQyrRrxRhSkdUch0O4ozes2NJY-Xk5Xy7kQoWaeC=w783-h346-no)

- 透過卷積，我們可以找出圖像上與濾波器具有相同特徵的區域

- 下圖可以看出，兩個不同濾波器(filter) 得到的特徵圖 (Feature map) 也不相同

  ![](https://lh3.googleusercontent.com/jDdKuAdis-N9WpvHChgR-7c5SvkEzxAg8sZC7pAxLDGuCmJYt1V2rr-QZ33RYwNVqnNIsRwSt-ekVaSiW1nPKEe88AHBzX66sl3W3UvcaN3fusPTE1JnW2c1ocyQObbt7pGpybpnE-yaL2xz0kg40_Ui-yZnN5nX3JQKmJj6PxvYNJ_IYF-eoDWX0ZXn8u7up0K7aD1AYhia7y2DBoH0qXiLOSAXCRxCxr2xJQhndEOd-7S8co1lBa_NvANDhVuFdmW5lfStCFop1uq1mDr2ioAP1r2R_uqtX8J5Jyo0kqte00_8VlvJ-fVazut4JmYGvlBj9SQCM6u_hHB4SJwGnP1mlBnPehScx1f7ltYbt_kTBVXu-D6Ku8W8ErasCpi_AGfobDYOUk1bbgSs-VY9lo3Hmisa4HRvVn1kEy0WJDLti_FxaWO81Wv2PU_UsSpHMzxRiEPbwoCcVJkMNaRBVKPTRcaytxlErGWSm9HtiLgO7ryT_JAKhg3e17Oi-JSzol_Skwx-MyITnZP9w4I-_ako3mjbEXOIlI_BUTmeaL0yPzTpFeY2Mvz8cwm5eQmgQwxU-vf7YA57ht2VCy0KuOkx5T425DpKTSnPzPDdhFfJgEy20TDDueGmbv9qhINGEfMgyARJAUvEOH2gQ2dslKrcnBbH61Zas8qWuXWI2DKg6StuxeQFZtT4KLlEKvC-amNeFpkLxPxO4Uw4JAA0n4R5=w902-h399-no)

- 由於我們不會只用一個濾鏡，所以最後會得到多張卷積後的資料，稱為卷積層

  ![](https://lh3.googleusercontent.com/pw/ACtC-3cos6e3pA2q2lcYZ0542k-3NX49PrwbAFm1PszNu5K7GCgvOalyllwUQEYRQpMNCs3kVxLBMj4YKd2I1EkdcV14SuP0MDmhgCILIAObzlZIKfznF8JBe_wcu8JVtxdFU6jlbLBAtUWSrf7JhRVfcYo2=w1160-h534-no?authuser=0)



### 濾波器 (filter)

- 我們已經了解濾波器是⽤來找圖像上是否有同樣特徵

- 那濾波器 (filter) 中的數字是怎麼得來的呢?

- 其實是透過資料學習⽽來的! 這也就是 CNN 模型中的參數 (或叫權重weights)

- CNN 會⾃動從訓練資料中學習出適合的濾波器來完成你的任務 (分類、偵測等)

- 透過⼀層⼜⼀層的神經網路疊加，可以看到底層的濾波器再找線條與顏⾊的特徵，中層則是輪廓與形狀 (輪胎)，⾼層的則是相對完整的特徵 (如⾞窗、後照鏡等)

  ![](https://lh3.googleusercontent.com/TL-Uar2ZnG9ksRMP1ujhOzs4A8nM0vVaNypSOdcg9N-Ktc2N2gZa8M79gkcTM95bnOmYrsw3MMCi50bTP4U2dVcISYVlofD_AuXAEEEBk3NHzjLS15kyHLI9IMe4BwDf2_gea0eBYfavSs4aAsIY14f6iQ6cmQ9EGxxUVB6REbUrQAklxkzUnIUeMxjbUpFKF4BmHWIudo1y6fi2TJ39oNQphWg8lz6WXY9ymKkoG7p2iJewOR5SX7kh-qDUjGC4UXXkA4HcOoreX4Ik7VQmVPO6b2o07TbhyA_j8tmNH1GvxJQUZLkDaAHVPUBGRtmssZZ5kc0t5wQOfalFqseGIAmEI_q8dAWjH848NkhtqthMhymzGZRprtxiRwh-pV6iJp0yaq3LNywhjTHzJCTu9uAv6j6abgbbHDkjLEZVEqlaMpp9DlO10ze1wpQUVSjwIgCB4OUc5YE4Gw0NZ7QKD5Az-VGIy9zOS389SvH9MoGM2ssMlcqPgeLo3Sd-ac_eL3SuBI6BMeuOzY_6bkqt5yJcxEAJZeWcNAF4AgJLabU_UzXwjEHbrxTb6jF6_ai0-cB7GXTdkZJtbUQClxiKvgyMSWB3XmDBhUJj74-03mCve60pfbpze4_BpV4erdOG3bVfxz7ArudyNUdAUTV6GP8hJCJxTL5eVy-tBakYizy3iNoM_WLFTTaG3fhGEjWEeSmp_9ZsNHUbPYyPV14B8EsE=w667-h432-no)

### 重要知識點複習

- 卷積神經網路⽬前在多數電腦視覺的任務中，都有優於⼈類的表現
- 卷積是透過濾波器尋找圖型中特徵的⼀種數學運算
- 卷積神經網路中的濾波器數值是從資料中⾃動學習出來的
- 卷積後由於資料量的減少，對於模型在學習特徵時會有幫助，而且也能夠加速訓練的時間

### 參考資料

- [ILSVRC 歷屆的深度學習模型]([https://chtseng.wordpress.com/2017/11/20/ilsvrc-%E6%AD%B7%E5%B1%86%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%A8%A1%E5%9E%8B/](https://chtseng.wordpress.com/2017/11/20/ilsvrc-歷屆的深度學習模型/))
- [卷積神經網路原理 - 中文](https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_convolutional_neural_networks_work.html)
- [CNN for beginner’s guide](https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)



## 卷積神經網路架構細節

> - 介紹 CNN
> - 說明 CNN 為何適⽤於 Image 處理
> - 卷積層中的捲積過程是如何計算的？為什麼卷積核是有效的？

### 深度神經網路的特例–CNN（卷積網路）

- 傳統的DNN（即Deep neural network）最⼤問題在於它會忽略資料的形狀。
  - 例如，輸入影像的資料時，該 data 通常包含了⽔平、垂直、color channel 等三維資訊，但傳統 DNN 的輸入處理必須是平⾯的、也就是須⼀維的資料。
  - ⼀些重要的空間資料，只有在三維形狀中才能保留下來。
    - RGB 不同的 channel 之間也可能具有某些關連性、⽽遠近不同的像素彼此也應具有不同的關聯性
- Deep learning 中的 CNN 較傳統的 DNN 多了 Convolutional（卷積）及池化（Pooling） 兩層 layer，⽤以維持形狀資訊並且避免參數⼤幅增加。
- Convolution 原理是透過⼀個指定尺⼨的 window，由上⽽下依序滑動取得圖像中各局部特徵作為下⼀層的輸入，這個 sliding window 在 CNN 中稱為稱為 Convolution kernel (卷積內核)
- 利⽤此⽅式來取得圖像中各局部的區域加總計算後，透過 ReLU activation function 輸出為特徵值再提供給下⼀層使⽤

### Convolution卷積網路的組成

- Convolution Layer 卷積層

  由於我們會用許多個 filter 來掃描原始資料，因此會產生出許多張 convolution的結果

- Pooling Layer 池化層

- Flatten Layer 平坦層

- Fully connection Layer 全連接層

  ![](https://lh3.googleusercontent.com/qzEb-nbGYufLKkRCC9Qbep2gf19FUfFe4AyEoyheoo8B_XFiPLnMOhvCch9l0fAXjRTuTQ1-4Eq1Cmg8D86JFsxiizMvostFdeYPFnn9p1nLHZWPxV_6QvaXExOXbIgQ-26wpgsj7Q_VOH0-B0SW1Bot4evlcjedIcurhXPyxpLSMLn6ZVSfCu8kFjK0ZHnIGMakpsrtP8Ce7B3bbEpZNVF3N7ev6-nNLpD1pRAcU_kQnJ52PGBfPXUHYBC7mwYYdMOXBS87KyFPGfjIHkseAHu7bxHHYCXEKDDMDWPQmB1UT0e9p-hvS2M8Fn0l5R8hj73VZu9hug85Ze0dGk8ptqiGNmjnap6bRqJORp3CzciWjWBXHY2x7zCMvUZaZip3iMhb814U18XNjz30GgF0bsVMIyUwBtWPsaM64KOVN2YlCTaRRcreIAfUi3T6z2kEn9YGHmOkSRJkhNJdvhCgqpq_hgACQCBCdwGLm5Z1YdTnV4mowXPyOxVB3jRHQBL6hBEiSgjC5EHkdhqazl_XVUGFTrldJMHJrYDOWpyPvY3aqkQYgygVs3O-5trpg8k1FFpxdzO0iqKUvdWL0aqSQTUV036eTs5fYJkxFDaedAelCzv-lcq3X6S2u3J12rDjFlghDhmmgggImJ5OrXMVA0wTt1660hU7EIAc8hGmaD34kWBx91bNeFPeda5l4d4BBNwHZa9bVdnAQVhWSRDEE2ff=w987-h288-no)

### 卷積如何做

![](https://lh3.googleusercontent.com/drvULzuFu6GPIrUM-Y40spO1zI2SBFJuq-bMCmMzAffHSirXtmPSxom4N3cBOwqNx8ebFSoYeoqsLgeb8G56WtnQ8QOSX_sNimWHZhwnfJPt_yBEEH_hMEmg35kTwp4eermyfSupIuwtfI3RS1D0qbl-DHfNGQz1VRHa0YIkLRkn4op0ea6L-_aLvsWZN8owMTaIabUtpV3O4ZZOHFMjFcItztRq4z2lI53YhWRYWWI7UnF5P8PP_UJ_PhC_EaBHn82UAldpP5m8RWnvZLFLMMPMZahGkLnxFyLdrE0npKUTVlZAny4aaaAxX8gRVExOfF3MhKf3cpVppwi8YkVWlfrNGDYgTUU2qW0Mqhw0h0W4sjoMEqP5pbvK1ey6YSSFqSWZ5Cs0964kWxNB3njxjXhBXm-wnUPPIHQvRVDUwkvmaLmE-tMLci-ZyLuYc0-Wm8M8he19mQfJW2NQssg4pI0luYV11pjka3hewx808MPZ7xFyKiPvMUGdBbOisWlgR7WA8nwG6vnzm1hU9X8oJudJ1_JUkkVqho2cjmzE0IQhufdJ_tZW5iuF_th58GpTZOgMaegjO5UzTTxBTRV2GKLKeGOolvqs7UeLA5aebzCS0_cbWcL2Fzf--ZXxDUlMTmYVRjjxH2C614NpYQKk2qLoPsUrdJYlHOH6mCWBuuZ5LmoaEOd1lkYkwYdAa9xdaTcoGB6ZAzXE6n3BvhDDxeap=w910-h341-no)

![](https://lh3.googleusercontent.com/tmLoLUsX-x_RyWoUd2g4-ho2lhGxjn5cKYd09jq-gLzYSswkP0ctdZvekMhLNgHykJ2wrBHZ5OOskTPJrnxMKyjHWm_e654zS1R8nR6JrPUnFmYRcyA760XU00B8RRPiWol3T6F6h0xaMI1aYH81mL4yq0rrddIuNlVxOoXustYwavYNlklkhvrE5SgbWYuT80GWcoJItij42boY3NAHj--j8LZzeEl-kvVaqUcORc0fNmw4TyUae6-VS44DYRt7BZrQH50RmFBC3tmJpNegNqgzlmFDta7CgppZPf5vVTnACIkDrs9y90xGsf2gHxjFNt2gdU7oCWy7z-pYpCI6vEldjrjvKd2j23TMkyi1ipVwhOpNcVue0w9oca9Chu10RBuvDXjN4GQuWXTrbHb79rTjfeYv9Zdttj1XnLd8fGt_e5rTihREvgK9BCsQtzALriSNAYipVm_ND8mqB-paKPP6nEQFrYlqtkKBpk6osQa9vOOMjoDMDK2wcnxWLTN5jvuVKAYd2MetQipQJqId6sBtKTL-PXZq4h5Jbpqr8OHYIGWYZcybkpZf7yNuDepV7V5bkaiA8EU-NDW3hPjxXPYbrAq3q4iIJeazDsGO-7Gfh5nBRMm4UG72rdb7dFyz-57oJKToPgNP8YbXi0aYKrPLnluQa8sN4CIobWgFGWvLinJVnD53byW026RzdxG5pVL2UnB2e_J_ZO2UDwxWRdjb=w905-h452-no)

![](https://lh3.googleusercontent.com/Wd6Y94qHSuh62CaN8nwbnm4y7ofu4n6heYDvK6pDe1xAOmcJsMtdSXCcTY4EnmRepERiH0tjwr2YZPly8dZmfXOmmntWMDaOI1pQglEFvysZjH1FIqOGt_g3QVKqv8A7SEegqZS9z3nCLAQSaBWOHujtlBxBEGlw0jVxuQ32gsjgqSd3n9c1Be12UkFkGVmMmvpiO24UsfVi6cr0WD0XZ6SiD3JjTuGouDgnQ1nhQBspYI2JEx8Nmdd4E2pc1ZLq7sBLXj4keclkSC6NFSKZGNIXZltRR2-XRY5HQbNFXEK2mS37JQ3iQs4Mwk4Z5Hp4smX66uwCiMirvS9_RSRAi5gAuPBxx3DT07lGk83nkBmseOb90KyPkUFkFPqksUenfFSFHcuEn_NWMTq3kIdkJrivfsLHEWKC1VwFekl9kq_5Ym_H47mvjO7BOEeKfn34C8fqS9Ac1xfAd2vV2bIIlsHoPJSecbX8RQtfytrnwH_Ci6i5AbEsSTu1oRmfSvQdJy2DizRsguoxfIFbXz5CpSOKqZ6ehGpRGDP0ma7Evo0iXQ77_4ouniLqNyuTNRD9YHukBlfK4tx03YKsjgmgY_cnlenKDyydZtYBkAI_5bLFxLyc1W7UYjWwihehhTRcbiQzgfhUO8PI84D5XFuCTMyFbmf7uy-wrByqDoLe94pUzDu4kQY3Ex7AbnFPte8UqjUc1zIQoaW38wrEchiT_Wm6=w934-h470-no)

### Pooling Layer_池化層

- Pooling layer 的功能很單純，就是將輸入的圖片尺⼨縮⼩（⼤部份為縮⼩⼀半）以減少每張 feature map 維度並保留重要的特徵，其好處有：

  - 特徵降維，減少後續 layer 需要參數。
  - 具有抗⼲擾的作⽤：圖像中某些像素在鄰近區域有微⼩偏移或差異時，對Pooling layer 的輸出影響不⼤，結果仍是不變的。
  - 減少過度擬合 over-fitting 的情況。

- Pooling的方法
  - Max Pooling
  - Min Pooling
  - Sum Pooling
  - average Pooling

![](https://lh3.googleusercontent.com/uIwV6uJAbcMLEsyIhMiDeXKZLk86vLlzw8t-eOtKhQLMH9HVivx0eO1qXUOhZ3Jx2DonHb9Oj2BvO2n82rcceeUswAkHw_4iXpQyK16r9KOqRh-kM_tCeXenYXigzWA0c7EsWnbcfGaj5KJoF9CbZwNIViqToygjCrGJNQLwJZMXZS8T5_Pu3mKslSc2LIJQHvCVukRgvdJ03zlx9eiNAUiNBtFkWUv7Zw-HWpwyX2bWW0hxJen2NBSsZj96gwJ860O7pFbR_ZSrdY7eSBEkw_0arfooc2waJvisKjurGMEJrR7sjBe-gof5UoOfDHbibV9HRqutNOQ6pGeCPztNIH9GJ5svGpMEF7hBfg1BTNjkCJMa15hKTpOLIBnYHWZVwObEANBy1-xpH63I-8BpXWkpGh_PRBJvikpejgjdHor3-CTkxC6zkTxaPCvgBJhV2AgGeDrEa0wbMNxjbvjdCR-Wl4ZkrJEm2BhU2U3vwX1jJOsTFz4C9A58oTANEpscvvgxrn20Az6_qvFxDYipFGe6inLeYE1_gBDyitXxyAqIKolY_k9pwJAc41CIihkSAtAUcDBXknFt5X42RlURrgx8NAbqBm5LeXcr1XLO0JGIugJCSYTBOhJb_OowHyfShTaGCy9Bo-fWBPXKJ-Ecu-gH-ljo1oun24hUpTFvzpeyxtFSdtPotP22TcATOcgh0sFM7s3D2ZLvNaCBH_iWf0_u=w811-h203-no)

- 參考資料
  - [Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf)

### Flatten_平坦層

Flatten：將特徵資訊丟到 Full connected layer 來進⾏分類，其神經元只與上⼀層 kernel 的像素連結，⽽且各連結的權重在同層中是相同且共享的

![](https://lh3.googleusercontent.com/OrQ0qDtyYpv-q2IFxDRmN12AzF6bPwghdXZan2BaVmjiYKCTV2-RQcq1wbxseojea0tFlbEjjArH5xzM99UQAseh-lAJyga56I41cEStFx2mm986kM41iLH2V1FIoYyas9uzCsj2qUEq_6_af95ExlPUOg6bBthGfmM3abXz3GIIvxPqoK_gcZFbPTv8TniDkmD_vww11J2l5wvfGes7WYFcyAFEDuLzQVfBJpOdcBZvdxOH0xr4XyoGR-aDaX5LB49s0hI2d0WvfUPaeNKS03EDGrJcyOEwB2wMho0rMsPEAZfQUhxv8jDOkSgW21TA8Ju600SoDE23drbg0E2yNHfhkEHKslJpsRafE9KkyvLMc_q8738gb_4jGXm55hhHuseBx7-s27EEVwlCUyoReQsKufDhkjeQC12lqbVmTVGEPcgtW-OKMeK1ZF13LqykxR3_w2YfZfKA_QxwhdnBvWXrdXvqGyzHHvxA5tk5B-DCD0OZ5jU1A8Ni9q_ojiGkBIocrPU5BQMFEDawJ0GknKC-fKxWsk4_R4jFVBiR2rSOMFEMNufUfGs0wYllzKCM5nfJoxrKM2jISnNBqgJUisXFGoKwA-LkhmrUgCQkb2jwSXQ-dBbTgBKZq4yur8cc0HRMs7JvGuYD6_EfeVLsdvIaxQolIRd6ciJaXl_BF3omjrhCYuTfbPASdfo94a0s4IIYFzWyA_YBdvfbF8-PkQdX=w790-h229-no)

![](https://lh3.googleusercontent.com/n_wK6hmlxuSdv_U0cenKX6XuqD9vba5H5CG0FPvzKAUJb1bTx9evLtQ_0Xmjaw9jghr1XgvORvtoVOjs5sK4Wng1wnJ6JG-kYKu1m2wBN9tLebKOVqHi50u9F07Bm1s5aH8PHmjTPnG7kKbg5uWxQZSBQDdPJxfqscciLhQ1Iak8DGaKhDD7dBpaNGDJBU6EXAadf9zsbZaaWD6IClBfghjdh-Xk9CMexD6YRbO39ULGgIi70cMasejSaYhtruHsxhH-iu2ytm4vvtFsJrZ9FGmjan6Aahc8AcNYgmhbpfFpzBVS2MIna1BkvwEkRMBJJSsU2lKCDhx9IvWK5ni3RVHL6zuqspt0Zn5FOvBR-SEeEzY7WHcwjH8f7o4vv-09xbkwy8frvfGd0P3_gp0R7OHMPr7LgWnbykkbO6drzgcQ01L4Rv_wg4QsCeF88Rro0th2HHqHCBR91jfG52aO2MZjJ6mtMV5dbqSqLz8SPz1ZxsVE_cNjACI3s5cqCB1eAgT4F3H5_4H4yhLB6QVqfoSxkYUEA4ZgUxeqzfh-uAAHE6FBtu05LH18W4DGIwAdfk8x5naIwnhzTk5GqYf923HT8h6p-yu0bmghw-SeX7bOWDjEl0S95Sv8s-4XgLFPWaGG6TZYnz1FfrrRYtvintYVeQJ_WHND4tb4SSYCu2r9xy54niIIFVZpkCIlA63vAkmd4gfPpDUVM5Rtqz2fj2zZ=w412-h312-no)

### Fully connected layers_全連接層

- 卷積和池化層，其最主要的⽬的分別是提取特徵及減少圖像參數，然後將特徵資訊丟到 Full connected layer 來進⾏分類，其神經元只與上⼀層 kernel的像素連結，⽽且各連結的權重在同層中是相同且共享的

  ![](https://lh3.googleusercontent.com/8IPH6bO4KWocQuLqdgxc660jX5FxHj-THiYaUobADhuxINrPusKV1iHzEujQlRjHQBOJq_SLU7itcX-AE2KRtH3tUyI2RmdquC6v1mwojJ-nGUmZIcT-dcQSCyxNdiXcYUAMCG_tnMQT9ZnZ28gbFzDVyAvbbStV8P81PqnCKC0IBmGnWR0Fpj9f9I4sg_5lAgHPo_fFwZQoYSdM1bCOcdfvAS8T6T-6LjZHOG5h1RS-SJHciSfS6hvz9V5C5AuZyqRzZNnkztb4yrFKGGybndfYAA5es4KOtlE87TnuqMtpoHMqhE4Oeh74_BlMcP27b1tB0dEjSaLujPK736NtdCfRpjbullrAXelXQik1e10E5erPVayMeE6C2CqEH2abYE5cyJny5dA4DE_7VqV41g-B1nuvNIzHVGE2J1KOAaupRAAEC-K2v8_8js7ydqs5jQiYTij8vzVFBWi3SGvA3UrYqNrZP5QoWSxxOic-HJMD0KG1JiOfBZLUUiyeIZ3z6Qp459SfIl72ZVYH0HoyQUc_aM1Jwkw193qp8iqiYfLlgrAjPzr24qkfcTcvY5xfp9I_hhjqhDrJ9RhMk_JvfDVySEPDjtfPIYqOHuWG7e7dnyVJWAZT89gPPut48siFO_opocmODFeLa8Ud-lgd1prX9GIhxnq5s89iSOsfxuwNHkJW6fHiDa4r8U4hz2dBvHGTob26WxyrXbd0_KAJ-uuy=w794-h274-no)

### CNN in Keras

```python
# 卷積層
## 建立一個序列模型
model = models.Sequential()
## 建立一個卷積層，32個內核，內核大小3x3，輸入影像大小28x28x1
model.add(layers.Conv2D(32, (3,3), input_shape=(28, 28,1)))

# 池化層
model.add(MaxPooling2D(2,2)) ## 池化引數：劃分的尺寸
## 建立第二個卷積層，池化層，注意，這裡不需要再輸入input_shape
model.add(layers.Conv2D(25, (3,3)))

# 平坦層
model.add(Flatten())
# 投入全連線網路與輸出
model.add(Dense(output_dim=100))
model.add(Activation='relu')
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))
```

![](https://lh3.googleusercontent.com/PAKVFBDKSIEMBVtizp4aMYPw_F6MC-zv1cLI4EdccDJykF3szfEjyeAmAX4we-tMSoC4VMPi1nwqd3A9a75lnyygbUtkcbiZ7s_7zSSl32vXovCtFQ9Xki2DYPGKhiRb8uyUmst-rX8NFFbl_7YfyafWyUJA12Kw8NOkgGGtgHn44SvZqMWqi3kOFBBhKnfgJyRLxQTuX0j3yurvCSC2rnfkU6pDjeIIzZ3o-A3mWEdRzSwmRD-wVWbW4P_mY9Qkir3F_hC-3lxSZX_EXBpQIAP7kcX1mOCWWqyBnTL39C6jrtAsbsRez6KJz9z9lGfTjlzzm8vOKbzO3Z2QeFErbw1jV_2kNUgdguWZgfwW6Xw1jKwhefGQzBcmaSNkaac_mddQbOzZo6HOeNuiYy_LYAJFsm8wra2JbSio878ccHHRGUnXeXrKrdzi8v7bBPVE1fRhwDpM0sKrgufwh7FOWvRAdh3SOjIcVplNBSRBXKHk0cPo-vGWvuNfa-CBPv_4bUmFlrABzTL9pvibEugSYQElx6QzCEJxXAfTeS7_1XqisvMa8lR-m9MTj5HYBw0VP4fAkCxqoeLZX4YhWhOwzxTj2cSOMkTXXjnu0RGcS9h5AWw2kIL30frh1ELnJB2DFXDZjZ9283-GK1LNBXvnaAQm-1TbR2S7PLB1lwlRoJWxU47Lhk511Jxt075ZpsFgkX566UaDGpSTqVJpwBCyBNno=w827-h163-no)

### 重要知識點複習：What is Convolution

- 卷積是圖像的通⽤濾鏡效果。
  - 將矩陣應⽤於圖像和數學運算，由整數組成
  - 卷積是通過相乘來完成的
  - 像素及其相鄰像素 矩陣的顏⾊值
  - 輸出是新修改的過濾圖像
- Kernel 內核 (or 過濾器 filter)
  - 內核（通常）很⼩ ⽤於的數字矩陣 圖像卷積。
  - 「不同⼤⼩的內核」包含不同的模式 數字產⽣不同的結果
  - 在卷積下。 「內核的⼤⼩是任意的」但經常使⽤ 3x3 
- input 上的變化
  - 單⾊圖片的 input，是 2D， Width x Height
  - 彩⾊圖片的 input，是 3D， Width x Height x Channels
- filter 上的變化
  - 單⾊圖片的 filter，是 2D, Width x Height
  - 彩⾊圖片的 filter，是 3D, Width x Height x Channels 但2個 filter 的數值是⼀樣的
- feature map 上的變化
  - 單⾊圖片，⼀個 filter，是 2D, Width x Height 多個 filters，Width x Height x filter 數量彩⾊圖片，也是如此

## 卷積神經網路_卷積(Convolution)層與參數調整

>- 了解卷積神經網路(CNN )中的 卷積 (Convolution)
>- 卷積 (Convolution) 的 超參數(Hyper parameter )設定與應⽤

### 卷積 (Convolution) 的超參數(Hyper parameter)

- 卷積 (Convolution) 的 超參數(Hyper parameter )
  - 內核⼤⼩ (Kernel size )
  - 深度(Depth, Kernel的總數)
  - 填充(Padding)
  - 選框每次移動的步數(Stride)

### 填充或移動步數(Padding/Stride )的用途

- RUN 過 CNN，兩個問題

  - 是不是卷積計算後，卷積後的圖是不是就⼀定只能變⼩?

    - 可以選擇維持⼀樣⼤

  - 卷積計算是不是⼀次只能移動⼀格?

- 控制卷積計算的圖⼤⼩ - Valid and Same convolutions
  - padding = ‘VALID’ 等於最⼀開始敘述的卷積計算，圖根據 filter ⼤⼩和 stride ⼤⼩⽽變⼩
    - new_height = new_width = (W—F + 1) / S 
  - padding = ‘ Same’的意思是就是要讓輸入和輸出的⼤⼩是⼀樣的
    - pad=1，表⽰圖外圈額外加 1 圈 0，假設 pad=2，圖外圈額外加 2 圈 0，以此類推

## 卷積神經網路_池化(Pooling)層與參數調整

> - 了解 CNN Flow
> - 池化層超參數的調適

![](https://lh3.googleusercontent.com/LX3-xNVpk-hMOLsVbnf5vW2HRgw7uRKJSRK3v9Z9hwsWVwsZ9ZyDzVAsU6ytCSC_kXJhGCWMq0HEbxG_mWU09mvykvB4y7WO1SlMcNZ3IENPJnic_Z_4mb4qMRzcS9GEzzNwbzrvnkwu0VLqpBe_PEZK8OR_6_IPUKi3olhvAEqw3hQS8k-Oclc5U7jDJDRj7GHWGS-azYiRe74UNqMcS3LTijK39g--JYUt4PxiLX5fP-nliTmMgE5SVcq1sMYg_9iaI85pyNJKacO1xhMS8DQsJNW4d7_X_8FHXTTYSBiMCJ_cXtZWVc4kb0eVZSRIDQn2wrmjD5rHwTPs6BR1oZP2e0cvTwt5imCbb23i-kaPTd34IDYLCAegcO6E9M7mejCZWy3M4s4abVVwo9QrKe6r_PJ9XMgyovxcE8zfx9uSbL8UcKCipj2rOd9tHdP6Gj4SGwH63BQQtq99JFyrs6uWw9MZMY5B5qzIREO0wNFQYT8Fs-0oizsCnAfwi-vehaFtGhLeS6h4-mp6Kt3tSVsPyBoEScmAU7zLqkIXfwYWbLYsTw0Ib2qIzKBRu8eQyl8m7yGFT6Uc8YR1HRVZAXvNZgryAQlwlUXcmDn2X2c1anxN5X07S9XoSjOHR6gKFNJ7XnPUORWzbqWQCBbs7NxxwSOPdUVBosmAinkk0KNaOJRUpsBrxTnWASVHhTZcjt_dW6AzyDdgODKoyfe6Ls8v=w737-h186-no)

### 池化層(Pooling Layer) 超參數

- 前端輸入feature map 維度：W1×H1×D1
- 有兩個hyperparameters：
  - Pooling filter 的維度- F,
  - 移動的步數 S,
- 所以預計⽣成的輸出是 W2×H2×D2:
  - W2=(W1−F)/S+1W2=(W1−F)/S+1
  - H2=(H1−F)/S+1H2=(H1−F)/S+1
  - D2=D1

![](https://lh3.googleusercontent.com/tds-tymj8KzB55rCs1HcCIaFzqRDWWq0hwDiRlPi2S48evhu-AU1tzsgcz5V0iRMnSPGwC8BTYYSjIxrbYBa_GZy7yxl06LRxETEL6zvKkkAgD0-tzhc5uvlomfR2Ydk2R30OXptd6BE_C2XX-DHZBRUdL08QlsAFsnViHLusNIy9N8S5iyh5QkF7hJApg3jGtESkUzsQYfCTgRsoLx9ACUY2sizihaX-NdV-jqiAOIqaEEA_-t_W9fx60IpxZruHdiivMCOCz7Ew6mLeUnqUbBpJi2vuCiDH5cSkgZBCkE_6oYpNI8N45yAVaBvMKC5-1XsJndaqyerw2HWspaDYC2WUqwUwn_2vu-JNJ_gcnMS2yI3X-ooBRr4Vhc6jGaU7r9Bq3FFH4yFD-p-ZyaVpdxSSAMZV9i6ZTQ3SZJfosQqyRXBaASu3kGa6qaCQBswZSw2gcn2UeNE2Yd5y1ejRf16ts4kGDb_wZLHoOuorrlY2_5i8eFpQP3RSDd1PzQIbd3pJWFqDuy2viYJUlRWiFQN55TdY4YB5ft-miAOriR9JdAM3_n3z31D9UGgWWzzLwX8pynu2LItmnNtWwjydVSc7vg9lswj7Dg-B8_60NKzgARyOvi2Qd4jIxlXrTnM0Oo8LZMtRfx0LAsDRDpuIDvSY0zdyLoeMPgbtTevgIC9P1KZQYfHFmAwyas6hbN8pHcPv5fndINB--ThyfbTs8p_=w469-h365-no)

### 池化層(Pooling Layer ) 常用的類型

- Pooling Layer 常⽤的類型:
  - Max pooling (最⼤池化)
  - Average pooling (平均池化)

![](https://lh3.googleusercontent.com/XtPOc_RTqvqc9p1TmNbXz5euRhKczvRLb2gGtrw3ElsrG6GVDbwAVvuJgBSSnQMsm_VLet-_43IohnbKSwG_oy4zShFL3yOHpCFMAqNFfIWFvxkRN3GeVeayhR8Muuh2P838clh5tdlI4oxWD1Q3xY6X_52bIa94TeWMX4T1pJJTqH7oO-tBNF_-Epi0-KoDZO4Gi1OfEcK-GZe_aeEJxplghz_-_IVZYH-YLOyLwelXEbknuN9vZjDrtwCOUpxKRJhI7K4Je2jGwM3pbbhQgTM1GdE_nnTrs6gQwsFQtI268ev0sPqaqscZEhZvHP_9W4XcFTQUBv9DrwNTMmOe1hTUW70_AUw4wjtJtJLXFLJqNfsmnaGHZeGi0lftRoWFJ_UlkBh9MlncxsA5zoPP4Tkvm2dOdViCTDnu39v3UepGimgMiUj4TNnzmhjv3_7ZZtBX_ujGNkOLrNf9-TR-HXXiMfAGFSgXjsKxedBGlPX4ftOQMS0TF7sXqmue_WuMjuQYvUd874A-unFBON-7o__d6Zbu0nM_Naa9u9WZQ1iUUR0Ocyz02e5IQ7BluRcEZ-Vt2oZ3R0viXMIIY0htJIo6PhNIfOmm4ugvT6_zcgEYVgrOwK0kGy3ErQWMC3cqJ51yIHIhF_QHI5Jenai3QbdwPrUxzHp-Gne4YQiprgkJoLwgAXzj4j7nUopgh68Xr1G4aiR1PlZAFx8VW08aRRoj=w702-h304-no)



### 池化層(Pooling Layer) 如何調用

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None,
padding='valid', data_format=None)
```

> - pool_size：整數，沿（垂直，⽔平）⽅向縮⼩比例的因數。
>   - (2，2)會把輸入張量的兩個維度都縮⼩⼀半。
> - strides：整數，2 個整數表⽰的元組，或者是”None”。表⽰步長值。
>   - 如果是 None，那麼默認值是 pool_size。
> - padding："valid"或者"same"（區分⼤⼩寫）。
> - data_format：channels_last(默認)或 channels_first 之⼀。表⽰輸入各維度的順序
>   - channels_last 代表尺⼨是(batch, height, width, channels)的輸入張量，
>   - channels_first 代表尺⼨是(batch, channels, height, width)的輸入張量。

### 重要知識點複習：Convolution 跟 Pooling

- 卷積神經網路(CNN)特性
  - 適合⽤在影像上
    - 因為 fully-connected networking (全連接層) 如果⽤在影像辨識上，會導致參數過多(因為像素很多)，導致 over-fitting(過度擬合)
    - CNN 針對影像辨識的特性，特別設計過，來減少參數
    - Convolution(卷積) : 學出 filter 比對原始圖片，產⽣出 feature map (特徵圖, 也當成image)
    - Max Pooling (最⼤池化)：將 feature map 縮⼩
    - Flatten (平坦層)：將每個像素的 channels (有多少個filters) 展開成 fully connected feedforward network (全連接的前⾏網路)
  - AlphaGo 也⽤了 CNN，但是沒有⽤ Max Pooling (所以不同問題需要不同model)
- Pooling Layer (池化層) 適⽤的場景
  - 特徵提取的誤差主要來⾃兩個⽅⾯：
    1. 鄰域⼤⼩受限造成的估計值⽅差增⼤；
    2. 卷積層超參數與內核造成估計均值的偏移。
  - ⼀般來說，
  - average-pooling 能減⼩第⼀種誤差，更多的保留圖像的背景信息
  - max-pooling 能減⼩第⼆種誤差，更多的保留紋理信息

![](https://lh3.googleusercontent.com/rP0CKzIqgSC_yC3_GfAjTMPK2GboGrgtECSxtoBZZMYsZaujyzVWGlA-wUflCYP7O1wnDUI-ijrwEkmcW9zm90mqWXEIdsF-qY3CEGQ9yF-NADQeuK5-y1qHcgRIazo4IXSv0hcr8XJslKMJR_46pGBPHeuHD7PVNQjOe3EYDongCIADAnAEbkhF6AJoSkdJ-j7J7I1H1s3lDJ4z-dheClmVBP9VeO6_4O1ZFlnXglKQdcnauR8C87NLeLfIgnHcKIztN6OJAR6q2fZCpthNNzhbwjkBxXiNBiAjScwI0i45PqWO3YebNtRO2mgO5sEqwR_hGxdJKXKLmF04h3E7W9tKM1F3gMnRwwX0Mr6y7EskmKGMbBqaMTbuCTi_7r6E1jwlMyqHfL5HUydbaZPp59eYdkcRUj0x1GMThuT4RN9484rLRFGAP5RO7cCZqsu_oKefS7foU3g2wzMyIloWbROG-3Xv02Qm7Lt-PcMBZNSDNwocB9CpSpNCpiLDv_nYIGGwhkjXdDkZZ9nSIC6TEkEH7MH_X8uSczLRJtmTpz-tBt6xTsIWTNC52-G6RQ6EzDxpg6_GkAtTfuRl5zezNj5la02kRBhCkLDoNvmv9uxFlm0U5FdvWNYmpLSalMG9kkkQytMK6kof7aXAXLdT1fPDJ2ZEF2TldGxZQGAM2EjwYEORW-Fr0hUUBKBILHCIFz88r4whfii14F5MaUzIJ0-k=w867-h401-no)

### 參考資料

- [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)
- [3D Visualization of a Convolutional Neural Network](http://scs.ryerson.ca/~aharley/vis/conv/?source=post_page---------------------------)

## Keras 中的 CNN layers

### 卷積層 Convolution layer

- 卷積神經網路就是透過疊起⼀層⼜⼀層的卷積層、池化層產⽣的。

- 影像經過卷積後稱作特徵圖 (feature map)，經過多次卷積層後，特徵圖的尺⼨ (width, height) 會越來越⼩，但是通道數 (Channel) 則會越來越⼤

  ![](https://lh3.googleusercontent.com/kioKIV7Wkw_pJIhRZFxmA30OLMM3iBgVnyQFoqBDXBHATcUb521vp-bsj6vXvjMkcj4BHRGy3uYZULESUmwofhZUDJIuozsSykRFLpsld1IMftw56eWcW4BryHcYn9mqINjRbDeA-S_feRQlgdXO885xqZvouTpDm3oHr6EK3GF3NwTu3CEhPNEHIfcsPtjCQeRv6VWYI2oRyT51RjXE9g6A2GmXDOrGdi7nvwJjwFRGvgWD4cu9BagO_A1AUaXQHzeUnJIpZIMrX1_v2MjKhX_yC0wmtS1ZB-QRtIdOZY7Sj9W-QE_yEukkKjC9E0MfUU5DXuE0CNV93LBbbLTAkqqlHCj4Ye2QoA3RykHjj6dFIafvxZU11uLJYJB7XgkO73tAZEXV11K79nyy_Yor-JrCAM1Wn1XAPa2cY6SJ0CXSa48bgL5eghXeDIs0L3twWdzAsfiJ8yW6RH44voFuCOkrevlq2jrRXUPfnx97788wf1tlXpOj1S5K8NBIxZ6Rp0yju2C4qFXgXoI1NeUZiAZW0izc3Eo4eTXPBQiJzUyTm3Uj1VrjOyGFh_ql5haPf6V-LCwP8gjQcmdLe77Qni6Moz7Fngj9qHmxkbct9_IVwPhypfFO4RZhZ5S6Ppk0zlEjx1qDjJSaj9tgqF2cEzcLDP-q1TajzyeAU5oaQArK_89tPHdex7ev2pl_6NCtwTRQyfXwO1-S-8BMMuPJehpJ=w920-h441-no)

### Keras 中的 CNN layers- Conv2D

```python
from keras.layers import Conv2D
feature_maps = Conv2D(filters=128,
                      kernel_size=(3,3),
                      input_shape=input_image.shape)input_image)
```

上⽅的程式碼先 import Keras 中的 Conv2D，接下來對 input_image 進⾏2D 卷積，即可得到我們的特徵圖 feature maps

> - filters：濾波器的數量。此數字會等於做完卷積後特徵圖的通道數，通常設定為 2 的 n 次⽅
>
> - kernel_size：濾波器的⼤⼩。通常都是使⽤ 3x3 或是 5x5
>
> - input_shape：只有對影像做第⼀次卷積時要指定，之後 Keras 會⾃動計算input_shape
>
> - strides：做卷積時，濾波器移動的步長。此處的 stirides 就是 1 (⼀次移動⼀格)
>
> ![](![](https://lh3.googleusercontent.com/_JgUxL4UM9Jf7h1N3LHCFHLeGmEmJODQh3jPY6egML_qhQjfiBvXXCzk8vs_wRpo7NPqWCcfiOKqEmmRsQfBLGA2t-r1w-3RoZ7K9vXt4egLu8kkxUHmQdibpscPbDIHc2WPu3-sLO_qjMRvbwblX0hHqouwsBbNi1hCEx6K4WimO06fvO6ptH_JVJsGBRDKiRRCqvN_7gcMfnP5_ZLparMnXS5wnkqRfOZEvGeeJ5aPilZWEJHES2ZbBZJn3ipPbk1UUlpDiX1n_pbOqW2tcF7E8SLedkH9XNCxBrJuaiJfyBs1K-BCv0a-M-7zcKXFz4UjCPhKdEnZbGQ75nQ0fCKMPeEjOMsKXwmR5tOEQu8dfazocia9PbiiQqYJxFWCsjyBcALxagLfpJThEjSHdarzW6qBoEVw0uiiZ-_WPB9PWy1v6WIust7JMuvlGvmoe7udFAQBBt_lg6D5B6EpXyuY1Qu-6EhXYd8LNpDwZIhSWAGZbsHMVL2PAEb5IJfSGFusFxlNZjpyy-RQpPbVQ8siZhj8tlU99Ny2ASDYJ0D3LJmJ_kayMKs-ZUyFOZjB2f6QJO-0gtmCrDhldmAazNOt4OGXd-c26L2O207Sj5cd_MxPJKbtQOo3da41r0qX5D53Ec9ubCeMU12AS8iSKRqhDzhNTpAdNy2Iyy42xs8KlwoiOK6xxLhPUZgiB7PnkBYXOk56eDwcY_Et5dFAlqxG=w526-h384-no)
>
>   - padding：是否要對輸入影像的邊緣補值。此處的 padding=same (邊緣補⼀層 0)，稱為 same 的原因是因為做完 padding 再卷積後，輸出的特徵圖尺⼨與輸入影像的尺⼨不會改變
>
> ![](https://lh3.googleusercontent.com/0JU7Xm8kOZceKP9PUq3y9MEcl-H37D9YnnRgtk0IqX4h7HViShrKFZbCDyDmBVDRLVC5VCY0KZpmOfUV5NfX_97tMNof7eFPyIwgh-YbMf-opgOyAN1TEW_709hTZV5Oan0dBkovUnYjInmMDsNT759xEihco6KpdyuNnct8CFf3quoAWCP4jkd0BZL1nMMAXPFXPAJZXwmt7vV_jHRlO7rpryhMXmy8ImtSIJiZKOSfHUMMKpK3lsuRo0JnSeZh8lR4YOyU-WqhiJuQIXYlu7j1PQVR4q2MtEy5tWKQctHbKqJTJ2QfXM6uTCxeidlPnlAINXZS2uPrxJPaQLXMa4bYVU1yneekPsacU_9c_jBjjlHNnHsvvY7lHYg7vXR5jcB3mP5EuPSHugJ56_BIgVzqh90mgTK7Nl_WxT5PA21M-PlOtMDHkYI8hahuovvCF6Vhyx2YkRKk8cj1l7dxUSwCbHdE_7DJXFQhiaAdkBXKSaoyHxPQipgQV6qOgzt8hgtsfmi0zMWWi2txzdZpYJWa-qO6xtuIbxle0kfXHtE8nAdJJAZIu2J6qqP8XeijPbAUb-0uGOWH-e8fFfyBVMFQEWydLRXhSzfXSbYHRQxpftktsHHZatVUKhYJykU0u-VMYl3yTeh8GnDns_tFR7Hy0_04QMMbzNeilDlC1WyvMIptpB2LGzKqsiHkven_OFly2S9JaXW_kz-1VsiGh0nM=w666-h294-no)

### Keras 中的 CNN layers- SeparableConv2D

- 全名稱做 Depthwise Separable Convolution，與常⽤的 Conv2D 效果類似，但是參數量可以⼤幅減少，減輕對硬體的需求
- 對影像做兩次卷積，第⼀次稱為 DetphWise Conv，對影像的三個通道獨立做卷積，
  得到三張特徵圖；第⼆次稱為 PointWise Conv，使⽤ 1x1 的 filter 尺⼨做卷積。兩次卷積結合起來可以跟常⽤的卷積達到接近的效果，但參數量卻遠少於常⾒的卷積
  更多資訊可參考[Xception](https://www.icode9.com/content-4-93052.html)

![](https://lh3.googleusercontent.com/Jdlzq6xZeHISkNEROatbJVFljJwhBqHjRAl0oad9vjG3xTbkGHrqrG_AH5tRPpMNsj4U1MtdBSocbpFDDb9uH_u_QJa9o5zxgSvb3JIk1GXaGv-e-X184M9IF5JnGhUFTBfxA6ntHl2HJ8f32P_oJzFEm_pkN4OOIEaEgKnBYkUBtzm5KZ19ZTofn_DL33rN9Ibf8OeQwEBSnjDAUFOS5VbZ89DLXMOQ2mbG1SmvV-RFbOmp6MzCOEGtyeWrTrRrF1azbgRI9pl-eBokIAXe8UFL4s1vSijxqGfCiS64Ff4cMLmIzsNXQIMBw39YKT6f-Q_ROtM2fOTj3Es2P4lI4jdxzyuPwM3xXbeCmSJ6OvMDTLLXOme8OXyDZBDDu2ltuYgrijTvyeUfAE4rRMA5Dm6fMTqoP_PTfWSh1EYngz_wXCGEV3WUSo7s99iSa4NRSjPz7-wq_w8W8Gjm0wjoIuV2eENqJwSvf_U4vLi8MxqBg9ahfHS3khsWS29ogkXxNBDGpGtxF-6opHwn9AlJiUTYZD27XG5VkB7Ga5xSVfktVpuDH2Zk3ejI4AOxOJ-kVyZbxbFJ3mflJCJngGvVgo6-RUY6vYjHEwpd5f5TgSiFsd9uo1xPYfDKZJX51sGLiwtunxwuUXi3usPQl_SdfoAdvqwYd5EXXCTPE7KDrfWqQTWasbSvyTyOxLpCn00_jaFaSNYaBmHNZouQJqk1TKiW=w903-h229-no)

- 以下說明 SeparableConv2D 中的參數意義
  - filters, kernel_size, strides, padding 都與 Conv2D 相同
  - depth_multiplier : 在做 DepthWise Conv 時，輸出的特徵圖 Channel 數
    量會是 filters * depth_multiplier，預設為 1，上⾴的簡報即為 1 的DepthWise Conv 

### 參考教材

[ML Lecture 10: Convolutional Neural Network](https://www.youtube.com/watch?v=FrKWiRv254g)

- [Introduction to Convolutional Neural Networks](https://cs.nju.edu.cn/wujx/paper/CNN.pdf)

- [The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
- [A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)

## 使用 CNN 完成 CIFAR-10 資料集

> Coding 練習⽇

### Cifar-10

- 如同先前課程中的 Scikit-learn.datasets，深度學習的影像資料集以 MNIST(⼿寫數字辨識) 與 Cifar-10 (⾃然影像分類) 作為常⾒

- Cifar-10 是 10 個類別，影像⼤⼩為 32x32 的⼀個輕量資料集，非常適合拿來做深度學習的練習

  ![](https://lh3.googleusercontent.com/LLJ77yR3iEnd0eL7lNLDMT6g_k1ismemoKzCkbzBzX43ZA5Rvu6Td-_09F0yjrlOYvr7M9bp-mwEqvUs8ekxEf0NX1NASKU8zOlJrb0Y3RWMUMOvM9AqGsGyq3rtdAylOIot22flxzbI85ssGk9aRiQ6qCfb-Uv32_OEjPNw5yG71VA3Y84SsGFhbXsbCBzUxn6u3wrzwggvCu5xjI4RgzJoWlYlXoF0Ua-k6hI_aoE7uTByWBiLxbka7wu5MtLbK76A4TUssCViNZ3JoG6SVRUdqQl4JK3fBoFH05yjn8K0PFXOQ4K5FOs-RHdV4eYTlKIQenLbifMTFzyqz7MenPa97Jgjy9t6PApdWL-Cqt65OZvLol5ztUOqED7hEtMrSnGAZufybzkPAXyke8ib15XmNkkcKrWejg7uQIlSk0DrxSehLOeZS1gGcVPMUttyCTcXYE_fsthGDZxapnHEHKcLSSW9RuQByxyQxkLgzMWPagCShq9oZQ49jGa5IVFsxwvwMBEmVsf2P38bYTCu620-tyjwuMeHInWUzToQ_dH8ypbZj6OPnZ9nGJMzN3_rzCvleGTqp0HWrIytzkDVoVE0EoNnOPYoMI1RBPisngvMnrEzjwGWGuKkhbmgnlqxg6GzY91ZTekJ3O-ydkmLJG4h8DIgV03e8beeKOx577B8UtbFpOJC4ynkioPL_DgNnxPd3z9U5eoaxTBSHKUVUfi6=w606-h444-no)

## 訓練卷積神經網路的細節與技巧_處理大量數據

### 處理大量數據

- Cifar-10 資料集相對於常⽤到的影像來說是非常⼩，所以可以先把資料集全部讀進記憶體裡⾯，要使⽤時直接從記憶體中存取，速度會相當快
- 但是如果我們要處理的資料集超過電腦記憶體的容量呢？桌上電腦的記憶體多為 32, 64, 128 GB，當處理超⼤圖片、3D 影像或影片時，就可能遇到 Out of Memory error

### 批次 (batch) 讀取

- 如同訓練神經網路時，Batch (批次) 的概念⼀樣。我們可以將資料⼀批⼀批的讀進記憶體，當從 GPU/CPU 訓練完後，將這批資料從記憶體釋出，在讀取下⼀批資料
- 使⽤ Python 的 generator 來幫你完成這個任務！
- Generator 可以使⽤ next(your_generator) 來執⾏下⼀次循環假設有⼀個 list，其中有 5 個數字，我們可以撰寫⼀個 generator，⽤next(generator) 會⾃動吐出 list 的第⼀個數字，再⽤第⼆次 next 則會吐出第⼆個數字，以此類推
- 將原本 Python function 中的 return 改為yield，這樣 Python 就知道這是⼀個Generator 囉

## 訓練卷積神經網路的細節與技巧_處理小量數據

### 處理小量數據

- 實務上進⾏各種機器學習專案時，我們經常會遇到資料量不⾜的情形，常⾒原因：
  - 資料搜集困難或是成本極⾼
  - 資料標註不易
  - 資料品質不佳
- 除了繼續搜集資料以外，資料增強 (Data augmentation) 是很常⾒的⽅法之⼀

### 資料增強_Data augmentation

- 其實就是對影像進⾏⼀些隨機的處理如翻轉、平移、旋轉、改變亮度等各樣的影像操作，藉此將⼀張影像增加到多張

  ![](https://lh3.googleusercontent.com/5ek7O9ksulsQPCfqPCtHmP0bXTt0FLZMrzE2oiCzlMRMNXvjTkimxDYFWbRqm8Roqg76wUSnYq8hg5-YVGppwOIPrm6cTPb6TtwZJlR8h57-b-Zq5rfeQLCXHL6_VhoYLp3XyB3ydBups_0meEgzzZntjx084hw4gUnPd96nFveYvZvKoOqOs3B747BJ6guByqRWW0Ff_W8H-exhHl66TUkwDa45YW4OuPLovSEa8D8F4815AAhrVilFK6EM_15ZKXKxNtGbZI6FegDTvi31o30IaEplAaUK9aovyFWE7Sm86datbCpYPrw72JYLb3vHPR300SpRxasoA1RGmqShxk-CMnE1_vwg_34aMt_57DZWa4nFUTQ8IncBO3IgnfeucrTUh62YgWVpSiteRh5LzxDoqPw3O_rxLUp8Q9LGjip08DPNSDrwOPr3pemqrguAHsJovaSeoEN0uIF2fX7UOdAjCXKLxn8jomfQVAjbp0ipBeIXHVQd3XqFs0GsQC_hoR3x4obZPuQPogWv_Z8y1M99-NaE3QbAdSNEhz1ogGzwM39bPueSGdevLKuxr_xHMCF4UOagFsn27gQzEURFK_ZN1ET5rkQOTivfJMfdQFGAIQTCYOiVTPm7GpNmtATYmTDR5xRuRxxu7-mGzfCT4rGTkoMwJrTKT7gW7Y-Qg0Nz3vMDtXpnArCVEqQiCQLz_Gl6Ddks3CdreTJK_g7o-bye=w929-h422-no)

- 資料增強並非萬靈丹！

- 適度的資料增強通常都可以提升準確率。選⽤的增強⽅法則須視資料集⽽定

  - 例如⼈臉辨識就不太適合⽤上下翻轉，因為實際使⽤時不會有上下顛倒的臉部

- 另外需特別注意要先對資料做 train/test split 後再做資料增強！否則其實都是同樣的影像，誤以為模型訓練得非常好

### Data augmentation in Keras

- Keras 提供許多常⽤的資料增強函數⽅便使⽤

  ```python
  from keras.preprocessing.image import ImageDataGenerator
  
  Aug_generator = ImageGenerator(rotation_range=30,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
  ```

  > 以上的 Generator 會對圖片做隨機的旋轉正負 30 度，垂直&⽔平 平移 10% pixels 

- 如同名稱顯⽰的，這是⼀個 Generator，要使⽤next(generator) 來取出做完資料增強的影像

  ```python
  Augmented_image = next(Aug_generator)
  ```

### 補充資料

- [Keras ImageDataGenerator 範例與介紹](https://zhuanlan.zhihu.com/p/30197320)

- 若你覺得 Keras 官⽅的資料增強函數太少，也可以使⽤這個非常紅的套件: imgaug，有非常多實作好的影像增強函數，使⽤⽅法也與 Keras ⼀樣，⼗分⽅便

  ![](https://lh3.googleusercontent.com/-5RXZ1tDFyWe3l581HhGA06mLCMLJBiO41fIpJx6OQBZ-NUT4xy48hVHV__A8Q4NqrgQRQ22Hz_lf3saXKVNxtwlg4X8Jq6XjDW8F0Rg7Mwg0G8JvBtmSbyR31QCRmmQZf9cFM-gT8krVVOgGZmWcOY1M3JsrUOPmvAno3nu0zqMEg4KqSBoJJ1UAcuz-TrS-OTXgTlv2wHt52VBkDsVss3RUSuwAq_klOsSRk1Uxz7bgnltB8iQ78lj6gYYJsuhtWZQY55pjiHoVGIV9fIk28EcymIM4Yb_bhOSP4dubBSX8VAUQLDND_PQICIv9zqOQzKg4nKofW_3disemji_wWv8MY1dm1MTFvvU4OcqxbJQMboGBQVaIKT4QWwmkDbX9XJ3jq3tUfnrOMA9SaHUnFjOH0ibSGfiM4r2Dj8BJEoM02MnzY098Lfyt_lMT-OmJCTIhXFttFQy9S1Qq6c1oVmAl9-xvEaXD4BEugopEp24BDYgVm7Ej7WKK8m9zE2U-Piuyf8QUKIytRmHXFF2PajnyfrAO0Gry5YlBitgvpPyC4revo5tN9XMx7CJAjcwJxfuD99y4G_gLt5tg-t1DNWQA-im0T4pCLhf6ajutH998LVw5RySPvVa3WA659ymhDJSSIhoc_xMEZyuB7d12DqLoi2SJSb4SLhDixUTNRgDF4tjmE-vlu8rGC6S6pkAx19usdy1BoqSaWxDYXT2ZBvd=w694-h363-no)

## 訓練卷積神經網路的細節與技巧_遷移學習 transfer learning

### 遷移學習, Transfer Learning

- 資料量不⾜時，遷移學習也是很常⾒的⽅法
- 神經網路訓練前的初始參數是隨機產⽣的，不具備任何意義
- 透過在其他龐⼤資料集上訓練好的模型參數，我們使⽤這個參數當成起始點，改⽤在⾃⼰的資料集上訓練！

### 為何可以用遷移學習？

- 記得前⾯ CNN 的課程有提到，CNN 淺層的過濾器 (filter) 是⽤來偵測線條與顏⾊等簡單的元素。因此不管圖像是什麼類型，基本的組成應該要是⼀樣的
- ⼤型資料集訓練好的參數具有完整的顏⾊、線條 filters，從此參數開始，我們訓練在⾃⼰的資料集中，逐步把 filters 修正為適合⾃⼰資料集的結果。

### 參考大神們的網路架構

- 同學們對於要疊幾層 CNN，filters 數量要選擇多少？Stride, Pooling 等參數要設定多少？這些應該都很疑惑。
- 許多學者們研究了許多架構與多次調整超參數，並在⼤型資料集如ImageNet 上進⾏測試得到準確性⾼並容易泛化的網路架構，我們也可以從這樣的架構開始！

### 注意

- 以下的程式碼提供給有 GPU 且具有較⼤影像尺⼨資料集的同學參考，若沒有 GPU 的同學可以直接看本⽇的 jupyter notebook 程式碼學習即可
- Cifar-10 並不適合直接使⽤ transfer learning 原因是多數模型都是在ImageNet 上預訓練好的，⽽ ImageNet 影像⼤⼩為 (224,224,3)，圖像差異極⼤，硬套⽤的結果反⽽不好

### Transfer learning in Keras: ResNet-50

```python
from keras.application.resnet50 import ResNet50
resnet_model = ResNet50(input_shape=(224, 224, 3),
                       weights='imagenet', pooling='avg',
                       include_top=False)
```

- 我們使⽤了 ResNet50 網路結構，其中可以看到 weights='imagenet'，代表我們使⽤從 imagenet 訓練好的參數來初始化，並指定輸入的影像⼤⼩為 (224,224,3)
- pooling=avg 代表最後⼀層使⽤ [Global Average pooling](https://blog.csdn.net/Losteng/article/details/51520555)，把 featuremaps 變成⼀維的向量
- include_top=False 代表將原本的 Dense layer 拔掉，因為原本這個網路是⽤來做 1000 個分類的模型，我們必須替換成⾃⼰的 Dense layer 來符合我們⾃⼰資料集的類別數量

```python
last_featuremaps = resnet_model.output
flatten_featuremap = Flatten()(last_featuremaps)
output = Dense(num_classes)(flatten_featuremap)

New_resnet_model = Model(inputs=resnet_model.input, outputs=output)
```

- 上⼀⾴的模型我們已經設定成沒有 Dense layers，且最後⼀層做 GAP，使⽤resnet_model.output 我們就可以取出最後⼀層的 featuremaps

- 將其使⽤ Flatten 攤平後，再接上我們的 Dense layer，神經元數量與資料集的類別數量⼀致，重建立模型，就可以得到⼀個新的 ResNet-50 模型，且參數是根據 ImageNet ⼤型資料集預訓練好的

- 整體流程如下圖，我們保留 Trained convolutional base，並新建 New classifier(Dense 的部分)，最後 convolutional base 是否要 frozen (不訓練) 則是要看資料集與預訓練的 ImageNet 是否相似，如果差異很⼤則建議訓練時不要 frozen，讓 CNN的參數可以繼續更新

  ![](https://lh3.googleusercontent.com/V6DMVj5pAEL-4xYjWM8nR6wHbNKwtkr4dkfrkPSqhYeXFWpROWklf-OswBNFCWIYurKOL0hmrplua7rREzBzS68ReNTqYE4beK1EUBOkkVfSTRvcIUvDuXsXMoTBJJ9DdEtScjwc8xKGNjv0E2_37p4kldSPdWcQSyPePQCRy9cEVDAxkgFtarJh2q1QIY6ntCxvjsY3DS-zxD0bs-FnUCUOS3y-OIqQkNrCy1X5oJffUq6r4YujDJ61Z-Foq4alG90aqBeo2b9sSn-53hmSHXoZRHbHfxJCJaQxD45p-7fexqxt7wnfpCw5mlWS7KlsvE7eXvxIG0X_3rM2arEBm7tTexnhh6UpgAdA3yYX1-iNg2zWplTvhZPgrqhLmNr3OxiooQ9m0gNdEeFnZWkdgeXUtdho8efB5BAbh0bquY0fu1BxBRcOSkV1ntH_RHo0xL6fDm74SzT-cv8ECwvEQE719LXAqfs_pdaEL4blW1ble-QaneNOwn2sW5KLngtdls2JpBcuGQS7tD2lb8AX3xwyQDnz7maZ1tpVvncy0EmpWXcRaBDiIIFaPNBD9ss-rdjmJW7SizST3dGvpPLD6qZ0xeGPAGEDhyf0AXYZFvKp6Jf4yJNKAtyNDjDFx0yZgu1Y87eFmrY-O-Lf7x_MmN9FVYlQ_MfO_7MU6z0U_npQL95ZYNR23b0UiCcpRXPfrA6u5uTUcYEQrdksLsN4V0O1=w731-h494-no)

### 參考資料

1. [Keras 以 ResNet-50 預訓練模型建立狗與貓辨識程式](https://blog.gtwang.org/programming/keras-resnet-50-pre-trained-model-build-dogs-cats-image-classification-system/)
2. [Img feature extraction with pretrained Resnet](https://www.kaggle.com/insaff/img-feature-extraction-with-pretrained-resnet)
3. [Deep Learning Specialization by Andrew Ng — 21 Lessons Learned](https://towardsdatascience.com/deep-learning-specialization-by-andrew-ng-21-lessons-learned-15ffaaef627c)

