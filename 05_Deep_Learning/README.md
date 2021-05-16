# 深度學習

[深度学习调参有哪些技巧？](https://www.zhihu.com/question/25097993/answer/934100939)

https://keras-cn.readthedocs.io/en/latest/legacy/blog/word_embedding/

https://cvdl.cupoy.com/

## 摘要

- 類神經網絡 (Neural Network)

  - 在1956年的達特茅斯會議中誕⽣，以數學模擬神經傳導輸出預測，在初期⼈⼯智慧領域中就是重要分⽀
  - 因層數⼀多計算量就⼤幅增加等問題，過去無法解決，雖不斷有學者試圖改善，在歷史中仍不免⼤起⼤落
  - 直到近幾年在算法、硬體能⼒與巨量資料的改善下，多層的類神經網路才重新成為當前⼈⼯智慧的應⽤主流
  - 類神經的應⽤曾沉寂⼆三⼗年，直到 2012 年 AlexNet 在 ImageNet 圖像分類競賽獲得驚艷表現後，才重回主流舞台

- 類神經網路(NN)與深度學習的比較

  - 就基礎要素⽽⾔，深度學習是比較多層的類神經網路，但就實務應⽤的層次上，因著設計思路與連結架構的不同，兩者有了很⼤的差異性。
  - 算法改良
    - 網路結構：CNN 與 RNN 等結構在神經連結上做有意義的精省，使得計算⼒得以⽤在刀口上
    - 細節改良：DropOut (隨機移除) 同時有節省連結與集成的效果，BatchNormalization (批次正規化) 讓神經層間有更好的傳導⼒
    - 計算機硬體能⼒提升
    - 圖形處理器 (GPU) 的誕⽣，持續了晶片摩爾定律，讓計算成為可⾏
  - 巨量資料

    - 個⼈⾏動裝置的普及及網路速度的持續提升，帶來巨量的資料量，使得深度學習有了可以學習的素材
  - 解決問題
    - NN: 基礎回歸問題
    - DL:影像、⾃然語⾔處理等多樣問題

- 深度學習

  - **CNN**

    - 設計⽬標：影像處理

    - 結構改進：CNN 參考像素遠近省略神經元，並且⽤影像特徵的平移不變性來共⽤權重，⼤幅減少了影像計算的負擔

    - 衍伸應⽤：只要符合上述兩種特性的應⽤，都可以使⽤ CNN 來計算，例如AlphaGo 的 v18 版的兩個主網路都是 CNN

      ![](https://lh3.googleusercontent.com/-nIspNBfu9CfLh5HhbAUJzVdjFZxRiTqjNvj8P-8xnxL06njn__gz9CSXyC4qDgLXvunl1MjNVFwKtDXe73OmrJOhriU7wCf58l-pZcWHTXm5HjSg5m_iVnkKrvgOaZGf0KYbEn8fACjxcBWPP-mXTXlVpQfJXd60yRVEe9kVE7pOEQSGJmZqIJKBHpZV2jMOP-IU81FhPiWDennCBDH_7UXNPp0yeSBhZBs9NsP59-l0k5PMYhfEf7-6Phgu0kmrnEWeGrvYWxDyhjW8t5CW-kQK2CABYfmzTMGXCqC3aeLbj14WiDdzgJJSGegQ49k0HDsb4dIMFHvDovrCJlmtRAhDvShLrOc2PRg3qzPwZyoZsM9i9-twcQI5Ze2IaoHSNYZXpN9Nzz1sz186329yqyv3AOrggYkEwoUwifBZXWxndAB8tgsuerxaml1jVAsj1WYs7bdsHI842RsDgHPkbDMF9tqmnYks0dR7PnB-aBZtVoJIGJWOElPP3HkvCpJf_8xpb8fXc6ba600lRg0qnC_JgQh5j-Xoq-WXMT0-9_guMRHRZocoll32wWd6STZy6tCZjrubly3cilQb7IsuLjOrS0cVre53mtihFqDXkFxmC1PAe1TmEqrWUWGFX_NrcXWBp02WgF3TSqC2uk234HikN9dJW8uQJCOecmvJXksbdzIKTsPyA0GDIAVgiwn1cNZ6lxHN63ivVSdWtygKP9d=w904-h265-no)

  - **RNN**

    - 設計⽬標：時序資料處理，自然語言處理

    - 結構改進：RNN 雖然看似在 NN 外增加了時序間的橫向傳遞，但實際上還是依照時間遠近省略了部分連結

    - 衍伸應⽤：只要資料是有順序性的應⽤，都可以使⽤ RNN 來計算，近年在⾃然語⾔處理 (NLP) 上的應⽤反⽽成為⼤宗

      ![](https://lh3.googleusercontent.com/2KTKuCkCCpaAoD7D2BPzMqEzChQUNF9VanLKqzSs-39AeawzYH8sX0WIEGBwD1hX8Y_48wH2fwznTexTW69LDUFNd0QMBsCTrLdqs4B43ouQnhQZHECC06lQ3YYGZA29bsN18H3IBexYnS1uBxbPMXb6hvISUIVtFdxsnOtQZsyaeFpodF8IfbcCUBzTxehV9xiZL5vVmBtCWzkPVM_gNx8XtlxsXUc7VBBfgadP6pdo7lITk73mZIeakW01cA8aZ-Gp6GnnbCdjoR2YPCuTUS4tPRkEix46lLbI7QSixKZDdK0YAf2djYM6eYJVkOClu-tB0Ke-5W6uGJtFy3lxDyxZvdTNdsC8vY0QBu9L-LcCvwtuso_fc9zXXXLqdP8sLMnu3JXI1DnPPyKlbDKP_ZEYT-vNCCAzokLBpb0_BkeOVxX0qKlRpJJjYcPw7gvS9TUKkK70YcR84VxI5Vjfw6Zi2pqVnDNXXeQ12XbsONS1zFEa4a3oSoBomh85E80CeccO6HcmQeyMxtIm7edzAC_9EI8dbqEw2kXE1jBp3iC1YKqehhfOd7e9Mq416QCHhiPceVgz7rRic5F529ur3Nnmw-JMmzW_8omCps8N7a_FYF-oWCa816e0Sx8cnPZ1lPDU3saDbJJoIjnuGxHE2Z1NnxBEw9qxK_QcJAjY8uYBFT9kLgp-pbcO4LPuQyB4NCZsaXaSsShahmqkVFvf0LIM=w871-h338-no)



- 深度學習-巨觀結構
  - 輸入層：
    - 輸入資料進入的位置，即我們的解釋變數。
  - 輸出層：
    - 輸出預測值的最後⼀層，可以是連續資料，二分類資料或多分類資料。當資料是多分類的資料時，需要轉換成虛擬變數
  - 隱藏層：
    - 除了上述兩層外，其他層都稱為隱藏層
    - 隱藏層的流程
      1. 將 input 的 neuron * 權重 後加總
      2. 將加總後的資料透過activate function 進行線性與非線性的轉換，藉以放大有用的資料，或過濾掉雜訊。
    - 透過在 input 與 output 層之間加入Hidden Layer，每層Hidden Layer中的許多neuron，我們可以對input layer的資料做許多線性與非線性的轉換，自動做出許多特徵工程。藉以提升模型的解釋效度。
    - 但我們仍然需要對資料做特徵工程，藉以幫助模型更好的擬合模型。

![](https://lh3.googleusercontent.com/_NhfhS_ZVTpiHAh1-MkwiFMOsRrkBpe8faP8k9PD9ZA1nYpF2uFuMMF4HeeN3KdcVrfA3-ocXsFLdRue9Bi9oL8E8XrnTDkJG1fM4eVb-Ucqn1NGTNhHJ3YJ-iIISZHpjPMXv-O5ntebMKpXJ6qhbgBjVQEfKzeLSF-sRIj-n0ixtfg5im_fJjoTFwsY_qwI1AgL1FpT3zNLUHc-VPDA3nD5UrP_3DqkvI13SgCwrIbkutn_4dEEjbxyJ2GmnQuV_bSR6F0MdFSPufaxOagebEUffg30Kb_UjXHwmGdSGWE1rc2QYV0L_b87lmbgB4MoAJybEthq3QSIt6Cht6VARGxtyq5mNym3cGEsfgAsS5FHtZC5hoRHGyFwIlrKLRWkQq2mPh9yTPW1mT8PwFUDjd1V_faMD3LGLM97t7d-Zi5QLA_tiESE_dPoob8QmFNsL5JxpufMZR5tO-PZxZRof6idCDOboVY6JIR2jrXyvzbNliESNEEhtyD8sIT3NEdY3AqpoNNF0WKdM34P3nE5Dssg_pbpW6MAoxFXyqLR0VJBZfc2Fxqi8bpRWl13xri_ahlnBlOK_J_pD8azOI_A1nRbK4d0lSSF7jwFyWFCPLVkcUm_NdZ_xC5w4HLi8qJ2zqHjX-LTIjtPLVTbQ4NOoRtq2GqSRn6emsKoYwTgI4R4c30iBrU4XarNRqAfMUvu_9JIoMPJqyiQqLDa199eXStV=w648-h346-no)



- 深度學習-微觀結構

  由啟動函數轉換輸出，藉由預測與實際值差距的損失函數，⽤倒傳遞⽅式反覆更新權重，不斷縮小預測與實際結果間的落差，最終達成各種應⽤的學習⽬標。

  - **啟動函數(Activation Function)**

    位於神經元內部，將上⼀層神經元的輸入總和，轉換成這⼀個神經元輸出值的函數

  - **損失函數(Loss Function)**

    定義預測值與實際值的誤差⼤⼩

  - **倒傳遞Back-Propagation)**

    將損失值倒傳遞回，轉換成類神經權重更新的⽅法

![](https://lh3.googleusercontent.com/KKTJNQP8DqEs5gfYISfMqq5HSbXhzcYBYF1C7yMUhg0kkxspTjindFQE_7GOKJjh64rhAJCx6HAuhTNNS6HAFaOR3QafQYSay1aRk9UUxgUsWNcc_u2pMUGd-C-7X2hZQTlXlwJu-XyvbI77E8d2txrMiEYWWtl6UFftBJEzWYH8vyI35hHV03GCgPJHL4c_85MHBBMPzARqgIPYXGg-GrZHjaIuRHsynYyf4lLxrWnrIaFI8L1scRbrsd69MXurfyIvDNRge-QEuQk1z_jxfenjhekne_0gtL3EFJgudN0tvZA0ZxR8w3A4JuvrU5pY1SVNpMv6Dnn_asFXlNw9r--2yklkCVg21OCbNKGYrw8SGJVBx59qigXpvmjC3Ki7YLlsSHEb2r_JUA673KBYFeXka7wq3DfSjyMm7ePGSVqh7NlLRwFSTULzzVgxAMrWTcfyfpmFO810Yt5s66d5xL1oNjXkNu4ivl7-Xx6_wQyz8XrFIuR-_i2aGuuyr2w3tPmufl5l6bmlyt4GLV_-c0jp0mBuDVg-czHqS-BK5z9p4jRzctZBGveUmma2rA1BtMqAVP3H1wvBpqk4obwTAO40b29xuqLnh_H1gTp28IwaJcnT6X1Q4vAxULOnpdZZwGSvg5yj5T6-Wfus8XV8sNu-L4iWYSh3JueIK8XrCUN9CKBDhAQgLBqOhT7h0Ubp4juPeR69BpdpCoWUZRjeCtKk=w736-h313-no)

- 參考資料
  - [神经网络的Python实现（一）了解神经网络](https://www.jianshu.com/p/a2bc960ee325)
  - [神经网络的Python实现（二）全连接网络](https://www.jianshu.com/p/51c92cd4eb67)
  - [神经网络的Python实现（三）卷积神经网络](https://www.jianshu.com/p/b6b9a1b0aabf)
  - [Understanding Neural Networks](https://towardsdatascience.com/understanding-neural-networks-19020b758230)

## 深度學習體驗平台

- TensorFlow PlayGround 是 Google 精⼼開發的體驗網⾴，提供學習者在接觸語⾔之前，就可以對深度學習能概略了解
  - 體驗平台網址 : https://playground.tensorflow.org
- 平台上⽬前有 4 個分類問題與 2 個迴歸問題，要先切換右上問題種類後，再選擇左上的資料集

![](https://lh3.googleusercontent.com/J-JoDiRHF79zpo2aZzaTk-vy7pLoFocuEnE22qYpflyOmm7DaobZ9HpiE63ZmiBwid-k7AP2qDHOUv6FwjsWN7ZFBmo4UgowS_TwL7jyV-9VqtnWPMCqVeaQIfQ-orWa5SJoQS06bOWIkg-M20B3JWoKY4jN10NCUC9MZVykVlizcExcZeQRVYMiHdzo08Dr1nFW3AOAyVrM1fQqCnKDy-_NrV1q5dNYtmoBOEXXE-qiZnSqslm9ItmpiWcBm1WnUFwqD9T0qywD_0I40fRSNf6tVA1aX0pqwVS_Ia3VYTlOI20zz5adA6iH7fVXGhQxK3mP1Nn0Q0MHvuc2yByUr3KphFQP3GcFmqlg-GfHkSPM1fCEUAzxzmuGsdZrWxU3x2ES1MIlHYQG4fCYaUVJqvCZqEFvQUSEFo2q-4nSLYCpGjbU0vFi8Izj6eN7LqFp2-acwD-tiGqdKefDnPIYr-Qe0IEBa6esprPLXwoRYunHHsM4YqG1y-zaFaTI5pzXUYIs2NYZ0CAysN-lcwq1Hbdhxme0kuVV3f5hBv0r9e2vpUPCMx03Fk_zvJ_NmNA8tr8rl0yMzFAZbEV5XTWqiFhGE0_egfumvuRbNqxK-rmb2Za_tdTSemuI-SLPVBQIniNQQCsmnu78IqBEpcc2XJ2CMJhpg_ioz0I04ff4CUcIHFJkr8P9K5N7YrggKHVADN7uzdnsahvEPYCKNVhrUPHY=w959-h541-no)

- 練習項目

  - **練習1：按下啟動，觀察指標變化**

    - 全部使⽤預設值，按下啟動按鈕，看看發⽣了什麼變化?
      - 遞迴次數（Epoch，左上）：逐漸增加
      - 神經元（中央）：⽅框圖案逐漸明顯，權重逐漸加粗，滑鼠移⾄上⽅會顯⽰權重
      - 訓練/測試誤差：開始時明顯下降，幅度漸漸趨緩
      - 學習曲線：訓練/測試誤差
      - 結果圖像化：圖像逐漸穩定

     - 後續討論觀察，如果沒有特別註明，均以訓練/測試誤差是否趨近 0 為主，這種情況我們常稱為收斂

  - **練習2：增減隱藏層數**

    - 練習項目

      - 資料集切換：分類資料集(左下)-2 群，調整層數後啟動學習
      - 資料集切換：分類資料集(左上)-同⼼圓，調整層數後啟動學習
      - 資料集切換：迴歸資料集(左)-對⾓線，調整層數後啟動學習

      ![](https://lh3.googleusercontent.com/g0p-a-z2jhfk2uW6IZe6hAVeDstOJ08iRXhU4RNUu96veogjL6jGRgtshNJp76Xr_WxinGDuOiTbVyUx0EB16KNYt9jgCDU1jpscOrzlBWPBrEVECwCkzZlcvHIFFnzhAfGzKxd6EPffbAPc-nIFbFEGDslsT7AXrcB2VFfgMb69GXBAore7jfjNJ6ZZcbSFVvvu9smAOS4nd8rFdxzHNtzN2Uu5o9k4-ujs99o_bdXgm8cD6iKr3E6IQawCSiYsvgfigmn78ET9pUBeUuCnGge0p_491JIbHbIWw2l8EHQh8uWI4r2xNAAnZBlZoWlwiTICRTofREUOjDkC4Le6PMBoTan_GvqiubEgWgE7XBtkrSMO4WTIjCromybr0ViF2BGL9ti6hksIFftDi_RWNwtPtJf_hUlve5KaAHubiA22PrE3SWatLCZhgeN6IsfU-ruPys6KFgRPJ8UkZuVNkZ-_6H0wlcIDEFeaCCj9MztdmOTW137MjkeFrs3TZQ2L4fXuR9yHCfPFTkmexLkLGEFyBaipbX1Yy9z6XmHLpip5dTqSgX2I1GLv3_Ys0aHjZ5yhFRRR7QTp2reHroGdAUm3siwCTUl7TmCxjYjVfKGQdPs1XrGCrwG8b266v7HvGi1j-STsMDGYYiW7UBje8XRdCAQ42JUBsdx2snpza-Piosrti4MXlR_46P4H-yYgCw9zZQyVp51H_Qf4kBLny1Fo=w1018-h363-no)

    - 實驗結果

      - 2 群與對⾓線：因資料集結構簡單，即使沒有隱藏層也會收斂
      - 同⼼圓：資料及稍微複雜 (無法線性分割)，因此最少要⼀層隱藏層才會收斂

  - **練習3：增減神經元數**

    - 練習項目

      - 資料集切換：分類資料集(左上)-同⼼圓，隱藏層設為 1 後啟動學習
      - 切換不同隱藏層神經元數量後，看看學習效果有何不同？

    - 實驗結果

      - 當神經元少於等於兩個以下時，將無法收斂(如下圖)

        ![](https://lh3.googleusercontent.com/CPNG3ir92ZqfKRnt9BHtsb2i_zU3hVXzZZfKY59yR7BX8CLsQQwjg-rEPzRujgQ98bSXSFMq2QYlc_RxgiqsmK9Y91_wDChtZfpnxhoeRHKFaZlDG58bZV1Yxe7nLPZSg0ZKQqpy1PvoJjXIp_NA_-dLJGEpSpWbBfMCqHcVT3Emf1rzN3jZhTfQoDZ52QTroMqVdL9JTUivHVljzgxN6bIEBkLFmtYigCyVtG1X4O8t4PHSDON-tyiCcTPajTuamRkh5GZ-Wyav4VhxLHNFGGEMwYmreIrkS9xaBZ9RMaUcmbaXis3vpEYGV3VFTpn_7nsrgAP2E-Kdddy5VoJvphuj4WiJbDk-I9SzGFEBXUgQ2hru6zuUg2NOK4UycQR_UBxzqvEZ_WGPJUip9bQPXPpiiusEGr6PoNz13zeWT2CysSom9uAsNos18AApsHnKk_q3QdlxDm9rkPnBlX_rSwzTk-l-3Rm2F3wf7tBufm5VufAjpK41_WNSvuAskYouwLlNIw_qCYiBEiZazldPlKDDcyQYoj81hCi4OqcvHx4poSPqFtwdyG4v4kTpgo8_St78voRgvYi1p-MoodLFYYJU5xk3cUvs6uXxBJOE3HqkL0Ihf05yvkbBRBCIb4kbMlDbzdPSIXSCLbUyKuh-BS_RswH_l_ZCcF0GXamiAHl8uBPxTk7cvKwvoUgUS6MZcLZT4So5WifhCJkgUKoUcmlJ=w774-h286-no)

  - **練習4：切換不同特徵**

    - 練習項目

      - 資料集切換：分類資料集(左上)-同⼼圓，隱藏層 1 層，隱藏神經元 2 個
      - 切換 任選不同的 2 個特徵 後啟動，看看學習效果有何不同?

    - 實驗結果

      - 當特徵選到兩個特徵的平⽅時，即使中間只有 2 個神經元也會收斂

        ![](https://lh3.googleusercontent.com/3jpTlFx9xIf5oacDXExgBVTTFswVLWD5pLjo4sZRmc2MRMPGimX0jWYMC_8DB8v5_g3HU9eqRYCQqI12G0vwsV_zFIDGbX6X9jz93e6up5P96et6kLQ5OYQ3_VlTr6i7ecFZ_7J9Mydkz2vDqmcIMw1RORRyyjYWoYHBUO00U5dDU6aSqUrzmwKxf4d3fTeks7lhQTTbx5q-gh039AuQVDUMCU6uW5qY6cdvMm-nkpMYsmNjZc5dTpswz9L2m7VM3eQbrQArT1ihsGnD-sRVL9nUFEaOjrKrgtyCf4PxOQMCV0GU6aj5DFEmJr8pSw3EbyUFULYgRvrBFN8RWt7sVpCZH-R-CtZWJKa2WOMBTQQVhpX9L1t8eCYwHOryJ0lVDJVL7F8MzXR8SmJ9L47K9ejLs4vAqLqE0Y37K0R-aUhVJdL7eR-0VT_DpN6nk4N8h9mGjx3x60FjFMmsbkNHH6mA7AMSjJ5rh2d8mUHzEbM9Mqu_unzvENsYwS0WTAfIJWRjTlCNuc3x4ta901O619doeAmmBeu2ae2gCM7WUGabQbRiDynaq3ux78s3DxO4Ax1vF3IYJ7UGXBQpkUcS7hb7PHG-3PUkUYCtpcafwd2-qQJYj-J54X_kfbL8oeECcbGQhBqebdTOgLIMUGBAbm1dB-zNhzgDD59TSIWavGStmmgrNZ_-hiD7OWoIkNu1ap58SIxmzkAb7Tu_hGgQ5X_1=w755-h285-no)

  - **練習 5：切換批次⼤⼩**

    - 練習項目

      - 資料集切換:分類資料集(右下)-螺旋雙臂，特徵全選，隱藏層1層/8神經元
      - 調整 不同的批次⼤⼩後執⾏500次遞迴，看看學習效果有何不同?

    - 實驗結果

      - 批次⼤⼩很⼩時，雖然收斂過程非常不穩定，但平均⽽⾔會收斂到較好的結果

      - 實務上，批次⼤⼩如果極⼩的效果確實比較好，但計算時間會相當久，因此通常會依照時間需要⽽折衷

        ![](https://lh3.googleusercontent.com/RyGT7m_Fvz-35onVdJ_KNNvIab5rypkuCFk8jR5S_w12ACma0y0OqMkN4WMLhooyLOIa9rXAepJ3_0XyWUuU0dkIDC1ITAJMoVP8mbITk0NuowLLC4Y30Dx88GcZP9Xca6rURzhNJFL1C9pgeRusQ5ZTdmmzxum9IlWwUU3mWNad9VHerbZcL_qNnHZqIW5-kFifhrSc_5atzl8TCyrm3ljxjuouuvwaaS34W4c4ECm1-ivu9NxXWHz80UCiCy8HbHlSssWLb5vQ89nsxFZCMgCiCV1Uc2YEgpWXAf_M2zhjyc2x6h7LgF9JDfTzQGwJLldkbuRO-BP308Q_76vU-dhdkVB42gLUQF_F_fEo74OWC5GcCtOGhtJ0_FZ3OmUlUMr01JcggSslgmyubquK2FASbXvtqCrd4U4hNtUJ82s-rwdjbVzlqszjy3EBaQ0yT0O8fvB_do43RMKAaVtdMvXI6RxdkN4X_KqNkxOu6WwGV8N0KoYdAKcHxU4D4xuLj96QaGE55LtelAl0eK8-ypQHKCUwz4LA2msP7HjSdTEvnsCSMAEq9YEle5uGbIJSY9b3FNb8vdo-3mVpb86E4qw49HkCA51g2x4sV96YkRtw4H37xF_8Mej56BLQhw0rObVwtg3xO85zeOLf2GeUFDRzqhoam5EKN5U2xUWY6Rb9Z_KmpXuZXfV8WwTpgDpwUos_vKvU-c64mkBDSLvK3c9q=w418-h184-no)

  - **練習 6：切換學習速率**

    - 練習項目

      - 資料集切換：分類資料集(右下)-螺旋雙臂，特徵全選，隱藏層 1 層 /8 神經元，批次⼤⼩固定 10
      - 調整 不同的學習速率 後執⾏ 500 次遞迴，看看學習效果有何不同?

    - 實驗結果

      - 學習速率較⼤時，收斂過程會越不穩定，但會收斂到較好的結果

      - ⼤於 1 時 因為過度不穩定⽽導致無法收斂

        ![](https://lh3.googleusercontent.com/K3yK4ia66stGDckpdgBi5eqAgDoAXVKOqRqYrc2tDgRtKN0gORUslM36dC74BXG2rirCGVnXC3oKQ-teSEGpBMNcD73OnzNyJDyh4GVY5Q-8rYk8pEf2EiA5IndphBSin4t_xI9tMu3uy2EszdofJ0BTHbc6eZBhNuBUy8qg78xZczzih6Hex8OM-WLw8JeBUAUf-7ND55yEgdF5Stpy1xvb2HSIyjb6le7nvNP1KEVK9KL62vEjw4ayKqE5SnqatYbw7Oykos6l8kx-wracNEbQ9Ow5f9xzmqKMFU5d6cx3lv5VQr2ozOL_l4MlmBtkqPIQKFkBUXQMpQGMOsy_4tQUwT82JAyJwZX_gpAEmGdAisXfVJn3eza0cH8nHXk6E9xS9HBGsH0szQ7cPOlZI-Jru75e1QEDzuX6R8_2oCSrQFgTm19PcQ9Z4ntkndcY0o_qJs-3gzPAYasw_ZfcivXEF8vFLGT-UBaU_U0ff89PmcE8Ypt14dmXIRM8NEIHm7E06pISOZgfWzrhWPgDCtvGPa6IyaqZcE6Ln2nRpxzItaMHVPDBH56cSZdG_7iKw5Wfv-vvVkHMOj_NdJeWaUaZJmLgDMZA-HROluUjgbz9YvQHSOrEZcbGNUwC4e_9aLjmiidlsAnnJVi4f8N8tvlbZvdpHBOV43eQj-VFlEBwfeWKnftNbfhHxvyijHg9VO_LP1oIR7ecBgJZOXSuN-mn=w417-h334-no)

  - **練習 7：切換啟動函數**

    - 練習項目
      - 資料集切換 : 分類資料集(右下)-螺旋雙臂，特徵全選，隱藏層 1層 /8 神經元，批次⼤⼩固定 10，學習速率固定 1
      - 調整不同的啟動函數 後執⾏500次遞迴，看看學習效果有何不同?
    - 實驗結果
      - 在這種極端的情形下，Tanh會無法收斂，Relu很快就穩定在很糟糕的分類狀態，惟有Sigmoid還可以收斂到不錯的結果
      - 但實務上，Sigmoid需要⼤量計算時間，⽽Relu則相對快得很多，這也是需要取捨的，在本例中因位只有⼀層，所以狀況不太明顯

  - **練習 8：切換正規化選項與參數**

    - 練習項目
      - 資料集切換:分類資料集(右下)-螺旋雙臂，特徵全選，隱藏層1層/8神經元，批次⼤⼩固定 10，學習速率固定 0.3，啟動函數設為 Tanh
      - 調整不同的正規化選項與參數後執⾏500次遞迴，看看學習效果有何不同?

    - 實驗結果
      - 我們已經知道上述設定本來就會收斂，只是在較⼩的 L1 / L2 正規劃參數下收斂比較穩定⼀點
      - 但正規化參數只要略⼤，反⽽會讓本來能收斂的設定變得無法收斂，這點 L1 比 L2情況略嚴重，因此本例中最適合的正規化參數是 L2 + 參數 0.001
      - 實務上：L1 / L2 較常使⽤在非深度學習上，深度學習上效果有限

- **重點摘錄**

  - 雖然圖像化更直覺，但是並非量化指標且可視化不容易，故深度學習的觀察指標仍以損失函數/誤差為主
  - 對於不同資料類型，適合加深與加寬的問題都有，但加深適合的問題類型較多
  - 輸入特徵的選擇影響結果甚鉅，因此深度學習也需要考慮特徵⼯程
  - 批次⼤⼩越⼩ : 學習曲線越不穩定、但收斂越快
  - 學習速率越⼤ : 學習曲線越不穩定、但收斂越快，但是與批次⼤⼩不同的是，學習速率⼤於⼀定以上時，有可能不穩定到無法收斂
  - 當類神經網路層數不多時，啟動函數 Sigmoid / Tanh 的效果比 Relu 更好
  - L1 / L2 正規化在非深度學習上效果較明顯，⽽正規化參數較⼩才有效果

- **參考資料**

  - [深度学习网络调参技巧](https://zhuanlan.zhihu.com/p/24720954)



## Keras 簡介與安裝

- Keras 是易學易懂的深度學習套件

  - Keras 設計出發點在於容易上⼿，因此隱藏了很多實作細節，雖然⾃由度稍嫌不夠，但很適合教學
  - Keras 實作並優化了各式經典組件，因此即使是同時熟悉TensorFlow 與Keras 的老⼿，開發時也會兩者並⽤互補

- Keras包含的組件有哪些?

  - Keras 的組件很貼近直覺，因此我們可以⽤ TensorFlow PlayGround 體驗所學到的概念，分為兩⼤類來理解 ( 非⼀⼀對應 )
  - 模型形狀類
    - 直覺概念：神經元數 / 隱藏層數 / 啟動函數
    - Keras組件 : Sequential Model / Functional Model / Layers
  - 配置參數類
    - 直覺概念：學習速率 / 批次⼤⼩ / 正規化
    - Keras組件 : Optimier / Reguliarizes / Callbacks

- 深度學習寫法封裝

  - TensorFlow 將深度學習中的 GPU/CPU指令封裝起來，減少語法差異，Keras 則是將前者更進⼀步封裝成單⼀套件，⽤少量的程式便能實現經典模型

- Keras的後端

  - Keras 的實現，實際上完全依賴 TensorFlow 的語法完成，這種情形我們稱 TensorFlow 是 Keras 的⼀種後端(Backend)

- Keras/TensorFlow的比較

  |          | Keras        | Tensorflow                     |
  | -------- | ------------ | ------------------------------ |
  | 學習難度 | 低           | 高                             |
  | 模型彈性 | 中           | 高                             |
  | 主要差異 | 處理神經層   | 處理資料流                     |
  | 代表組件 | Layers/Model | Tensor / Session / Placeholder |

- 安裝方法

  - 由於 Anaconda 已經會自動幫忙安裝 CUDA / cuDNN 因此我們只需要創造一個 tensorflow 的虛擬環境，啟動 jupyter 後輸入直接安裝 gpu 版的 tensorflow 即可!

  - 安裝完後，建議重新啟動電腦 / restart kernel 後再跑相關語法!

    ```python
    conda create --name tensorflow
    activate tensorflow
    jupyter lab
    conda install tensorflow-gpu
    conda install keras
    ```

- Ref

  - [Keras: The Python Deep Learning library](https://keras.io/)
  - [别再使用pip安装TensorFlow了！用conda吧](https://zhuanlan.zhihu.com/p/46579831)



## Dataset 介紹與應⽤

- **CIFAR10**

  - CIFAR10 small image classification
- Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.

```python
from keras.datasets import cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- **CIFAR100**

  - CIFAR100 small image classification

  - Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

    ```python
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    ```

- **MNIST database**

  - MNIST database of handwritten digits

  - Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

    ```python
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnsit.load_data() 
    ```

- **Fashion-MNIST**

  - Fashion-MNIST database of fashion articles

  - Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST. 

    ```python
    from keras.datasets import fashion_mnsit
    (x_train, y_train), (x_test, y_test) = fashion_mnsit.load_data()
    ```

- **Boston housing price**

  - Boston housing price regression dataset
  - Dataset taken from the StatLib library which is maintained at Carnegie Mellon University.
  - Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. Targets are the median values of the houses at a location (in k$).

  ```python
  from keras.datasets import boston_housing
  (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
  ```

- **IMDB電影評論情緒分類**

  - 來⾃ IMDB 的 25,000 部電影評論的數據集，標有情緒（正⾯/負⾯）。評論已經過預處理，每個評論都被編碼為⼀系列單詞索引（整數）。

  - 單詞由數據集中的整體頻率索引

    - 整數“3”編碼數據中第 3 個最頻繁的單詞。
    - “0”不代表特定單詞，⽽是⽤於編碼任何未知單詞

  - 說明

    - path：如果您沒有本地數據（at '~/.keras/datasets/' + path），它將被下載到此位置。
    - num_words：整數或無。最常⾒的詞彙需要考慮。任何不太頻繁的單詞將oov_char在序列數據中顯⽰為值。
    - skip_top：整數。最常被忽略的詞（它們將 oov_char 在序列數據中顯⽰為值）。
    - maxlen：int。最⼤序列長度。任何更長的序列都將被截斷。
    - 種⼦：int。⽤於可重複數據改組的種⼦。
    - start_char：int。序列的開頭將標有此字符。設置為 1，因為 0 通常是填充字符。
    - oov_char：int。這是因為切出字 num_words 或 skip_top 限制將這個字符替換。
    - index_from：int。使⽤此索引和更⾼的索引實際單詞。

    ```python
    from keras.datasets import imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz', 
                                                          num_words=None,
                                                          skip_top=0, 
                                                          maxlen=None, 
                                                          seed=113, 
                                                          start_char=1, 
                                                          oov_char=2, 
                                                          index_from=3)
    ```

- 路透社新聞專題主題分類

  - 來⾃路透社的 11,228 條新聞專線的數據集，標註了 46 個主題。與 IMDB 數據集⼀樣，每條線都被編碼為⼀系列字索引

  ```python
  from keras.datasets import reuters
  (x_train, y_train), (x_test, y_test) = reuters.load_data(path='reuters npz', 
                                                           num_words=None,
                                                           skip_top=0,
                                                           maxlen=None,
                                                           test_split=0.2,
                                                           seed=113,
                                                           start_char=1,
                                                           oov_char=2,
                                                           index_from=3)
  ```

- [Imagenet](http://www.image-net.org/about-stats)

  Imagenet數據集有1400多萬幅圖片，涵蓋2萬多個類別；其中有超過百萬的圖片有明確的類別標註和圖像中物體位置的標註。

  Imagenet數據集是目前深度學習圖像領域應用得非常多的一個領域，關於圖像分類、定位、檢測等研究工作大多基於此數據集展開。Imagenet數據集文檔詳細，有專門的團隊維護，使用非常方便，在計算機視覺領域研究論文中應用非常廣，幾乎成為了目前深度學習圖像領域算法性能檢驗的“標準”數據集。數據集大小：~1TB（ILSVRC2016比賽全部數據）

- [COCO](http://mscoco.org/)

  COCO(Common Objects in Context)是一個新的圖像識別、分割和圖像語義數據集。

  COCO數據集由微軟贊助，其對於圖像的標註信息不僅有類別、位置信息，還有對圖像的語義文本描述，COCO數據集的開源使得近兩三年來圖像分割語義理解取得了巨大的進展，也幾乎成為了圖像語義理解算法性能評價的“標準”數據集。

- 資料集應用

  - 適⽤於⽂本分析與情緒分類
    - IMDB 電影評論情緒分類
    - 路透社新聞專題主題分類
  - 適⽤於 Data/Numerical 學習
    - Boston housing price regression dataset
  - 適⽤於影像分類與識別學習
    - CIFAR10/CIFAR100
    - MNIST/ Fashion-MNIST
  - 針對⼩數據集的深度學習
    - 數據預處理與數據提升

- 參考資料

  1. [Keras: The Python Deep Learning library](https://github.com/keras-team/keras/)
  2. [Keras dataset](https://keras.io/datasets/)
  3. [Predicting Boston House Prices](https://www.kaggle.com/sagarnildass/predicting-boston-house-prices)




## Keras Sequential API

- 序列模型

  - 序列模型是多個網路層的線性堆疊。
  - Sequential 是⼀系列模型的簡單線性疊加，可以在構造函數中傳入⼀些列的網路層：

  ```python
  from keras.models import Sequential
  from keras.layers import Dense, Activation
  model = Sequential([Dense(32, _input_shap=(784,)), Activation(“relu”)
  ```

  - 也可以透過 .add 疊加

  ```python
  model = Sequential()
  model.add(Dense(32, _input_dim=784))
  model.add(Activation(“relu”))
  ```

- Keras 框架回顧

  ![](https://lh3.googleusercontent.com/iFg9DGhNe0unCJaWaNzBXd0zH8PYJroNj3IevU9CQ6l0B98ySSxILtc-O--OSYuGpvQQgGSXEFCpRwVicQtPYjPJe66ab0wn-GuaiSSfxCyIMtcRc3yuxBMwGo3YVB6Lu8-gmo1fv-uTMZkJsEV-mnOuPBdCc-Wk84Z2tcAOSaA4YUMONlapCzj7mxA4kL9Ri2NLTHmgcrF3fRF5IUqmr-_fYY0P5qxdQOlRjuDKDFWnWzYhvFnuLnC-ZMFldiGihjLPledSwrUuwpcBjD4UI05ViQKixCG1QmEWQ4azMPsW_T-15dxBxNdO1a256yJJFLf9Nn4zep58Oc6FucTl6sekFcz3bkpyxbCe1mR-FxGFwB22ThoNnM1V2ftBEEN8W5kLk2wUcXW7frJ4qs7FvOGczcsucvPZ1VGWKzA-ZwFVLxVn0EY897TkrGp93C0rU2Jd9YYGzTZkxAc3F0fS8S9_V6l8TnjaZoHLVxsdEo3i207-diKILeyMaE4f-XN_GP8olxcMhzFOYAD_-LrkhuJJg2ZSRwO_ROBnpwk7iUNCL0cAApVUdzmwSt9ch3Q3QVB7lcSpt8RXwJQFLx75V_cmZZH4juoTrZ1WhvZz50YpJmzTXGDRTgSzIkdkXj3k-7WH6W1OeqgdHZ0EKeThLicQm23NbJSkVpwLLvIvfR9yL8DfVlxaEzmD5UftOzupkBYnnjri91YyjvXaNecXCEtLKJ-rELaaDQ0zCZoDyGfEQgI=w1305-h697-no)

- 指定模型的輸入維度

  - Sequential 的第⼀層(只有第⼀層，後⾯的層會⾃動匹配)需要知道輸入的shape
    - 在第⼀層加入⼀個 input_shape 參數，input_shape 應該是⼀個 shape 的 tuple 資料類型。
    - input_shape 是⼀系列整數的 tuple，某些位置可為 None
    - input_shape 中不⽤指明 batch_size 的數⽬。

  - 2D 的網路層，如 Dense，允許在層的構造函數的 input_dim 中指定輸入的維度。
  - 對於某些 3D 時間層，可以在構造函數中指定 input_dim 和 input_length 來實現。
  - 對於某些 RNN，可以指定 batch_size。這樣後⾯的輸入必須是(batch_size, input_shape)的輸入

- 常用參數

  - Dense：實現全連接層

    ```python
    Dense(units,activation,use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
    ```

  - Activation：對上層輸出應⽤激活函數

    ```python
    Activation(activation)
    ```

  - Dropout：對上層輸出應⽤ dropout 以防⽌過擬合

    ```python
    Dropout(ratio)
    ```

  - Flatten:對上層輸出⼀維化

    ```python
    Flatten()
    ```

  - Reahape:對上層輸出 reshape  

    ```python
    Reshape(target_shape)
    ```

- 前述流程 / python程式 對照

  ```python
  
  ```

- Sequential 模型

  - Sequential 序列模型為最簡單的線性、從頭到尾的結構順序，⼀路到底
  - Sequential 模型的基本元件⼀般需要：
    1. Model 宣告
    2. model.add，添加層
    3. model.compile,模型訓練
    4. model.fit，模型訓練參數設置 + 訓練
    5. 模型評估
    6. 模型預測

  ```python
  model = Sequential()
  model.add(Conv2D(64, 3,3), padding='same', input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  ```

- 參考資料

  [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)



## Keras Functional API

- Keras 函數式模型接⼝是⽤⼾定義多輸出模型、非循環有向模型或具有共享層的模型等複雜模型的途徑

- 定義復雜模型（如多輸出模型、有向無環圖，或具有共享層的模型）的⽅法。

- 所有的模型都可調⽤，就像網絡層⼀樣

  - 利⽤函數式API，可以輕易地重⽤訓練好的模型：可以將任何模型看作是⼀個層，然後通過傳遞⼀個張量來調⽤它。注意，在調⽤模型時，您不僅重⽤模型的結構，還重⽤了它的權重。
  - 具有多個輸入和輸出的模型。函數式 API 使處理⼤量交織的數據流變得容易。
    - 試圖預測 Twitter 上的⼀條新聞標題有多少轉發和點贊數
    - 模型的主要輸入將是新聞標題本⾝，即⼀系列詞語。
    - 但是為了增添趣味，我們的模型還添加了其他的輔助輸入來接收額外的數據，例如新聞標題的發布的時間等。
    - 該模型也將通過兩個損失函數進⾏監督學習。較早地在模型中使⽤主損失函數，是深度學習模型的⼀個良好正則⽅法。
  - 函數式API 的另⼀個⽤途是使⽤共享網絡層的模型。
    - 來考慮推特推⽂數據集。我們想要建立⼀個模型來分辨兩條推⽂是否來⾃同⼀個⼈（例如，通過推⽂的相似性來對⽤⼾進⾏比較）。
    - 實現這個⽬標的⼀種⽅法是建立⼀個模型，將兩條推⽂編碼成兩個向量，連接向量，然後添加邏輯回歸層；這將輸出兩條推⽂來⾃同⼀作者的概率。模型將接收⼀對對正負表⽰的推特數據。
    - 由於這個問題是對稱的，編碼第⼀條推⽂的機制應該被完全重⽤來編碼第⼆條推⽂（權重及其他全部）。

- 函數式API 與 順序模型

  - 模型需要多於⼀個的輸出，那麼你總應該選擇函數式模型。
    - 函數式模型是最廣泛的⼀類模型，序貫模型（Sequential）只是它的⼀種特殊情況。
  - 延伸說明
    - 層對象接受張量為參數，返回⼀個張量。
    - 輸入是張量，輸出也是張量的⼀個框架就是⼀個模型，通過 Model 定義。
    - 這樣的模型可以被像 Keras 的 Sequential ⼀樣被訓練

- Keras 函數式模型接⼝是⽤⼾定義多輸出模型、非循環有向模型或具有共享層的模型等複雜模型的途徑

- 延伸說明

  - 層對象接受張量為參數，返回⼀個張量。
  - 輸入是張量，輸出也是張量的⼀個框架就是⼀個模型，通過 Model 定義。
  - 這樣的模型可以被像 Keras 的 Sequential ⼀樣被訓練

- 如何設定

  使⽤函數式模型的⼀個典型場景是搭建多輸入、多輸出的模型

  ```python
  from keras.layers import Input
  from keras.models import Model
  main_input = Input(shape=(100,), dtype='int32', name='main_input')
  ```

- 參考資料

  [Getting started with the Keras functional API](https://keras.io/getting-started/functional-api-guide/)



## Multi-layer Perception

- Multi-layer Perceptron (MLP)：MLP 為⼀種監督式學習的演算法，可以使⽤非線性近似將資料分類或進⾏迴歸運算

- 多層感知機其實就是可以⽤多層和多個 perception 來達到最後⽬的

- 在機器學習領域像是我們稱為 multiple classification system 或是 ensemble learning

- 深度神經網路(deep neural network, DNN)，神似⼈⼯神經網路的 MLP

- 若每個神經元的激活函數都是線性函數，那麼，任意層數的 MLP 都可被約簡成⼀個等價的單層感知器

- 多層感知機是⼀種前向傳遞類神經網路，⾄少包含三層結構(輸入層、隱藏層和輸出層)，並且利⽤到「倒傳遞」的技術達到學習(model learning)的監督式學習，以上是傳統的定義。

- 現在深度學習的發展，其實MLP是深度神經網路(deep neural network, DNN)的⼀種special case，概念基本上⼀樣，DNN只是在學習過程中多了⼀些⼿法和層數會更多更深。

  - 以NN的組成為例

    ![](https://lh3.googleusercontent.com/GnGBa_1pWSHeVGOgOY_tsmB5NG6YuL3RJGj7Z_dgbry9QOh1d6P1ni-eNFebNIZq9CK_YoS9nE9CaVu6VVgwHnG9d04TuhPKFucr8Jc_Zo2yA899tJlLIOnlI5thVFYarAy7K4ZtXufr-h5QKKmQIB7YeOo0Vmlu46C_Xl0I5QewePgFlkCEjQ2q6gvmsyJv9V2ZtItKKIRXLfflJ35NkjixOd5ylSSvOTLzOg_ww_gX24RiTnnens64-wJRZzhB_aMghCDMBqbLbBzrWo3ONdHUbNTKZXntJJEE_HitEQPq1S2LfqjfltUiOrh3lrDMuxv8wvrUGM1i3TxXj7lqdZkrdI4wwEQzTYnLYCjWjaoqVQdnADSS02aQ08Omv6AZvjg7rejeCaS1pCvpg72z-_ebWeismdZgy7euMeFutk0EfyVgTO4LYuNYBVTLLXe779TypM9ccd7K1pHtmeQegyO14dlkx7irtbMB_rvFjnx19XuPSZ2b_RTdhg3CJXMWiuWHRwvevJCibKFDkHkgvnImZEEGBh7ihbySvyLWaeas6aeoRZfOhyzsZ-O-1nkwDQQA05Y_yESMPCGIqjcbSlEUFerhhZAQc3z3CLOdJEwATyUtbjDFCmYvpU1ArA838q-hLOt84c2mKkVZPkkYH8qw-qQ1WUhxuhJ9HrlLyMa98p9PYrayBKwRZwoggtGWIOyYmUkYjn83AceNVd7GCbSa=w873-h385-no)

  - Multi-layer Perceptron(多層感知器): NN 的結合

    ![](https://lh3.googleusercontent.com/eLjfhWJKXjRojUHbqqLb_kbKz_GetRM5K9QR8R1Rfj0IM9Fj6jf3FzFd5Wqt3fJY7UT-_MMSwHnUxl50TNyITdegGXOyuopEcl-d-Qo4YF6bkpLwq8lzb3Oe0NTyCH0Cz3ptxHvFQhhFNX-mIfolWjIXgLZ451WwECqXrmF2gCZlF4obsfEAP3lYdTuKJ_4iA36RHk-UgTaFSV7uvncxas1facXfYDurR9W4CzPuY0wykjp4OdIlDrj1nfDwIzNbsE7XpSvTOMJHiOsrSitNOLcLoGj0VxdnObi8NrrCUPg2LcxkcyfUPttdMMZCRcMj_to3K4_ADLnoxUrlWiWOr4S1an0CUpm24EWTly-3IiaHIxJVYsnYT0CRrxRzlhai2bVQUKUJHAWxBgMidQO0x69kcaeOkR8Ocx1XePamoeI2YihqRBqdzQB2K03HKjsMFXJ_Q7A3j-DilVwsFWc1w3GiG4PO9UekQCCRVj-XTKTnSrpHeDyEcx5Ufzf1uHePnAdf35-s8RU8ZZoUgxD5GbZhD7tbDZtGJFaoxKJNevao_tOY3n5W4Eq9mcDzHpKxPqzZ-Ar1XfTURfCrTPfefROKnMxQDg4-sLrepkvmobyyhG3d_syEJkJfYXyCx7lSYIU8Je0fhx76_MT6Ar-tCy-8dxO1kGcwZ-bO-ppXu7KfXBqUs6BnptK2BGiDpToeNk2ryBVxby3axOFbkl6C9-_B19BZMtTTwVPMuFSZyDQyA1U=w598-h372-no)

- **MLP優缺點**

  - **優點**
    - 有能⼒建立非線性的模型
    - 可以使⽤ partial_fit​ 建立 realtime模型
  - **缺點**
    - 擁有⼤於⼀個區域最⼩值，使⽤不同的初始權重，會讓驗證時的準確率浮動
    - MLP 模型需要調整每層神經元數、層數、疊代次數
    - 對於特徵的預先處理很敏感

- 參考資料

  - [機器學習-神經網路(多層感知機 Multilayer perceptron, MLP)運作方式](https://medium.com/@chih.sheng.huang821/機器學習-神經網路-多層感知機-multilayer-perceptron-mlp-運作方式-f0e108e8b9af)

  - [多層感知機](https://zh.wikipedia.org/wiki/多层感知器)



## 損失函數

- 機器學習中所有的算法都需要最⼤化或最⼩化⼀個函數，這個函數被稱為「⽬標函數」。其中，我們⼀般把最⼩化的⼀類函數，稱為「損失函數」。它能根據預測結果，衡量出模型預測能⼒的好壞
- 損失函數中的損失就是「實際值和預測值的落差」，根據不同任務會需要使用不同的損失函數，並將損失降到最低
- 損失函數⼤致可分為：分類問題的損失函數和回歸問題的損失函數

- 損失函數為什麼是最小化?
  - 期望：希望模型預測出來的東⻄可以跟實際的值⼀樣
  - 預測出來的東⻄基本上跟實際值都會有落差
    - 在回歸問題稱為「殘差(residual)」
    - 在分類問題稱為「錯誤率(errorrate)」

  - 損失函數中的損失就是「實際值和預測值的落差」
  - 在以下的說明中，y 表⽰實際值，ŷ 表⽰預測值



- 損失函數的分類介紹

  - **mean_squared_error**

    - 就是最⼩平⽅法(Least Square) 的⽬標函數-- 預測值與實際值的差距之平方值。
    - 另外還有其他變形的函數, 如 mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error.

    $$
    \sum (\hat y - y)^2/N
    $$

      - 使⽤時機為處理 y 為數值資料的迴歸問題

      - Keras 上的調⽤⽅式：

        ```python
        from keras import losses
        model.compile(loss= 'mean_squared_error', optimizer='sgd')
        # 其中，包含 y_true， y_pred 的傳遞，函數式表達如下：
        keras.losses.mean_squared_error(y_true, y_ped)
        ```

        

  - **Cross Entropy**

    - 當預測值與實際值愈相近，損失函數就愈⼩，反之差距很⼤，就會更影響損失函數的值要⽤ Cross Entropy 取代 MSE，因為，在梯度下時，Cross Entropy 計算速度較快。

    - 使⽤時機：

      - 整數⽬標：Sparse categorical_crossentropy
      - 分類⽬標：categorical_crossentropy
      - ⼆分類⽬標：binary_crossentropy

    - Keras 上的調⽤⽅式：

      ```python
      from keras import losses
      model.compile(loss= 'categorical_crossentropy', optimizer='sgd’)
      # 其中, 包含 y_true， y_pred 的傳遞, 函數是表達如下：
      keras.losses.categorical_crossentropy(y_true, y_pred)
      ```

  - **Hinge Error (hinge)**

    - 是⼀種單邊誤差，不考慮負值，同樣也有多種變形，squared_hinge, categorical_hinge

    - 適⽤於『⽀援向量機』(SVM)的最⼤間隔分類法(maximum-margin classification)

    - Keras 上的調⽤⽅式：

      ```python
      from keras import losses
      model.compile(loss= 'hinge', optimizer='sgd’)
      # 其中，包含 y_true，y_pred 的傳遞, 函數是表達如下:
      keras.losses.hinge(y_true, y_pred) 
      ```

    

  - **⾃定義損失函數**

    - 根據問題的實際情況，定制合理的損失函數

    - 舉例：預測果汁⽇銷量問題，如果預測銷量⼤於實際銷量則會損失成本；如果預測銷量⼩於實際銷量則會損失利潤。

      - 考慮重點：製造⼀盒果汁的成本和銷售⼀盒果汁的利潤不是等價的

      - 需要使⽤符合該問題的⾃定義損失函數⾃定義損失函數為：
        $$
        loss = \sum nf(y_, y)
        $$

- 損失函數表⽰若預測結果 y ⼩於標準答案 y_ ，損失函數為利潤乘以預測結果 y 與標準答案之差

- 若預測結果 y ⼤於標準答案 y_，損失函數為成本乘以預測結果 y 與標準答案之差⽤

- Tensorflow 函數表⽰為：
      

  ```python
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y-y_), PROFIT*(y_-y)))
  ```

- 參考資料

  1. [TensorFlow筆記-06-神經網絡優化-損失函數，自定義損失函數，交叉熵](https://blog.csdn.net/qq_40147863/article/details/82015360)
  2. [Hinge_loss](https://en.wikipedia.org/wiki/Hinge_loss)
  3. [使用損失函數](https://keras.io/losses/)



## 激活函數

- 激活函數定義了每個節點（神經元）的輸出和輸入關係的函數為神經元提供規模化非線性化能⼒，讓神經網絡具備強⼤的擬合能⼒

  ![](https://lh3.googleusercontent.com/y0MVNA4d2-mbKs7-sudJaUprzfYPbaETic2CF3NOAAdKrPhmxwUpLLN8Xu489mdR11Ah6XgTCm1z3xJODrZgvhiNbR-8JvQnebV6Rv8swA4l1bq3wnYnNZtH6Yveq-dcTjdyH7dCNMpqefgV83ZL5Ucj8lgJOtN1Hdk3u7qtHdcX7q6rSBC17pfKuvD1YyPz7BRbAPrBcFCyHGvMPyM1U5hl_-iRyJmvjugnIS7Aq02-oNGgCjd1OpXjRwNjnVHlH491kYPin6yJEswfXj_4OZocoFhaKheyTsbYFGOsD9No_NYK1EI3aIUx_A49IOUEtGvwabySVSwALw9DyPpaH_tPc4GqPugz2kIyKGe4Hp8SMDIqr5CQjKeNtnlsMtbIcGnJ2orBxpQ1qs7R0vLJn4iqJAkrCvgAHz0nrMUvG_jphpKl5IEsWQbtK1u0wEwZesLzb5ZjjyFp0N2-oH3o3gFAKW4eLIp0GNtyouWulrUqGNp3tsrXEel4bXKVMXZK45BRBNL5ley5egWblLBZuoBfsG0BQfY71dzTB9F-_hOxotIlKjTWBYVsUTcro8ym6cZ3IQfHoitje0T2Bl8VQBL7n3REhs4wcgZyW_r_TQExah7CZKiXI69Fjn_ExETqz7Yv6d47PVTvXMDzNOlN-vqgXxSA-zQo7Ryl89ETt2jfuEo7lIIEer7xD3EoNot7Se5oYirLcmMHd1CJVmAbKPAN=w951-h324-no)

- 輸出值的範圍

  - 當激活函數輸出值是有限的時候，基於梯度的優化⽅法會更加穩定，因為特徵的表⽰受有限權值的影響更顯著
  - 當激活函數的輸出是無限的時候，模型的訓練會更加⾼效

- 激活函數的作用

  - 深度學習的基本原理是基於⼈⼯神經網絡，信號從⼀個神經元進入，經過非線性的 activation function
  - 如此循環往復，直到輸出層。正是由於這些非線性函數的反复疊加，才使得神經網絡有⾜夠的 capacity 來抓取複雜的 pattern
  - 激活函數的最⼤作⽤就是非線性化，如果不⽤激活函數的話，無論神經網絡有多少層，輸出都是輸入的線性組合

  - 激活函數的另⼀個重要特徵是：它應該是可以區分，以便在網絡中向後推進以計算相對於權重的誤差（丟失）梯度時執⾏反向優化策略，然後相應地使⽤梯度下降或任何其他優化技術優化權重以減少誤差

- 常用激活函數介紹

  - Threshold Function
    $$
    \begin{cases}1, \quad if \quad x >= 0 \\0, \quad if \quad x <  0\end{cases}
    $$

- **Sigmoid**

  - 特點是會把輸出限定在 0~1 之間，在 x<0 ，輸出就是 0，在 x>0，輸出就是 1，這樣使得數據在傳遞過程中不容易發散
    - 兩個主要缺點
      1. Sigmoid 容易過飽和，丟失梯度。這樣在反向傳播時，很容易出現梯度消失的情況，導致訓練無法完整
      2. Sigmoid 的輸出均值不是 0

  - Sigmoid 將⼀個 real value 映射到（0,1）的區間，⽤來做⼆分類。
    - 用於二分類的輸出層

  $$
    f(z) = \frac{1}{1+exp(-z)}
  $$

- **Softmax**

  - Softmax 把⼀個 k 維的 real value 向量（a1,a2,a3,a4….）映射成⼀個（b1,b2,b3,b4….）其中 bi 是⼀個 0～1 的常數，輸出神經元之和為 1.0，所以可以拿來做多分類的機率預測
    - 為什麼要取指數
      1. 第⼀個原因是要模擬 max 的⾏為，所以要讓⼤的更⼤。
      2. 第⼆個原因是需要⼀個可導的函數
    - ⼆分類問題時 sigmoid 和 softmax 是⼀樣的，求的都是 cross entropy loss

  $$
    \sigma (z)_j = \frac {e^{zj}}{\Sigma^K_{k=1}e^{ek}}
  $$

- **Tanh(Hyperbolic Tangent)**

  - 也稱為雙切正切函數，取值範圍為 [-1,1]。
    - 在特徵相差明顯時的效果會很好，在循環過程中會不斷擴⼤特徵效果
    - 將輸入值壓縮到 -1~1 的範圍，因此它是 0 均值的，解決了 Sigmoid 函數的非 zero-centered 問題，但是它也存在梯度消失和冪運算的問題。
    - 幾乎所有場合都可以使用

  $$
    tanh(x)=2sigmoid(2x)-1
  $$

- **ReLU**

  - 修正線性單元（Rectified linear unit，ReLU）

    - 在 x>0 時導數恆為1
      - 對於 x<0，其梯度恆為 0，這時候它也會出現飽和的現象，甚⾄使神經元直接無效，從⽽其權重無法得到更新（在這種情況下通常稱為 dying ReLU）
      - Leaky ReLU 和 PReLU 的提出正是為了解決這⼀問題
      - 使用時機：二分類的問題選擇sigmoid，其餘預設選ReLU

    $$
      f(x) = max(0, x)
    $$

- **ELU**

  - ELU 函數是針對 ReLU 函數的⼀個改進型，相比於 ReLU 函數，在輸入為負數的情況下，是有⼀定的輸出的
    - 這樣可以消除 ReLU 死掉的問題
    - 還是有梯度飽和和指數運算的問題

  $$
    f(x) =\begin{cases}x\quad \quad \quad, x > 0 \\a(e^x-1),\quad x \leq  0\end{cases}
  $$

- **PReLU**

  - 參數化修正線性單元（Parameteric Rectified Linear Unit，PReLU）屬於 ReLU 修正類激活函數的⼀員。

- **Leaky ReLU**

  - 當 α=0.1 時，我們叫 PReLU 為Leaky ReLU，算是 PReLU 的⼀種特殊情況

  > PReLU 以及 Leaky ReLU 有⼀些共同點，即爲負值輸入添加了⼀個線性項。

- **Maxout**

  - 是深度學習網絡中的⼀層網絡，就像池化層、卷積層⼀樣，可以看成是網絡的激活函數層
    - 神經元的激活函數是取得所有這些「函數層」中的最⼤值
    - 擬合能⼒是非常強的，優點是計算簡單，不會過飽和，同時⼜沒有 ReLU 的缺點
    - 缺點是過程參數相當於多了⼀倍

  $$
    f(x) = max(wT1x+b1, wT2x+b2)
  $$

![](https://lh3.googleusercontent.com/cZQ4cAyBhFGdA8uLvz2L0rLU26hz9J2IBpczsHBBRs4MZRETk6-IeDOlE1M_5L2cb1u17TE3vzYti1DGkIUGyW5tNurRh3ag23vHpzVQ88UPiJbQfCtJLsKLEOXORPhpd_rs_qMtlyLrf9jRyQEf5AvR2f4RQzXfTW_j_IfJMCG9Kncf4wNkyZj7amI0N0K37DVdrNToswaPVW7MgR7lRYZS-1zlKOlbrtxBSprfic7EvBC6lzD5O8nCWDfcwBVJzCoZh6eNRVT4Ps37IOLyfUtGZBQv9fVS-XuKGzO6vdJdzPHMPLZLZDMKQDr-ZVoEQ8qQvSSSQbupyen2_SG2ynRaXmZxBne38uI0qgbV3BUU1VM5-LlJvbRvcRfJU-4ys3YLAEwihkqmfS0aPHGqzP-E9zax96TRxa-Kg6Nd7PQ4ofFgOcbU9EZL5WoWGcrKFGZnVHG0tVJ8bQHgoPYCQD0RIIf3V-OnQ5gYfmhTFEbqS6MostV114_8wByuu6grIJKTydpp2PLGkndRpFvJd96ZOqtpi4HnqYC48zpsAbin0qCA7LgFG4XWPPhCq1YEP-TjaXMJoMx5Yfx4PqcsH5ni0MD1eoC8hOIhKdKvCVnLtSy04NrDAGdU0IuGEiHVKWKE-wbvPQ4SC6PslaCssRs45wIKi3Jzq0ClTatP2_tlRVQdL1_GJu4OPFrwq4LlD7cQQ7SMGhwdX8thDT6fHm7Nv6pULt19W9X6xTGoNXOnHDMnR2MEXOY=w1000-h428-no)

- 如何選擇正確的激活函數

  - 根據各個函數的優缺點來配置

    - 如果使⽤ ReLU，要⼩⼼設置 learning rate，注意不要讓網絡出現很多「dead」 神經元，如果不好解決，可以試試 Leaky ReLU、PReLU 或者Maxout

  - 根據問題的性質

    - ⽤於分類器時，Sigmoid 函數及其組合通常效果更好
    - 由於梯度消失問題，有時要避免使⽤ sigmoid 和 tanh 函數。ReLU 函數是⼀個通⽤的激活函數，⽬前在⼤多數情況下使⽤
    - 如果神經網絡中出現死神經元，那麼 PReLU 函數就是最好的選擇
    - ReLU 函數建議只能在隱藏層中使⽤

  - 考慮 DNN 損失函數和激活函數

    - 如果使⽤ sigmoid 激活函數，則交叉熵損失函數⼀般肯定比均⽅差損失函數好；
    - 如果是 DNN ⽤於分類，則⼀般在輸出層使⽤ softmax 激活函數
    - ReLU 激活函數對梯度消失問題有⼀定程度的解決，尤其是在CNN模型中。

  - 梯度消失 Vanishing gradient problem

    - 原因：前⾯的層比後⾯的層梯度變化更⼩，故變化更慢
    - 結果：Output 變化慢 -> Gradient⼩ -> 學得慢
    - Sigmoid，Tanh 都有這樣特性
    - 不適合⽤在 Layers 多的DNN 架構

    ![](https://lh3.googleusercontent.com/x3CN8LTxGqX31Q4PX1ArsP4KTFLs4YHnV4rntTHfzJi96IR1QXt2n4LJz3pxU-Tr4XjZmFTppnQ_47VVA-uIeYdm3G7fjOh33YE2tU_BMb9uGhCMYZIXprGXy99MrsurSwlJyMlsgJzkKdefTyJeCereiNOjG30wT-AzFQXcvGnOOZDOKjJr-4gRvK_76kxUn2F8GDwS95pXtAQ97SEKKtMMHiVulST0k8MdZI1zy6dUqiisFvM1SgvgPbP0r-7xeZf5vs1jOgYg4sXrrJGLrgXla28jJ6EGRDby3NLyjesNm6nAzQoUiPP4hnp4zLvwtdyVxr-VmKzjGuk0gs62GzmUbqIdUXDtiA83mvxSVRkuzJrwVCZTHYtmpTW3woctaNs-7_UwsJ-ITvzu16TY3HfHBNHJY2uivi3RoEYHDroRe0tsrxonAMH5dtFYELcxaUVNxYCiPEyjg4utMR2NnTEAxmE69hDcB-W87Ppduc5eCBW3DU2aWfvg6J1WmOLwDOHg3NjU188XfVja5Vlv-8KRCqQYzOkjMNNoUTUFdlzhQIvCa4yAImSxqG-oSln1Wn98OnjRd6dUoFGm5rEMsmfi6y9UKwTO0AvVCYVa4xoRYY5b1I47TRs-XzfxNuL2snGrBpeG0o42oqpMlwv1FVc1H0rbFNyh8eb6ejeObUkpZtQBU0qXX1Q8tF1DgYb_jffmgz7YiHZK6xjZ-7G0cGYL=w565-h422-no)

- 前述流程 / python程式對照

  - 激活函數可以通過設置單獨的激活層實現，也可以在構造層對象時通過傳遞 activation 參數實現：

    ```python
    from keras.layers import Activation,Dense
    model.add(Dense(64,activation=‘tanh’))
    ```

  - 考慮不同Backend support，通過傳遞⼀個元素運算的Theano / TensorFlow / CNTK函數來作為激活函數：

    ```python
    from keras import backend as k
    model.add(Dense(64,activation=k.tanh))
    model.add( Activation(k.tanh))
    ```

- 參考資料

  1. [神經網絡常用激活函數總結](https://zhuanlan.zhihu.com/p/39673127)
  2. [Reference 激活函數的圖示及其一階導數](https://dashee87.github.io/data science/deep learning/visualising-activation-functions-in-neural-networks/)
  3. [Maxout Network](https://arxiv.org/pdf/1302.4389v4.pdf)
  4. [Activation function](https://en.wikipedia.org/wiki/Activation_function)



## Learning rate

![](https://lh3.googleusercontent.com/P2ufOXOvaucTCVxEprN6gsJw6ngC_BYQ8u0CwVGybjSX1_Y5jneX2iMz0UvVK3XjL5dE6O7TZM7wSK_xMwXfHQqdO9g1ZUkSclTis6Ig1KXiWj5tPxLgdN_aKyxoe6Fm-W7pSsGmcLtXMwrUEl3oBGr3VluJwrpP3m5nBPowY1PmUX1o-ruYWrz3WsWBJkJbmWzx9e-Bd0VNL5PS0n6D6ahyv85dR4fjZRAmjZG8eKENEQ5hBRVrt9SQN2PrbqYliwRpvViYyYy4mT9zbzqrsKstQ33jxbbButqxCWJQPZrLqY2ZIHpXl1rP2me5wZnBkk4TDOhfkbDEHX6XTO-IugoBGc4e3-PYAgcM4AV12PXkhZ2lLlxxdkCoLcaev3vNiiIoEnm10XMSKGQMxDq6mVafP2NCFrSj5g9OnFx4VyN4uDQ7SWoUnCTd11MvTrDk88sdkLqoGBUg0Puaye6lfrRPq1qbbC_psvhYy226FTThYVkiOy5YH7YKa4ebiXUBUzuokOFhiesDxp1esCPZ2o2m5rWW1za5RRh1OawWqNtpE8PZCh5ryTf5sNH5Bvht_81uDV_Ehwctza_MrpwoT5wT_wSLoOEaGEZ75RaRONUPqISCc_z80Dvyv4iUiKb3TYyBX6U4RSJI8D-xrfQZihSSx2BRQvQwjRGrZLXLZJQvUiwMEhzSOeME3WTjPwAaxBJh8UVyGynL9PZfKKPan64G=w1027-h427-no)

- η ⼜稱學習率，是⼀個挪動步長的基數，df(x)/dx是導函數，當離得遠的時候導數⼤，移動的就快，當接近極值時，導數非常⼩，移動的就非常⼩，防⽌跨過極值點

- Learning rate 選擇，實際上取值取決於數據樣本，如果損失函數在變⼩，說明取值有效，否則要增⼤ Learning rate

- 機器學習算法當中，優化算法的功能，是通過改善訓練⽅式，來最⼩化(或最⼤化)損失函數，當中最常用的優化算法就是梯度下降

  - 通過尋找最⼩值，控制⽅差，更新模型參數，最終使模型收斂

    - 梯度下降法的過程

      - ⾸先需要設定⼀個初始參數值，通常情況下將初值設為零(w=0)，接下來需要計算成本函數 cost
      - 然後計算函數的導數-某個點處的斜率值，並設定學習效率參數(lr)的值。
      - 重複執⾏上述過程，直到參數值收斂，這樣我們就能獲得函數的最優解

      ![](https://lh3.googleusercontent.com/CC-uYdn_fZGk1DvGv8T5nWD7RrZH4bUSlGOmoVkwDhp9Jc3rLzgJAplUvK3KtutQp-0fgi1QGd66g69nIJLTnRJvB_uC3ukERZU6cc-CP867UEQF1WxVOn7ueeuDfMHRWEDDA-0G-J_iH52FiH-_UB0Si6kYIXBcjPev-Bz-RQ3WEZid1CNGqmrfYbpC5jJnTjVhl7ObTUyMrnZ_wvVhFd1qHL8SOr_CmkOEs729q_JH5xYBZZCNUzicRZXfSOFp7dxxS6iTtfv3l54Qz1VvYpRdshhVqUdSHIjsKU79O7JsgvlvyL42_O-prcjfhvOnhRGe7dp0SN_jSu4Uq_lE4rc2W1CJcMTijyBirEV06qVGSoZU_ceKkPQWxU5OPZW7pbNmSdV7jFucDxmeeV3odIZn13MQgxHA-rAqJuZ-7Yonbfn54dse2K_9hXJp5_oLLDL3sJFTe-8tiMxBxFLNPv6gEgjs---3H26NzWjO45aNudxjMZcNY6GF_2rZSX8EexUWGSD6R8_LZQKv2oFzfNEf-dZhocYiQJ9D2XYz8W1RBLdzp5UBnkcCzC_u0OafQX4RBjX9lif0INkuJoLRzdVBssXAOF0psy3VaNikl8YjSNbPkfwS_8UKQ4-YXFoyD8cHu_Om0iWLA4ldqZHO0jMeEldrWIn0D0ygwCb8Os642Mm9VUmBUCMLemBri0wdjndESfcAunD5nll04dscG3oM=w757-h323-no)

  - ⽬的：沿著⽬標函數梯度下降的⽅向搜索極⼩值（也可以沿著梯度上升的⽅向搜索極⼤值）

  - 要計算 Gradient Descent，考慮

    - Loss = 實際 ydata – 預測 ydata 

      ​     = w* 實際 xdata – w*預測 xdata (bias 為 init value，被消除)

    - Gradient =  ▽f($\theta$) (Gradient = $\partial$L/$\partial$w)

    - 調整後的權重 = 原權重 – $\eta$(Learning rate) * Gradient
      $$
      w ←w- η  ∂L/∂w
      $$
      ![](https://lh3.googleusercontent.com/LB71izAdksVkRZbpJbVVFJgEgIqct32yMOt507joZoWxT_FZuR-eLDsfdpI2v5Ef8Xxs4xPqIRCh5xs9p7zRz60vF_rDqNcJgOSDIpLG84ywRWBwGCWoS34eLDN7McFlKqiiwgtR48jAjprHsVEnry5Qwxc23ybeEabEM71m7MSSmkBPzJe04jHF74N2rygeNx9gI2tmKe4CkJ7_OmGHp5kBSyFVOMIWw9ohist9hmv-IQ2d_5OWrIJcee77Ag_hW9f8xi86iQMmF7qPHP0DnUBQle3lfdZUfYCCRyCknsKC-CKxW90Z4Sj9Mm3TRxWOg1XlolifQIE_Ht7p_zUuRhkz01qR6J8S3ECDaKdrsKP_4zOKF0t93bPUGdMirN5R3uwNEdRwcpB5ztjjHV-Kh5yBzID-KWBLWRFLZ_WFcKlnPhor0CkRliQMfHitJUDz-6eDl56Vpqri5Be-oaOB_bWP-bjSN94G1UbC2fbBsNRM47qP1SZHIpNIsTmT6-LlciRrrTsmgxzBDGYAoszRezlPEsTu9XqyeR4g4bSfDFUldIGxSHyJaYeQZ54jXbKwC-TAV3-UsS1pjDUqvfr8kD3LXuZjxvu9Tp8J8tVVjyBF9Wf3iHluuerCRk14255h_mf9Tg9aHmdj8M3WKhtGTEpihvk9rxkb17gh5EZBAeITbzlmkHffnjGLP3lvthLCTGUKgoBs9ATD5agI_hvjZ0n7=w845-h376-no)

- 梯度下降法的缺點包括：

  - 靠近極⼩值時速度減慢

  - 直線搜索可能會產⽣⼀些問題

  - 可能會「之字型」地下降

- 避免局部最佳解(Avoid local minima)

  - Gradient descent never guarantee global minima
  - Different initial point will be caused reach different minima, so different results

- 在訓練神經網絡的時候，通常在訓練剛開始的時候使⽤較⼤的 learning rate，隨著訓練的進⾏，我們會慢慢的減⼩ learning rate ，具體就是每次迭代的時候減少學習率的⼤⼩，更新公式

  - 學習率較⼩時，收斂到極值的速度較慢。
  - 學習率較⼤時，容易在搜索過程中發⽣震盪
  - 相關參數:
    - decayed_learning_rate 哀減後的學習率
    - learning_rate 初始學習率
    - decay_rate 哀減率
    - global_step 當前的 step
    - decay_steps 哀減週期

- η ⼜稱學習率，是⼀個挪動步長的基數，df(x)/dx是導函數，當離得遠的時候導數⼤，移動的就快，當接近極值時，導數非常⼩，移動的就非常⼩，防⽌跨過極值點

- 學習率對梯度下降的影響

  - 學習率定義了每次疊代中應該更改的參數量。換句話說，它控制我們應該收斂到最低的速度或速度。

  - ⼩學習率可以使疊代收斂，⼤學習率可能超過最⼩值

    ![](https://lh3.googleusercontent.com/yzU1_yVv-CP0T3gGicaPdMjSO4Ppo9wOpgpWiZ7I_81vlGlzqrxQc9eW8EKTaUYKxEoz7KgF1ey1grQQSOmUbNe0G__lUO7BFw9Wwcd3Kdyao8LPkYw9Dp7a7VsNqpYrb-U6-DJ_EinUd9CQCl1ld6CIJI2n6RM_kJiwJ6xa0sY8DoOCEm5G_ABDiFyTyZOth-0V0m-2Ynnp4X2_al0Axku5X_CCTokrS3czLz3z3iHjVuaR_PUdaI1uTDwRLX6Uf39fn76akqB3-MjvzMnOdWcn0nqHQGcrywjbJeXu1U0My1Ltgf-51aEhzFhy2yM3NhYxtjgRKP-T3AUVTLPMWfmNYi7uC5dC2s3fEKNy_rtbNsfSXJ5mO1S_gxzKGn4awbPUEaR2lcvixJn9WS2X8neCa7AcuuG3Z7pUTRGOMZlgdHsnhTuGAZSWG6UVqUB2RVeAJDvKNj3x5qy5qf3WDk0p3sbwfzEfZVtmFqHsTl2E94X5qdU_AbGmmUh7vKS67ND5NcoiqkXlSULGxtv5MvlQhNlpMlbMisgmyTBjsxUK8JL4BH7ocAvjSv40O9wp7VdxY8kQmHvCbBgzdKd-kFtnkxg5XmYEQ1stCSr_9loJxPTO8WBAgfQXQGHv6Q-d6P1Mxf_eGVNCF2fUwxo-BTHEDbVgfJin6AOuTVPXAnzakc4MZ6xbcvtCtc2uWUdHtrg2d328wmnlH2WRFKpg-9qY=w1031-h276-no)

- ⾃動更新 Learning rate - 衰減因⼦ decay

  - 算法參數的初始值選擇。初始值不同，獲得的最⼩值也有可能不同，因此梯度下降求得的只是局部最⼩值；當然如果損失函數是凸函數則⼀定是最優解。

  - 學習率衰減公式

    - lr_i = lr_start * 1.0 / (1.0 + decay * i)

    - 其中 lr_i 為第⼀迭代 i 時的學習率，lr_start 為初始值，decay 為⼀個介於[0.0, 1.0]的⼩數。從公式上可看出：

      > decay 越⼩，學習率衰減地越慢，當 decay = 0時，學習率保持不變
      >
      > decay 越⼤，學習率衰減地越快，當 decay = 1時，學習率衰減最快

  - 使⽤ momentum 是梯度下降法中⼀種常⽤的加速技術。Gradient Descent 的實現：SGD, 對於⼀般的SGD，其表達式為

    - 隨著 iteration 改變 Learning
      - 衰減越⼤，學習率衰減地越快。 衰減確實能夠對震盪起到減緩的作⽤

  - momentum

    - 如果上⼀次的 momentum 與這⼀次的負梯度⽅向是相同的，那這次下降的幅度就會加⼤，所以這樣做能夠達到加速收斂的過程
    - 如果上⼀次的 momentum 與這⼀次的負梯度⽅向是相反的，那這次下降的幅度就會縮減，所以這樣做能夠達到減速收斂的過程

  $$
  X ← x -a * dx
  $$

  - ⽽帶 momentum 項的 SGD 則寫⽣如下形式：

  $$
  v = \beta * v -a*dx 
  $$

  $$
  X ← x + v
  $$

  

  - 其中 ß 即 momentum 係數，通俗的理解上⾯式⼦就是，如果上⼀次的 momentum（即ß ）與這⼀次的負梯度⽅向是相同的，那這次下降的幅度就會加⼤，所以這樣做能夠達到加速收斂的過程

![](https://lh3.googleusercontent.com/na5BUBlKpeVl5D7ka3yaYVZZnPlyD87vReJ0sag9DlARk0pBqPmwzcbM6XNf5njtnpYNzDls59s64kHCrbTaqmINw4K-N6DTtLFZeqP6lLFySMOFGKaAsnAJ3zn2czG504P9yqhPJvYKmUx0Z8FyDkK9tPywCq3Gnd97Vj_SubPwH22C1T_qdmcxJiY_7lb-KsHFFa86v2UsRbdRn3RKVBl588dHBrkWtQ-9SiPkNAH3IbatBTewW0CxZ2SvCuuPWOEqfD42fd8LNab9nOf--4vPqdpmyGU2JTHU5aJZU0wK9yHsE3y4mdBosrpmYXs4k8o5pVAA6YN6XmEKOPPU6P0PhlGB_B_x71-ZeNtJFJjLM_aLYK3gSEH2FhRjN9P13upQ3HBFd-I0D2p7ubVAdvOHNsKXB7r6P1QpaoeXuGTsCUBt5PjBCacwWOPolCUJRXk8QtNcX_hQUuMgsJiGdVV7Sr9qd6YL8UrRkmikbNtFCTtZi-22B9h-0wa6J6tzSmkYmFWPoXN8Tnmxqn6akj6PQ7HgWMsZYSMESfS439t5rGTuQgsVjfIpe_oUYEB5pPR9lxKHMFzqBKlcxjGCPtkXevIvVDHGKdUJuCToy7kK-mgrdKok7ihNAB1dEBMfS9UROA1kWJ4cVHZ-o0kgC3DY1b2MufUD08_sES0g3zfs6Zb_NX8W0opxR2mFGqxHeqjG2iuk5Nsz3U6ieuxs5goM=w924-h411-no)



- 要使⽤梯度下降法找到⼀個函數的局部極⼩值，必須向函數上當前點對應梯度（或者是近似梯度）的反⽅向的規定步長距離點進⾏疊代搜索。
  - avoid local minima
    - Item-1：在訓練神經網絡的時候，通常在訓練剛開始的時候使⽤較⼤的learning rate，隨著訓練的進⾏，我們會慢慢的減⼩ learning rate
      - 學習率較⼩時，收斂到極值的速度較慢。
      - 學習率較⼤時，容易在搜索過程中發⽣震盪
    - Item-2：隨著 iteration 改變 Learning
      - 衰減越⼤，學習率衰減地越快。 衰減確實能夠對震盪起到減緩的作⽤
    - Item-3：momentum
      - 如果上⼀次的 momentum 與這⼀次的負梯度⽅向是相同的，那這次下降的幅度就會加⼤，所以這樣做能夠達到加速收斂的過程
      - 如果上⼀次的 momentum 與這⼀次的負梯度⽅向是相反的，那這次下降的幅度就會縮減，所以這樣做能夠達到減速收斂的過程

- 參考資料
  - [知乎 - Tensorflow中learning rate decay](https://zhuanlan.zhihu.com/p/32923584)
  - [機器/深度學習-基礎數學篇(一):純量、向量、矩陣、矩陣運算、逆矩陣、矩陣轉置介紹](https://medium.com/@chih.sheng.huang821/機器學習-基礎數學篇-一-1c8337179ad6?source=post_page---------------------------)
  - [機器/深度學習-基礎數學(二):梯度下降法(gradient descent)](https://medium.com/@chih.sheng.huang821/機器學習-基礎數學-二-梯度下降法-gradient-descent-406e1fd001f?source=post_page---------------------------)
  - [機器/深度學習-基礎數學(三):梯度最佳解相關算法(gradient descent optimization algorithms)](https://medium.com/@chih.sheng.huang821/機器學習-基礎數學-三-梯度最佳解相關算法-gradient-descent-optimization-algorithms-b61ed1478bd7?source=post_page---------------------------)
  - [五步解析机器学习难点—梯度下降](https://zhuanlan.zhihu.com/p/27297638)
  - [機器學習-梯度下降法](https://www.jianshu.com/p/31740cd2ca48)
  - [gradient descent using python and numpy](https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy)
  - [梯度下降算法的參數更新公式](https://blog.csdn.net/hrkxhll/article/details/80395033)



## BackPropagation

- 反向傳播（BP：Backpropagation）是「誤差反向傳播」的簡稱，是⼀種與最優化⽅法（如梯度下降法）結合使⽤的該⽅法對網路中所有權重計算損失函數的梯度。這個梯度會反饋給最優化⽅法，⽤來更新權值以最⼩化損失函數。
- 反向傳播要求有對每個輸入值想得到的已知輸出，來計算損失函數梯度。因此，它通常被認為是⼀種監督式學習⽅法，可以對每層疊代計算梯度。反向傳播要求⼈⼯神經元（或「節點」）的啟動函數可微
- BP 神經網路是⼀種按照逆向傳播算法訓練的多層前饋神經網路
  - 優點：具有任意複雜的模式分類能⼒和優良的多維函數映射能⼒，解決了簡單感知器
    不能解決的異或或者⼀些其他的問題。
    - 從結構上講，BP 神經網路具有輸入層、隱含層和輸出層。
    - 從本質上講，BP 算法就是以網路誤差平⽅⽬標函數、採⽤梯度下降法來計算⽬標函數的最⼩值。

  - 缺點：
    - 學習速度慢，即使是⼀個簡單的過程，也需要幾百次甚⾄上千次的學習才能收斂。
    - 容易陷入局部極⼩值。
    - 網路層數、神經元個數的選擇沒有相應的理論指導。
    - 網路推廣能⼒有限。

  - 應⽤：
    - 函數逼近。
    - 模式識別。
    - 分類。
    - 數據壓縮
  - 流程
    - 第1階段：解函數微分
    - 每次疊代中的傳播環節包含兩步：
      - （前向傳播階段）將訓練輸入送入網路以獲得啟動響應；
      - （反向傳播階段）將啟動響應同訓練輸入對應的⽬標輸出求差，從⽽獲得輸出層和隱藏層的響應誤差。
    - 第2階段：權重更新
      - Follow Gradient Descent
      - 第 1 和第 2 階段可以反覆循環疊代，直到網路對輸入的響應達到滿意的預定的⽬標範圍為⽌。

![](https://lh3.googleusercontent.com/X7h72dPGnJkyEVh2FcaGVXPpWroXzN2c83A0FICa_H6cfZ0zo87rIQXzkRhYOeClI2Hltfff3N-FDh3tdwMmyY2dMkUQ_xdGkvgOI7OGqMo8GWpHLq8MS32NjpL-TfwV3YoBpSc4P2F7bFe1hzN3SlHRPA3eFrrbjqbcmYQWqRKF-kf1U2sjUcnkTqwezqpslaS_dvJUUSl2O05t4gHMXruC1leS5y4cLc4gyHpWvJW7aFWG-J-GZaSSyoGlT18XtzI_GHIphbUWZaDtMrVKw8QGe5JPKvatzf2EpH-JkIpDM_mcAaqfmZqvJ7OnkudJ0PgKPgk4JYdcBeAp8nLDcOC9_EyXKDcoP5Rr5VfMU650H2ANm3n-1JetXAO7NxE7_uAIei__GnbHhwnhr5jY3uuo6lVdtI8e49fiG0x9WtOiNqFcEx-U7e8qyOmhQGxQJhrjTWlM9X7jANtk_gVL7wMAl8sT12tjPILiniOdO-RA5a165dtUa7TQMaJOdhIYM4V25ChE-Vd8bplunqXlKGUkQA6BGkjljZC3q-ERPmgEroq_O0MVc0pgpWKM48_xYwchw6jOUAleSscs1Dm4yDT2F88klIZoHljIrRoIha5kqWqMcIb7rZMhJoEQzTjHY7oPoTD6r-1kOw0DGpX3_fL4JKPnTJD7_C2mJlSegj_FKHFmO8VXZ9wsSU3qxmsx2ziNMBwCWCfvo2a-pGoaItde=w660-h450-no)

- 參考資料

1. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
2. [BP神經網絡的原理及Python實現](https://blog.csdn.net/conggova/article/details/77799464)
3. [完整的結構化代碼見於](https://github.com/conggova/SimpleBPNetwork.git)
4. [深度學習-BP神經網絡(python3代碼實現)](https://blog.csdn.net/weixin_41090915/article/details/79521161)



## Optimizers

- 什麼是Optimizer

  - 機器學習算法當中，⼤部分算法的本質就是建立優化模型，通過最優化⽅法對⽬標函數進⾏優化從⽽訓練出最好的模型
  - 優化算法的功能，是通過改善訓練⽅式，來最⼩化(或最⼤化)損失函數 E(x)
  - 優化策略和算法，是⽤來更新和計算影響模型訓練和模型輸出的網絡參數，使其逼近或達到最優值

- 常用的優化算法

  - Gradient Descent

    - 最常⽤的優化算法是梯度下降

      - 這種算法使⽤各參數的梯度值來最⼩化或最⼤化損失函數E(x)。

    - 通過尋找最⼩值，控制⽅差，更新模型參數，最終使模型收斂

    - 複習⼀下，前面提到的 Gradient Descent

      > wi+1 = wi - di·ηi, i=0,1,…
      >
      > - 參數 η 是學習率。這個參數既可以設置為固定值，也可以⽤⼀維優化⽅法沿著訓練的⽅向逐步更新計算
      > - 參數的更新分為兩步：第⼀步計算梯度下降的⽅向，第⼆步計算合適的學習

  - Momentum

    > ⼀顆球從⼭上滾下來，在下坡的時候速度越來越快，遇到上坡，⽅向改變，速度下降

    $$
    V_t ← \beta V_{t-1}-\eta \frac{\partial L}{\partial w}
    $$

    - $V_t$:方向速度，會跟上一次的更新相關
    - 如果上⼀次的梯度跟這次同⽅向的話，|Vt|(速度)會越來越⼤(代表梯度增強)，W參數的更新梯度便會越來越快，
    - 如果⽅向不同，|Vt|便會比上次更⼩(梯度減弱)，W參數的更新梯度便會變⼩

    $$
    w ← w + V_t
    $$

    - 加入 $V_t$ 這⼀項，可以使得梯度⽅向不變的維度上速度變快，梯度⽅向有所改變的維度上的更新速度變慢，這樣就可以加快收斂並減⼩震盪

  - SGD

    - SGD-隨機梯度下降法(stochastic gradient decent)
    - 找出參數的梯度(利⽤微分的⽅法)，往梯度的⽅向去更新參數(weight)

    $$
    w ← w - \eta \frac{\partial L}{\partial w}
    $$

    - w 為權重(weight)參數，
    - L 為損失函數(loss function)， 
    - η 是學習率(learning rate)， 
    - ∂L/∂W 是損失函數對參數的梯度(微分)
    - 優點：SGD 每次更新時對每個樣本進⾏梯度更新， 對於很⼤的數據集來說，可能會有相似的樣本，⽽ SGD ⼀次只進⾏⼀次更新，就沒有冗餘，⽽且比較快
    - 缺點： 但是 SGD 因為更新比較頻繁，會造成 cost function 有嚴重的震盪。

    ```python
    keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    ```

    - lr：學習率
    - Momentum 動量：⽤於加速 SGD 在相關⽅向上前進，並抑制震盪。
    - Decay(衰變)： 每次參數更新後學習率衰減值。
    - nesterov：布爾值。是否使⽤ Nesterov 動量

    ```python
    from keras import optimizers
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
    model.add(Activation('softmax’))
    
    # 實例化⼀個優化器對象，然後將它傳入model.compile()，可以修改參數
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
                         
    # 通過名稱來調⽤優化器，將使⽤優化器的默認參數。
    model.compile(loss='mean_squared_error', optimizer='sgd')
    ```

    - Train the ANN with Stochastic Gradient Descent

      1. Randomly initialise the weights to small numbers close to 0(but not 0).
      2. Input the first observation of your dataset in the input layer, each feature in one input node.
      3. forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neurons activation is limited by the weights. Propagate the activations until getting the predicted result y.
      4. Compare the predicted result to the actual result. Measure the generated error.
      5. Back-Propagation: from right to left, the error is back-propagated.Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.
      6. Repeat Step 1 to 5 and update the weights after each observation(Reinforcement Learning).
         Or: ReapeatStep 1 to 5 but update the weights only after a batch of observations(Batch Learning).
      7. When the whole training set set pass through the ANN, that makes an epoch.Redo more epochs.

      ```python
      
      ```

    # splitting the dataset into the Training set and Test set

      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X, y. test_size=0.2, random_state=42)

      # Feature Scaling

      from sklearning.preprocessing import StandardScaler
      sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)

      # importing the Keras libraries and packages

    import keras
      from keras.comdels import Sequential
      from keras.layers import Dense

      # Initialising the ANN

      clf = Sequential()

    # Adding the input layer and the first hidden layer

      clf.add(Dense(units=X_train.shape[1], # 第一層的units數為變數的數量
                   kernel_initializer = 'glorot_uniform')
              

             )

      ```
      
      ```

  - mini-batch gradient descent

    - batch-gradient，其實就是普通的梯度下降算法但是採⽤批量處理。
      - 當數據集很⼤（比如有100000個左右時），每次 iteration 都要將1000000 個數據跑⼀遍，機器帶不動。於是有了 mini-batch-gradient——將 1000000 個樣本分成 1000 份，每份 1000 個，都看成⼀組獨立的數據集，進⾏ forward_propagation 和 backward_propagation。

  - 在整個算法的流程中，cost function 是局部的，但是W和b是全局的。

    - 批量梯度下降對訓練集上每⼀個數據都計算誤差，但只在所有訓練數據計算完成後才更新模型。
    - 對訓練集上的⼀次訓練過程稱為⼀代（epoch）。因此，批量梯度下降是在每⼀個訓練 epoch 之後更新模型。

  ![](https://lh3.googleusercontent.com/pSAfVKPxy0VBBeI_5sAFsNORpc6TocZpoZF6uEddMy1fwNyeG83nGh_jz0FRqQsKrfW4mt-30Yn1Refw5k026Lk_ZckL_5bchE90SOpO1xrlPd9bvp-rTxj27vsVjJfsTnb-v3TZ39YqPsUySoCyG5JfQwavcb7-YEBjwB37v6feLBp5LbLfL9eX8TkKpFQpcP5WLepnCDUsDsztOqqsBwlG2YnU8Bmnh_MOJSR67U3Q7bjPbwVP-Le78u-zhv6QZW0GmLip8ugjqXRVGKTkL_Z5xqlCRPiDUugTSZSioJheg_U070Qg5WK9VgechLxomLG6xiaTsQxzF7DG6XATFeXEe6Tvcm5wGPf22eO5tY3p-7APlGmZfeGls8CdxHw6sHzlk0Yki2rFqqwsUZ2yUEY9aCVMaXEnYq9y6-ptna4XAWkmeoyX_zBqyrzWKuVjgHRPUhOdM2MOyrH55Om8PwvWk-mIczqndAs46npxhoR_MNwDPPojp2sXtafG9njZgGWK2NcblGt60JWA2Isl_1FnwUqY_mbNyRivLUGJHA0kx-gux0SvJg3fRC-wbVIJyGdWQ4Ur53f06Vt5-CIKnvXiDBVkI6mXOZkXF_whZ47XzGLC15FCNxg2_zcrVrsnhtVjvjazRuNegMfKHNNiuACC_2HYN65yqWwvF0TyxXVeFr1aMnqw3xgnFDLQpr7hLcZ-ePdZreRkDTNVHY67vrPj=w354-h485-no)

    - 參數說明

      - batchsize：批量⼤⼩，即每次訓練在訓練集中取batchsize個樣本訓練；
      - batchsize=1;
        - batchsize = mini-batch;
        - batchsize = whole training set
      - iteration：1個 iteration 等於使⽤ batchsize 個樣本訓練⼀次；

    - epoch：1個 epoch 等於使⽤訓練集中的全部樣本訓練⼀次；

      > Example:
      > features is (50000, 400)

    > labels is (50000, 10)
    > batch_size is 128
    > Iteration = 50000/128+1 = 391

  - 怎麼配置mini-batch梯度下降

    - Mini-batch sizes，簡稱為「batch sizes」，是算法設計中需要調節的參數。
      - 較⼩的值讓學習過程收斂更快，但是產⽣更多噪聲。
      - 較⼤的值讓學習過程收斂較慢，但是準確的估計誤差梯度。
      - batch size 的默認值最好是 32 盡量選擇 2 的冪次⽅，有利於 GPU 的加速
    - 調節 batch size 時，最好觀察模型在不同 batch size 下的訓練時間和驗證誤差的學習曲線
    - 調整其他所有超參數之後再調整 batch size 和學習率

  - Adagrad

    - 對於常⾒的數據給予比較⼩的學習率去調整參數，對於不常⾒的數據給予比較⼤的學習率調整參數

      - 每個參數都有不同的 learning rate,

      - 根據之前所有 gradient 的 root mean square 修改

      $$
      \theta^{t+1} = \theta - \frac{\eta}{\sigma^t}g^t
      $$

    $$
      \sigma^t = \sqrt{\frac{(g^0)^2+...+(g^t)^2}{t+1}}
    $$

    > Root mean square (RMS) of all Gradient 

  - 優點：Adagrad 的優點是減少了學習率的⼿動調節

    - 缺點：它的缺點是分⺟會不斷積累，這樣學習率就會收縮並最終會變得非常⼩。

  ```python
    keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
  ```

  - lr：float >= 0. 學習率.⼀般 η 就取 0.01

    - epsilon： float >= 0. 若為 None，默認為 K.epsilon().

  - decay：float >= 0. 每次參數更新後學習率衰減值

    ```python
    from keras import optimizers
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
    model.add(Activation('softmax’))
                         
    #實例化⼀個優化器對象，然後將它傳入model.compile() , 可以修改參數
    opt = optimizers. Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    ```

  ```
  - RMSprop
  
    - RMSProp 算法也旨在抑制梯度的鋸⿒下降，但與動量相比， RMSProp 不需要⼿動配置學習率超參數，由算法⾃動完成。更重要的是，RMSProp 可以為每個參數選擇不同的學習率。
  
    - RMSprop 是為了解決 Adagrad 學習率急劇下降問題的，所以
  
    $$
    \theta^{t+1} = \theta ^t - \frac{\eta} {\sqrt{r^t}}g^t
  $$
  
  $$
    r^t = (1-p)(g^t)^2+pr^{t-1}
  $$
  
    - 比對Adagrad的梯度更新規則：分⺟換成了過去的梯度平⽅的衰減平均值
  
  ​```python
    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) 
    This optimizer is usually a good choice for recurrent neural networks.
  ```

  - lr：float >= 0. Learning rate.

  - rho：float >= 0.

    - epsilon：float >= 0. Fuzz factor. If None, 

  - defaults to K.epsilon().

    - decay：float >= 0. Learning rate decay over each update

    ```python
    from keras import optimizers
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
    model.add(Activation('softmax’))
    #實例化⼀個優化器對象，然後將它傳入model.compile() , 可以修改參數
    opt = optimizers.RMSprop(lr=0.001, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt) 
    ```

  - Adam

    - 除了像 RMSprop ⼀樣存儲了過去梯度的平⽅ vt 的指數衰減平均值，也像momentum ⼀樣保持了過去梯度 mt 的指數衰減平均值,「 t 」：

    $$
    m_t=\beta_1m_t + (1-\beta_1)g_t
    $$

    $$
    v_t=\beta_2m_t + (1-\beta_2)g^2_t
    $$

    - 計算梯度的指數移動平均數，m0 初始化為 0。綜合考慮之前時間步的梯度動量。

    - β1 係數為指數衰減率，控制權重分配（動量與當前梯度），通常取接近於1的值。默認為 0.9

    - 其次，計算梯度平⽅的指數移動平均數，v0 初始化為 0。β2 係數為指數衰減率，控制之前的梯度平⽅的影響情況。類似於 RMSProp 算法，對梯度平⽅進⾏加權均值。默認為 0.999 

    - 由於 m0 初始化為 0，會導致 mt 偏向於 0，尤其在訓練初期階段。所以，此處需要對梯度均值 mt 進⾏偏差糾正，降低偏差對訓練初期的影響。

    - 與 m0 類似，因為 v0 初始化為 0 導致訓練初始階段 vt 偏向 0，對其進⾏糾正

    $$
    \hat m_t = \frac{m_t}{1-\beta_1^t}
    $$

    $$
    \hat v_t = \frac{v_t}{1-\beta_2^t}
    $$

    - 更新參數，初始的學習率 lr 乘以梯度均值與梯度⽅差的平⽅根之比。其中默認學習率lr =0.001, eplison (ε=10^-8)，避免除數變為 0。

    - 對更新的步長計算，能夠從梯度均值及梯度平⽅兩個⾓度進⾏⾃適應地調節，⽽不是直接由當前梯度決定

    ```python
    from keras import optimizers
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
    model.add(Activation('softmax’))
    #實例化⼀個優化器對象，然後將它傳入 model.compile() , 可以修改參數
    opt = optimizers.Adam(lr=0.001, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt) 
    ```

    - lr：float >= 0. 學習率。
    - beta_1：float, 0 < beta < 1. 通常接近於 1。
    - beta_2：float, 0 < beta < 1. 通常接近於 1。
    - epsilon：float >= 0. 模糊因數. 若為 None, 默認為 K.epsilon()。
    - amsgrad：boolean. 是否應⽤此演算法的 AMSGrad 變種，來⾃論⽂ 「On the Convergence of Adam and Beyond」
    - decay：float >= 0. 每次參數更新後學習率衰減值



- 如何選擇優化器

  - 隨機梯度下降（Stochastic Gradient Descent）

    SGD 指的是 mini batch gradient descent 優點：針對⼤數據集，訓練速度很快。從訓練集樣本中隨機選取⼀個 batch 計算⼀次梯度，更新⼀次模型參數。

    - 缺點：
      - 對所有參數使⽤相同的學習率。對於稀疏數據或特徵，希望盡快更新⼀些不經常出現的特徵，慢⼀些更新常出現的特徵。所以選擇合適的學習率比較困難。
      - 容易收斂到局部最優 Adam：利⽤梯度的⼀階矩估計和⼆階矩估計動態調節每個參數的學習率。
    - 優點：
      - 經過偏置校正後，每⼀次迭代都有確定的範圍，使得參數比較平穩。善於處理稀疏梯度和非平穩⽬標。
      - 對內存需求⼩
      - 對不同內存計算不同的學習率

  - AdaGrad 採⽤改變學習率的⽅式

  - RMSProp：這種⽅法是將 Momentum 與 AdaGrad 部分相結合，⾃適應調節學習率。對學習率進⾏了約束，適合處理非平穩⽬標和 RNN。

    - 如果數據是稀疏的，就⽤⾃適⽤⽅法，如：Adagrad, RMSprop, Adam。

  - Adam 

    結合 AdaGrad 和 RMSProp 兩種優化算法的優點，在 RMSprop 的基礎上加了 bias-correction 和momentum，隨著梯度變的稀疏，Adam 比 RMSprop 效果會好。

    對梯度的⼀階矩估計（First Moment Estimation，即梯度的均值）和⼆階矩估計（SecondMoment Estimation，即梯度的未中⼼化的⽅差）進⾏綜合考慮，計算出更新步長。

- 參考資料

  - [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html)
  - [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)
  - [A Neural Network in 13 lines of Python (Part 2 - Gradient Descent)](https://iamtrask.github.io/2015/07/27/python-network-part2/)

- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

  - [TensorFlow中的優化器](https://www.tensorflow.org/api_guides/python/train)

- [keras中的優化器](https://keras.io/optimizers/)

  - [Adam优化器如何选择](https://blog.csdn.net/qq_35860352/article/details/80772142)

## 訓練神經網路的細節與技巧

- Validation and overfit

  - Overfitting：訓練⼀個模型時，使⽤過多參數，導致在訓練集上表現極佳，但⾯對驗證/測試集時，會有更⾼的錯誤率。

    ![](https://lh3.googleusercontent.com/wySmre4Z4or2gnibj57qWRWZuo6kaADcnuD-2McBnAg-DuqT5LkPZhYFwL1Mrt6oBOv-WrAE8OOaxav5U_CQ1fKes3l63YoljkduOzXqhNnvhE2kyTc4TUx5hk6MRviqaESPetK865aHpt7sjyAkJ2uyLMSWZc6xmqkTrk-9pFSL1XN0y5jgi8SNOIWX78i5rqCSTiIAlrUoOzMQ2LHBqn0gUpEbADWv7toequIDS6Fhfsthpn1cjZ35MWTIKkpuulPgavdAZTp46EHr9j_GVhyLSTUnRJ1pgoL2tL9ErvYzzq38Y-_KdFpyhHNGu0q31lFuM8t8Dqcly-tiV-TFD9yLzsC77s7jpHNKT_W7r1835hcmTgsaUnSvqoWvEfGxBtljNNE3ntToIiHSg3ERkI3S5FfIvN1SgfyQqJpc_hqARwzkpuIj1JgTB2TnHxBig4H-dqtkcdNGUXV9Why1y7G1VAYN78urhVqXAMXkp1QANAftf-XXEOfGQ789JVbg5cRpa2cmVNUP6skO0f9dXv9y1b9PaQvxPSCZ-6oH0lZhvlMHbpRMYWZJJheLucmSz4TzRe6T9qMEBVRuccOP4hGkRjA97Q1NiGwQWb-D7XMlZ1OQ2DGsYIrvvNWIEKtQT1Z3r4B_PGrV85fZ5e1IWv62uj17mfJ6ThsaVWzgy0XwmVZt6nBMEFZBVGFZ0DJbNjCl6fnf6j0AJDoLCDC8BIPJ=w406-h340-no)

  - 過度擬合 (overfitting) 代表

  - 訓練集的損失下降的遠比驗證集的損失還來的快

  - 驗證集的損失隨訓練時間增長，反⽽上升

    ![](https://lh3.googleusercontent.com/ZyMV6_I4wI0eKi-E6uBTfIpR60EFGuNbHsMqhMaqno1cooNxN0fTXeVcqlCCwY5Re7RbvAhdKRnqTk6p7TMeSvnN2XuORYEcRBFxMMLg5TyNnrBc_Z_v-gWpC7NDhp_rHK0dcvXAPCD6k5zssWsxCCWaXJEbt2p1mxPV4l-HJ_EKj2uikR7AaQu1NWFOxvK8YpuSERDDJdgI15IsuxevUM6d5H4gW_jWCMQf0FAPLKlYiHXwLl9_sxmjnjf__ce_aLm4ZzaDDFGXvzJM0lEpP5jtDYIIRKvweoISwKwgMjVW8vmEpffZj4Efx9p8w2ZELxdI1PAqR1Ah6b_q70AggX0pu8EoMBbKHxgb9k5ZTiAUXqHNjiBDKiHifgZxF1M26-9jG9fllJIdmKRGM3dREAnlX4h9M31f0t4vruP6Dvqf6bxg_nJ6NXITxXh55Mw3cfIm10PMCfu4xc-lVoKd_-DTXtBCNk-LRyX_axDmb0Ua_NrYTir3C_95ewfH6AabtmVF9SMHC_y9pQVtx9B5C_ILvf5u3OnYYldjlbUsP1ENQlhba3AlMlaVHWROlHTGrQoBG6mttIlZqbw780eKBdA-P6mA670mcASksFY7_fwXqcEvCmgAooEdZfgIT4CXmzVmLHwpjACB41BRI0-Wsr9Sbt_I6W_aegIzFAARVhuYqbdmxS9Vnwq9y1ZJkiK-I_U-XUO_JR1QN34mJIuI6FgU=w528-h364-no)

  - 在 Keras 的 model.fit 中，加入 validation split 以檢視模型是否出現過擬合現象。

    ```python
    model.fit(x_train, y_train, # 訓練資料
              epochs=EPOCHS, # EPOCHS
              batch_size=BATCH_SIZE, #BATCH_SIZE
              validation_data=(x_valid, y_valid), # 驗證資料
              shuffle=True) # 每epoch後，將資料順序打亂
    
    model.fit(x_train, y_train, # 訓練資料
              epochs=EPOCHS, # EPOCHS
              batch_size=BATCH_SIZE, #BATCH_SIZE
              validation_split=0.9, # 驗證資料
              shuffle=True) # 每epoch後，將資料順序打亂
    ```

    > 注意：使⽤ validation_split 與 shuffle 時，Keras 是先⾃ x_train/y_train 取最後 (1-x)% 做為驗證集使⽤，再⾏ shuffle

  - 在訓練完成後，將 training loss 與 validation loss 取出並繪圖

    ```python
    train_loss = model.history.history['loss']
    valid_loss = model.history.history['val_loss']
    
    train_acc = model.history.history['acc']
    valid_acc = model.history.history['val_acc']
    ```

    ![](https://lh3.googleusercontent.com/aRcKSR_WJasIBDgHERPDSP2Lsj-FYplyPHzdhPhgu2Fu3I6WdAHAsRDk9_n6SwHXD2Koqy66RVuzf4req3PEUWY_ZFwFBoORZy62tkn_FJiuQBtO8BncqGY-SqZ61y3t_oFjzdkqhKWgxVT-wD8jpELgCdmkElw1Unx74ey6U1tGzsKGe1xhICNUfgxv7EwgsRF_ynSw0fcIzEoRc6DhFAvXE3mW315m9elVL5n8eXS5xGVN3roQ3yrnhBxX0BGDtF_GQBt9kvpvQCtwpT_-QAwpdqk39YCXUVsGlLPoabsz55qIkIq-oQbu1b62WQF7bc9aMGVXAa1rhx-a6oQc4Kvwyg3kbnaz09TddTjKke-M36Vu6xkfB4pfT-brCh_t8gwcMaSxbI0Xc_zUvgz1ai3tGVUrDN4fFZLtzqvqw5H2UnvGVs-au6gD6SElpEtGvyJ1aykTkwSYVoq1vbs0d07zKpsIWwO5GdPAJyx6ybUXK3AnWI5Rk8UTZzaEfzjpQkyl4nvqVzBLrTrJyatcWpWsiebw1F87iR37b8daGyXJn2roKGTaEHfo1ak_T2kOTVgHfnbn4czypgd3JWPiCcYj_N64hZ6f3TOVPHnokr-k64HVLZnSgrEqt8P95SK76hNDD74B8f-jpKW6thgjZS7E0ywn5D6hHOWWFh3pnOx2Y0p7Uy3U6UCjNBTjsbbxkPk4qbYyb8JAszwDWinrAIhx=w547-h386-no)

  - 參考資料

    - [Overfitting – Coursera 日誌](https://medium.com/@ken90242/machine-learning學習日記-coursera篇-week-3-4-the-c05b8ba3b36f)
    - [EliteDataScience – Overfitting](https://elitedatascience.com/overfitting-in-machine-learning#signal-vs-noise)
    - [Overfitting vs. Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)



- 訓練模型前的檢查

  - 訓練模型的時間跟成本都很⼤ (如 GPU quota & 你/妳的⼈⽣)

  - 要做哪些檢查

    1. 使⽤的裝置：是使⽤ CPU or GPU / 想要使⽤的 GPU 是否已經被別⼈佔⽤?

       - nvidia-smi 可以看到⽬前可以取得的GPU 裝置使⽤狀態

         > windows作業系統的nvidia-smi在以下路徑，進入該路徑後輸入nvidia-smi即可使用
         >
         > ![](https://lh3.googleusercontent.com/Aih5DxOZU1a-woo1a6_VhqDnX__WD6LDjTYmkpi33vll6Xd-rZbSmkzwShlQD3aQ09uwNZOBReeZDRJzC3aSL283DXMyrJ7sTNq4EL37n67_K_fjpvvkW401dxkHMfmrONMUiHskQ8XOLRu_HbGAiR1Y0AzcluB8yRVV01NmT0jQ_q3I5GFnPEgE9do8-HykIOXJw95aGkrkbO9QSU1hY_5hR8dhHwf_UNtu6sdzBzKY-_vPj9spwC_hTP8Qw3IUBggnIlFw9zju1lBVNJSyICrquamYKsaJyh-iskKKzplILoF6QedfU30SF5m8v1c9A7FzKjslqt4Ct71bxWgLKO-PwDlkT7O_1VcjYUPidKMoo57BzoXHl6HVvEqX163Rlg-YKVVbo9RS55WW4hy1Vy_CoMXJL9d7iepMJNs_VLe8V3yLqxBQe7tvpFI6aOlDsmxP2LtB1zWYt5lul5URHamxeqcJiPD-0gteNAyaGzjjB0TDac3J4YPhajh8W5WYHVdIuu5bpY8joNOe92J8bFxkbvSK5w-kOxCSKY_DAahBLEYSEX_i5oqGHHDGDIPzb4Nuy7mOILMHDJNYEPHuMERoEiYv0Ye0PESUoQI22zodcG7xIUtcwesaU0-19Lbu66sEHeV0D06SvF5r40v47s3Wh-h2N-sguUUMk7UOvXPGYvBcNufXlWFuVkQvnlZqGxOJCvdILERI96cOO3s_tuhW=w653-h423-no)

    2. Input preprocessing：資料 (Xs) 是否有進⾏過適當的標準化?

       透過 Function 進⾏處理，⽽非在 Cell 中單獨進⾏避免遺漏、錯置

    3. Output preprocessing：⽬標 (Ys) 是否經過適當的處理?(如 onehotencoded)

       透過 Function 進⾏處理，⽽非在 Cell 中單獨進⾏避免遺漏、錯置

    4. Model Graph：模型的架構是否如預期所想?

       model.summary() 可以看到模型堆疊的架構

    5. 超參數設定(Hyperparameters)：訓練模型的相關參數是否設定得當?

       將模型/程式所使⽤到的相關參數集中管理，避免散落在各處

  - 參考資料

    - [選擇 GPU 裝置與僅使用部分 GPU 的設定方式](https://github.com/vashineyu/slides_and_others/blob/master/tutorial/gpu_usage.pdf)
    - [養成良好 Coding Style: Python Coding Style – PEP8](https://www.python.org/dev/peps/pep-0008/)
    - [Troubleshooting Deep Neural Network – A Field Guide to Fix your Model](http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf)

- Learning rate effect

  - 學習率過⼤：每次模型參數改變過⼤，無法有效收斂到更低的損失平⾯

  - 學習率過⼩：每次參數的改變量⼩，導致

    1. 損失改變的幅度⼩
    2. 平原區域無法找到正確的⽅向

    ![](https://lh3.googleusercontent.com/PTMBO-EvAdGWmYwCXXJwfog7zX3RKk1Eucro_zCNSdpsP0zL7wvHzJERYZXaAFctNBilQj1OWTDLhxWvIHrdZuyPfQ3DAMMIO9crEa5tGiBs_jf8hG2_CD4-8RoTSBCeyOMH2KfPWNgbOCxW7W6ivmwO4oAKyJc_nLJjA02c1533_ihyEDsH6W9MbO8oTXeY4N1UpIE6nfVGEeKeAWMU8pu0ytFzzVUk2HOOZxSjkq6Cy2qPGQAEelGGbGqGU8NGPU-aCiSGedGnkPr5-3uNqSTgDmScBuF8aS6YrgoHP8pqpVKjYUpNDpj-udf_7VxXBfuGlJb6pkeF0efnf42TOJ9mQZm3nB0uZA0gkzNcoc9oj4abMglGubk3oOEG_I4MaAHoMzJyvZNuIdkqDsWeOdPmJn-NzN_mdrfMzPoRL86x0ijqKv4m52DzlwKZcLMJaRGmvMzErs1-i3ZaDXkxlfRoqbV2G2_H84um4VNC-AVQ7brz-VFTy2wvf8s6K4GuRNkFGDckpffC3OTMsDK5S0Gjzj_wly3YsjN3JnD8lYtvg5OsPXjyVUCJOa_j8o9LnQ9DeUMG-CzrwpNTKhZ_8Ke7LO56-uDh2MQsgEXyehYnVDnzt5vnA7_EjYIoe_hCPo8BAKVsTuc8bE8vXzGRiONgOefRQP749yj5b4pSqKY6uM0gF3sqffvfVB8hfclnOnDRJALkgDWUrb60DJbCEDkU=w1029-h349-no)

  - Options in SGD optimizer 

    - Momentum：動量 – 在更新⽅向以外，加上⼀個固定向量，使得真實移動⽅向會介於算出來的 gradient step 與 momentum 間。

      - Actual step = momentum step + gradient step

    - Nesterov Momentum：拔草測風向

      - 將 momentum 納入 gradient 的計算

      - Gradient step computation is based on x + momentum

      ![](https://lh3.googleusercontent.com/Y-oEhhWLyFGrfALnvn-9KpR_R23c8pDkE5h2KUxL1eNokjwC_LXAP0QoXW0cLIsi6Slrcc057wH45VU80m441RtY6VOWeBCBJfGnAEeGmonX0u9oBApYfvlUnah0NTpfLGMGW2XJ3vyVCE5nrMBMW_pACUAb24f1cvPdAOpo-IAJgbe4BGqt-cVST8a1FZlb_qYBwzY1Pwt2WrRytVIbfn5e1Z2l_zAtU-8lr4TlbCzRniWAGooX4aSaIAzdplXR4HczqOeAGbsyGMXDbGiTzluk7dJ7o3XF-pJGfQsXJYUdxV1H2NxxvI6CT0pAVUZTRV00VwnyEQd1AkkXzW8JRvdy_f1UOrqz86_fbhufezO927YgvRTyTOyW7NZp6rNEVhRyKJSsMRc3VvjV_As_KU_feCuJvoIaLq0vGWJlX1R6S87DgR6AyLP8D7FOn2Wkd8HyfSToumyOZ33FvgIkYKDkpYKSqdWYcOdUD0OMWSeI1OWDacOXv0_bP2W32nRp5w8ZblukMgEjN_uL3IFS2WS4-2lWH5nDh8i_LtvUdBPp26DR9s71SquD6iQul8RguwXd0HCqxBU2szbU-T2aKtZrfoFJlgKqErTVcZtHeGE7ux3dU3NaqRPqgXVXhBPOiApFkuAt2jzbxbNV1qonAropAYUSBLZwnTR3QV_Pw0OiCqvONY1MBrBxlmuYzsGqNeFH2gTrYIvtzCYMy6coFfsa=w1063-h301-no)

  - 在 SGD 中的動量⽅法

    - 在損失⽅向上，加上⼀定比率的動量協助擺脫平原或是⼩⼭⾕

  - 參考資料

    - [優化器總結 by Joe-Han](https://blog.csdn.net/u010089444/article/details/76725843)

- Regularization

  - Cost function = Loss + Regularization
  - 透過 regularization，可以使的模型的weights 變得比較⼩
  - Regularizer 的效果：讓模型參數的數值較⼩ – 使得 Inputs 的改變不會讓 Outputs 有⼤幅的改變。
  - wi 較⼩ 
    ➔ Δxi 對 ŷ 造成的影響(Δ̂ y)較⼩ 
    ➔ 對 input 變化比較不敏感
    ➔ better generalization

  ```python
  from keras.regularizers import l1
  input_layer = keras.layers.Input(...)
  keras.layers.Dense(units=n_units,
                     activation='relu',
                     kernel_regularizer=l1(0.001))(input_layer)
  ```

  ![](https://lh3.googleusercontent.com/6SLWFn1vxOrvvMds-C7SihSsG1dOJcYQJyyq-DhrLKHvpZtnLvF53qxQ_rB5xmO3AnPCM8U1RdmgBTBkkaeZQaf5cY2O1ycp4NsiCpHNP2d0APJn_1LC5dDWckhVzV2DrLsgNpgDTfwLwxdtRUgI9Rr79mOj9HBBsctvLcJs1NAl-OL2DnT0bCsxuLrdRV-cDFd3c906g7ca3z5pIK7N4p0f4bLossiMiFaOGc-x-GUsDYbdGK4K5Pe-cZcTNdHvH3NgDTwzwvF1IdVGNQd2WXkohn_1f1jOSauorkQo45WJgfh0HPap7gZ5nvTs5ZBT2uH48f014HFIIoWIjJiHB2AjxZPraFTPH1zTLjCLAUzn8Fx3H2x6Mijvp0TBSLnDJL7-JWxdj6CO4dVSkCmt7Qro-MTUIxzCLZMhbMpsU6MwpqpIn52CtmAIR9eWmNWsBbdapnk9j8QHfNcZ7My8Emmkv8CBY-xAMeUhEFXSCu7mm8SRc2rGp2Q3ZF2Q6dOjmArHZuOxMr_qu3F3pJW-hBNSpg_xhUhJIdJ9p5cPgaSxISK0Zn2ydhEXb_Rnk7_o7C-XzbfhOHd6nb0WtqsTNI90968DVrLHdGpK6hgAIfbFQxwe-4vQ2_obRn0GPazia1Emw2jpTuUPPMrb513kHvfl6EFizF1a31pHRGwgWtfNcASNDApzluYflAU_IklfGEAFGvnD_sOS8pIAPLcAcydt=w611-h472-no)

  - 參考資料
    - [Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
    - [Machine Learning Explained：Regularization](http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/)
    - [機器學習：正規化 by Murphy](https://murphymind.blogspot.com/2017/05/machine.learning.regularization.html)

- Dropout

  - Dropout：在訓練時隨機將某些參數暫時設為 0(刻意讓訓練難度提升)，強迫模型的每個參數有更強的泛化能⼒，也讓網路能在更多參數組合的狀態下習得表徵。
  - 在訓練過程中，在原本全連結的前後兩層 layers ，隨機拿掉⼀些連結(weights 設為 0)
    - 解釋1：增加訓練的難度 – 當你知道你的同伴中有豬隊友時，你會變得要更努⼒學習
    - 解釋2：被視為⼀種 model ⾃⾝的ensemble ⽅法，因為 model 可以有 2^n 種 weights combination

  ![](https://lh3.googleusercontent.com/vkLYD68AKR-OcpD4tS2tCWvwt-6hZ4qy8MM6RajC2guOV4jWp_2_NtCIhBJRiuU9uDEfCiKweWeolok69vWGSgWruASv4mdzjEMAxrEp6aZ8hXGJQxrySirZOdT00_ol4qWArYMTNtVjCZWJiTyp631G5RSK6iraVun38qxiI8ksOXGsAHePv0iL5UGDZ8GYFx9gp4D-Ys1lRntokQ5x2NORUiFGaiH2cHUILjJadMTac9sCwxu6_z-Ac9O3FOBl7cCgKaPBFFugH6Fd9_lvfb8phe0335-8RR-lZlYd6oSR_wYzM8hd09WWby9iUt-vuPt_5nEac1HSB7zzrJ7Q9cQ8P62GIlMDsZW-dfkmE6sQihnfICkZnpDffzbgFNIpdgvLzTsJBXJjxnMWkle2oT9NF4em8IPOfr-sOeM_MSkVaIFJQrssCFOS2UAQQmIvemgiVNJHURzMxPmkkfYjR1BVLv4P5tRfT9VLLXnkjFhpjIE_qCTV5O4O0WOxJlldwLdYzYBSnRZhULusRD5BP-N9NTG4al82GVOdj4ftJJWxYBrASdbtirq53oShQk17VQPiURdVEHw5rnMFufmVyMXByXwZGdKewCAszDWGJfRTvJcYpX1OF16aGMNjI8JhZ5zzOwqWxtiQ0heGJ5O3JQ90STvVgLfFSaJKlk3mrYW-Afo_9HxzYJet4aq41hcZZcf1Sews-J_nY74jZ9MclN4o=w725-h334-no)

  ```python
  from keras.layers import Dropout
  x = keras.layers.Dense(units=n_units,
                         activation='relu')(x)
  x = Dropout(0.2)(x)
  # 隨機在一次 update 中，忽略 20% 的neurons間的connenction
  ```

  - 參考資料
    - [理解 Dropout – CSDN](https://blog.csdn.net/stdcoutzyx/article/details/49022443) 
    - [Dropout in Deep Learning](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)
    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)



- Regularizatioan

  - 對於 Input 的數值，前⾯提到建議要 re-scale
    - Weights 修正的路徑比較會在同⼼圓⼭⾕中往下滑

  ![](https://lh3.googleusercontent.com/u4ozWUFeEgyfpSYmG1I1ablnk-E-Y1wAsED9CKgjt6Vb3qNQUb0YngMngoUtBrJlrGf_BnzOAU3EppM9LEsIG8zoszH8zBEmADAio2smH37kc3JsC1vt3bDYiHcU0gCgmoaf2WArBjH8J0TNKJe6zYVG8vpgWi9P_piV2M3vQiXJZkFufNfvDGD0aKGp8q_kLIGzqkgZu6hXGYtAmrRjsbcaKVE27R_AdPbz-zLbxlSmgMW8uy76rjYAXsfvm47OQ5NSS5Mtni_KcBD3RO995G02SA4IJZgjjA7701DJcFSVVSj5UoKUdJxcoTv42XmdM1gl7kO3Ml86x9NPJKcbE9Oo6vhrHP1gOqezvPlaM_Ie1edyK76ynh2yRDEOA72R41NEUJ1lP6UPo5TJO_1IgJU3EIDRcd_1TcHviFbcs41h-CiUsmn3cH3T-1t0FWU1Pe_8TJbkmvFoLgW79NYE_Nc9zSqiIMxrUQEQhmz651IVUH8nHDvnl02Uj74eyslBsTD4N17SClimh5Xn2icfSIvJXAZoeW3stGxGeTQyOE0lWP6co0Fjeksb1jqMGjPkI-56z3vVCuf0_JBql-U9dW7wlqHM5q6x4J4pi2xNaV10mAt4SW0VHXyMOrMeJS1Y3cwwGPx2vQcVIMYDtokXHKDXFcORZ5Iwg6AUlvcQH-i6npaFoaVIrG_fLm8nowA2Qxu3cEr3EfESPJJ4L8AuY_ye=w925-h309-no)

  - 只加在輸入層 re-scale 不夠，你可以每⼀層都 re-scale !

- Batch Normalization

  - Batch normalization：除了在 Inputs 做正規化以外，批次正規層讓我們能夠將每⼀層的輸入/輸出做正規化
  - 各層的正規化使得 Gradient 消失 (gradient vanish)或爆炸 (explode) 的狀況得以減輕 (但最近有 paper對於這項論點有些不同意)
  - 每個 input feature 獨立做 normalization
  - 利⽤ batch statistics 做 normalization ⽽非整份資料
  - 同⼀筆資料在不同的 batch 中會有些微不同
  - BN：將輸入經過 t 轉換後輸出
    - 訓練時：使⽤ Batch 的平均值
    - 推論時：使⽤ Moving Average
  - 可以解決 Gradient vanishing 的問題
  - 可以⽤比較⼤的 learning rate加速訓練
  - 取代 dropout & regularizes
    - ⽬前⼤多數的 Deep neural network 都會加

  ```python
  from keras.layers import BatchNormalization
  x = keras.layers.Dense(units=n_units,
                         activation='relu')(x)
  x = BatchNormalization()(x)
  ```

  - 參考資料
    - [為何要使用 Batch Normalization – 莫凡 python](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/)
    - [Batch normalization 原理與實戰 – 知乎](https://zhuanlan.zhihu.com/p/34879333)
    - [Batch Normalization： Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
    - [深度學習基礎系列（九）| Dropout VS Batch Normalization? 是時候放棄Dropout了 深度學習基礎系列（七）| Batch Normalization](https://www.itread01.com/content/1542171910.html)

- EarlyStopping

  - Callbacks function：在訓練過程中，我們可以透過⼀些函式來監控/介入訓練

  - Earlystopping：如果⼀個模型會overfitting，那 training 與 validation 的表現只會越差越遠，不如在開始變糟之前就先停下

  - 在 Overfitting 前停下，避免 model weights 被搞爛

  - 注意：Earlystop 不會使模型得到更好的結果，僅是避免更糟

    ![](https://lh3.googleusercontent.com/-6BtDcbtyb79-m8O_YBqrawpJDfBAZafriKFlMdxm9VVhvtCZE3vb37HZivlfknuQfAM5CTGXEp7NPiFrsNL0kNoSKX1CqUTbrWxNSJucy8HeRwplqnH5IcHE6UFqj2Urblk62LEutx7NvHa3dolFFaXGQgI60e79y5Lp3r-WK2bkKWchbtfQZt1F0U5t0Z83hg8qxPt8SGJtUNSADvLtvVV0T479skMhDv5VZVE2OIwgD7s3J8lnSPKh9LOdKcjcHaE9Gff-ewiK5OoLwZYkiHmOt_vti9EuJV2ryHzV2Etlp0V5sn1MvK7v2F2S96Zr44Lc3Yl2k_IGtyZHzxIc6mVN6TEKIYFS80D7S24_nGmE2JtOVerPQ_nBEaMyXdjkD9CgFaF9pNZtKR4cuDcV1dykEhOZxPLMR2g6C3HAqso2LZ9jdjQyaKRLttfdZ_bFU5V7713-NtdM9pWgHFNilGPQPJenHcD3G7gTvabmJRy6ip9nGi5KDLMfQPmyvVJG7Xl_H2K8OBy46dvwYLqepHpIDRAtF9_N6uzfJiNBVxuABeF68ETU0V-V_UyZQJU4TkypzKnSJXMyw0B_I8Oohcs3DuY4OznyjXiEIi6N6OTNR94FsdDEigrrFBjtyahpl9ZybNFEOVsZBg85ASBBd7v1PJ3cMfxJz5yjoNpaNDecWZtuca0rNceNZ-yfrE6grpI2juaJpVaiMzXIOSCke2i=w589-h428-no)

    ```python
    from keras.callbacks import EarlyStopping
    earlystop = EarlyStopping(monitor='val_loss', # waht to monitor
                              patience=5, # epochs to wait
                              verbose=1) # print information
    
    model.fit(x_train, y_train,
             epochs=EPOCHS,
             batch_size=BATCHIZES,
             validation_data=(x_test, y_test),
             shuffle=Ture,
             callbacks=[earlystop])
    ```

    

- 參考資料

- [Keras 的 EarlyStopping callbacks的使用與技巧 – CSND blog](https://blog.csdn.net/silent56_th/article/details/72845912)

- ModelCheckPoint

  - 為何要使⽤ Model Check Point?
    - ModelCheckPoint：⾃動將⽬前最佳的模型權重存下
  - Model checkpoint：根據狀況隨時將模型存下來，如此可以保證
    - 假如不幸訓練意外中斷，前⾯的功夫不會⽩費。我們可以從最近的⼀次繼續重新開始。
    - 我們可以透過監控 validation loss 來保證所存下來的模型是在 validation set表現最好的⼀個
  - 假如電腦突然斷線、當機該怎麼辦? 難道我只能重新開始?
    - 假如不幸斷線 : 可以重新⾃最佳的權重開始
    - 假如要做 Inference :可以保證使⽤的是對 monitor metric 最佳的權重

  ```python
  from keras.callbacks import ModelCheckpoint
  checkpoint = ModelCheckpoint('model.h5', # path to save
                               monitor = 'val_loss', # target to monitor
                               verbose = 1, # print information
                               save_best_only = True # save best checkpoint)
                               
  model.fit(x_train, y_train,
           epochs=EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=(x_test, y_test),
           shuffle=True,
           callbacks=[checkpoint])
  ```

  - 參考資料
    - [How to Check-Point Deep Learning Models in Keras](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
    - [ModelCheckpoint – Keras github](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633)

- Reduce Learning Rate

  - Reduce learning rate on plateau：模型沒辦法進步的可能是因為學習率太⼤導致每次改變量太⼤⽽無法落入較低的損失平⾯，透過適度的降低，就有機會得到更好的結果
  - 因為我們可以透過這樣的監控機制，初始的 Learning rate 可以調得比較⾼，讓訓練過程與 callback 來做適當的 learning rate 調降。
  - Reduce Learning Rate: 隨訓練更新次數，將 Learning rate 逐步減⼩
    - 因為通常損失函數越接近⾕底的位置，開⼝越⼩ – 需要較⼩的Learning rate 才可以再次下降
  - 可⾏的調降⽅式
    - 每更新 n 次後，將 Learning rate 做⼀次調降 – schedule decay
    - 當經過幾個 epoch 後，發現 performance 沒有進步 – Reduce on plateau

  ![](https://lh3.googleusercontent.com/KsS-dlezf2E2VPokBOWBpgw1fYvWC-1Vtvj0rI4Xwv7X4f1zXa-GdQpmg5esDz7xlUIeXN8bAQrDPquMoK64B_6uFnJyM4OryKbiLWhh-uQ_KpHiA2wq-s-jxAjfNkiWJae2j-FlIU0SwD7xkM33iQ9hjm5rFqrZHqI3yWxbv6ACkucG6c7vGleBRuvCM5pLWOpLtRTbHYd8B0B81LfnAaMzlgfuTlbLxbqHnFOQMhZDbvOK8BaqOMAfIAOTIPSM6tV4HONZJCnfKOWA6GvAeWo_WLIv9nDoLHgLIo0I7m9tOMQz-FlluWBq9LrmmumU67eqwSdWP-5-Yexl40dDNbeJJ0kDrNpeC2-FWoeAjgiBe7K2A_YtcT76q7DbmMC41wh3U05KI9S71IW5TgmzxJkwLr7-2-ckKfSxaU5qAh68sqSKnJ4tTyVJAG5Dv69m6LopX761cYsvFOTvspnLZeaisKEnJOciwv92xqOozHwNHfpRlf8x9h6g9VHjSc-MCGaMBqPCJafNM_wecvBQyXMBbL4IryuKxHDn8XgYIUKjYMGU6tuB_uBlJRaJbNKpxdrczwXuxNB5NbLfIEfB8qCYYRvh_VAwjsyrgdLUe3SySYnBxOCm8nSdswAQgDa1PiXbVlAcJ4HF8MuoNlH8xXNVOzHEow1GsegTxzODlA7Lg-t7WPNmFjCVI5FueNjHXoXsLNiO0ZqA-IdjCaLL5Wo_=w976-h323-no)

  ```python
  from keras.callbacks import ReduceLROnPlateau
  reduce_lr = ReduceLROnPlateau(factor=0.5,
                                min_lr=1e-12,
                                monitor='val_loss',
                                patience=5,
                                verbose=1)
  
  model.fit(x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=[reduce_lr])
  ```

  - 參考資料
    - [Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1) 
    - [Why Reduce Learning Rate so quickly WILL fail](https://stats.stackexchange.com/questions/282544/why-does-reducing-the-learning-rate-quickly-reduce-the-error)
    - [Keras source code of reduce-learning-rate](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L906) 

- 撰寫自己的callbacks 函數

  - Callbacks 可以在模型訓練的過程中，進⾏監控或介入。Callbacks 的時機包含
    - on_train_begin：在訓練最開始時
    - on_train_end：在訓練結束時
    - on_batch_begin：在每個 batch 開始時
    - on_batch_end：在每個 batch 結束時
    - on_epoch_begin：在每個 epoch 開始時
    - on_epoch_end：在每個 epoch 結束時

  - 通常都是在 epoch end 時啟動


  ```python
from keras.callbacks import Callback
class My_Callback(Callback):
    def on_train_begin(self, logs={}):
        return
    
    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, log={}):
        return
    
    def on_epoch_end(self, log={}):
        return
    
    def on_batch_begin(self, batch, logs{}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return
  ```

  

  - 參考資料

    - [Keras 中保留 f1-score 最高的模型 – 知乎](https://zhuanlan.zhihu.com/p/51356820)
    - [How easy is making custom Keras Callbacks](https://medium.com/@upu1994/how-easy-is-making-custom-keras-callbacks-c771091602da)
    - [Keras Callbacks — Monitor and Improve Your Deep Learning](https://medium.com/singlestone/keras-callbacks-monitor-and-improve-your-deep-learning-205a8a27e91c)

- Loss function

  - 在 Keras 中，除了使⽤官⽅提供的 Loss function 外，亦可以⾃⾏定義/修改 loss function
  - 在 Keras 中，我們可以⾃⾏定義函式來進⾏損失的運算。⼀個損失函數必須
    - 有 y_true 與 y_pred 兩個輸入
    - 必須可以微分
    - 必須使⽤ tensor operation，也就是在 tensor 的狀態下，進⾏運算。如K.sum …
  - 所定義的函數
    - 最內層函式的參數輸入須根據 output tensor ⽽定，舉例來說，在分類模型中需要有 y_true, y_pred
    - 需要使⽤ tensor operations – 即在 tensor 上運算⽽非在 numpy array上進⾏運算
    - 回傳的結果是⼀個 tensor

  ```python
  import keras.backend as K
  def dice_coef(y_true, y_pred, smooth):
      # 皆須使⽤ tensor operations
      y_pred = y_pred >= 0.5
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      intersection = K.sum(y_true_f * y_pred_f)
      
      # 最內層的函式 – 在分類問題中，只能有y_true 與 y_pred，其他調控參數應⾄於外層函式
      return (2. * intersection + smooth) / (K.sum(y_true_f) = K.sum(y_pred_f) + smooth)
  
  # 輸出為 Tensor
  def dice_loss(smooth, thresh):
      def dice(y_true, y_pred):
          return -dice_coef(t_true, y_pred, smooth, thresh)
      return dice
  ```

  - 參考資料
    - [如何自定義損失函數 – CSDN blog](https://blog.csdn.net/A_a_ron/article/details/79050204)
    - [How to use a custom objective function for a model – keras github](https://github.com/keras-team/keras/issues/369)
    - [Issue for load model if using custom loss function – keras issue](https://github.com/keras-team/keras/issues/3977)

## 傳統電腦視覺與影像辨識

> - 了解⽤傳統電腦來做影像辨識的過程
> - 如何⽤顏⾊直⽅圖提取圖片的顏⾊特徵

- 影像辨識的傳統⽅法是特徵描述及檢測，需要辦法把影像像素量化為特徵（特徵⼯程），然後把特徵丟給我們之前學過的機器學習算法來做分類或回歸。

![](https://lh3.googleusercontent.com/obqRVJFHA5BjE2xl8veo5Ctmjw8NHmekXAUyRy-uJVPEYtaYDFTekBzvuqdfH06VdvIOSv_d4pqBPiN3oe5_sopPH2eZ6QLCbdUZ4Yfc_pKuObFRzXugdXKnGQelbr-4i2_baD6p8xOWziehQjZty6zaMZAh5JVdCqKA3K2uxMtXKc0qX3NVK46Xq5Xh5YCkr21j6dUdbsE0PVSWBUlyxy1Boz3v6_NXUc36IBbsQumtdYH1ttv8o_1Fr7D13pugYNpb1ge8pXl2nAcLdSwlCxhuKgJYFTZAP5YeNq_wiRCj4uMwf__Ex0IR6PoWXk2QpnChaKfXXyB7lGHgVnn_k1t9VE-oW8bimk2FaRe5tF8kj902RiJA4KQJ2xv3oxPi8fXYSWHQXlyE2kNTKFx90mvLlwgUs8KJSUx9ZNL1YUAO6Dyb9cB6stCm1D0K342zQrFh3t8GM-lsDdmMkLjFfrici4w02u8nWiXrnjKYaHF7vCrem5RsvpVh4Kg5gUfOaHoTpHVV6BVCqw8bd4MTMtDqpY-0HnMFT8_CLGHfBitP6M57iN0RALSD9XblGl7X-LDaP40oMhsw7edzoLX5Kr3kR99D-VAQ3BIjHSsI2lH-bErzeYrVE2_hUG66gZRX9JqSwqeoYtkboZRZ_-7y4Rlze6DoNbG_xYXtX8IVABwk3_M8B-PW4vBt7U3EOZnHKmUNH2vtpdeGgAjj-EbjH4OQ=w1013-h226-no)

### 傳統電腦視覺提取特徵的⽅法

- 為了有更直觀的理解，這裡介紹⼀種最簡單提取特徵的⽅法

  - 如何描述顏⾊？
  - 顏⾊直⽅圖

- 顏⾊直⽅圖是將顏⾊信息轉化為特徵⼀種⽅法，將顏⾊值 RGB 轉為直⽅圖值，來描述⾊彩和強度的分佈情況。舉例來說，⼀張彩⾊圖有 3 個channel， RGB，顏⾊值都介於 0-255 之間，最⼩可以去統計每個像素值出現在圖片的數量，也可以是⼀個區間如 (0 - 15)、(16 - 31)、...、(240 -255)。可表⽰如下圖

  ![](https://lh3.googleusercontent.com/jA_qqEPxxF2wmpYIlWpgg2t3eJpFgjaqfUABumhFitJXkNHbGMrZx4EDr3Muk9LNUej5JFxsNa9AXsgURxjnCRY80_qJsqg6YTKg1SjIPC4p9RnSHtyu8OpBJuw9KJiO7uhLTUtx5zGrMs37LgAMvGI6QuCeKgnkWBusUspqS6isUMdWia4hYoQ-IAwYoqh7IK1RpYFQtoZ8vR_xxMTMh34mTGUz0r39y1EE2gU4aFu0bkGnbcvTkCTauVDJM1VKhcq6As8nwV--SYa7sqPWqmq7apdgVQhrq_JVJVhPEQQqqAU5DtZ3s2OpvGXOdbXp0Q8kFs71WSxFHZqRY6Q73FNUCTfvzM0pPMvhDuqM2nQfoMVwRXfI-uUhcFIhLRgHErG4y65FOuZk_Mu6uR5Eyg_MZV1Kc-GjHSTBUB0HsGzBBw01FGIdW1N8yTDfr5bJVPAq3bbmThP35SQmJBFjWlHFhWklwi0q-LxVJD8yiCkUQwTvRbHWdmY9EdFXmsA8_NTfeKWYNmik39wZiljHDQs49KwZ9DAhsqR8P5AWmG9nW1Kn8x_z8DZgl-ibiIpd3S_hNx2JTQEKNBzsU1NzlzXp70iXmt0CGLlmf_lk-2HqByuS7D37GAdaQ-rJqPXhyXT_dAf4wxM-5jyZGFBWGBVhvQ5Z9zj5yKYZf8aOZSNlVIHS4hbLt8WT7RE67kHOw9rJdHQe4jJb6KMT7XsZ8OhN=w623-h485-no)

### 重要知識點複習

- 傳統影像視覺描述特徵的⽅法是⼀個非常「⼿⼯」的過程， 可以想像本⽇知識點提到的顏⾊直⽅圖在要辨認顏⾊的場景就會非常有⽤
- 但可能就不適合⽤來做邊緣檢測的任務，因為從顏⾊的分佈沒有考量到空間上的信息。
- 不同的任務，我們就要想辦法針對性地設計特徵來進⾏後續影像辨識的任務。

### 參考資料

1. [圖像分類 | 深度學習PK傳統機器學習](https://cloud.tencent.com/developer/article/1111702)
2. [OpenCV - 直方圖](https://chtseng.wordpress.com/2016/12/05/opencv-histograms直方圖/)
3. [OpenCV 教學文檔](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
4. [Udacity free course: Introduction To Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)

## 傳統電腦視覺與影像辨識_HOG

> - 體驗使⽤不同特徵來做機器學習分類問題的差別
> - 知道 hog 的調⽤⽅式
> - 知道 svm 在 opencv 的調⽤⽅式

- 嘗試比較⽤ color histogram 和 HOG 特徵分類 cifar10 準確度各在 training 和 testing data 的差別

### 重要知識點複習

- 靠⼈⼯設計的特徵在簡單的任務上也許是堪⽤，但複雜的情況，比如說分類的類別多起來，就能明顯感覺到這些特徵的不⾜之處
- 體會這⼀點能更幫助理解接下來的卷積神經網路的意義。

### 參考資料

1. [Sobel 運算子 wiki](https://zh.wikipedia.org/wiki/索貝爾算子)
2. [基於傳統圖像處理的目標檢測與識別(HOG+SVM附代碼)](https://www.cnblogs.com/zyly/p/9651261.html)
3. [知乎 - 什麼是 SVM](https://www.zhihu.com/question/21094489)
4. 程式碼範例的來源，裡面用的是 mnist 來跑[https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py]
   - [範例來源裡使用的 digit.png 檔案位置](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png)
5. [支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489)

6. [第十八节、基于传统图像处理的目标检测与识别(HOG+SVM附代码)](https://www.cnblogs.com/zyly/p/9651261.html)





