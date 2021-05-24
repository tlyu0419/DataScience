## 時間序列模型

- 重視資料的先後順序
- 觀測值之間彼此不獨立
- 不關注變量間的因果關係，而是變量在時間上的發展變化規律

## Prophet套件

- 在Prophet算法裡面，作者同時考慮了季節項、趨勢項、節假日項與剩餘項
  $$
  y(t) = g(t) + s(t) + h(t) + \varepsilon_t
  $$

  - $g(t)$ 表示趨勢項，他表示時間序列在非週期上面的變化趨勢
  - $s(t)$ 表示週期項，或稱為季節項，一般來說是以周或者年為單位
  - $h(t)$ 表示節假日項，表示在當天是否存在節假日
  - $\varepsilon_t$ 表示誤差項或者稱為剩餘項

- Prophet 的算法就是通過擬合這幾個項，然後把他們累加起來得到時間序列的預測值

### 趨勢項$g(t)$

### 季節性趨勢

### 節假日效應

- 在現實環境中，除了週末同樣有很多節假日，而且不同的國家有著不同的假期。在 Prophet 裡面，通過維基百科裡面對各個國家的節假日描述，hdays 收集了各個國家的特殊節假日。除了節假日之外，使用者還可以根據自身的情況來設置必要的假期，例如雙十一網購節、年終大促等等。

- 由於每個節假日對時間序列的影響程度不一樣，例如春節，國慶日則是七天假期。對於勞動節等假期來說則假日較短。因此不同的節假日可以看成相互獨立的模型，並且可以為不同的節假日設置不同的前後窗口值，表示該節假日會影響前後一段時間的時間序列。
- 用數學的語言來說，與第 $i$ 個節假日來說 $D_i$ 表示該節假日的前後一段時間。為了表示節假日效應，我們需要一個相應的指示函數(indicator function)，同時需要一個參數 $k_i$ 來表示節假日的影響範圍。



### 模型擬合(Model Fitting)

在 Prophet 中，用戶一般可以設置以下四種參數，如果不想設置的話，使用 Prophet 默認的參數即可

1. Capacity: 在增量函數為邏輯回歸時需要設置的容量值。

2. Change Points: 通過 n_changepoints 和 changepoint_range 來進行等距的變點設置，也可以通過人工設置的方式來指定時間序列的變點

3. 季節性與節假日: 可以根據實際的業務需求來指定相應的節假日

4. 光滑參數: 

   - $\tau = $ changepoint_prior_scale 可以用來控制趨勢的靈活度
   - $\sigma = $ seasonality_prior_scale 用來控制季節項的靈活度
   - $v=$ holidays prior scale 用來控制節假日的靈活度

   

```python
```





- Ref
  - [Classical Time Series Forecasting Methods in Python (Cheat Sheet)](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)
  - [FACEBOOK 时间序列预测算法 PROPHET 的研究](https://zr9558.com/2018/11/30/timeseriespredictionfbprophet/)





- 方法介紹
- 使用方式
- 成效
- 缺點/限制
- 