[TOC]

# Explainable AI

1. Partial Dependence Plot (PDP)

2. Individual Conditional Expectation (ICE)  

3. Permuted Feature Importance  

4. Global Surrogate  

5. Local Surrogate (LIME)  

6. Shapley Value (SHAP)

## Partial Dependence Plot (PDP)

方便研究人員了解一個或兩個特徵對於模型預測結果的影響狀況

### Ref

- [8.1 Partial Dependence Plot (PDP) | Interpretable Machine Learning (christophm.github.io)](https://christophm.github.io/interpretable-ml-book/pdp.html)



## Individual Conditional Expectation (ICE)

- 與 PDP 相似的目的。PDP 是繪製整體平均的狀況，而 ICE 會顯示個別樣本的結果





## Permutation Feature Importance

透過把 features 的值打亂(用壞)後，觀測對於模型預測結果的影響。如果動到模型認為重要的變數會對模型的預測結果有較大的改變

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

### Ref

- [All Models are Wrong, but Many are Useful Learning a Variable’s Importance by Studying an Entire Class of Prediction Models Simultaneously](https://arxiv.org/pdf/1801.01489.pdf)
- [sklearn.inspection.permutation_importance — scikit-learn 1.2.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance)



## Global Surrogate

- 另外訓練一個可解釋的模型來逼近黑盒模型的預測結果

- 由於另外訓練的模型是以黑盒模型的預測結果作為目標變數，使用時需要留意新訓練的模型的預測結果會多增加一個誤差(相較於直接以答案作為目標變數)
  
  - 額外的誤差可以透過 MAE, R^2 等指標進行測量

## 

## Local Surrogate (LIME)

- LIME（Local Interpretable Model-agnostic Explanations） 是在解釋個別樣本而不是整體模型的變數重要性。透過 LIME 可以觀察到個別樣本被模型預測的重要變數與原因

- Ref
  
  - [marcotcr/lime: Lime: Explaining the predictions of any machine learning classifier (github.com)](https://github.com/marcotcr/lime)

## Shapley Value (SHAP)

- 可以用來解釋個別樣本的預測結果也可以用來解釋整體變數的重要性

- 

- Ref
  
  - [slundberg/shap: A game theoretic approach to explain the output of any machine learning model. (github.com)](https://github.com/slundberg/shap)
  
  - [机器学习模型可解释性进行到底 —— SHAP值理论（一）_悟乙己的博客-CSDN博客_shap原理](https://blog.csdn.net/sinat_26917383/article/details/115400327)
