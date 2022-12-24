[TOC]

# Explainable AI

## Permutation Feature Importance

計算當某個feature的值被重新排列後，模型誤差增加多少

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```



### Ref

[All Models are Wrong, but Many are Useful Learning a Variable’s Importance by Studying an Entire Class of Prediction Models Simultaneously](./papers/All Models are Wrong, but Many are Useful Learning a Variable’s Importance by Studying an Entire Class of Prediction Models Simultaneously.pdf)
