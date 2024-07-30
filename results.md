## 1. Random Forest
### 1.0 RandomUndersample+SMOTEnc
Train Set:
               precision    recall  f1-score   support

           0       0.96      0.91      0.94     87895
           1       0.42      0.63      0.50      8861

    accuracy                           0.89     96756
   macro avg       0.69      0.77      0.72     96756
weighted avg       0.91      0.89      0.90     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.96      0.91      0.93     22055
           1       0.38      0.58      0.46      2134

    accuracy                           0.88     24189
   macro avg       0.67      0.74      0.69     24189
weighted avg       0.91      0.88      0.89     24189

Target Threshold @ 0.61:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95     22055
           1       0.46      0.44      0.45      2134

    accuracy                           0.91     24189
   macro avg       0.70      0.70      0.70     24189
weighted avg       0.90      0.91      0.90     24189

Default Predictive Equality: 0.60
Target Predictive Equality: 0.53

![RF_CV-smote.png](./imgs/results/RF_rd-smote.png)

### 1.1 NearMiss+SMOTEnc
TIME: 8min 40s

best params for rf: {'randomforestclassifier__n_estimators': 100, 'randomforestclassifier__max_features': 'log2', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__criterion': 'entropy'} 

Train Set:
               precision    recall  f1-score   support

           0       0.98      0.93      0.95     87895
           1       0.53      0.76      0.62      8861

    accuracy                           0.92     96756
   macro avg       0.75      0.85      0.79     96756
weighted avg       0.93      0.92      0.92     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.97      0.93      0.95     22055
           1       0.49      0.73      0.59      2134

    accuracy                           0.91     24189
   macro avg       0.73      0.83      0.77     24189
weighted avg       0.93      0.91      0.92     24189

Target Threshold @ 0.56:
               precision    recall  f1-score   support

           0       0.97      0.95      0.96     22055
           1       0.56      0.65      0.60      2134

    accuracy                           0.92     24189
   macro avg       0.76      0.80      0.78     24189
weighted avg       0.93      0.92      0.93     24189

Default Predictive Equality: 0.62
Target Predictive Equality: 0.54

![RF_CV-smote.png](./imgs/results/RF_nm-smote.png)

### 1.2 IHT(rf)+SMOTEnc (F1)
best params for rf: {'randomforestclassifier__n_estimators': 80, 'randomforestclassifier__max_features': 'log2', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__criterion': 'gini'} 

Train Set:
               precision    recall  f1-score   support

           0       0.99      0.97      0.98    508064
           1       0.32      0.69      0.44      9855

    accuracy                           0.97    517919
    macro avg       0.66      0.83      0.71    517919
    weighted avg       0.98      0.97      0.97    517919

Default Threshold:
               precision    recall  f1-score   support

           0       0.99      0.97      0.98     56407
           1       0.32      0.67      0.43      1140

    accuracy                           0.96     57547
    macro avg       0.66      0.82      0.71     57547
    weighted avg       0.98      0.96      0.97     57547

Target Threshold @ 0.43:
               precision    recall  f1-score   support

           0       0.99      0.95      0.97     56407
           1       0.23      0.73      0.35      1140

    accuracy                           0.95     57547
    macro avg       0.61      0.84      0.66     57547
    weighted avg       0.98      0.95      0.96     57547

Predictive Equality: 0.73

这里很奇怪，使用default threshold进行分类阈值时候甚至有更低的FPR（3%），即更少的误报。

![RF_CV-smote.png](./imgs/results/RF_iht(rf)-smote.png)

### 1.3 IHT(xgboost)+SMOTEnc (F1) 
TIME: 8min 23s

best params for rf: {'randomforestclassifier__n_estimators': 80, 'randomforestclassifier__max_features': 'log2', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__criterion': 'entropy'} 
Train Set:
               precision    recall  f1-score   support

           0       0.99      1.00      0.99     87952
           1       0.99      0.90      0.94      8812

    accuracy                           0.99     96764
   macro avg       0.99      0.95      0.97     96764
weighted avg       0.99      0.99      0.99     96764

Default Threshold:
               precision    recall  f1-score   support

           0       0.99      1.00      0.99     22008
           1       0.98      0.87      0.92      2183

    accuracy                           0.99     24191
   macro avg       0.98      0.93      0.96     24191
weighted avg       0.99      0.99      0.99     24191

Target Threshold @ 0.23:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22008
           1       0.66      0.97      0.78      2183

    accuracy                           0.95     24191
   macro avg       0.83      0.96      0.88     24191
weighted avg       0.97      0.95      0.96     24191

Default Predictive Equality: 0.00
Target Predictive Equality: 0.86

![RF_CV-smote.png](./imgs/results/RF_iht(xgboost)-smote.png)

### 1.4 IHT(LGB)+SMOTEnc (F1)
Train Set:
               precision    recall  f1-score   support

           0       0.99      1.00      1.00     87895
           1       1.00      0.94      0.97      8861

    accuracy                           0.99     96756
   macro avg       1.00      0.97      0.98     96756
weighted avg       0.99      0.99      0.99     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.99      1.00      1.00     22055
           1       1.00      0.92      0.96      2134

    accuracy                           0.99     24189
   macro avg       1.00      0.96      0.98     24189
weighted avg       0.99      0.99      0.99     24189

Target Threshold @ 0.15:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22055
           1       0.67      0.99      0.80      2134

    accuracy                           0.96     24189
   macro avg       0.83      0.97      0.89     24189
weighted avg       0.97      0.96      0.96     24189

Default Predictive Equality: 0.00
Target Predictive Equality: 0.79

![RF_CV-smote.png](./imgs/results/RF_iht(lgb)-smote.png)


### 1.5 IHT(XGB)+GAN
Train Set:
                precision    recall  f1-score   support

           0       0.97      1.00      0.99     87914
           1       1.00      0.97      0.99     87914

    accuracy                           0.99    175828
    macro avg       0.99      0.99      0.99    175828
    weighted avg       0.99      0.99      0.99    175828

Default Threshold:
               precision    recall  f1-score   support

           0       0.97      1.00      0.99     22053
           1       0.99      0.72      0.84      2140

    accuracy                           0.97     24193
    macro avg       0.98      0.86      0.91     24193
    weighted avg       0.98      0.97      0.97     24193

Target Threshold @ 0.11:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22053
           1       0.65      0.96      0.77      2140

    accuracy                           0.95     24193
    macro avg       0.82      0.95      0.87     24193
    weighted avg       0.97      0.95      0.95     24193

Default Predictive Equality: 0.09
Target Predictive Equality: 0.39

![RF_CV-smote.png](./imgs/results/gan(iht)_rf.png)

## 2. XGBoost
### 2.0 RandomUndersample+SMOTEnc
Train Set:
               precision    recall  f1-score   support

           0       0.95      0.93      0.94     87895
           1       0.46      0.56      0.51      8861

    accuracy                           0.90     96756
   macro avg       0.71      0.75      0.73     96756
weighted avg       0.91      0.90      0.90     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.95      0.93      0.94     22055
           1       0.43      0.55      0.48      2134

    accuracy                           0.90     24189
   macro avg       0.69      0.74      0.71     24189
weighted avg       0.91      0.90      0.90     24189

Target Threshold @ 0.60:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95     22055
           1       0.47      0.47      0.47      2134

    accuracy                           0.91     24189
   macro avg       0.71      0.71      0.71     24189
weighted avg       0.91      0.91      0.91     24189

Default Predictive Equality: 0.55
Target Predictive Equality: 0.54

![RF_CV-smote.png](./imgs/results/xgboost_rd-smote.png)

### 2.1 NearMiss+SMOTEnc
Train Set:
               precision    recall  f1-score   support

           0       0.97      0.98      0.98     87895
           1       0.82      0.74      0.78      8861

    accuracy                           0.96     96756
   macro avg       0.90      0.86      0.88     96756
weighted avg       0.96      0.96      0.96     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.97      0.98      0.97     22055
           1       0.75      0.69      0.72      2134

    accuracy                           0.95     24189
   macro avg       0.86      0.83      0.85     24189
weighted avg       0.95      0.95      0.95     24189

Target Threshold @ 0.28:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96     22055
           1       0.61      0.80      0.69      2134

    accuracy                           0.94     24189
   macro avg       0.79      0.87      0.83     24189
weighted avg       0.95      0.94      0.94     24189

Default Predictive Equality: 0.33
Target Predictive Equality: 0.43

![RF_CV-smote.png](./imgs/results/xgboost_nm-smote.png)

上面的图是27000个0类测试集，下面的是22000个0类的
![RF_CV-smote.png](./imgs/results/xgboost_nm-smote2.png)


### 2.2 IHT(rf)+SMOTEnc (F1)

Train Set:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99    508064
           1       0.73      0.70      0.71      9855

    accuracy                           0.99    517919
    macro avg       0.86      0.85      0.85    517919
    weighted avg       0.99      0.99      0.99    517919

Default Threshold:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99     56407
           1       0.64      0.63      0.63      1140

    accuracy                           0.99     57547
    macro avg       0.82      0.81      0.81     57547
    weighted avg       0.99      0.99      0.99     57547

Target Threshold @ 0.09:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     56407
           1       0.24      0.79      0.37      1140

    accuracy                           0.95     57547
    macro avg       0.62      0.87      0.67     57547
    weighted avg       0.98      0.95      0.96     57547

Default Predictive Equality: 0.53
Target Predictive Equality: 0.54

![RF_CV-smote.png](./imgs/results/xgboost_iht(rf)-smote.png)

### 2.3 IHT(xgboost)+SMOTEnc (F1)
TIME: 6min 54s

best params for xgb: {'xgbclassifier__subsample': 0.8, 'xgbclassifier__n_estimators': 150, 'xgbclassifier__min_child_weight': 2, 'xgbclassifier__max_depth': 4, 'xgbclassifier__learning_rate': 0.2, 'xgbclassifier__colsample_bytree': 0.4} 
Train Set:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     87952
           1       1.00      0.97      0.98      8812

    accuracy                           1.00     96764
   macro avg       1.00      0.98      0.99     96764
weighted avg       1.00      1.00      1.00     96764

Default Threshold:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22008
           1       0.99      0.96      0.97      2183

    accuracy                           1.00     24191
   macro avg       0.99      0.98      0.99     24191
weighted avg       1.00      1.00      1.00     24191

Target Threshold @ 0.04:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22008
           1       0.66      0.99      0.79      2183

    accuracy                           0.95     24191
   macro avg       0.83      0.97      0.88     24191
weighted avg       0.97      0.95      0.96     24191

Default Predictive Equality: 0.23
Target Predictive Equality: 0.61

![RF_CV-smote.png](./imgs/results/xgboost_iht(xgboost)-smote.png)

## Insights 1: 
发现使用iht进行下采样后进行smote能对比原始方法（nearmiss+smote）大量减少fpr，即减少将
0类误分类为1的数量，且都是default threshold下就可以达到很低的fpr（小于我们人工设定的target fpr 0.05）

这也导致default下模型评估效果也很好（特别是这个策略下的xgboost，precision和recall都达到0.6以上）

IHT方法倾向于保留那些对分类器来说较难区分的样本，这使得模型在训练过程中更加专注于这些“困难”样本。这可以提高模型的整体鲁棒性，特别是在处理那些容易被错误分类的负类样本时，进而降低FPR。

另一个问题是：使用RF做IHT的分类器只能达到50：1,使用xgb则可以达到10：1,并且之后基于rf的iht采样后的数据使用RF训练或者xgb训练的结果都远不如基于
xgb的iht采样策略。

原因分析：在使用IHT下采样时，XGBoost可能更有效地识别和保留重要的少数类样本，同时去除不重要的多数类样本。这是因为XGBoost在建模时能够更精细地调整权重和样本选择策略，从而使得下采样后的数据更具代表性。
相比之下，随机森林在下采样时可能不如XGBoost那样精细，导致最终下采样的比例和数据质量不如预期，从而影响后续模型的性能。

### 2.4 IHT(xgb)+GAN (F1)

best params for xgb: {'xgbclassifier__subsample': 0.8, 'xgbclassifier__n_estimators': 150, 'xgbclassifier__min_child_weight': 4, 'xgbclassifier__max_depth': 8, 'xgbclassifier__learning_rate': 0.2, 'xgbclassifier__colsample_bytree': 1.0} 


Train Set:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     87914
           1       1.00      1.00      1.00     87914

    accuracy                           1.00    175828
   macro avg       1.00      1.00      1.00    175828
weighted avg       1.00      1.00      1.00    175828

Default Threshold:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22053
           1       1.00      0.95      0.97      2140

    accuracy                           1.00     24193
   macro avg       1.00      0.97      0.99     24193
weighted avg       1.00      1.00      1.00     24193

Target Threshold @ 0.01:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22053
           1       0.66      0.99      0.79      2140

    accuracy                           0.95     24193
   macro avg       0.83      0.97      0.88     24193
weighted avg       0.97      0.95      0.96     24193

Default Predictive Equality: 0.08
Target Predictive Equality: 0.36

![RF_CV-smote.png](./imgs/results/gan(iht)_XGB.png)


## 3. LGB
### 3.0 RandomUndersample+SMOTEnc
Train Set:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95     87895
           1       0.51      0.50      0.51      8861

    accuracy                           0.91     96756
   macro avg       0.73      0.73      0.73     96756
weighted avg       0.91      0.91      0.91     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95     22055
           1       0.47      0.48      0.48      2134

    accuracy                           0.91     24189
   macro avg       0.71      0.72      0.71     24189
weighted avg       0.91      0.91      0.91     24189

Target Threshold @ 0.51:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95     22055
           1       0.48      0.47      0.47      2134

    accuracy                           0.91     24189
   macro avg       0.71      0.71      0.71     24189
weighted avg       0.91      0.91      0.91     24189

Default Predictive Equality: 0.49
Target Predictive Equality: 0.48

![RF_CV-smote.png](./imgs/results/lgb_rd-smote.png)

### 3.1 NearMiss+SMOTEnc
Train Set:
               precision    recall  f1-score   support

           0       0.97      0.98      0.98     87895
           1       0.77      0.72      0.75      8861

    accuracy                           0.95     96756
    macro avg       0.87      0.85      0.86     96756
    weighted avg       0.95      0.95      0.95     96756

Default Threshold:
               precision    recall  f1-score   support

           0       0.97      0.98      0.97     22055
           1       0.73      0.69      0.71      2134

    accuracy                           0.95     24189
    macro avg       0.85      0.83      0.84     24189
    weighted avg       0.95      0.95      0.95     24189

Target Threshold @ 0.31:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96     22055
           1       0.60      0.78      0.68      2134

    accuracy                           0.94     24189
    macro avg       0.79      0.87      0.82     24189
    weighted avg       0.94      0.94      0.94     24189

Default Predictive Equality: 0.38
Target Predictive Equality: 0.49

![RF_CV-smote.png](./imgs/results/lgb_nm-smote.png)

### 3.2 IHT(xgb)+SMOTEnc (F1)
TIME: 13min 9s

best params for lgb: {'lgbmclassifier__subsample': 0.8, 'lgbmclassifier__n_estimators': 140, 'lgbmclassifier__min_child_weight': 1, 'lgbmclassifier__max_depth': 8, 'lgbmclassifier__learning_rate': 0.2, 'lgbmclassifier__colsample_bytree': 0.8} 


Train Set:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     87866
           1       1.00      1.00      1.00      8905

    accuracy                           1.00     96771
    macro avg       1.00      1.00      1.00     96771
    weighted avg       1.00      1.00      1.00     96771

Default Threshold:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22103
           1       0.99      0.95      0.97      2090

    accuracy                           1.00     24193
    macro avg       0.99      0.98      0.99     24193
    weighted avg       1.00      1.00      1.00     24193

Target Threshold @ 0.01:
               precision    recall  f1-score   support

           0       1.00      0.95      0.98     22103
           1       0.66      0.99      0.79      2090

    accuracy                           0.96     24193
    macro avg       0.83      0.97      0.89     24193
    weighted avg       0.97      0.96      0.96     24193

Default Predictive Equality: 0.14
Target Predictive Equality: 0.39

![RF_CV-smote.png](./imgs/results/lgb_iht(xgb)-smote.png)


### 3.3 IHT(lgb)+SMOTEnc (F1)
Train Set:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     87895
           1       1.00      1.00      1.00      8861

    accuracy                           1.00     96756
   macro avg       1.00      1.00      1.00     96756
weighted avg       1.00      1.00      1.00     96756

Default Threshold:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22055
           1       1.00      0.98      0.99      2134

    accuracy                           1.00     24189
   macro avg       1.00      0.99      0.99     24189
weighted avg       1.00      1.00      1.00     24189

Target Threshold @ 0.00:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22055
           1       0.65      1.00      0.79      2134

    accuracy                           0.95     24189
   macro avg       0.83      0.97      0.88     24189
weighted avg       0.97      0.95      0.96     24189

Default Predictive Equality: 0.00
Target Predictive Equality: 0.74


![RF_CV-smote.png](./imgs/results/lgb_iht(lgb)-smote.png)

### 3.4 IHT(xgb)+GAN
Train Set:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     87914
           1       1.00      1.00      1.00     87914

    accuracy                           1.00    175828
   macro avg       1.00      1.00      1.00    175828
weighted avg       1.00      1.00      1.00    175828

Default Threshold:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22053
           1       1.00      0.95      0.97      2140

    accuracy                           1.00     24193
   macro avg       1.00      0.98      0.99     24193
weighted avg       1.00      1.00      1.00     24193

Target Threshold @ 0.01:
               precision    recall  f1-score   support

           0       1.00      0.95      0.97     22053
           1       0.65      0.99      0.79      2140

    accuracy                           0.95     24193
   macro avg       0.83      0.97      0.88     24193
weighted avg       0.97      0.95      0.96     24193

Default Predictive Equality: 0.00
Target Predictive Equality: 0.35

![RF_CV-smote.png](./imgs/results/gan(iht)_lgb.png)