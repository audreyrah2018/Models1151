import numpy as np
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import datetime
from sklearn.model_selection import StratifiedKFold
import operator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#2. Read/load data set
   
DATA_PATH = r'C:\Users\Davis\Desktop\Audrey'
df = pd.read_csv(DATA_PATH+r'\Myproject1151.csv' , header=0)

#N=df.isnull().sum()

#p=df.dtypes
df.drop(['Original_Quote_Date', 'Field12', 'CoverageField8', 'CoverageField9', 'SalesField7', 'PersonalField16', 'PersonalField17', 'PersonalField18', 'PersonalField19', 'PropertyField5', 'PropertyField7', 'PropertyField14', 'PropertyField28', 'PropertyField30', 'PropertyField31', 'PropertyField33', 'PropertyField37', 'GeographicField64', 'Original_Quote_Date'], axis='columns',inplace=True)
df.drop('Field6', axis=1, inplace=True)
df.drop('Field10', axis=1,inplace=True)
df.drop('GeographicField63', axis=1,inplace=True)

p=df.dtypes

####
features=df.drop('QuoteConversion_Flag',axis=1)
target=df['QuoteConversion_Flag']

X = features
y = target

X_train,X_valid, y_train,y_valid = train_test_split(X,y,shuffle=True,random_state=10, test_size=0.1)
print(X_train.shape,X_valid.shape,len(y_train),len(y_valid))



# 4.2 train a linear model using Perceptron
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score,precision_recall_fscore_support,classification_report,confusion_matrix

model_pc = Perceptron(tol=1e-4, random_state=42,penalty='l2') # initialize perceptron model 
model_pc.fit(X_train,y_train) # train model (learning)
y_pred_pc = model_pc.predict(X_valid)


#4.3 evaluate performance by printing f1 score, confusion matrix and classification report
confusion_matrix(y_valid,y_pred_pc)
f1_score(y_valid,y_pred_pc)
print(classification_report(y_valid,y_pred_pc))


#5. RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=10,n_jobs=-1,max_depth=10,verbose=1)
# must understand how to set parameters in the API


model_rf.fit(X_train,y_train) # train model
y_pred_rf = model_rf.predict(X_valid)

y_pred_rf = model_rf.predict(X_valid)
y_pred_rf_prob = model_rf.predict_proba(X_valid)
y_pred_rf_prob
confusion_matrix(y_valid,y_pred_rf)
f1_score(y_valid,y_pred_rf)
    
#6. Build and train model 
    
X = features
y = target
print(X.shape,len(y))  

X_train,X_valid, y_train,y_valid = train_test_split(X,y,shuffle=True,random_state=10, test_size=0.1)
print(X_train.shape,X_valid.shape,len(y_train),len(y_valid))

X.shape,len(y)

X_train,X_test, y_train,y_test = train_test_split(X,y,shuffle=True,random_state=10, test_size=0.1)
print(X_train.shape,X_test.shape,len(y_train),len(y_test))

X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)



model_lgb = lgb.LGBMClassifier(
            n_jobs=4,
            n_estimators=100000,
            boost_from_average='false',
            learning_rate=0.01,
            num_leaves=64,
            num_threads=4,
            max_depth=-1,
            tree_learner = "serial",
            feature_fraction = 0.7,
            bagging_freq = 5,
            bagging_fraction = 0.7,
            min_data_in_leaf=100,
            silent=-1,
            verbose=-1,
            max_bin = 255,
            bagging_seed = 11,            )

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
    
auc_scores = []
models = [] 
for i, (train_idx,valid_idx) in enumerate(kf.split(X_train,y_train)):
    
    print('...... training {}th fold \n'.format(i+1))
    tr_X = X_train.loc[train_idx]
    tr_y = y_train.loc[train_idx]
    
    va_X = X_train.loc[valid_idx]
    va_y = y_train.loc[valid_idx]
    
    model = model_lgb 
    model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (va_X, va_y)], eval_metric = 'auc', verbose=500, early_stopping_rounds = 50)
    ##I chanegd numbers to finish earlier
        
    pred_va_y = model.predict_proba(va_X,num_iteration=model_lgb.best_iteration_)[:,1]
    auc = roc_auc_score(va_y,pred_va_y)
    print('current best auc score is:{}'.format(auc))
    auc_scores.append(auc)
    models.append(model)
    
    
    
print('the average mean auc is:{}'.format(np.mean(auc_scores)))
fts = X_train.columns.values
len(fts)    
fts_imp = dict(zip(fts,models[0].feature_importances_))
fts_imp = sorted(fts_imp.items(), key=operator.itemgetter(1),reverse=True)
fts_imp[:20]
    
#6 feaure importance
feature_imp = pd.DataFrame({'Value':models[0].feature_importances_,'Feature':X_train.columns})
plt.figure(figsize=(40, 20))
sns.set(font_scale = 5)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[:20])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
plt.show()
        
