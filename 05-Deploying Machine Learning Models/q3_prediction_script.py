#!/usr/bin/env python
# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle


# Variables definition
features = ['tenure', 'monthlycharges', 'contract']
C = 1.0
n_splits = 5
output_file = f'homeworkModel_C={C}.bin'


########################### Function Definition ###########################

def train(df_train, y_train, C=1.0):
    dicts = df_train[features].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model



def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')
    
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:, 1]
    
    return y_pred

###########################################################################


# Load and prepare data from CSV
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

cols = list(df.dtypes[df.dtypes == 'object'].index)

for c in cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# Prepare dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('Validation results:')    
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# Train the model
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values 

auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')



# Load model and DV from bin file
model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
	model = pickle.load(f_in)

with open(dv_file , 'rb') as f_in:
	dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X = dv.transform([customer])

prediction = model.predict_proba(X)[0, 1]
print(f'prediction={prediction}')

