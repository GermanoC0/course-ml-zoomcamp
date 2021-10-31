#!/usr/bin/env python
# coding: utf-8



# Dataset of Pizza Price
# 
# This dataset for beginners for practice
# 
# https://www.kaggle.com/alyeasin/predict-pizza-price
# 
# The model could be used for pizza price prediction.
# 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

# VARIABLES
random_state_value=42
output_file = f'model.bin'


# LOAD THE DATA
df = pd.read_csv('Pizza-Price.csv')


# CLEAN COLUMNS
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = ['restaurant', 'extra_cheeze', 'extra_mushroom', 'size_by_inch',
       'extra_spicy', 'price']




# Convert the Price array from Taka to Euro value
df['price'] = df['price'].astype('float32')
df['price'] = df['price'] * 0.010



# PREPARE DATASETS (60-20-20)

numerical = ['size_by_inch']
categorical = ['restaurant', 'extra_cheeze', 'extra_mushroom', 'extra_spicy']

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state_value)

df_full_train = df_full_train.reset_index(drop=True)

y_full_train = df_full_train.price.values

del df_full_train['price']

full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(full_train_dicts)

features = dv.get_feature_names()
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)


# TRAIN THE MODEL

xgb_params = {
    'eta': 0.3,
    'max_depth': 1,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=10)


# SAVE THE MODEL TO FILE

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The Model is saved to {output_file}')



