#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

# Name of the model
output_file = f'model.bin'

# Function responsible for loading the CSV data
def dataset_loading(file):
    df = pd.read_csv(file)
    return df

# Function responsible for cleaning the entire dataset
def dataset_cleaning(df):
    df = df.drop(['job_id'], axis = 'columns')
    
    df['title_lenght'] = (np.where(df['title'].isnull(), 0, df['title'].str.len())).astype('int64')
    df['location_available'] = np.where(df['location'].isnull(), 'not_available', 'available')
    df['department_available'] = np.where(df['department'].isnull(), 'not_available', 'available')
    df['salary_range_available'] = np.where(df['salary_range'].isnull(), 'not_available', 'available')
    df['company_profile_available'] = np.where(df['company_profile'].isnull(), 'not_available', 'available')
    df['company_profile_lenght'] = (np.where(df['company_profile'].isnull(), 0, df['company_profile'].str.len())).astype('int64')
    df['description_available'] = np.where(df['description'].isnull(), 'not_available', 'available')
    df['description_lenght'] = (np.where(df['description'].isnull(), 0, df['description'].str.len())).astype('int64')
    df['requirements_available'] = np.where(df['requirements'].isnull(), 'not_available', 'available')
    df['requirements_lenght'] = (np.where(df['requirements'].isnull(), 0, df['requirements'].str.len())).astype('int64')
    df['benefits_available'] = np.where(df['benefits'].isnull(), 'not_available', 'available')
    df['benefits_lenght'] = (np.where(df['benefits'].isnull(), 0, df['benefits'].str.len())).astype('int64')
    df = df.drop(['title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits'], axis = 'columns')

    df.fillna("not_available",inplace=True)

    cols = list(df.dtypes[df.dtypes == 'object'].index)
    for c in cols:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    return df


# Load CSV data
df = dataset_loading('fake_job_postings.csv')
# Clean data
df = dataset_cleaning(df)

# Define variables
numerical = ['telecommuting', 'has_company_logo', 'has_questions', 'company_profile_lenght', 'description_lenght',
            'requirements_lenght', 'benefits_lenght', 'title_lenght']
categorical = ['location_available', 'department_available', 'salary_range_available', 'company_profile_available',
               'description_available', 'requirements_available', 'benefits_available', 'employment_type',
               'required_experience', 'required_education', 'industry', 'function']         
target = ['fraudulent']

# Define dataset (60-20-20)
# Full dataset (80-20)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.fraudulent.values
del df_full_train['fraudulent']

full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(full_train_dicts)
features = dv.get_feature_names()
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)

# Prepare params for xgboost model
xgb_params = {
    'eta': 0.3,
    'max_depth': 10,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Train the model
model = xgb.train(xgb_params,
                  dfulltrain,
                  num_boost_round=175)


# Export the model and DV to bin file
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f'The Model is saved to {output_file}')





