# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:11:12 2023

@author: Gaspar Avit Ferrero
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
import pickle

# from fbprophet import Prophet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# --- Read and preprocess data --- #

# Read data
dataset = pd.read_csv('data.csv', index_col=0)

# Drop duplicates
dataset.drop_duplicates(inplace=True)

# Drop item name
dataset.drop(['item_name'], axis=1, inplace=True)

# Change data types
dataset = dataset.rename(columns={"day": "date"})
dataset.date = pd.to_datetime(dataset.date)
dataset.orders_quantity = dataset.orders_quantity.astype(int)
dataset.sales_quantity = dataset.sales_quantity.astype(int)

# Impute missing data
imputer = SimpleImputer(missing_values=np.nan,
                        strategy='constant', fill_value=0)
dataset.revenue = imputer.fit_transform(
    np.array(dataset['revenue']).reshape(-1, 1))


# --- Feature Engineering --- #

# Sales features
dataset['item_profit'] = (
    dataset.suggested_retail_price - dataset.purchase_price).round(2)

# Time features
dataset['year'] = dataset.date.dt.year
dataset['month'] = dataset.date.dt.month
dataset['week'] = dataset.date.dt.isocalendar().week
dataset.week = dataset.week.astype(int)
dataset['day'] = dataset.date.dt.day
dataset['quarter'] = dataset.date.dt.quarter
dataset['day_of_year'] = dataset.date.dt.dayofyear
dataset['day_of_week'] = dataset.date.dt.dayofweek
dataset["weekend"] = dataset.date.dt.dayofweek > 4


# --- Model --- #

# Develop one model for each item
items = dataset.item_number.unique()

for item in items:
    df_item = dataset[dataset.item_number == item].copy()

    print("\nItem ", item, ":")

    # Sort by date and then drop it
    df_item.sort_values(by=['date'], inplace=True)
    df_item.drop(['date'], axis=1, inplace=True)

    # Define features and target
    target = pd.concat([pd.Series([0]), df_item['sales_quantity'].iloc[1:]])
    # features = df_item.drop('sales_quantity', axis=1)
    features = df_item.copy()

    # Splits
    train_size = int(0.8 * len(df_item))
    val_size = int(0.1 * len(df_item))

    X_train = features[:train_size]
    y_train = target[:train_size]

    X_val = features[train_size:train_size+val_size]
    y_val = target[train_size:train_size+val_size]

    X_test = features[train_size+val_size:]
    y_test = target[train_size+val_size:]

    # XGBoost model
    # Tuning parameters - using default metrics
    params = {'max_depth': 5, "booster": "gbtree",
              'objective': 'reg:squarederror'}
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    # Training the model
    xgboost = xgb.train(params, dtrain, 1000, evals=watchlist,
                        early_stopping_rounds=100, verbose_eval=False)

    # Making predictions for test set
    preds = xgboost.predict(dtest)

    # RMSE of model
    rms_xgboost = sqrt(mean_squared_error(y_test, preds))
    print("Root Mean Squared Error for XGBoost:", np.round(rms_xgboost, 3))
    mae_xgboost = sqrt(mean_absolute_error(y_test, preds))
    print("Mean Absolute Error for XGBoost:", np.round(mae_xgboost, 3))

    # Plot feature importances
    feature_important = xgboost.get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    feature_importances = pd.DataFrame(data=values, index=keys, columns=[
        "score"]).sort_values(by="score", ascending=False)
    feature_importances.nlargest(10, columns="score").plot(
        kind='barh')  # plot top 10 features

    # Train model with maximum available data
    dtrain = xgb.DMatrix(pd.concat([X_train, X_val]),
                         pd.concat([y_train, y_val]))
    dval = xgb.DMatrix(X_test, y_test)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    # Training the model
    xgboost = xgb.train(params, dtrain, 1000, evals=watchlist,
                        early_stopping_rounds=100, verbose_eval=False)

    # Save model
    if not os.path.exists('./saved_models/'):
        os.makedirs('./saved_models/')

    pickle.dump(xgboost, open('./saved_models/' + str(item) + '.pkl', "wb"))
