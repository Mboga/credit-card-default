# mlflow for MLOPS
import os
import mlflow
import xgboost as xgb
from pathlib import Path
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from mlflow.models.signature import infer_signature
#use logging

# read train, val test csv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def data_loader(data_path):
    datasets ={}

    for split in ['train','val','test']:
        file_path = data_path/f"{split}.csv"

        if not file_path.exists():
            raise FileNotFoundError("Missing Expected File: {file_path} ")
        
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {split}.csv with {df.shape[0]} rows and  {df.shape[1]} columns")

        # data quality checks: columns: check for column consistency against the training set
        if split == 'train':
            expected_columns = df.columns.to_list()
        else:
            if not df.columns.equals(pd.Index(expected_columns)):
                missing = set(expected_columns)- set(df.columns)
                extra = set(df.columns) - set(expected_columns)
                raise ValueError(f"Column mismatch in {split}.csv! \nMising: {missing} \nExtra: {extra}")

        datasets[split] =df

    return datasets['train'],datasets['val'],datasets['test']


# To do: create module
def extract_features(raw_data):
    raw_data = raw_data.copy()

    # 1. Payment-to-Bill Ratios 
    # We use np.where to handle the -1 case specifically
    denom_1 = raw_data['BILL_AMT1'] + 1
    raw_data['pay_bill_ratio_1'] = np.where(denom_1 != 0, raw_data['PAY_AMT1'] / denom_1, 0)
    
    # For the average, we calculate the denominator first
    avg_bill_denom = raw_data[['BILL_AMT1','BILL_AMT2','BILL_AMT3']].mean(axis=1)
    # If the average bill is 0, the ratio is 0. No more .replace(0, 1) needed.
    raw_data['pay_bill_ratio_avg'] = np.where(avg_bill_denom != 0, 
                                             raw_data[['PAY_AMT1','PAY_AMT2','PAY_AMT3']].mean(axis=1) / avg_bill_denom, 
                                             0)

    # 2. Payment Delay Score 
    delay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    raw_data['total_delays'] = raw_data[delay_cols].clip(lower=0).sum(axis=1)
    raw_data['recent_delays'] = raw_data[['PAY_0','PAY_2']].clip(lower=0).sum(axis=1)

    #  3. Credit Utilization
    denom_util = raw_data['LIMIT_BAL'] + 1
    raw_data['utilization'] = np.where(denom_util != 0, raw_data['BILL_AMT1'] / denom_util, 0)

    # 4. Payment Momentum 
    raw_data['pay_trend'] = raw_data['PAY_AMT6'] - raw_data['PAY_AMT1'] 

    # 5. Demographic Risk 
    raw_data['age_sex_risk'] = raw_data['AGE'] * (raw_data['SEX'] == 2) 

    return raw_data


def prepare_model_data(df, target_col='default', drop_cols = ['ID']):
    
    #extract features
    df_processed = extract_features(df)

    # Separate target
    y = df_processed[target_col].copy()

    # drop unwanted columns, ID
    X = df_processed.drop(columns=[target_col] + drop_cols, errors ='ignore')

    return X,y