from scipy import stats
import numpy as np
from itertools import combinations
import random


# Function to remove outliers from X & y 
def remove_outliers(features_df,z_score=2.75):

    
    X = features_df
    mask = (np.abs(stats.zscore(X)) < z_score).all(axis=1)
    print(f"Removed {len(X) - len(X[mask])} outlier samples out of {len(X)} from feature and target dataframes")
    return X[mask]


# Eg: train_X_or, train_y_or = remove_outliers(train_X, train_y)
def create_ratio(df, col1, col2):
    col_name = f"{col1}_{col2}_ratio"
    df[col_name] = df[col1]/df[col2]
    return col_name

def feature_engineering_create_ratios(df):
    """Do not pass target in the df. It will be scaled."""
    df["Height"] = df["Height (inches)"] * 2.54
    df.drop(["Height (inches)"], inplace=True, axis=1)
    columns = list(df.columns)
    ratio_list = combinations(columns, 2)
    ret = df.copy()
    for col1, col2 in ratio_list:
        create_ratio(ret, col1, col2)
    return ret

def row_augmentation(train_df, num_aug=5):
    df=train_df.copy()
    rand_vals_0 = lambda : 1 + random.randint(1, 10)/100
    rand_vals_1 = lambda : 1 - random.randint(1, 10)/100
    for i in range(5):
        df = df.append(df * random.choice([rand_vals_0, rand_vals_1])())
    return df