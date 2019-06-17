


from scipy import stats

# Function to remove outliers from X & y 
def remove_outliers(features_df,target_df,z_score=2.75):
    X = features_df
    y = target_df
    if len(X) != len(y):
        print(f"Error: Number of samples in feature and target dataframe are not same : {len(X)} and {len(y)} ")
        return
    mask = (np.abs(stats.zscore(X)) < z_score).all(axis=1)
    print(f"Removed {len(X) - len(X[mask])} outlier samples out of {len(X)} from feature and target dataframes")
    return X[mask],y[mask]

# Eg: train_X_or, train_y_or = remove_outliers(train_X, train_y)
