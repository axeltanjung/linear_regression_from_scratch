import numpy as np
import pandas as pd

from ml_from_scratch.model_selection import KFold
from ml_from_scratch.linear_model import LinearRegression

def mean_squarred_error(y_act, y_pred):
    return np.mean((y_act-y_pred)**2)

# Import data
data = pd.read_csv("data/auto.csv")
X = data.drop(columns=["mpg"])
y = data["mpg"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Reset Index
X_train_res = X_train.reset_index(drop=True)
y_train_res = y_train.reset_index(drop=True)

# Train --> Train & Valid --> KFold CV
# Test
# List Experiment
cols_list = [['displacement'],
             ['horsepower'],
             ['weight'],
             ['displacement', 'horsepower'],
             ['displacement', 'weight'],
             ['horsepower', 'weight'],
             ['horsepower', 'weight','displacement']]

# Object Splitting
kf = KFold(n_splits=5)

# KFold CV
mse_train_list = []
mse_valid_list = []
for cols in cols_list:
    mse_train_cols = []
    mse_valid_cols = []
    # Splitting KFold
    for i, (ind_train, ind_test) in enumerate(kf.split(X_train_res)):
        # Extract data
        X_train_ = X_train_res[cols].loc[ind_train]
        y_train_ = y_train_res.loc[ind_train]
        X_valid_ = X_train_res[cols].loc[ind_test]
        y_valid_ = y_train_res.loc[ind_test]

        # Create model
        reg = LinearRegression()
        reg.fit(X_train_, y_train_)

        # Predict
        y_pred_train = reg.predict(X_train_)
        y_pred_valid = reg.predict(X_valid_)

        # Calculate error
        mse_train = mean_squarred_error(y_train_, y_pred_train)
        mse_valid = mean_squarred_error(y_valid_, y_pred_valid)

        mse_train_cols.append(mse_train)
        mse_valid_cols.append(mse_valid)

    # print(f"cols : {cols}")
    # print(mse_train_cols)
    # print(mse_valid_cols)
    # print("")

    mse_train_list.append(np.mean(mse_train_cols))
    mse_valid_list.append(np.mean(mse_valid_cols))

# Write summary
summary_manual_df = pd.DataFrame({"cols" : cols_list,
                                 "MSE_training" : mse_train_list,
                                 "MSE_valid" : mse_valid_list})

print("Manual k-Fold CV")
print(summary_manual_df)
print("")

# Best model
ind_best = summary_manual_df["MSE_valid"].argmin()
col_best = summary_manual_df.loc[ind_best]["cols"]
print(f"Best model feature : {col_best}")
print(f"Best valid score : {summary_manual_df['MSE_valid'].min()}")

# Train the best model
print("")
print("Re-train the best model")
X_best = X_train[col_best]
linreg_best = LinearRegression()
linreg_best.fit(X_best, y_train)

# Predict the test data
X_test_best = X_test[col_best]
y_pred_test = linreg_best.predict(X_test_best)

# Calculate MSE
mse_best = mean_squarred_error(y_test, y_pred_test)
print(f"MSE best model: {mse_best}")