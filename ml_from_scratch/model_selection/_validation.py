import numpy as np
import copy

from ..model_selection import KFold
from ..metrics import mean_squarred_error

def cross_val_score(
    estimator,
    X,
    y,
    cv=5,
    scoring="mean_squarred_error"
):
    """
    Evaluate a score by Cross Validation

    Parameters
    ---------
    estimator : estimator object implementing "fit"
        The object to use to fit the data

    X : array-like of shape (n_samples, n_features)
        The data to fit

    y : array-like of shape (n_samples,)
        Target variable to try to predict

    scoring : str, default = "mean_squarred_error"
        A scoring function

    cv : int, default=5
        The k of k-fold cross validation
    """
    # Extract data
    X = np.array(X).copy()
    y = np.array(y).copy()

    # Object Splitting
    kf = KFold(n_splits=cv)

    # KFold CV
    scoring = mean_squarred_error[scoring]
    mse_train_list = []
    mse_valid_list = []
    # Splitting KFold
    for i, (ind_train, ind_test) in enumerate(kf.split(X)):
        # Extract data
        X_train = X[ind_train]
        y_train = y[ind_train]
        X_valid = X[ind_test]
        y_valid = y[ind_test]

        # Create model
        mdl = copy.deepcopy(estimator)
        mdl.fit(X_train, y_train)

        # Predict
        y_pred_train = mdl.predict(X_train)
        y_pred_valid = mdl.predict(X_valid)

        # Calculate error
        mse_train = mean_squarred_error(y_train, y_pred_train)
        mse_valid = mean_squarred_error(y_valid, y_pred_valid)

        mse_train_cols.append(mse_train)
        mse_valid_cols.append(mse_valid)

    #     mse_train_list.append(np.mean(mse_train_cols))
    #     mse_valid_list.append(np.mean(mse_valid_cols))

    # # Write summary
    # summary_manual_df = pd.DataFrame({"cols" : cols_list,
    #                                 "MSE_training" : mse_train_list,
    #                                 "MSE_valid" : mse_valid_list})

    # print("Manual k-Fold CV")
    # print(summary_manual_df)
    # print("")

    # # Best model
    # ind_best = summary_manual_df["MSE_valid"].argmin()
    # col_best = summary_manual_df.loc[ind_best]["cols"]
    # print(f"Best model feature : {col_best}")
    # print(f"Best valid score : {summary_manual_df['MSE_valid'].min()}")

    # # Train the best model
    # print("")
    # print("Re-train the best model")
    # X_best = X_train[col_best]
    # linreg_best = LinearRegression()
    # linreg_best.fit(X_best, y_train)

    # # Predict the test data
    # X_test_best = X_test[col_best]
    # y_pred_test = linreg_best.predict(X_test_best)

    # # Calculate MSE
    # mse_best = mean_squarred_error(y_test, y_pred_test)
    # print(f"MSE best model: {mse_best}")