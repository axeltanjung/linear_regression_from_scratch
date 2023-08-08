import numpy as np

from ._base import LinearRegression

class Ridge(LinearRegression):
    """
    Linear least square with L2 regularization.

    Minimizes the objective function:
    ||y - Xw||^2 + alpha * ||w||^2

    Parameters
    ---------
    alpha : float, default=1.0
        Constant that multiples the L2 penalty,
        Controlling the regularization stenght.
        'alpha' must be a non-negative float i.e. [0, inf]

        When 'alpha=0', the objective is equivalent to
        ordinary least squares.

    fit_intercept : bool, default=True
        Wheater to calculate the intercept for this model.
        If set to False, no intercept will be used in calculations
        i.e. data is expected to be centered (manually)

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Estimated coef. for the linear regression problem

    intercept_ : float
        Independent term in linear model.
        Set to 0.0 if 'fit_intercept = False'
    """
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True
    ):
        super().__init__(
            fit_intercept=fit_intercept
        )
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit linear model

        Parameters
        ---------
        X : {array-like} of shape (n_samples, n_features)
            Training Data

        y : {array-like} of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract size
        n_samples, n_features = X.shape

        # Create the design matrix, A
        if self.fit_intercept:
            # Create A
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            # Create A
            A = X

        # Solve for the model parameter
        AT_A = A.T @ A

        if self.fit_intercept:
            alpha_I = self.alpha * np.identity(n_features+1)
            alpha_I[-1, 1] = 0.0 # To exclude intercept being trained as well
        else:
            alpha_I = self.alpha * np.identity(n_features)

        AT_y = A.T @ y
        theta = np.linalg.inv(AT_A + alpha_I) @ AT_y

        # Extract model parameters
        if self.fit_intercept:
            self.coef_ = theta[:n_features]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0