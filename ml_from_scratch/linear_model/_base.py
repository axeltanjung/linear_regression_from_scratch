import numpy as np

class LinearRegression:
    '''
    Ordinary least square Linear Regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Wheater to calculate the intercept for this model.
        If set to False, no intercept will be used in calculations
        i.e. data is expected to be centered (manually)

    Attributes
    ----------
    coef_ : array of shape (n_features, )
        Estimated coef for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.
        Set to 0.0 if 'fit_intercept = False'   
    '''

    def __init__(
        self,
        fit_intercept = True
    ):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Normalisasi data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract size
        n_samples, n_features = X.shape

        # Design matrix = A --> [1, x_1, x_2, ...]
        # [1] gabungkan dengan X
        A = np.column_stack((X,
                            np.ones(n_samples)))

        # Model parameter
        theta = np.linalg.inv(A . T @ A)@ A.T @ y
        theta

        # Extract
        if self.fit_intercept:
            self.coef_ = theta[:n_features]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta[:n_features]
            self.intercept_ = 0
        # A.T   (n_feature + 1, n_samples)
        # A     (n_samples, n_feaute + 1)
        # A.T A (n_feature + 1, n_feature + 1)
        # ....  (n_feature + 1, n_samples)
        # y     (n_samples, )
        # theta (n_feature + 1,)

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ---------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : array, shape (n_samples)
            Returns predicted values        
        """
        # Normalisasi input
        X = np.array(X).copy()

        # Masukkan ke fungsi linear
        # y = w.X + b
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred
