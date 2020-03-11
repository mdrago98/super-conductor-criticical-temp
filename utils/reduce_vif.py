from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas import DataFrame


class ReduceVIF(BaseEstimator, TransformerMixin):
    """
    A pipeline object that removes values having a high VIF.
    """
    def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):
        self.thresh = thresh

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        return self

    def transform(self, X, y=None):
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=10.0):
        """
        A function to calculate the vif of all columns in the frame.
        Adapted from https://stats.stackexchange.com/a/253620/53565
        :param X: the inputs
        :param thresh: the threshold over 10 should be removed
        :return:
        """
        dropped = True
        X = DataFrame(X)
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X