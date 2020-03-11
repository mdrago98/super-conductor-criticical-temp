from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import DataFrameSelector
from utils.reduce_vif import ReduceVIF


def get_pipeline(inputs) -> Pipeline:
    """
    A function to return a standard pipeline containing a standard scaler
    :param inputs: the input features
    :return:
    """
    return Pipeline([
        ('selector', DataFrameSelector(inputs)),
        ('std_scaler', StandardScaler()),
        # ('vif', ReduceVIF())
        # ('pca', PCA(.95))
    ])


def apply_pipeline(data, x_col, y_col):
    """
    A helper function to apply the pipeline
    :param data: the data
    :param x_col: inputs
    :param y_col: outputs
    :return: the transformed data
    """
    y = data[y_col]
    data = DataFrame(get_pipeline(x_col).fit_transform(data), columns=x_col)
    data['critical_temp'] = y
    x, y = data[x_col], data[y_col]
    return data, x, y
