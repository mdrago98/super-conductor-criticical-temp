from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    A helper class to select attributes from a dataframe and split into input data to be used in conjunction
    with a pipeline
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        return self

    def transform(self, inputs):
        return inputs[self.attribute_names].values
