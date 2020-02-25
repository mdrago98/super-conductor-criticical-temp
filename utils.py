from pandas import DataFrame
from typing import Callable
from sklearn.linear_model import LinearRegression


def simple_regression_analysis(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame,
                               sort_criteria: Callable, linear_model: Callable = LinearRegression,
                               input_dimensions: int = 1) -> list:
    """
    A function that returns a list of regression models of the cross product between the inputs and outputs.
    :param y_test:
    :param x_test:
    :param y_train:
    :param x_train:
    :param input_dimensions: the size of the inputs
    :param linear_model: the linear model to compute the variations for
    :param sort_criteria: a callable for ranking the
    :return: a list of models ranked from best to worst
    """
    model_variants = [[input_variable, *y_train.columns] for input_variable in x_train.columns]
    models = [
        evaluate_model(fit_model(x_train[variant[:input_dimensions]], y_train[variant[input_dimensions:]], linear_model),
                       x_test[variant[:input_dimensions]], y_test[variant[input_dimensions:]], sort_criteria)
        for variant in model_variants]
    return sorted(models, key=lambda x: x[1])


def fit_model(x_train: DataFrame, y_train: DataFrame, linear_model: Callable) -> LinearRegression:
    """
    A function to fit a linear model on the data
    :param x_train: the x input
    :param y_train: the train
    :param linear_model: the linear model to fit
    :return:
    """
    model = linear_model()
    model.fit(X=x_train, y=y_train)
    return model


def evaluate_model(model: LinearRegression, x_validation: DataFrame, y_true: DataFrame, metric: Callable) -> tuple:
    """
    A function to evaluate a given model given a metric
    :param metric: a callable function to evaluate the model
    :param model: the model callable
    :param x_validation: the x input validation
    :param y_true: the expected output
    :return: a tuple representing the model and it's score
    """
    predictions = model.predict(x_validation)
    error = metric(y_true=y_true, y_pred=predictions)
    return model, error


# data = read_csv(path_join('./', 'data', 'train.csv'))
# X, y = data[data.columns.difference(['critical_temp'])], data['critical_temp']
# features = ['wtd_entropy_atomic_radius', 'entropy_Valence', 'wtd_std_fie', 'std_atomic_radius', 'entropy_atomic_mass',
#             'range_ThermalConductivity', 'entropy_fie', 'std_fie', 'wtd_entropy_FusionHeat', 'wtd_std_atomic_radius',
#             'std_ThermalConductivity', 'wtd_std_ThermalConductivity', 'range_atomic_radius', 'entropy_FusionHeat',
#             'wtd_entropy_Valence', 'range_fie', 'wtd_entropy_atomic_mass', 'entropy_atomic_radius',
#             'number_of_elements']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# simple_regression_analysis(X_train, DataFrame(y_train), X_test, DataFrame(y_test), mean_squared_error)
