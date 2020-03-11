from typing import Callable

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from numpy import sqrt

from utils.plotting import plot_residual


def calculate_mean_rsme(all_data, x_cols, y_cols, model, train_size: float = 0.7,
                        iterations: int = 25, error_function: Callable = mean_squared_error) -> tuple:
    """
    A utility function to take the out of sample rmse amongst a sample of test input outputs
    :param error_function:
    :param iterations:
    :param train_size:
    :param model: the model to calculate out of sample rmse
    :return: a tuple representing the model and the mean rmse
    """
    scores = []
    predictions = None
    y_test = None
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(all_data[x_cols], all_data[y_cols], shuffle=True, test_size=1-train_size)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores += [error_function(y_test, predictions)]
    figure, _ = plot_residual(y_test.to_numpy(), predictions, f'Residual Plot')
    return model, sqrt(sum(scores) / iterations), figure


def calculate_rmse(mean_square_error: float) -> float:
    """
    A utility function that converts the mean square error to root mean square
    :param mean_square_error: the mean square error
    :return: the root mean square
    """
    return sqrt(mean_square_error)
