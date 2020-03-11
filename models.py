from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, validation_curve, \
    learning_curve
from numpy import sqrt, cumsum, round, arange, logspace
import numpy as np
from pipeline import apply_pipeline
from utils import calculate_mean_rsme
from os.path import join as path_join
import logging
from matplotlib import rc
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from textwrap import wrap

logging.root.setLevel(logging.INFO)
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
from utils.plotting import plot_residual, plot_learning_curve

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)

params = {
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'figure.autolayout': True,
}
plt.rcParams.update(params)


def train_linear_regression_using_mean_rmse(data: DataFrame, x_col: list, y_col: list):
    """
    A function to train a baseline linear regression model using out of sample rmse
    :param data: the dataset
    :param x_col: the x inputs
    :param y_col: the y outputs
    :return: the model, score, residual plot
    """
    model, score, figure = calculate_mean_rsme(data, x_col, y_col, LinearRegression())
    logging.info(f'Out of sample rmse Linear Regression score {score}')
    figure.show()
    return model, score, figure


def train_regression_using_standardisation(data, x_col, y_col, model):
    """
    A function to train a linear regression model with standardisation
    :param model:
    :param data: the dataset
    :param x_col: the x inputs
    :param y_col: the y outputs
    :return:
    """
    data, x, y = apply_pipeline(data, x_col, y_col)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    fig, ax = plot_residual(y_test.to_numpy(), prediction, 'Linear Regression using Standardisation')
    fig.show()
    return model, sqrt(mean_squared_error(y_test, prediction)), fig


def apply_pca(x, y, variance_threshold: float = 0.95):
    """
    A function to apply principle component analysis
    :param variance_threshold: the variance threshold
    :param x: the x inputs
    :param y: the y outputs
    :return:
    """
    pca = PCA(variance_threshold)
    x_reduced = pca.fit_transform(x)
    variance: list = cumsum(round(pca.explained_variance_ratio_, decimals=4) * 100)
    return variance, x_reduced, y


def train_linear_regression_pca(x_col, y_col, variance_threshold: float = 0.95):
    """
    A linear regression model trained on PCA
    :param variance_threshold: the variance to keep whilst transforming the data
    :param data: the dataset
    :param x_col: the x inputs
    :param y_col: the y outputs
    :return:
    """
    variance, x_reduced, y = apply_pca(x_col, y_col, variance_threshold)
    kf_10 = KFold(n_splits=5, shuffle=True, random_state=2)
    regr = LinearRegression()
    scores: list = []
    benchmark = cross_val_score(LinearRegression(), x_col, y, cv=kf_10, scoring='r2').mean()
    for i in arange(1, len(variance)):
        score = cross_val_score(regr, x_reduced[:, :i], y, cv=kf_10,
                                     scoring='r2').mean()
        scores += [score]
        logging.info(f'Adding {i} dimension. Previous Score: {scores[-1]}')
    fig, ax1, = plt.subplots()
    ax1.plot([benchmark for _ in range(len(variance))], color='r')
    ax1.plot(scores, '-v')
    for ax in fig.axes:
        ax.set_xlabel('Number of principal components in regression')
        ax.set_ylabel('R2')
    return fig, ax1


def estimate_ridge(data, x_col, y_col, n_alphas=200):
    data, x, y = apply_pipeline(data, x_col, y_col)
    kf_10 = KFold(n_splits=5, shuffle=True, random_state=2)
    alphas = logspace(-10, -2, n_alphas)

    scores = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=True)
        score = -1 * cross_val_score(ridge, x, y, cv=kf_10,
                                     scoring='r2').mean()
        scores += [score]
    fig, ax1, = plt.subplots()
    ax1.plot(scores, '-v')
    for ax in fig.axes:
        ax.set_xlabel('alpha')
        ax.set_ylabel('R2')
    fig.show()


def fit_random_forrest_regressor(x, y) -> tuple:
    """
    A function to evaluate a regression model and plot a learning curve
    :param x: the x inputs
    :param y: the y the y inputs
    :return:
    """
    kf_10 = KFold(n_splits=5, shuffle=True, random_state=2)
    regr = RandomForestRegressor()
    train_sizes, train_scores, test_scores = learning_curve(regr, x, y, cv=kf_10,
                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                            scoring='r2', n_jobs=4)
    figure, axis = plt.subplots(1, 1, figsize=(10, 10))
    axis = plot_learning_curve(train_scores, test_scores, axes=axis, title='Random Forrest Learning Curve')
    return regr, figure, axis


# data = read_csv(path_join('./', 'data', 'train.csv'))
# issparse(data.to_numpy())
# # print(data['critical_temp'].describe())
# # data[['entropy_Density', 'entropy_ElectronAffinity', 'entropy_FusionHeat', 'entropy_ThermalConductivity',
# #       'entropy_Valence', 'entropy_atomic_mass']].hist()
# # plt.show()
# # # train_linear_regression_pca(data, data.columns.difference(['critical_temp']), ['critical_temp'])
# # # train_regression_using_standardisation(data, data.columns.difference(['critical_temp']), ['critical_temp'],
# # #                                        LinearRegression(normalize=True))
# # # model, error, fig = train_regression_using_standardisation(data, data.columns.difference(['critical_temp']), ['critical_temp'],
# # #                                        RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], scoring='neg_mean_squared_error')
# #
# # # estimate_ridge(data, data.columns.difference(['critical_temp']), ['critical_temp'])
# data, x, y = apply_pipeline(data, data.columns.difference(['critical_temp']), ['critical_temp'])
# model, figure, axis = fit_random_forrest_regressor(x, y)
# figure.show()
# # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
