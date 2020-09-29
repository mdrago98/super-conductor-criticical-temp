from pandas import read_csv
from os.path import join as path_join

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from models import fit_random_model_with_cv, estimate_lin_with_reg
from pipeline import apply_pipeline
from matplotlib import pyplot as plt
from numpy import sqrt
from utils.plotting import plot_residual

data = read_csv(path_join('./', 'data', 'train.csv'))
data, x, y = apply_pipeline(data, data.columns.difference(['critical_temp']), ['critical_temp'])

lin_reg_model, figure, axis = fit_random_model_with_cv(x, y, title='Linear regression Learning Curve')
figure.show()
figure.savefig('figs/linearlearningcurve.png')

ridge_estimate, figure, axis = estimate_lin_with_reg(x, y)
figure.suptitle('Cross validation scores against different alpha for regularisation')
figure.show()
figure.savefig('figs/alpharidge.png')

ridge_model, figure, axis = fit_random_model_with_cv(x, y, model=Ridge(alpha=200, fit_intercept=True), title='Ridge regression Learning Curve')
figure.show()
figure.savefig('figs/Ridgelearningcurve.png')

# lasso_estimate, figure, axis = estimate_lin_with_reg(x, y, Lasso)
# figure.suptitle('Cross validation scores against different alpha for lasso regularisation')
# figure.show()
# figure.savefig('figs/alphalasso.png')
#
# lasso_model, figure, axis = fit_random_model_with_cv(x, y, model=Lasso(alpha=200, fit_intercept=True), title='Lasso regression Learning Curve')
# figure.show()
# figure.savefig('figs/lassolearningcurve.png')
# print(lasso_model.coefs)

random_forest, figure, axis = fit_random_model_with_cv(x, y, model=RandomForestRegressor(), title='Random forest regression Learning Curve')
figure.show()
figure.savefig('figs/randomforestlearningcurve.png')

data_test = read_csv(path_join('./', 'data', 'test.csv'))
data_test, x_test, y_test = apply_pipeline(data_test, data.columns.difference(['critical_temp']), ['critical_temp'])
y_test = y_test.to_numpy()
figure, ax = plt.subplots(3, figsize=(10, 10))
lin_reg_pred = lin_reg_model.predict(x_test)
print(f'Linear Regression RMSE {sqrt(mean_squared_error(y_test, lin_reg_pred))}, R2: {r2_score(y_test, lin_reg_pred)}')
plot_residual(y_test, lin_reg_pred, 'Linear Regression', (figure, ax[0]))
ridge_pred = ridge_model.predict(x_test)
print(f'Ridge Regression RMSE {sqrt(mean_squared_error(y_test, ridge_pred))}, R2: {r2_score(y_test, ridge_pred)}')
plot_residual(y_test, ridge_pred, 'Ridge Regression', (figure, ax[1]))
random_forest_pred = random_forest.predict(x_test)
print(f'Random Forest Regression RMSE {sqrt(mean_squared_error(y_test, random_forest_pred))}, R2: {r2_score(y_test, random_forest_pred)}')

plot_residual(y_test, random_forest_pred, 'Random Forest Regression', (figure, ax[2]))
figure.show()
figure.savefig('figs/residual.png')
