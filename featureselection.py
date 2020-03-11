from pandas import read_csv
from os.path import join as path_join

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from models import train_linear_regression_pca
from pipeline import apply_pipeline
from utils.reduce_vif import ReduceVIF
import logging

logging.root.setLevel(logging.INFO)

data = read_csv(path_join('./', 'data', 'train.csv'))
data, x, y = apply_pipeline(data, data.columns.difference(['critical_temp']), ['critical_temp'])

fig, ax = train_linear_regression_pca(x, y)
fig.show()
fig.savefig('figs/pcasteps.png')

vif = ReduceVIF()
x_reduced = vif.fit_transform(x)
kf_10 = KFold(n_splits=5, shuffle=True, random_state=2)
regr = LinearRegression()
vif_score = cross_val_score(regr, x_reduced, y, cv=kf_10,
                            scoring='r2').mean()
benchmark = cross_val_score(regr, x, y, cv=kf_10,
                            scoring='r2').mean()
logging.info(f'Benchmark regression training scores {benchmark} vif_score {vif_score}')
logging.info(f'Benchmark regression generalisation scores {benchmark} vif_score {vif_score}')
