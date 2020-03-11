from pandas import read_csv
from os.path import join as path_join

from pandas.plotting import scatter_matrix
from scipy.sparse import issparse
from matplotlib import pyplot as plt
import logging

from models import train_linear_regression_pca
from pipeline import apply_pipeline

logging.root.setLevel(logging.INFO)

data = read_csv(path_join('./', 'data', 'train.csv'))

logging.info(f'Is matrix sparse {issparse(data.to_numpy())}')
scatter_matrix(data[['entropy_Density', 'entropy_ElectronAffinity', 'entropy_FusionHeat', 'entropy_ThermalConductivity',
                     'entropy_Valence', 'entropy_atomic_mass', 'critical_temp']], figsize=(20, 20))
plt.savefig('figs/scattermatrix.png')
logging.info(f'Applying Standardisation pipeline')
data, x, y = apply_pipeline(data, data.columns.difference(['critical_temp']), ['critical_temp'])
# x.hist(figsize=(20, 20))

data.corr()['critical_temp'].to_csv('correlations.csv')
logging.info(data.corr())


plt.imshow(data.corr(), cmap='hot', interpolation='nearest')
plt.savefig('figs/heatmap.png')
fig, ax = train_linear_regression_pca(x, y)
fig.show()
fig.savefig('figs/pcasteps.png')
# transformer = ReduceVIF()
# x = transformer.fit_transform(x, y)
# print(x.head)

