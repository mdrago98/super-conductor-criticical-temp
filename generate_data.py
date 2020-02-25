from sklearn.model_selection import train_test_split
from os.path import join as path_join
from pandas import read_csv
import random

SEED = 42
TEST_SIZE = 0.2

data_dir = path_join('./', 'data', 'all.csv')
data = read_csv(data_dir)
X, y = data[data.columns.difference(['critical_temp'])], data['critical_temp']
random.seed(SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

training = X_train.join(y_train)[list(data.columns)].to_csv(path_join('./', 'data', 'train.csv'), index=False)
testing = X_test.join(y_test)[list(data.columns)].to_csv(path_join('./', 'data', 'test.csv'), index=False)
