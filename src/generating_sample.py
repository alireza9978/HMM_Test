import numpy as np

from src.evaluation import evaluate_model_two
from visualization import plot_data_color
from hmmlearn import hmm
from my_hmm.hmmt import GaussianHMMT

np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.33, 0.33, 0.34])
model.transmat_ = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
model.means_ = np.array([[0.0], [4.0], [8.0]])
model.covars_ = np.array([[[0.5]], [[0.5]], [[0.5]]])

# generate data with defined hmm model
X, Z = model.sample(100)
# plot_data_color(X, Z)

# add nan to data
temp_indexes = np.random.choice(np.arange(X.shape[0]), 10)
X[temp_indexes] = np.nan
new_x = np.array(X, copy=True)

# trying to learn a new model with data that contains nan
new_model = GaussianHMMT(n_components=3, covariance_type="full")
new_model.fit_wit_nan(new_x)
y = new_model.predict(new_x)
y = np.array(y)
evaluate_model_two(y, Z, 3)
