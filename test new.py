import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA, PCA


fps = 60
source = 'rPPGVideo.xlsx'

df = pd.read_excel(source, index_col=0)
df_list = df.values.tolist()

S = np.c_[df_list[0], df_list[1], df_list[2]]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

fchgvbjh