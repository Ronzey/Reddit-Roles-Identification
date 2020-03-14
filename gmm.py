
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse

root_path = 'D:/Research/roleIdentification/dataset'
month = '2019_01'
author_rep_path = '{}/authors_rep/authors_{}_95.csv'.format(root_path,month)



df = pd.read_csv(author_rep_path)
df_without_header = df.ix[0:,1:]
X = df_without_header.values
#a = np.isnan(X)
pd.DataFrame(X).fillna(0, inplace=True)
#b = np.isnan(X)

gmm = GMM(n_components=10).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 4], X[:, -3], c=labels, s=40, cmap='viridis')
plt.show()

#probs = gmm.predict_proba(X)
#print(probs[:5].round(3))

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Eclipse"""
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    #Draw
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
gmm = GMM(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)

# BIC AIC
n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()

print('1')

#
