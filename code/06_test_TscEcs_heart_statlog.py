import numpy as np
import benchmarks
import TscEcs as cut
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = np.loadtxt(fname='data_UCI/heart_statlog.data', delimiter=' ')
X, labels_true = data[:, 0:13], data[:, 13]

high_dim_data = X

scaler = StandardScaler()
data_scaled = scaler.fit_transform(high_dim_data)

pca = PCA(n_components=5)
low_dim_projection = pca.fit_transform(data_scaled)

X = low_dim_projection

start = time.time()
labels_pred, comps = cut.tscEcs(X, global_scale=1.2, alpha=1, beta=1, gama=1, iter_max_num=2, allow_outliers=True, verbose=False)
end = time.time()

print('heart_statlog:')
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)

benchmarks.write_plot('results/test_TscEcs_heart_statlog.png', plt)
plt.show()