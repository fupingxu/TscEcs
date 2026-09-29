import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data_UCI/new_thyroid.data', delimiter=',')
X, labels_true = data[:, 1:6], data[:, 0]

start = time.time()
labels_pred, comps = cut.tscEcs(X, global_scale=0.9, alpha=1, beta=1, gama=1, iter_max_num=3, allow_outliers=True, verbose=False)
end = time.time()

print('new_thyroid:')
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)

benchmarks.write_plot('results/test_TscEcs_new_thyroid.png', plt)
plt.show()