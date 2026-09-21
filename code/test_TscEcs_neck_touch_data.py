import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data/neck_touch_data.txt', delimiter=',')
X, labels_true = data[:, [0, 1]], data[:, 2]

start = time.time()
# labels_pred, comps = cut.tscEcs(X, global_scale=1, alpha=1, beta=1, gama=1, iter_max_num=4, allow_outliers=True, verbose=True)
labels_pred, comps = cut.tscEcs(X, global_scale=1, alpha=1, beta=1, gama=1, iter_max_num=4, allow_outliers=True, verbose=False)
end = time.time()
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/test_TscEcs_neck_touch_data.png', plt)
plt.show()
'''
cut.tscEcs(X, global_scale=1, alpha=1, beta=1, gama=1, iter_max_num=4, allow_outliers=True, verbose=True)
Dice    Jaccard Precision       Recall
1.000   1.000   1.000           1.000
'''
