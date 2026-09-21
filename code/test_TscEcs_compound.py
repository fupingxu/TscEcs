import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data/compound.txt', delimiter=',')
X, labels_true = data[:, [0, 1]], data[:, 2]

start = time.time()
# labels_pred, comps = cut.tscEcs(X, global_scale=1.3, alpha=0, beta=0, gama=0.7, iter_max_num=1, allow_outliers=True, verbose=True)
labels_pred, comps = cut.tscEcs(X, global_scale=1.3, alpha=0, beta=0, gama=0.7, iter_max_num=1, allow_outliers=True, verbose=False)
end = time.time()
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/test_TscEcs_compound.png', plt)
plt.show()
exit()
'''
cut.tscEcs(X, global_scale=1.3, alpha=0, beta=0, gama=0.7, iter_max_num=1, allow_outliers=True, verbose=True)
Dice    Jaccard Precision       Recall
0.983   0.967   0.980           0.986
'''
