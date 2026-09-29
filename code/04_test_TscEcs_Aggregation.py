import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data2/Aggregation.txt', delimiter='\t')
X, labels_true = data[:, [0, 1]], data[:, 2]

start = time.time()
labels_pred, comps = cut.tscEcs(X, global_scale=6, alpha=1, beta=1, gama=1, iter_max_num=1, allow_outliers=False, verbose=False)
end = time.time()

print('Aggregation:')
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/test_TscEcs_Aggregation.png', plt)
plt.show()
'''
cut.tscEcs(X, global_scale=6, alpha=1, beta=1, gama=1, iter_max_num=1, allow_outliers=False, verbose=True)
Dice    Jaccard Precision       Recall
0.856   0.749   0.749           1.000
'''