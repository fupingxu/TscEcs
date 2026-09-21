import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data/t8.8k.txt', delimiter=',')
X, labels_true = data[:, [0, 1]], data[:, 2]

start = time.time()
# labels_pred, comps = cut.tscEcs(X, global_scale=1, alpha=1, beta=0, gama=0, iter_max_num=1, allow_outliers=True, verbose=True)
labels_pred, comps = cut.tscEcs(X, global_scale=1, alpha=1, beta=0, gama=0, iter_max_num=1, allow_outliers=True, verbose=False)
end = time.time()
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/test_TscEcs_t8.8k.png', plt)
plt.show()
exit()
'''
cut.tscEcs(X, global_scale=1, alpha=1, beta=0, gama=0, iter_max_num=1, allow_outliers=True, verbose=True)
Dice    Jaccard Precision       Recall
0.979   0.959   0.987           0.972
'''

