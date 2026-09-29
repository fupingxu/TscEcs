import numpy as np
import benchmarks
import TscEcs as cut
import time

data = np.loadtxt(fname='data2/D31.txt', delimiter='\t')
X, labels_true = data[:, [0, 1]], data[:, 2]

start = time.time()
labels_pred, comps = cut.tscEcs(X, global_scale=0.08, alpha=1, beta=1, gama=1, iter_max_num=1, allow_outliers=False, verbose=False)
end = time.time()

print('D31:')
print('Total Process: %.3f (sec)' % (end - start))

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/test_TscEcs_D31.png', plt)
plt.show()
'''
cut.tscEcs(X, global_scale=0.08, alpha=1, beta=1, gama=1, iter_max_num=1, allow_outliers=False, verbose=True)
Dice    Jaccard Precision       Recall
0.725   0.569   0.853           0.631
'''
