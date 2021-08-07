import sys
sys.path.append('..')

import plotext as plt
from coniferest.datasets import MalanchevDataset, Label

dataset = MalanchevDataset(inliers=100, outliers=10,
                           regions=(Label.R, Label.R, Label.A))

plt.colorless()
plt.plotsize(70, 20)
plt.title('MalanchevDataset(inliers=100, outliers=10, regions=(R,R,A))')
index = dataset.labels == Label.R
plt.scatter(*dataset.data[index, :].T, marker='.', color='none')
plt.scatter(*dataset.data[~index, :].T, marker='*', color='none')
plt.show()
