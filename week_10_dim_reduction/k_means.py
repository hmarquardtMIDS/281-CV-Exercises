import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

# generate data in two clusters
N = 25  # number of data points
x_pos = np.random.rand(N, 2)
x_neg = -x_pos
X = np.vstack((x_pos, x_neg))
N = 2 * N

# display everything
plt.plot(X[:, 0], X[:, 1], 'o')

Niter = 5  # number of iterations

c1 = X[random.randint(0, N-1), :]  # initial cluster-1 center
c2 = X[random.randint(0, N-1), :]  # initial cluster-2 center

for i in range(Niter):

    # determine class assignment
    m = np.zeros(N)
    for j in range(N):
        if (np.linalg.norm(X[j, :] - c1) < np.linalg.norm(X[j, :] - c2)):
            m[j] = 1
        else:
            m[j] = 2

    # estimate class centers
    ind1 = np.where(m == 1)
    ind2 = np.where(m == 2)
    c1 = np.mean(X[ind1, :], axis=1)[0]
    c2 = np.mean(X[ind2, :], axis=1)[0]

    # display
    plt.scatter(X[:, 0], X[:, 1], c=m, cmap=plt.cm.Paired)  # data
    plt.plot(c1[0], c1[1], 'bo')  # cluster-1 center
    plt.plot(c2[0], c2[1], 'ro')  # cluster-2 center
    plt.show()
    sleep(0.5)
    # clear_output(wait=True)
    clear_output()
    plt.close()
