import numpy as np
import matplotlib.pyplot as plt


def distance(X, mu):
    # calculate the euclidean distance between numpy arrays X and mu
    (m, n) = X.shape
    d = np.zeros(m)
    ##### insert your code here #####
    d = np.sum(np.square(X - mu), axis=1)
    return d


def findClosestCentres(X, mu):
    # finds the centre in mu closest to each point in X
    (k, n) = mu.shape  # k is number of centres
    (m, n) = X.shape  # m is number of data points
    C = list()
    for j in range(k):
        C.append(list())
        ##### insert your code here #####
    d = (np.square(mu[:, None] - X).sum(axis=2).T).argmin(axis=1)
    for i in range(d.size):
        C[d[i]].append(i)
    return C


def updateCentres(X, C):
    # updates the centres to be the average of the points closest to it.
    k = len(C)  # k is number of centres
    (m, n) = X.shape  # n is number of features
    mu = np.zeros((k, n))
    ##### insert your code here #####
    for i in range(k):
        mu[i] = np.sum((X[C[i]]), axis=0) / len(C[i])
    return mu


def plotData(X, C, mu):
    # plot the data, coloured according to which centre is closest. and also plot the centres themselves
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[C[0], 0], X[C[0], 1], c='c', marker='o')
    ax.scatter(X[C[1], 0], X[C[1], 1], c='b', marker='o')
    ax.scatter(X[C[2], 0], X[C[2], 1], c='g', marker='o')
    # plot centres
    ax.scatter(mu[:, 0], mu[:, 1], c='r', marker='x', s=100, label='centres')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    fig.savefig('graph.png')


def main():
    print('testing the distance function ...')
    print(distance(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [1, 2]])))
    print('expected output is [0,8]')

    print('testing the findClosestCentres function ...')
    print(findClosestCentres(np.array([[1, 2], [3, 4], [0.9, 1.8]]), np.array([[1, 2], [2.5, 3.5]])))
    print('expected output is [[0,2],[1]]')

    print('testing the updateCentres function ...')
    print(updateCentres(np.array([[1, 2], [3, 4], [0.9, 1.8]]), [[0, 2], [1]]))
    print('expected output is [[0.95,1.9],[3,4]]')

    print('loading test data ...')
    X = np.loadtxt('data.txt')
    [m, n] = X.shape
    iters = 10
    k = 3
    print('initialising centres ...')
    init_points = np.random.choice(m, k, replace=False)
    mu = X[init_points, :]  # initialise centres randomly
    print('running k-means algorithm ...')
    for i in range(iters):
        C = findClosestCentres(X, mu)
        mu = updateCentres(X, C)
    print('plotting output')
    plotData(X, C, mu)


if __name__ == '__main__':
    main()