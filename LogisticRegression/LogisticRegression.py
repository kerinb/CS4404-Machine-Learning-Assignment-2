import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
    fig, ax = plt.subplots(figsize=(12, 8))

    positive = y > 0
    negative = y < 0
    ax.scatter(X[positive, 0], X[positive, 1], c='b', marker='o', label='Healthy')
    ax.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label='Not Healthy')
    ax.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label='Not Healthy')

    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')
    fig.savefig('graph.png')


def predict(X, theta):
    # calculates the prediction h_theta(x) for input(s) x contained in array X
    pred = np.array(np.sign(np.array(np.dot(X, theta))))
    return pred


def computeCost(X, y, theta):
    # function calculates the cost J(theta) and returns its value
    m = 1 / y.size
    cost = sum(np.log(1 + np.exp(-y * np.dot(X, theta)))) * m
    return cost


def computeGradient(X, y, theta):
    # calculate the gradient of J(theta) and return its value
    exponent_value = np.exp(-y * np.dot(X, theta))
    (m, n) = X.shape

    grad = np.array([sum(-y * X[:, i] * (exponent_value / (1 + exponent_value))) / m for i in range(0, n)])
    return grad


def gradientDescent(X, y):
    # iteratively update parameter vector theta

    # initialize variables for learning rate and iterations
    alpha = 0.1
    iters = 10000
    cost = np.zeros(iters)
    (m, n) = X.shape
    theta = np.zeros(n)

    for i in range(iters):
        theta = theta - alpha * computeGradient(X, y, theta)
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def normaliseData(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return (x / scale, scale)


def addQuadraticFeature(X):
    # Given feature vector [x_1,x_2] as input, extend this to
    # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
    ##### insert your code here #####
    x = X[:, 0] * X[:, 0]
    # X = np.append(X, x**2, axis=1)
    X = np.column_stack((X, x))
    return X


def computeScore(X, y, preds):
    # for training data X,y it calculates the number of correct predictions made by the model
    score = 0
    a = 0
    score = sum(np.array([a + 1 for i in range(0, y.size) if y[i] == preds[i]]))
    return score


def plotDecisionBoundary(Xt, y, Xscale, theta):
    # plots the training data plus the decision boundary in the model
    fig, ax = plt.subplots(figsize=(12, 8))
    # plot the data
    positive = y > 0
    negative = y < 0
    ax.scatter(Xt[positive, 1] * Xscale[1], Xt[positive, 2] * Xscale[2], c='b', marker='o', label='Healthy')
    ax.scatter(Xt[negative, 1] * Xscale[1], Xt[negative, 2] * Xscale[2], c='r', marker='x', label='Not Healthy')
    # calc the decision boundary
    x = np.linspace(Xt[:, 2].min() * Xscale[2], Xt[:, 2].max() * Xscale[2], 50)
    if (len(theta) == 3):
        # linear boundary
        x2 = -(theta[0] / Xscale[0] + theta[1] * x / Xscale[1]) / theta[2] * Xscale[2]
    else:
        # quadratic boundary
        x2 = -(theta[0] / Xscale[0] + theta[1] * x / Xscale[1] + theta[3] * np.square(x) / Xscale[3]) / theta[2] * \
             Xscale[2]
    # and plot it
    ax.plot(x, x2, label='Decision boundary')
    ax.legend()
    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')
    fig.savefig('pred.png')


def main():
    # load the training data
    data = np.loadtxt('health.csv')
    X = data[:, [0, 1]]
    y = data[:, 2]
    X = addQuadraticFeature(X)

    # plot it so we can see how it looks
    plotData(X, y)

    # add a column of ones to input data
    m = len(y)
    Xt = np.column_stack((np.ones((m, 1)), X))
    (m, n) = Xt.shape  # m is number of data points, n number of features

    # rescale training data to lie between 0 and 1
    (Xt, Xscale) = normaliseData(Xt)

    # calculate the cost when theta is zero
    print('testing the prediction function ...')
    theta = np.arange(1, n + 1)
    print('when x=[1,1,1] and theta is [1,2,3]) predictions = ', predict(np.ones(n), theta))
    print('when x=[-1,-1,-1] and theta is [1,2,3]) prediction = ', predict(-np.ones(n), theta))
    print('approx expected predictions are 1 and -1')
    input('Press Enter to continue...')
    print('testing the cost function ...')
    theta = np.zeros(n)
    print('when theta is zero cost = ', computeCost(Xt, y, theta))
    print('approx expected cost value is 0.693')
    input('Press Enter to continue...')

    # calculate the gradient when theta is zero
    print('testing the gradient function ...')
    print('when theta is zero gradient = ', computeGradient(Xt, y, theta))
    print('approx expected gradient value is [-0.024,-0.037,-0.049]')
    input('Press Enter to continue...')

    # perform gradient descent to "fit" the model parameters
    print('running gradient descent ...')
    theta, cost = gradientDescent(Xt, y)
    print('after running gradientDescent() theta=', theta)
    print(
        'approx expected value is [1.11,2.42,2.29] for a linear boundary, and [3.08,2.97,3.69,-5.36] when using a quadratic boundary (or values with about the same ratio)')

    # plot the prediction
    plotDecisionBoundary(Xt, y, Xscale, theta)

    preds = predict(Xt, theta)
    score = computeScore(Xt, y, preds)
    print('accuracy = {0:.2f}%'.format(score / len(y) * 100))
    print('approx expected value is 68%, increasing to 78% when quadratic boundary is used')

    # plot how the cost varies as the gradient descent proceeds
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.semilogy(cost, 'r')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('cost')
    fig2.savefig('cost.png')


if __name__ == '__main__':
    main()
