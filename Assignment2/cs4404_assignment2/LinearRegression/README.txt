1. Introduction
In this exercise, you will implement linear regression and get to see it work on real data.

The file stockprices.csv contains the dataset we will use, it consists of stock prices of Amazon and Google.

The main.py program has been set up to load this data for you.  This program also makes calls to functions predict(), computeCost() and gradientDescent() that you will write.   You are only required to modify these functions, no other part of the program.

You are encouraged to discuss these programming assignments with other students. However, you must write the code yourself - do not look at any source code written by others and do not share your source code with others.

After completing the exercise, you can submit your solution for grading.  You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.

Note well: all of the parameters passed into functions are numpy arrays.  Be sure to use numpy arrays in your own code, and avoid using numpy matrices.  Unfortunately, the numpy array data type and the numpy matrix data type behave very differently, and in a confusing way that can lead to hard to find bugs, so do be careful !

2. Plotting the data
As a first step it is usually a good idea to plot the data to see how it looks.  In main.py the data is loaded using:
    data=np.loadtxt('stockprices.csv',usecols=(1,2))
    X=data[:,0]
    y=data[:,1]

and then plotted using:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X, y, label='Training Data')
    ax.set_xlabel('Amazon')
    ax.set_ylabel('Google')
    ax.set_title('Google stock price vs Amazon')
    fig.savefig('graph.png')

Run the main.py program.  The above code will save the plot in a file called graph.png, which will appear as a tab on the left-hand pane once the program has been run:


3. Linear Regression
The objective of linear regression is to minimise the cost function:
where m is the number of data points and prediction h_θ is:
We can use gradient descent to find the values of the parameters θ.  Each gradient descent iteration uses the update
 Note that we add an additional first "feature" to input x and set it 1.

In main.py this additional column is added by the line:

Xtrain = np.column_stack((np.ones((m, 1)), X))


We also rescale the data so that the values lies between 0 and 1, to help with the numerical stability of the calculations.  This is done by the code:

(Xt,Xscale) = normaliseData(Xtrain)

(yt,yscale) = normaliseData(ytrain)


3.1 Computing the prediction
Your first coding task is to add code to complete the function predict() in main.py.  This function takes as input a set of data point(s) X (X is a vector if there is more than one data point) and parameter vector θ.  It outputs the prediction(s) h_θ(x) for these input(s).   Try to avoid using for loops since these are inefficient in python.  And remember to use numpy arrays rather than matrices.

As a check, for input X=[1,1] and θ=[1,2] the prediction should be 3.  For input X=[[1,1],[5,5]] (i.e two input points [1,1] and [5,5]) the prediction should be [3,15].

3.2 Computing the cost
Your next coding task is to add code to complete the function computeCost() in main.py.   This function takes the data X and y as input and also the vector θ.  It returns the value of cost function J(θ).  Again, try to avoid using for loops and remember to use numpy arrays rather than matrices.

As a check, for θ = (0,0) the cost should be calculated to be 0.318 when you run main.py.

3.3 Computing the gradient
Your third coding task is to add code to complete the function computeGradient().  This function takes X, y and θ as input and returns the gradient of the cost J(θ).

As a check, for θ = (0,0) the gradient should be calculated to be (-0.79,-0.59) when you run main.py

3.4 Putting it together
The function gradientDescent() iteratively updates θ by calling computeGradient() so as to find the parameters which minimise the cost function J(θ).    You should make sure that you understand how this code works.

Run main.py and for this dataset gradientDescent() should return approximately the values θ= (0.34,  0.61).   The program will also plot the predictions using these parameter values on the plot in pred.png, and the result should look something like this:
As a check that the gradient descent update is working as expected its also a good idea to plot out how the cost function changes at each update - it should decrease.   The program plots this in file cost.png, which should look something like this:

3.5 Plotting the cost function
To help you understand the cost function J(θ) better, the program also plots the cost over a 2-dimensional grid of θ values (remember θ is a vector).   This plot is shown in J.png.

The purpose of this plot is to show you that how J(θ) varies with changes in θ_0 and θ_1. The cost function J(θ) is bowl-shaped and has a global minimum.  This minimum is the optimal point for θ_0 and θ_1, and each step of gradient descent moves closer to this point.

You should now submit your assignment for marking.


4. Optional Exercises (ungraded)
The following exercises are optional/ungraded, but you are strongly encouraged to complete them.

4.1 Choosing the learning rate
In main.py the learning rate α is fixed at 0.02 in function gradientDescent().   How does the changing the learning rate α affect the result ?

Try values of the learning rate α on a log-scale, at multiplicative steps of about 3 times the previous value (e.g., 0.3, 0.1, 0.03, 0.01 and so on). You may also want to adjust the number of iterations you are running if that will help you see the overall trend in the curve.

Notice the changes in the convergence of the cost in cost.png as the learning rate changes. With a small learning rate, you should find that gradient descent takes a very long time to converge to the optimal value.  Conversely, with a large learning rate, gradient descent might not converge or might even diverge.

4.2 Multiple inputs/features
Previously, you implemented gradient descent for a regression problem with a single input/feature. Now your task is to modify the code in functions computeCost() and computeGradient() to allow use of any number of inputs/features i.e. which take inputs X with two or more columns (remember the first column is always all ones).

You can test your code using the data set housing.csv.  This data consists of housing prices, the first column is the area of the house (in square feet), the second is the number of bedrooms, and the third column is the price of the house in US dollars.   Use the first two columns as the inputs/features for predicting the sale price.   Its a good idea to modify the plotting code to produce a 3D plot of price vs area and number of bedrooms so that you can visualise the training data, and also the predictions to check that they look reasonable.

4.3 Compare with sklearn
Compare the predictions you obtain using your own code with those obtained using the sklearn library for python.   The relevant documentation for sklearn is here:

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

The following code calls sklearn to fit a linear regression model:

from sklearn import linear_model

model = linear_model.LinearRegression()

model.fit(data[:,0].reshape(-1,1),data[:,1].reshape(-1,1))

print(model.coef_)


Note that sklearn automatically normalises the data and add a column of ones, so we don't need to do that.

Now generate some predictions and plot them:


Xtest_sk = np.linspace(data[:,0].min(), data[:,0].max(), 100)

ytest_sk = model.predict(Xtest_sk.reshape(-1,1)).flatten()

ax.plot(Xtest_sk, ytest_sk, 'r', label='Prediction')

fig.savefig('pred_sk.png')


How does this compare with your own code's predictions ?  It should be much the same.

4.4 Evaluating the accuracy of predictions
One simple way to estimate how good the predictions made by our model are likely to be is to split our data into two parts: one part (the training data) used for training/estimating the parameters θ and the other (the test data) for evaluating the accuracy of the prediction.

Modify the function splitData() so that 90% of the data is used for training and the remaining 10% for testing.  Then rerun main.py.  The plot in pred.png should change to show the test data only, and the corresponding model predictions.

How does the way we split the data into training and test parts affect things ?  Some possibilities:

    We might select the first (or last) 90% of the data points as training data.
    We might select 90% of the data uniformly at random (with or without replacement) to use as training data, and use the remainder as test data.


