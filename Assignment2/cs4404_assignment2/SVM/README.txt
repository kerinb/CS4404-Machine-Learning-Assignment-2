1. Introduction
In this assignment you will implement a support vector machine classifier for the health data set used previously, and compare with use of logistic regression.

2. Support Vector Machine
Recall that for a SVM the prediction is
for parameter vector θ and input x.  Also, the cost function is:
and the gradient of the cost is a vector where the jth element is:

2.1 Implementing an SVM
As usual, your first task is to complete the code in functions predict(), computeCost() and computeGradient() that implement h_θ(x), J(θ) and dJ/dθ_j respectively.   Make sure that these functions work for any number of features and data points.   Try to avoid using for loops since these are inefficient in python.  And remember to use numpy arrays rather than numpy matrices.

Use the same code for plotData(), computeScore() and addQuadraticFeature() as for the previous logistic regression assignment (just cut and paste it).

2.2 Choosing 𝜆
The cost J(θ) depends on parameter 𝜆, and so the value of 𝜆 affects the values of θ that minimise J(θ).  Try some values for 𝜆 and see how they affect the value of θ obtained after running gradient descent e.g. try values of 0, 0.01, 0.1, 1, 10.  Keep an eye on the plot in cost.png to make sure that the gradient descent is converging, and adjust α if necessary.

You may find that the results are quite sensitive to the value of 𝜆, and also to the learning rate α, the initial value for θ and the number of iterations used in gradientDescent().

You should find that θ=[3.08,2.97,3.69,-5.36] gives an accuracy of around 78%, and similarly for other θ values in the same ratio.  Observe that this is much the same accuracy as when we used a logistic regression model - it's often the case that the two models produce similar accuracy when using the same features since the only difference between them is in the cost function used.

2.3 Using sklearn to train the SVM
While gradient descent works for training an SVM, it can be fiddly to tune the learning rate α etc to obtain good convergence.  Packages such as sklearn therefore usually use more robust methods for training the model.   Try this out using the following code to replace the call to gradientDescent():

from sklearn import svm

model = svm.SVC(C=1,kernel='linear')

model.fit(Xt,y)

theta=np.concatenate((model.intercept_,(model.coef_.ravel())[1:4]))

cost=[1,2] # to avoid plotting error


Make sure you understand what this code is doing.  The parameter C in the sklearn implementation corresponds to 1/𝜆 in your implementation.

2.4. Using kernels with an SVM
Your next task is to explore the use of a Gaussian kernel with the SVM.

The Gaussian kernel is a similarity function:
that measures the “distance” between a pair of examples, (x(i),x(j)). The Gaussian kernel is also parameterized by a bandwidth parameter σ which determines how fast the similarity metric decreases (to 0) as the examples are further apart.   For example:
Using a kernel in the SVM is a different way of allowing a nonlinear decision boundary.

Add the following code to the end of function main() in main.py:

model = svm.SVC(C=0.5, gamma=0.75,kernel='rbf')

model.fit(Xt,y)

plotDecisionBoundary2(Xt,y,Xscale,model)

preds = model.predict(Xt)

score = computeScore(Xt,y,preds)


This code trains an SVM with a Gaussian kernel (also referred as as an RBF kernel) and parameters C=1/𝜆=0.5 and gamma=1/σ^2=0.75.  It then calls plotDecisionBoundary2() which calculates the decision boundary and plots it in file pred2.png.  Make sure that you fully understand the code above, and also the code in plotDecisionBoundary2().   Why could we not use function plotDecisionBoundary() ?

Run main.py with this new code.  The output in pred2.png should look something like the following:
and the accuracy should be reported as around 78% (similar to when a quadratic boundary was used).

2.5. Selecting C and gamma (𝜆 and σ) using cross-validation
In the last section we manually selected C=1/𝜆=0.5 and gamma=1/σ^2=0.5.   In general, when 𝜆 is too small (C is too large) there is a risk of overfitting the data, so obtaining poor prediction accuracy when presented with new data, and when 𝜆 is too large (C is too small) there is a risk of underfitting the data and again obtaining poor predictive accuracy.    Similarly for gamma.

Since these are important parameters, how can we choose their values more systematically ?

A common approach is to loop through a range of values for C and gamma and for each value we estimate the predictive accuracy using cross-validation.  We select the values of C and gamma which give the best predictive accuracy.  The predictive accuracy is the accuracy of our model when presented with new data (i.e. different from the training data).  We can estimate this using cross-validation.  That is, by randomly splitting the data that we have into a part used for training the model and a part used for testing the model, training the model and measuring its accuracy and then repeating this several times (i.e. for several different partitions of the data).    We can let sklearn do this for us using its cross_val_score() function, see:

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

Your task is to write code that (i) scans over a range of values of C and gamma, (ii) calls cross_val_score() to estimate the prediction accuracy for each pair of values, (iii) plots this data (a 3D plot is needed since we have two values C and gamma plus the corresponding accuracy score.  Then use this code to select appropriate values for C and gamma.

Here is an example of the sort of plot you should generate
From this plot we can see that choosing C=7 and gamma=0.2 is probably reasonable.  With these values the accuracy is reported as 80%.

You should now submit your assignment for marking.

Solution code is given below (this task is ungraded), but its best to have a good try and writing code yourself before looking at the solutions.  And, of course, if using the solution code then make sure that you take the time to understand it fully.

Solution code:

from sklearn.model_selection import cross_val_score

C_s, gamma_s = np.meshgrid(np.logspace(-2, 1, 20), np.logspace(-2, 1, 20))

scores = list()

i=0; j=0

for C, gamma in zip(C_s.ravel(),gamma_s.ravel()):

    model.C = C

    model.gamma = gamma

    this_scores = cross_val_score(model, Xt, y, cv=5)

    scores.append(np.mean(this_scores))

scores=np.array(scores)

scores=scores.reshape(C_s.shape)

fig2, ax2 = plt.subplots(figsize=(12,8))

c=ax2.contourf(C_s,gamma_s,scores)

ax2.set_xlabel('C')

ax2.set_ylabel('gamma')

fig2.colorbar(c)

fig2.savefig('crossval.png')




3. Optional Exercise: Sentiment analysis (ungraded)
In this exercise we'll build an SVM that takes movie review text and classifies it as either having a positive or negative sentiment towards the movie.

We'll use a subset of the Polarity data set for this, which is based on review text from the IMDb web site.  This text has already been pre-processed to clean it up and has been manually labelled as being either positive or negative.  For example, the initial text from one negative review is:

    that's exactly how long the movie felt to me.  there weren't even nine laughs in nine months.  it's a terrible mess of a movie starring a terrible mess of a man , mr. hugh grant

and here's a positive one:

    every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling .

3.1 Loading the data
The full data set is available at  http://www.cs.cornell.edu/people/pabo/movie-review-data/, but we'll use a subset consisting of 800 reviews that is contained in file polar_data.json.  This data (review text plus label) has been transformed to json format to make it easier to work with.   Use the following python code to read it:

import json

f = open('polar_data.json', 'r')
d=json.load(f)
X=d['data']
y=d['labels']


and to see an example review use:

print('sentiment:', 'positive' if y[1]>0 else 'negative')

print('review text:\n%.210s'%X[1])


which should give output:

sentiment: positive
review text:
every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling .

The next step is to split the data into training and test sets.  We can use sklearn to do this using:

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)


which holds 10% of the data back for testing and uses 90% for training.  See http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

3.2 Mapping the review text to a feature vector

Each review is a list of words, and reviews contain different numbers of words, so we can't use these directly with an SVM.   We need to transform each review into a fixed-size numerical vector.  There are several approaches for mapping text to vectors (so-called vectorization) and in this exercise we'll use what's called a "bag of words" model.

We first construct a dictionary D consisting of all the words appearing in the review data.   We then prune this dictionary by removing very frequent words (such as "a", "the", "and") since these are usually not very informative and perhaps also removing very rare words.

For each review we now construct a vector of length |D| which has element x_i set equal to the number of times the i'th word in dictionary D appears in the review.   For example, suppose D=["terrible","mess","man","action"] and the review text is "a terrible mess of a movie starring a terrible mess of a man".  Then the vector associated with the review is [2,2,1,0] since the word "terrible" occurs twice, the word "mess" occurs twice, the work "man" appears once and so on.  We do not include common words such as "a" and "of" in the dictionary and so these do not contribute to the vector.

Note that each review is described by word occurrences while completely ignoring the relative position information of the words in the review text.  Hence why this is called a bag of words model.  Alternative approaches based on n-grams can retain more information on word position but at the cost of greater complexity (the feature vector length can get much larger!)

We can use sklearn to do this work for us using the following code:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
Xtrain = vectorizer.fit_transform(Xtrain)
Xtest = vectorizer.transform(Xtest)


The parameter stop_words='english' causes words such as "a" and "of" to be pruned from the dictionary.  The parameter max_df=0.8 parameter causes words which appear in more than 80% of documents to also be pruned.  For more details see:

http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

When applied to the review data, the above code results in a dictionary that contains around 25000 words.  We can see the resulting vector associated with some text using for example:

print(vectorizer.transform(["a terrible mess of a movie starring

 a terrible mess of a man , mr. hugh grant"]))


which outputs something like the following:

  (0, 9746)	1
  (0, 10884)	1
  (0, 14173)	2
  (0, 14737)	1
  (0, 21332)	1
  (0, 22496)	2

Since the vector is so large (>25K elements) only the 6 non-zero elements are listed, with their location within the vector and the value.

3.3 Training an SVM classifier

Now that we have training data consisting of inputs X and labels y we can use sklearn to train and SVM:

from sklearn import svm
model = svm.SVC(C=1.0,kernel='linear')
model.fit(Xtrain, ytrain)


To evaluate the models predictive performance we use the test data:

preds  = model.predict(Xtest)


We could use your function computeScore() to compute the accuracy of these predictions, but to gain some practice with sklearn we'll use their metrics.classification_report() function:


from sklearn.metrics import classification_report
print(classification_report(ytest, preds))


The output should look something like this:

                   precision    recall  f1-score   support
         -1       0.86      0.69      0.76        35
          1       0.79      0.91      0.85        45

avg / total   0.82      0.81      0.81        80

For more information on precision and recall see, for example:
https://en.wikipedia.org/wiki/Precision_and_recall.

The above performance data says that given review text the model predicts the right sentiment about 80% of the time, which is quite good for a simple classifier which has not been tuned.  We used parameters C=1.0 and max_df=0.2, but you should use cross-validation to decide on an appropriate values for these parameters.

You might also like to try the classifier out on your own review text e.g, scraped from the Rotten Tomatoes web site.



