1. Introduction     
In this exercise, you will implement logistic regression and apply it to two data sets.     
The first data set is health.csv.  This consists of blood measurements (columns 1 and 2) and health status (column 3, 1=heart disease present, -1=not present).     
The second data set consists of the two files digit_images.txt and digit_classes.txt.  The file digit_images.txt consists of pixel values from 1797 images of handwritten digits.         
The file digit_classes.txt contains the label for each image i.e where it shows a 0, 1, 2 etc.      
You are encouraged to discuss these programming assignments with other students. However, you must write the code yourself - do not look at any source code written by others and do not share your source code with others.        
After completing the exercise, you can submit your solution for grading.          
You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.        
Note well: all of the parameters passed into functions are numpy arrays.          
Be sure to use numpy arrays in your own code, and avoid using numpy matrices.         
Unfortunately, the numpy array data type and the numpy matrix data type behave very differently, and in a confusing way that can lead to hard to find bugs, so do be careful !      

2. Visualising the data     
Before starting to implement any learning algorithm, it is always good to visualize the data if possible.       
In main.py the data is loaded using:        
*data=np.loadtxt('health.csv')*     
*X=data[:,[0,1]]*
*y=data[:,2]*                   
The program then calls function plotData(X,y) to plot this data.  
Your first task is to complete the code in function plotData() so that it generates a plot similar to this:
![Alt text](https://storage.googleapis.com/replit/images/1506439430392_6207b8e58ad50bf53d2da000f4424312.png)        
Note that this is an optional (ungraded) exercise.  
We also provide our implementation below so you can copy it or refer to it. If you choose to copy our example, make sure you understand what each of its commands is doing      
*positive = y>0*
*negative = y<0*
*ax.scatter(X[positive,0], X[positive,1], c='b', marker='o', label='Healthy')*
*ax.scatter(X[negative,0], X[negative,1], c='r', marker='x', label='Not Healthy')*

3. Logistic Regression      
Recall that for logistic regression the prediction is:  
![Alt text](https://storage.googleapis.com/replit/images/1506459406918_0498aee95dbe6c0f2b20a9dbdc7e97fe.png)        
for parameter vector θ and input x.  Also, the cost function is:        
![Alt text](https://storage.googleapis.com/replit/images/1506441620448_ab6a733d9052a3e3983b222d6c9b9e85.png)        
and the gradient of the cost is a vector where the jth element is:      
![Alt text](https://storage.googleapis.com/replit/images/1506441954883_fc137b71acaa9606b47b0c331bf80c2f.png)        

    3.1 Implementing logistic regression    
    Similarly to the previous assignment, your task is to complete the code in functions predict(), computeCost() and computeGradient() that implement h_θ(x), J(θ) and dJ/dθ_j respectively.   
    Make sure that these functions work for any number of features and data points.         
    Try to avoid using for loops since these are inefficient in python.         
    And remember to use numpy arrays rather than numpy matrices.        
    As a quick check, for θ = [1,2,3] the cost should be calculated to be 0.693 and the gradient should be calculated to be [-0.024,-0.037,-0.049].
    To help with evaluating the accuracy of the predictions made by the model we need to count the number of correct predictions (i.e. correct 1/-1 labels).  
    Note that this is different from linear regression where we evaluate accuracy using the mean square error since the outputs are real valued.   
    Complete the code in function computeScore(), so that it calculate the number of correct predictions.         
    Try to avoid using for loops.

    3.2 Applying to health data
    Now run main.py using your code.  
    The value of θ learned after gradient descent should be approximately [1.11,2.42,2.29], or values in the same ratio (why do you think its only the ratio that matters in this example ?  
    Hint: take a look at the code in plotDecisionBoundary()).   
    To help visualise the results, the program plots the decision boundary and outputs this is pred.png.  
    Make sure that you fully understand the code in function plotDecisionBoundary().   
    Its output should look something like this: 
    ![Alt text](https://storage.googleapis.com/replit/images/1506442875343_4ebeb4910413cd8c4559bb1d23397902.png)        
    The prediction accuracy for the training data should be calculated to be about 68% i.e. about 68% of the labels of the training data points are correctly predicted by the model.

    3.3 Using a nonlinear decision boundary     
    A prediction accuracy of 68% is relatively low.  
    Looking at the training data it looks as if a curved decision boundary might be better.    
    Let's modify our model to allow us to study this.   
    For each training data point our input/features consist of the value of test1 and of test2 (i.e. columns 1 and 2 in health.csv).   
    To this we add a constant term 1, as usual., so our feature vector is x=[1,test1,test2].   
    We want to modify this to be [1,test1,test2,test1*test1] i.e we add an extra feature test1*test1 to the vector.   
    Your next task is therefore to modify the code in function addQuadraticFeature() to implement this change.  
    As a check, the output when calling addQuadraticFeature(X) with X=[1,2] should now be [1,2,1].  
    And when calling addQuadraticFeature(X) with X=[[1,2],[3,4]] the output should be [[1,2,1],[3,4,9]].    
    The function plotDecisionBoundary() already has the following code to handle the quadratic case, which you should make sure that you understand fully:  
    *x=np.linspace(Xt[:,2].min()*Xscale[2],Xt[:,2].max()*Xscale[2],50)*
    *x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1]+theta[3]*np.square(x)/Xscale[3])/theta[2]*Xscale[2]*
    *ax.plot(x,x2,label='Decision boundary')*   
    Now run main.py using your code.  
    The value of θ learned after gradient descent should be approximately [3.08,2.97,3.69,-5.36] (or any values in the same ratio).     
    The decision boundary plotted in pred.png should look something like this:
    ![Alt text](https://storage.googleapis.com/replit/images/1506460459612_5aa6c762fa2b4ea4dfc85947e7f779dc.png)
    and the prediction accuracy should now be about 78% i.e. about 10% better than when using a linear decision boundary    
    You should now submit your assignment for marking.
    
4. Optional Exercise:        
Build a classifier for handwritten digits (ungraded)
The following exercises are optional/ungraded, but you are strongly encouraged to complete them.        
Our task is to take as input an image of a handwritten digit and predict whether it is a 0,1,2 etc.     
The images are 8x8 blocks of pixels (for simplicity, they have been downsampled).       
We will build up to this via a series of smaller exercises.     
Solution code is given, but its best to have a good try and writing code yourself before looking at the solutions.        
And, of course, if using the solution code then make sure that you take the time to understand it fully.            
The training data is in the two files digit_images.txt and digit_classes.txt.       
The file digit_images.txt consists of pixel values from 1797 images of handwritten digits.  
The file digit_classes.txt contains the label for each image i.e whether it shows a 0, 1, 2 etc.  
The original data is available here:  
  
    http://yann.lecun.com/exdb/mnist/

    4.1 Visualising the training data
    As usual, let's begin by plotting the training data so that we can get a feel for it.

    Use the following code to load the training data:

    import numpy as np  
*n_samples=1797*    
*digit_images = np.loadtxt('digit_images.txt')*         
*digit_images = digit_images.reshape((n_samples, 8, 8))*            
*X = digit_images.reshape((n_samples, -1))*         
*y = np.loadtxt('digit_classes.txt')*       

    After executing this code the variable digits now contains the 1797 images, each image being an 8x8 block of pixels, and the variable y contains the label for each image i.e. 0,1,2 etc.  
For example, digits[1,:,:] is and 8x8 matrix containing the pixel values for the first image and y[1] is the corresponding label for the image.

    Your first task is to plot the first 8 images from the training data and save to the file digits.png.  
    You should obtain something like this:

    ![Alt text](https://storage.googleapis.com/replit/images/1506462632288_920d652118bd03808353001527021e3b.png)

    Solution code:

    *import matplotlib.pyplot as plt*       
    *images_and_labels = list(zip(digit_images, y))*        
    *fig, ax = plt.subplots(2, 4)*      
    *for index, (image, label) in enumerate(images_and_labels[:8]):*        
    *ax[index//4,index%4].axis('off')*                
    *ax[index//4,index%4].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')*       
    *ax[index//4,index%4].set_title('Training: %i' % label)*        
    *fig.savefig('digits.png')*     
    
    4.2 Logistic regression using the sklearn library
    Since the data set is relatively large, we will use the implementation of logistic regression from the sklearn library rather than the one that we developed above, since the sklearn implementation is optimised for speed.
    
    We begin by flattening each 8x8 image into a vector to use as the input/features for our model.  
    The above code already does this using

    *X = digits.reshape((n_samples, -1))*

    The next step is to create an instance of a logistic regression model from the sklearn library and train it.  
    The relevant documentation is available here:

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Notes:

        sklearn automatically adds a column of ones and scales the data, so we don't need to do that.
        It also automatically infers from the values of y that there are 10 classes to learn - you can check this by using print(model.coef_.shape) after the nodel has been trained.  
        This displays the shape of the parameter vector θ and should be (10, 64) i.e for each of the 10 classes there are 64 parameters (one for each pixel/feature in the 8x8 image).
        It's best to split the data into a part used for training the model and a part used for testing the model.  
        Suggested split is 90% for training and 10% for testing.  We can let sklearn do this for us using its train_test_split()function, see http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        sklearn implements a regularised version of logistic regression with quadratic penalty term 1/C*θ^T θ.  
        The weighting parameter C equals 1/Lambda in the notation used in the lectures.  
        Its value defaults to 1.  When evaluating the model (next exercise), do make sure to try a range of values for C to find one that works well.

    Solution code:
 
    *from sklearn import linear_model*
    *model = linear_model.LogisticRegression(C=1)*
    
    *# split the data into training and test parts*
    *from sklearn.model_selection import train_test_split*
    *X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.9)*
    
    *# now train the model ...*
    *model.fit(X_train,y_train)*
    *print(model.coef_.shape)*

    4.3 Evaluating the trained model    
    Now that we have trained the model we can use it to make predictions and evaluate their accuracy.

    Use the sklearn function metrics.classification_report() to measure the accuracy of the model predictions, see:

    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html.

    Your generated output should look something like this:

    *         precision    recall  f1-score   support
    
    *        0.0       0.99      0.97      0.98       158
    *        1.0       0.79      0.85      0.82       161
    *        2.0       0.99      0.89      0.94       167
    *        3.0       0.93      0.91      0.92       169
    *        4.0       0.96      0.98      0.97       163
    *        5.0       0.91      0.96      0.94       163
    *        6.0       0.96      0.94      0.95       159
    *        7.0       0.95      0.99      0.97       155
    *        8.0       0.80      0.75      0.77       164
    *        9.0       0.86      0.89      0.87       159
    *        avg / total       0.91      0.91      0.91      1618

    For more information on precision and recall see, for example:

    https://en.wikipedia.org/wiki/Precision_and_recall.

    You might also consider using the the function confusion_matrix(), see:

    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    which should produce output something like this:

    [[154   0   0   0   1   3   0   0   0   0]     
    [  0 137   0   0   1   1   0   0   9  13]  
    [  0  11 148   0   0   0   0   5   3   0]  
    [  0   1   0 153   0   1   0   1  11   2]  
    [  0   0   0   0 160   0   0   0   3   0]  
    [  0   0   0   1   0 157   1   0   1   3]  
    [  0   4   0   0   2   3 150   0   0   0]  
    [  0   0   0   0   0   0   0 153   0   2]  
    [  1  21   1   3   1   3   6   2 123   3]  
    [  1   0   0   8   1   4   0   0   4 141]]

    Solution code:  

    *from sklearn import metrics*       
    *predicted = model.predict(X_test)*     
    *print(metrics.classification_report(y_test, predicted))*       
    *print(metrics.confusion_matrix(y, predicted))*     
    
    