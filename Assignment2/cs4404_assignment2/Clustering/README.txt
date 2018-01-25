1.  K-means Clustering
In this assignment, you will implement the k-means clustering algorithm and apply it to compress an image.    You will first start on an example 2D dataset that will hopefully help you gain an intuition of how the k-means algorithm works. After that, you will use the k-means algorithm for image compression.

Recall that the k-means algorithm is a method for automatically clustering similar data examples together.    Pseudo-code for the algorithm is as follows:
This algorithm will always converge to some final set of centres, but the converged solution may not always be ideal and depends on the initial setting of the centres.  Therefore, in practice the algorithm is usually run a few times with different random initialisations.

1.1 Cluster assignment
In the cluster assignment phase of the algorithm we need to find the centre c^(i) that is closest to point x^(i).  Of course we first need to define what we mean by the distance between vectors  c^(i) and x^(i) before we can select the one that is closest.   We will use the Euclidean distance:
i.e. the sum of the square differences of the elements of the vectors.

Your first task is to complete the code in function distance().   This takes as input two 2D numpy arrays X and mu of size k rows and n columns.  The i'th row of X is point x^(i) and the i'th row of mu is c^(i).  The function outputs an array of size k which is the Euclidean distance between each of the k points.  For example, if the inputs are X=[[1,2],[3,4] and mu=[[1,2],[1,2]] then the output is [0, 8].

Your next task is to complete the code in findClosestCentres(). This function takes as input 2D numpy array of data points X and the locations mu of the k centres.   It outputs a list of lists C, with the jth list C[j] containing the indices of the points in X that are closest to centre c^(j).   For example, if the inputs are X=[[1,2],[3,4],[0.9,1.8]] and mu=[[1,2],[2.5,3.5]] then the output is C[0]=[0,2], C[1]=[1].  You can implement this using a loop over every data point and every centre, but as a bonus challenge try to use numpy functions to avoid at least one of these loops.

1.2 Update centres

The second phase of the algorithm recomputes the locations of the centres using:
Here C_j is the set of points assigned to centre j.   For example,  if two points x(3) and x(5) are assigned to centre j = 2, then you should update μ_2 = (x(3) + x(5))/2.

You should now complete the code in updateCentres().  This takes a 2D numpy array X of data points and a list of lists C assigning data points to centres.  It outputs a 2D numpy array with k rows containing the k updated centres.   For example, if the inputs are X=[[1,2],[3,4],[0.9,1.8]] and C=[[0,2],[1]] then the output should be [[0.95,1.9],[3,4]].

You can implement this function using a loop over the centres. You can also use a loop over the rows in X, but try to avoid this as it should make the code faster

1.3  Apply k-means to example dataset
After you have completed these functions run main.py to use them in a applying the k-means algorithm to the data in file data.txt.  This is a toy dataset, for testing purpose.

Do make sure you understand how the code in main.py works. Notice that the code calls the two functions you implemented in a loop.    Also that the code initialise the centres to random points drawn from the data set - this ensures that initially at least one point is close to each centre.

The program main.py also plots the data and the centres found by the k-means algorithm in file graph.png.  The output should look something like this:
You should now submit your assignment for marking.


2.   Optional Exercise: Image compression with k-means  (ungraded)
In the next part of the assignment we apply k-means to compress the following image:
In a 24-bit colour representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values.  An image often contains thousands of colours, e.g the image above contains more than 96,000 different colours.   We will use the k-means algorithm to cluster the colours and so reduce the number of colours to e.g. 16.   Fewer colours mean that the image can be stored more efficiently - just the RGB codes for the 16 colours plus 4 bits for each pixel (to specify which of the 16 colours to use) rather than 24 bits as before, so we expect to use around 6 times less storage.

 2.1 Visualising the data
As usual, we start by visualising our data.  The image is stored in image.txt in as an 427x640x3 array of 8bit values.   Use the following code to read in the image and display it in file image.png:

X=np.loadtxt('image.txt')


# convert to values between 0 and 1

X=np.array(X, dtype=np.float64) / 255


X=X.reshape(427,640,3)


fig, ax = plt.subplots(figsize=(12,8))

ax.axis('off')

ax.imshow(X)

fig.savefig('image.png')


The output should look like the image above.

2.2 Apply k-mean clustering to image
Use X=X.reshape(427*640,3) to convert the 427x640x3 array into a 273280x3 array.  Now run your k-means algorithm on this array X.  Try k=16 centres (corresponding to 16 colours) and iters=5 iterations.

If the code is a bit slow to run (which is quite likely) then an easy way to speed it up is to carry out the clustering on a random subset of the image data e.g. use:

X=X[np.random.choice(273280,1000),:]

to select a subset of 1000 points and then run the k-means algorithm on this subset.  It should be much faster.

The k-means algorithm output the centres mu and list C of centres to which each point in X belongs.  Now use the following code to construct a compressed image and display it:

for j in range(k):

    X[C[j],:]=mu[j,:]

X=X.reshape(427,640,3)

fig, ax = plt.subplots(figsize=(12,8))

ax.axis('off')

ax.imshow(X)

fig.savefig('compressed.png')


The output should look something like this:
The above image uses only 16 colours.  If instead you try k=64 so that the image has 64 colours (so 6 bits for each pixel instead of only 4 bits) then the output should look something like this:







