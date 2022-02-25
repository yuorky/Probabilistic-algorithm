import sys
import sys
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
import operator
import math

"""part 1"""
def MSE(y, y_pred):
    if len(y) != len(y_pred):
        sys.stderr.write("Error: y and y_pred do not have the same dimension ")

    mse_sum = 0
    for y_item, y_pred_item in zip(y, y_pred):
        mse_sum += (y_item - y_pred_item)**2

    mse = mse_sum/len(y)
    return mse


# to calculate the distance between two points
def l2_distance(x_train, x_test):
    if type( x_train ) is np.ndarray:
        dim = len( x_train )
        dist = 0
        for i in range( dim ):
            dist += (x_train[i] - x_test[i]) ** 2  # O(d)
    else:
        dist = 0
        dist += (x_train - x_test) ** 2

    return math.sqrt(dist)


class KNN():
    # initialization
    def __init__(self, x, y, k):
        assert (k <= len( x )), "k cannot be greater than the # of training samples"
        self.x = x
        self.y = y
        self.k = k

    # predict the test labels for a given datasets
    def predict(self, test_set):
        predictions = []
        for x_test in test_set:  # O(n_test)
            distances = []
            for idx, x_train in enumerate( self.x ):  # O(n_train)
                dist = l2_distance( x_train, x_test )

                # record both the y_label and the distance
                # use distance to sort
                distances.append( (self.y[idx], dist) )
            distances.sort( key=operator.itemgetter( 1 ) )

            v = 0
            for i in range( self.k ):  # O(k)
                v += distances[i][0]
            predictions.append( v / self.k )
        return predictions


"""part 2"""

# load the data
# you can change the file here
X_train_D = genfromtxt('X_train_D.csv', delimiter=',')
y_train_D = genfromtxt('Y_train_D.csv', delimiter=',')
X_test_D = genfromtxt('X_test_D.csv', delimiter=',')
y_test_D = genfromtxt('Y_test_D.csv', delimiter=',')

X_train_E = genfromtxt('X_train_E.csv', delimiter=',')
y_train_E = genfromtxt('Y_train_E.csv', delimiter=',')
X_test_E = genfromtxt('X_test_E.csv', delimiter=',')
y_test_E = genfromtxt('Y_test_E.csv', delimiter=',')


class RidgeRegression():
    def __init__(self, l2_lambda):
        self.l2_lambda = l2_lambda
        self.train_loss = []

    def fit(self, X, y):
        # This is for calulation of b
        X_with_intercept = np.c_[np.ones( (X.shape[0], 1) ), X]
        dimension = X_with_intercept.shape[1]

        # create the Identity matrix
        A = np.identity( dimension )
        # set the first diagonal element to 1 so that
        # do not include the lambda(alpha) for the intercept b
        A[0, 0] = 0

        A_biased = self.l2_lambda * A  # lambda*I
        weights = np.linalg.inv( X_with_intercept.T.dot( X_with_intercept ) + A_biased ).dot( X_with_intercept.T ).dot(
            y )

        # compute the loss
        self.train_loss.append(
            MSE( y, X_with_intercept.dot( weights ) ) + self.l2_lambda * np.dot( weights, weights.T ) )

        self.weights = weights
        return self

    def predict(self, X):
        weights = self.weights
        X_predictor = np.c_[np.ones( (X.shape[0], 1) ), X]
        self.predictions = X_predictor.dot( weights )
        return self.predictions


# Dataset D
clf_LR_D = RidgeRegression(l2_lambda=0)

clf_LR_D.fit(X_train_D, y_train_D)
y_predict_LR_D = clf_LR_D.predict(X_test_D)

KNN_prediction = [None] * 9

for k in range(1,10):
    classifier = KNN(X_train_D, y_train_D, k)
    KNN_prediction[k-1] = classifier.predict(X_test_D)

x_min = min(X_train_D)
x_max = max(X_train_D)

x_range = np.linspace(x_min, x_max, 500)[:, np.newaxis]

LR_D_prediction = clf_LR_D.predict(x_range)

KNN_1_D = KNN(X_train_D, y_train_D, 1)
KNN_1_D_prediction = KNN_1_D.predict(x_range)

KNN_9_D = KNN(X_train_D, y_train_D, 9)
KNN_9_D_prediction = KNN_9_D.predict(x_range)

plt.plot(x_range, LR_D_prediction, label='Linear regression')
plt.plot(x_range, KNN_1_D_prediction, label='KNN (k=1)')
plt.plot(x_range, KNN_9_D_prediction, label='KNN (k=9)')
plt.legend()
plt.title('Solutions for dataset D')
plt.ylabel('Prediction')
plt.xlabel('x value')
plt.savefig('Solutions for dataset D')
plt.show()

mse_knn = [None] *9
for i in range(len(KNN_prediction)):
    mse_knn[i] = MSE(y_test_D, KNN_prediction[i])

mse_lr = MSE(y_test_D, y_predict_LR_D)

plt.plot(np.arange(1,10), mse_knn, label='MSE for KNN')
plt.plot(np.arange(1,10), [mse_lr]*9, label='MSE for linear regression')
plt.legend()
plt.title('Test mean square error')
plt.xlabel('k value')
plt.ylabel('MSE')

plt.savefig('MSE for dataset D')
plt.show()

# Dataset E
clf_LR_E = RidgeRegression( l2_lambda=0 )

clf_LR_E.fit( X_train_E, y_train_E )
y_predict_LR_E = clf_LR_E.predict( X_test_E )

KNN_prediction = [None] * 9

for k in range( 1, 10 ):
    classifier = KNN( X_train_E, y_train_E, k )
    KNN_prediction[k - 1] = classifier.predict( X_test_E )

x_min = min( X_train_E )
x_max = max( X_train_E )

x_range = np.linspace( x_min, x_max, 500 )[:, np.newaxis]

LR_E_prediction = clf_LR_E.predict( x_range )

KNN_1_E = KNN( X_train_E, y_train_E, 1 )
KNN_1_E_prediction = KNN_1_E.predict( x_range )

KNN_9_E = KNN( X_train_E, y_train_E, 9 )
KNN_9_E_prediction = KNN_9_E.predict( x_range )

plt.plot( x_range, LR_E_prediction, label='Linear regression' )
plt.plot( x_range, KNN_1_E_prediction, label='KNN (k=1)' )
plt.plot( x_range, KNN_9_E_prediction, label='KNN (k=9)' )
plt.legend()
plt.title( 'Solutions for dataset E' )
plt.ylabel( 'Prediction' )
plt.xlabel( 'x value' )
plt.savefig( 'Solutions for dataset E' )
plt.show()

mse_knn = [None] *9
for i in range(len(KNN_prediction)):
    mse_knn[i] = MSE(y_test_E, KNN_prediction[i])

mse_lr = MSE(y_test_E, y_predict_LR_E)

plt.plot(np.arange(1,10), mse_knn, label='MSE for KNN')
plt.plot(np.arange(1,10), [mse_lr]*9, label='MSE for linear regression')
plt.legend()
plt.title('Test mean square error')
plt.xlabel('k value')
plt.ylabel('MSE')

plt.savefig('MSE for dataset E')
plt.show()


"""part 3"""

# load the data
# where you can change the path
X_train_F = genfromtxt('X_train_F.csv', delimiter=',')
y_train_F = genfromtxt('Y_train_F.csv', delimiter=',')
X_test_F = genfromtxt('X_test_F.csv', delimiter=',')
y_test_F = genfromtxt('Y_test_F.csv', delimiter=',')

clf_LR_F = RidgeRegression(l2_lambda=0)

clf_LR_F.fit(X_train_F, y_train_F)
y_predict_LR_F = clf_LR_F.predict(X_test_F)

KNN_prediction = [None] * 9

for k in range(1,10):
    classifier = KNN(X_train_F, y_train_F, k)
    KNN_prediction[k-1] = classifier.predict(X_test_F)

mse_knn = [None] *9
for i in range(len(KNN_prediction)):
    mse_knn[i] = MSE(y_test_F, KNN_prediction[i])

mse_lr = MSE(y_test_F, y_predict_LR_F)

plt.plot(np.arange(1,10), mse_knn, label='MSE for KNN')
plt.plot(np.arange(1,10), [mse_lr]*9, label='MSE for linear regression')
plt.legend()
plt.title('Test mean square error')
plt.xlabel('k value')
plt.ylabel('MSE')

plt.savefig('MSE for dataset F')
plt.show()


class KNN():
    # initialization
    def __init__(self, x, y, k):
        assert (k <= len( x )), "k cannot be greater than the # of training samples"
        self.x = x
        self.y = y
        self.k = k

    # predict the test labels for a given datasets
    def predict(self, test_set):
        predictions = []
        dist_list = [None] * len( test_set )
        aa = 0
        for x_test in test_set:  # O(n_test)
            distances = []
            for idx, x_train in enumerate( self.x ):  # O(n_train)
                dist = l2_distance( x_train, x_test )

                # record both the y_label and the distance
                # use distance to sort
                distances.append( (self.y[idx], dist) )
            distances.sort( key=operator.itemgetter( 1 ) )

            v = 0
            for i in range( self.k ):  # O(k)
                v += distances[i][0]
            dist_list[aa] = distances
            aa += 1
            predictions.append( v / self.k )
        return predictions, dist_list


classifier = KNN(X_train_F, y_train_F, 4)
pred, dist = classifier.predict(X_test_F)