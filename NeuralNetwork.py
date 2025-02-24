import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv('mnist_train.csv') 

data = np.array(data)
m,n = data.shape # number of rows and columns in the data, n is the label column
np.random.shuffle(data) # shuffle the data so that the training and dev data is random

#split the data into training and dev data

data_dev = data[0:1000].T
Y_dev = data_dev[0] #first row of training data 
X_dev = data_dev[1:n] #first column of training data
X_dev = X_dev/255 #normalizing the data

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255
_,m_train = X_train.shape # number of training examples

def init_params():
    W1 = np.random.rand(10,784) - 0.5 # random weights for first layer, 10 neurons, 784 features
    b1 = np.random.rand(10,1) - 0.5 # random bias for first layer and 10 neurons for second layer
    
    W2 = np.random.rand(10,10) - 0.5 # random weights for second layer
    b2 = np.random.rand(10,1) - 0.5 # random bias for second layer
    
    return W1, b1, W2, b2
    

def reLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propagation(W1, b1, W2, b2,X):
    Z1 = W1.dot(X) + b1
    A1 = reLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1)) #creates a matrix of zeroes of size Y.size and Y.max+1 i.e 0-9
    one_hot_Y[np.arange(Y.size), Y] = 1 #it says for each row of one_hot_Y, set the value of the column to 1 where the column is the value of Y
    one_hot_Y = one_hot_Y.T #now each column is an example
    return one_hot_Y

def der_reLU(z):
    return z > 0 #returns 1 if z > 0 else 0 

def back_propagation(Z1, A1, Z2,A2,W1,W2,X,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * der_reLU(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)    
    db1 = 1/m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0: 
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(f"Accuracy: {get_accuracy(predictions, Y)*100:.3f}%")
    return W1, b1, W2, b2

W1, b1, W2 , b2 = gradient_descent(X_train, Y_train, 500, 0.1)  #Gave an 84% accuracy on the training data

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#test_prediction(90, W1, b1, W2, b2) #test prediction for a random index in the training data

#testing the model on the dev data
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2) 
accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Accuracy on dev data: {accuracy*100:.3f}%")

#testing the model on the test data
data_test = pd.read_csv('mnist_test.csv')
data_test = np.array(data_test)
data_test = data_test.T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test/255
test_predictions = make_predictions(X_test, W1, b1, W2, b2)
test_accuracy = get_accuracy(test_predictions, Y_test)
print(f"Accuracy on test data: {test_accuracy*100:.3f}%")

#Testing the model for random data

for i in range(10): #Testing 10 images from dataset
    test_prediction(i,W1,b1,W2,b2)