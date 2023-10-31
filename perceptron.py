import numpy as np


def load_data(file_data):
    data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
    X = []
    Y = []
    for row in data:
        temp = [float(x) for x in row]
        temp.pop(-1)
        X.append(temp)
        Y.append(int(row[-1]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def prediction(a):
    return np.where(a<=0, -1, 1)

def perceptron_train(X,Y):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    epoch = 3
    update = True
    
    while (update or epoch>=0):
        update = False
        for idx, X_i in enumerate(X):
                activation = np.dot(X_i, weights) + bias
                # print("activation = ", activation)
                if(activation*Y[idx] <= 0):
                    update = True
                    weights += Y[idx] * X_i
                    # print(weights)
                    bias += Y[idx]
                    # print(bias)
        epoch= epoch -1
                    
   

    w=[]
    w.append(weights)
    w.append(bias)
    print("weights and bias=", w)
    return (w)

def perceptron_test(X, Y, weights, bias):
    n_samples, n_features = X.shape
    acc = 0
    for idx, X_i in enumerate(X):
            activation = np.dot(X_i, weights) + bias
            # print("activation = ", activation)
            y_predict = prediction(activation)
            # print("y_predict = ", y_predict)

            if(y_predict == Y[idx]):
                acc += 1
    acc = acc/n_samples
    return(acc)




