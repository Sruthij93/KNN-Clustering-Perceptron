import numpy as np
from scipy.spatial import distance

def find_nn(X_train,dist, K):
    """obtaining first k nearest neighbors"""
    
    nearest_neighbors = []
    for i in range(K):
        nearest_neighbors.append(X_train[i])

    for j in range(len(X_train)-K):
        if(dist[K-1]==dist[K+j]):
            nearest_neighbors.append(X_train[K+j])

    nearest_neighbors = tuple(nearest_neighbors)
    return (nearest_neighbors)

def distance_sort(X_train, dist):
    dist, X_train = (list(t) for t in zip(*sorted(zip(dist, X_train))))
    return dist, X_train

def KNN_test(X_train, Y_train, X_test, Y_test,K):

    #mapping X_train to the respective labels- Y_train in a dictionary
    X_train_indices = {}
    for i in range(X_train.shape[0]):
        X_train_indices[tuple(X_train[i])] = Y_train[i]

    temp_accuracy = 0 
    
    #finding distances between sample points and 1 test sample, sorting the distances and choosing KNN
    for i in range(X_test.shape[0]):
        dist= []
        for j in range(len(X_train)):
            dist.append(distance.euclidean(X_train[j], X_test[i]))
           
        # print("all distances ==", dist, "\n")

        X_trainc = X_train.copy()
        X_trainc = X_trainc.tolist()

        #sorting distances and X_train points
        dist, X_trainc = distance_sort(X_trainc, dist)
        # print("sorted dist = ", dist,"\n")
        # print("sorted X_trainc = ", X_trainc,"\n")
        # print("X test=", X_test[i],"\n")
        
        
        nearest_neighbors = find_nn(X_trainc, dist, K)
        #getting prediction
        prediction = 0
        for m in range(K):
            prediction += X_train_indices[tuple(nearest_neighbors[m])]
        # print("prediction= ", prediction)

        #accuracy test
        if (prediction == Y_test[i]):
            temp_accuracy += 1
    accuracy = temp_accuracy/len(X_test)
    return(accuracy)
    
    
def find_best_K(X_train, Y_train, X_val, Y_val ):
    best_accuracy = 0
    for z in range(len(X_train)):
        accuracy = KNN_test(X_train,Y_train,X_val,Y_val,z+1)
        # print(f"accuracy for {z+1} is {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k= z+1
    return(best_k)




