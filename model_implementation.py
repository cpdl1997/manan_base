import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dataset_generator as dg

##################################################################################################################################################################
#model
def model_obj(layer_size, lr, max_it=100):
    clf = MLPClassifier(hidden_layer_sizes=layer_size,
                    random_state=5,
                    max_iter=max_it,
                    verbose=True,
                    learning_rate_init=lr)
    return clf


##################################################################################################################################################################
#utility functions
def sigmoid(x): 
    return 1 / (1 + math.exp(-x))


def create_test_train(dataset, test_size):
    data=pd.read_csv(dataset)
    data = normalize(data)
    X=data.drop('Y', axis=1)
    Y=data['Y']
    Y[Y>=0.5] = 1
    Y[Y<0.5] = 0

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
    return x_train, x_test, y_train, y_test



def normalize(dataset):
    for i in dataset.columns:
        dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
    return dataset


##################################################################################################################################################################
#main function
def implement_model():
    #paramaters
    test_size = 0.3
    learning_rate = 0.001
    hidden_layer_size = (6,5)
    lower=-20
    upper= 20
    max_epochs = 500

    ch = dg.generateDatapoints(lower, upper)
    if(ch!='1' and ch!='2' and ch!='3'):
        print("Incorrect Input.")
    else:
        ch=int(ch)
        if(ch==1):
            dataset = 'dataset1.csv'
        elif(ch==2):
            dataset = 'dataset2.csv'
        else:  
            dataset = 'dataset3.csv'

    #training
    x_train, x_test, y_train, y_test = create_test_train(dataset, test_size)
    
    model = model_obj(hidden_layer_size, learning_rate, max_epochs)
    model.fit(x_train,y_train)

    # Calcuate test accuracy
    print("Checking for test data... ")
    ypred = model.predict(x_test)
    score = accuracy_score(y_test,ypred)
    print("Test Accuracy: ", score)




implement_model()