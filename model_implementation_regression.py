import pandas as pd
import numpy as np
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dataset_generator as dg

##################################################################################################################################################################
#model
def model_obj(train, layer_size, lr):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(256, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()
    return NN_model


##################################################################################################################################################################
#utility functions
def create_test_train(dataset, test_size):
    data=pd.read_csv(dataset)
    data = normalize(data)
    X=data.drop('Y', axis=1)
    Y=data['Y']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
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

    dg.generateDatapoints(lower, upper)
    ch = input("Choose the dataset you want:\n1) dataset1.csv\n2) dataset2.csv\n3) dataset3.csv\nEnter your choice: ")
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
    model = model_obj(x_train, hidden_layer_size, learning_rate)
    model.fit(x_train,y_train, epochs=500)

    # Calcuate test accuracy
    print("Checking for test data... ")
    score = model.evaluate(x_test,y_test)
    print("Test Accuracy: ", score)




implement_model()