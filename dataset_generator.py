import math
import numpy as np
import os
import random

##################################################################################################################################################################
#actual functions to be found
def actual_function1(inp1, inp2):
    return 2*(inp1**2)+inp2*5 + 4



def actual_function2(inp1, inp2):
    return 1/(inp1**2+math.sqrt(abs(inp2))*3+1) + math.sin(3*inp1*inp2+5)



def actual_function3(inp1, inp2, inp3, inp4, inp5):
    return (5/(abs(inp1**abs((inp2)))+1) + np.random.normal(inp3, abs(inp4)))*inp5


##################################################################################################################################################################
#generate datapoints for testing counterfactuals
def generateDatapoints(lower=-100, upper=100):
    ch = input("Enter which function you would like to generate a dataset for:\n1) Simple Polynomial Function in two variable\n2) Complex Function in two variable\n3) Function in 5 variables with Gaussian Noise\nEnter your choice: ")
    if(ch!='1' and ch!='2' and ch!='3'):
        print("Incorrect Input. Restarting...")
        generateDatapoints(lower, upper)
    else:
        ch=int(ch)
        n = int(input("Enter the number of datapoints: "))
        
        if(ch==1):
            #remove file if it exists
            try:
                os.remove('dataset1.csv')
            except OSError:
                pass
            #create file
            f = open('dataset1.csv', 'w')
            f.write('X1,X2,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(actual_function1(inp1, inp2))+'\n')
            f.close()
        elif(ch==2):
            #remove file if it exists
            try:
                os.remove('dataset2.csv')
            except OSError:
                pass
            #create file
            f = open('dataset2.csv', 'w')
            f.write('X1,X2,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(actual_function2(inp1, inp2))+'\n')
            f.close()
        else:
            #remove file if it exists
            try:
                os.remove('dataset3.csv')
            except OSError:
                pass
            #create file
            f = open('dataset3.csv', 'w')
            f.write('X1,X2,X3,X4,X5,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                inp3 = random.randint(lower, upper)
                inp4 = random.randint(lower, upper)
                inp5 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(inp3)+','+str(inp4)+','+str(inp5)+','+str(actual_function3(inp1, inp2, inp3, inp4, inp5))+'\n')
            f.close()
        
        print('Dataset Generated Successfully.')
    return str(ch)