import sys
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from os.path import join
import csv
from sklearn.preprocessing import OneHotEncoder
#import MLPClassifier
from sklearn.neural_network import MLPClassifier

def getAcc(yPredic,yTest):
    cor=0
    inc=0
    for i in range(len(yTest)):
        if yPredic[i]==yTest[i]:
            cor+=1
        else:
            inc+=1
    return 100*(cor/(cor+inc))

#one hot encoding for Y in r(1 to r) classes
def one_hot_encoding(Y,number_of_classes):

    for i in range(len(Y)):
        temp = np.zeros(number_of_classes)
        temp[int(Y[i])-1]=1
        Y[i]=temp
    return Y

def getConfusionMatrix(test_data_Y_predicted,test_data_Y_max,num_classes):
    confusion_matrix = np.zeros((num_classes,num_classes))
    for i in range(len(test_data_Y_predicted)):
        confusion_matrix[test_data_Y_predicted[i]][test_data_Y_max[i]]+=1
    return confusion_matrix

def print_matrix_for_latex(matrix,header,outpath,file_name,write_mode):
    file = open(join(outpath,file_name),write_mode)
    file.write("\n"+header+"\n")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            file.write(str(matrix[i][j])+" & ")
        file.write("\\\\ \n")
    file.close()

# det data 728 in X and last in y
def get_data(data_file):
    file = open(data_file)
    csvreader = csv.reader(file)
    X=[]
    Y=[]
    lim =np.inf
    cur =0
    for row in csvreader:
        if cur==lim:
            break
        X.append(row[:-1])
        #y 
        Y.append(row[-1])
        cur+=1
    file.close()
    X = np.array(X).astype(np.float64)
    return X/255, np.array(one_hot_encoding(Y,10)).astype(np.float64)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def relu(x):
    y = np.zeros(x.shape)
    y[x <= 0] = 0
    y[x > 0] = x[x > 0]
    return y

def relu_derivative(x):
    y=np.zeros(x.shape)
    y[x<=0] = 0.0
    y[x>0] = 1.0
    return y

def calculate_loss_function(test_data_Y,activation,objective_function='MSE'):
    loss = 0.0
    if objective_function == 'MSE':
        loss = np.sum((test_data_Y-activation[-1])**2)/(test_data_Y.shape[0]*2)
    elif objective_function == 'cross_entropy':
        loss = -np.sum(test_data_Y*np.log(activation[-1])+(1-test_data_Y)*np.log(1-activation[-1]))/test_data_Y.shape[0]
    #print accuracy
    # print("train acc",getAcc(np.argmax(activation,axis=1),np.argmax(test_data_Y,axis=1)))
    # print(activation[-1][0])
    return loss


def init_weights_biases_he_uniform(hidden_layer_size_list,number_of_classes,train_data_X):
    weights = []
    biases = []
    # append input and output layer to hidden layer size list
    
    total_units = [train_data_X.shape[1]] + hidden_layer_size_list + [number_of_classes]
    for i in range(1, len(total_units)):
        weights.append(np.random.randn(
            total_units[i], total_units[i - 1]) * (2 / total_units[i - 1]) ** 0.5)
        biases.append(np.zeros(total_units[i]))
    return weights,biases

def init_weights_biases_xavier_normalized(hidden_layer_size_list,number_of_classes,train_data_X):
    weights = []
    biases = []
    # append input and output layer to hidden layer size list
    # do xavier normalized initialization
    total_units = [train_data_X.shape[1]] + hidden_layer_size_list + [number_of_classes]
    for i in range(1, len(total_units)):
        weights.append(np.random.randn(
            total_units[i], total_units[i - 1]) * (6 / (total_units[i - 1] + total_units[i])) ** 0.5)
        biases.append(np.zeros(total_units[i]))
    return weights,biases

def forward_propagation(weights,biases,test_data_X,activation_function='sigmoid'):
    activation=[test_data_X]
    for i in range(len(weights)):
        # print(activation[-1].shape,weights[i].shape,biases[i].shape)
        z = np.matmul(activation[i], weights[i].T) + biases[i]
        if activation_function == 'sigmoid':
            activation.append(sigmoid(z))
        elif activation_function == 'relu':
            if i == len(weights)-1:
                activation.append(sigmoid(z))
            else:
                activation.append(relu(z))
    return activation


def back_propagation(weights,biases,train_data_Y,activation,i,batch_size,activation_function='sigmoid',objective_function='MSE'):
    #implement backpropagation for both sigmoid and relu
    #calculate delta for output layer
    delta = []
    if objective_function == 'MSE':
        delta.append((train_data_Y[i:i+batch_size]-activation[-1])*sigmoid_derivative(activation[-1]))
    elif objective_function == 'cross_entropy':
        delta.append((train_data_Y[i:i+batch_size]-activation[-1]))
    #calculate delta for hidden layers
    for j in range(len(weights)-1,0,-1):
        if activation_function == 'sigmoid':
            delta.append(np.matmul(delta[-1],weights[j])*sigmoid_derivative(activation[j]))
        elif activation_function == 'relu':
            delta.append(np.matmul(delta[-1],weights[j])*relu_derivative(activation[j]))
    delta.reverse()
    return delta

    # delta=[]
    # if activation_function == 'sigmoid':
    #     delta.append((train_data_Y[i:i+batch_size]-activation[-1])*sigmoid_derivative(activation[-1]))
    #     for i in range(len(weights)-1,0,-1):
    #         delta.append(np.dot(delta[-1],weights[i])*sigmoid_derivative(activation[i]))
    #     delta.reverse()
    # elif activation_function == 'relu':
    #     delta.append((train_data_Y[i:i+batch_size]-activation[-1])*sigmoid_derivative(activation[-1]))
    #     for i in range(len(weights)-1,0,-1):
    #         delta.append(np.dot(delta[-1],weights[i])*relu_derivative(activation[i]))
    #     delta.reverse()
    # return delta




# implement a generic neural network architecture to learn a model for multi-class classification using backpropagation and mini batch gradient descent

def neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list,learning_rate,epochs,lot = 1e-4,adaptive_learning_rate=False,activation_function ="sigmoid",objective_function = "MSE"):
    # initialize weights and biases
    if activation_function=="sigmoid":
        weights,biases = init_weights_biases_xavier_normalized(hidden_layer_size_list,number_of_classes,train_data_X)
    elif activation_function=="relu":
        weights,biases = init_weights_biases_he_uniform(hidden_layer_size_list,number_of_classes,train_data_X)

    # training
    prev_epoch_loss = float('inf')
    for epoch in range(epochs):
        # print("Epoch: ",epoch)
        curr_epoch_loss = 0
        for i in range(0,train_data_X.shape[0],batch_size):
            #forward pass
            activation=forward_propagation(weights,biases,train_data_X[i:i+batch_size],activation_function)
            
            #backpropagation
            delta=back_propagation(weights,biases,train_data_Y,activation,i,batch_size,activation_function)

            #update weights and biases
            learning_rate_adaptive = learning_rate
            if adaptive_learning_rate:
                learning_rate_adaptive = learning_rate/np.sqrt(epoch+1)
            for i in range(len(weights)):
                weights[i]+=learning_rate_adaptive*np.dot(delta[i].T,activation[i])/batch_size
                biases[i]+=learning_rate_adaptive*np.sum(delta[i],axis=0)/batch_size
        
            #use lot to set limit such that changes in loss are less than lot
            curr_epoch_loss+=calculate_loss_function(train_data_Y[i:i+batch_size],activation[-1],objective_function)
        curr_epoch_loss/=(train_data_X.shape[0]/batch_size)
        # print("Loss: ",curr_epoch_loss)
        if abs(prev_epoch_loss-curr_epoch_loss)<lot:
            break
        prev_epoch_loss=curr_epoch_loss


        #calculate loss
        # loss = calculate_loss_function(weights,biases,train_data_X,train_data_Y,activation_function)
        # print("Loss: ",loss)

    #print train accuracy
    yPredic = np.argmax(predict(weights,biases,train_data_X),axis=1)
    yTest = np.argmax(train_data_Y,axis=1)
    # print("Train Accuracy: ",getAcc(yPredic,yTest))
    return weights,biases



def predict(weights,biases,test_data_X,activation_function='sigmoid'):
    activation=forward_propagation(weights,biases,test_data_X,activation_function)
    return activation[-1]

def parta(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    # implement a generic neural network architecture to learn a model for multi-class classification using backpropagation and mini batch gradient descent
    # training neural network
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [5]
    learning_rate = 0.5
    epochs = 100
    weights,biases = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list,learning_rate,epochs)
    #predicting on test data and train data
    test_data_Y_predicted = predict(weights,biases,test_data_X)
    test_data_Y_predicted = np.argmax(test_data_Y_predicted,axis=1)
    test_data_Y_max = np.argmax(test_data_Y,axis=1)
    train_data_Y_predicted = predict(weights,biases,train_data_X)
    train_data_Y_predicted = np.argmax(train_data_Y_predicted,axis=1)
    train_data_Y_max = np.argmax(train_data_Y,axis=1)
    #calculating accuracy
    test_accuracy = getAcc(test_data_Y_predicted,test_data_Y_max)
    train_accuracy = getAcc(train_data_Y_predicted,train_data_Y_max)
    #print accuracies in output file a.txt
    output_file = open(output_folder_path+"/a.txt","w")
    output_file.write("Train Accuracy: "+str(train_accuracy)+"\n")
    output_file.write("Test Accuracy: "+str(test_accuracy)+"\n")
    output_file.close()


def partb(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    # Vary the number of hidden layer units from the set {5, 10, 15, 20, 25}.
    #plot the accuracy on the training and the test sets, time taken to train the network.
    # Additionally, report the confusion matrix for the test set, for each of the above parameter values.
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [5,10,15,20,25]
    learning_rate = 0.1
    epochs = 1000
    #plotting and reporting accuracy
    train_accuracy_list = []
    test_accuracy_list = []
    time_list = []
    output_file = open(join(output_folder_path,"b.txt"),"w")
    for hidden_layer_size in hidden_layer_size_list:
        start_time = time.time()
        weights,biases = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,[hidden_layer_size],learning_rate,epochs)
        end_time = time.time()
        time_list.append(end_time-start_time)
        test_data_Y_predicted = predict(weights,biases,test_data_X)
        test_data_Y_predicted = np.argmax(test_data_Y_predicted,axis=1)
        test_data_Y_max = np.argmax(test_data_Y,axis=1)
        train_data_Y_predicted = predict(weights,biases,train_data_X)
        train_data_Y_predicted = np.argmax(train_data_Y_predicted,axis=1)
        train_data_Y_max = np.argmax(train_data_Y,axis=1)
        test_accuracy = getAcc(test_data_Y_predicted,test_data_Y_max)
        train_accuracy = getAcc(train_data_Y_predicted,train_data_Y_max)
        test_accuracy_list.append(test_accuracy)
        train_accuracy_list.append(train_accuracy)
        #report accuracy
        output_file.write("\nHidden Layer Size: "+str(hidden_layer_size)+": \n")
        output_file.write("Train Accuracy: "+str(train_accuracy)+"\n")
        output_file.write("Test Accuracy: "+str(test_accuracy)+"\n")
        output_file.write("Time Taken: "+str(end_time-start_time)+"\n")
        #report confusion matrix
        output_file.write("Confusion Matrix: \n")
        confusion_matrix = getConfusionMatrix(test_data_Y_predicted,test_data_Y_max,number_of_classes)
        print_matrix_for_latex(confusion_matrix,"confusion matrix for part 2_b",output_folder_path,"b_conf_mat","a")
        output_file.write(str(confusion_matrix)+"\n")



    output_file.close()
    #plotting using matplotlib
    plt.plot(hidden_layer_size_list,train_accuracy_list,label="Train Accuracy")
    plt.plot(hidden_layer_size_list,test_accuracy_list,label="Test Accuracy")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_folder_path+"/b1_accuracy.png")
    plt.clf()
    plt.plot(hidden_layer_size_list,time_list)
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time")
    plt.savefig(output_folder_path+"/b2_time.png")
    plt.clf()
    
def partc(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    # Use adaptive learning rate 
    # Vary the number of hidden layer units from the set {5, 10, 15, 20, 25}.
    #plot the accuracy on the training and the test sets, time taken to train the network.
    # Additionally, report the confusion matrix for the test set, for each of the above parameter values.
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [5,10,15,20,25]
    learning_rate = 0.1
    epochs = 1000
    #plotting accuracy
    train_accuracy_list = []
    test_accuracy_list = []
    time_list = []
    output_file = open(join(output_folder_path,"c.txt"),"w")
    for hidden_layer_size in hidden_layer_size_list:
        start_time = time.time()
        weights,biases = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,[hidden_layer_size],learning_rate,epochs,adaptive_learning_rate=True)
        end_time = time.time()
        time_list.append(end_time-start_time)
        test_data_Y_predicted = predict(weights,biases,test_data_X)
        test_data_Y_predicted = np.argmax(test_data_Y_predicted,axis=1)
        test_data_Y_max = np.argmax(test_data_Y,axis=1)
        train_data_Y_predicted = predict(weights,biases,train_data_X)
        train_data_Y_predicted = np.argmax(train_data_Y_predicted,axis=1)
        train_data_Y_max = np.argmax(train_data_Y,axis=1)
        test_accuracy = getAcc(test_data_Y_predicted,test_data_Y_max)
        train_accuracy = getAcc(train_data_Y_predicted,train_data_Y_max)
        test_accuracy_list.append(test_accuracy)
        train_accuracy_list.append(train_accuracy)
        # report accuracy
        
        output_file.write("\nHidden Layer Size: "+str(hidden_layer_size)+": \n")
        output_file.write("Train Accuracy: "+str(train_accuracy)+"\n")
        output_file.write("Test Accuracy: "+str(test_accuracy)+"\n")
        output_file.write("Time Taken: "+str(end_time-start_time)+"\n")
        #report conf matrix
        confusion_matrix = getConfusionMatrix(test_data_Y_predicted,test_data_Y_max,number_of_classes)
        #print confusion matrix in latex format
        print_matrix_for_latex(confusion_matrix,"confusion matrix for part 2_c",output_folder_path,"c_conf_mat","a")
        output_file.write(str(confusion_matrix)+"\n")
    output_file.close()
        
        
    #plotting using matplotlib
    plt.plot(hidden_layer_size_list,train_accuracy_list,label="Train Accuracy")
    plt.plot(hidden_layer_size_list,test_accuracy_list,label="Test Accuracy")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_folder_path+"/c1_accuracy.png")
    plt.clf()
    plt.plot(hidden_layer_size_list,time_list)
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time")
    plt.savefig(output_folder_path+"/c2_time.png")
    plt.clf()
    # #printing confusion matrix
    # for hidden_layer_size in hidden_layer_size_list:
    #     weights,biases = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,[hidden_layer_size],learning_rate,epochs)
    #     test_data_Y_predicted = predict(weights,biases,test_data_X)
    #     test_data_Y_predicted = np.argmax(test_data_Y_predicted,axis=1)
    #     test_data_Y_max = np.argmax(test_data_Y,axis=1)
    #     confusion_matrix = getConfusionMatrix(test_data_Y_predicted,test_data_Y_max,number_of_classes)
    #     print_matrix_for_latex(confusion_matrix,"confusion matrix for part 2_c",output_folder_path,"c_conf_mat","a")
    #     output_file = open(join(output_folder_path,"/c.txt"),"a")
    #     output_file.write(str(confusion_matrix))
    #     output_file.close()
    
def partd(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
# Implement a network with 2 hidden layers with 100 units each. Experiment with both ReLU and sigmoid activation units as described above.
# Report training and test accuracies in each case
# Report the confusion matrix for the test set
    test_data_Y_max = np.argmax(test_data_Y,axis=1)
    train_data_Y_max = np.argmax(train_data_Y,axis=1)
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [100,100]
    learning_rate = 0.1
    epochs = 100
     
    #getting accuracy for sigmoid and relu
    #relu
    relu_time = time.time()
    weigths_relu,biases_relu = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list,learning_rate,epochs,adaptive_learning_rate=True, activation_function="relu")
    relu_time = time.time()-relu_time
    test_data_Y_predicted_relu = predict(weigths_relu,biases_relu,test_data_X,activation_function="relu")
    test_data_Y_predicted_relu = np.argmax(test_data_Y_predicted_relu,axis=1)
    train_data_Y_predicted_relu = predict(weigths_relu,biases_relu,train_data_X,activation_function="relu")
    train_data_Y_predicted_relu = np.argmax(train_data_Y_predicted_relu,axis=1)
    test_accuracy_relu = getAcc(test_data_Y_predicted_relu,test_data_Y_max)
    train_accuracy_relu = getAcc(train_data_Y_predicted_relu,train_data_Y_max)
    #sigmoid
    sigmoid_time = time.time()
    weigths_sigmoid,biases_sigmoid = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list,learning_rate,epochs,adaptive_learning_rate=True, activation_function="sigmoid")
    sigmoid_time = time.time()-sigmoid_time
    test_data_Y_predicted_sigmoid = predict(weigths_sigmoid,biases_sigmoid,test_data_X,activation_function="sigmoid")
    test_data_Y_predicted_sigmoid = np.argmax(test_data_Y_predicted_sigmoid,axis=1)
    train_data_Y_predicted_sigmoid = predict(weigths_sigmoid,biases_sigmoid,train_data_X,activation_function="sigmoid")
    train_data_Y_predicted_sigmoid = np.argmax(train_data_Y_predicted_sigmoid,axis=1)
    test_accuracy_sigmoid = getAcc(test_data_Y_predicted_sigmoid,test_data_Y_max)
    train_accuracy_sigmoid = getAcc(train_data_Y_predicted_sigmoid,train_data_Y_max)

    #printing accuracy
    output_file = open(output_folder_path+"/d.txt","w")
    #for sigmoid
    output_file.write("Sigmoid Activation Function Train accuracy: "+str(train_accuracy_sigmoid)+"\n")
    output_file.write("Sigmoid Activation Function Test accuracy: "+str(test_accuracy_sigmoid)+"\n")
    output_file.write("Sigmoid Activation Function Time: "+str(sigmoid_time)+"\n")
    #for relu
    output_file.write("Relu Activation Function Train accuracy: "+str(train_accuracy_relu)+"\n")
    output_file.write("Relu Activation Function Test accuracy: "+str(test_accuracy_relu)+"\n")
    output_file.write("Relu Activation Function Time: "+str(relu_time)+"\n")

    #printing confusion matrix
    #for sigmoid
    confusion_matrix_sigmoid = getConfusionMatrix(test_data_Y_predicted_sigmoid,test_data_Y_max,number_of_classes)
    print_matrix_for_latex(confusion_matrix_sigmoid,"confusion matrix for part 2_d sigmoid",output_folder_path,"d_conf_mat","a")
    output_file.write("Sigmoid Activation Function Confusion Matrix: \n"+str(confusion_matrix_sigmoid)+"\n")
    #for relu
    confusion_matrix_relu = getConfusionMatrix(test_data_Y_predicted_relu,test_data_Y_max,number_of_classes)
    print_matrix_for_latex(confusion_matrix_relu,"confusion matrix for part 2_d relu",output_folder_path,"d_conf_mat","a")
    output_file.write("Relu Activation Function Confusion Matrix: \n"+str(confusion_matrix_relu)+"\n")
    output_file.close()

def parte(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    #increasing the number of hidden layers in your network from 2 to 5 with 50 units
    # Plot the training and test accuracy against number of hidden layers
    # plotting exercise for ReLU as well as sigmoid activation in the hidden layers
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [50,50,50,50,50]
    learning_rate = 0.1
    epochs = 100
    # vary number of hidden layers from 2 to 5 and plot the accuracy
    #relu
    test_data_Y_max = np.argmax(test_data_Y,axis=1)
    train_data_Y_max = np.argmax(train_data_Y,axis=1)
    test_accuracy_relu = []
    train_accuracy_relu = []
    for i in range(2,6):
        weigths_relu,biases_relu = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list[:i],learning_rate,epochs,adaptive_learning_rate=True, activation_function="relu")
        test_data_Y_predicted_relu = predict(weigths_relu,biases_relu,test_data_X,activation_function="relu")
        test_data_Y_predicted_relu = np.argmax(test_data_Y_predicted_relu,axis=1)
        train_data_Y_predicted_relu = predict(weigths_relu,biases_relu,train_data_X,activation_function="relu")
        train_data_Y_predicted_relu = np.argmax(train_data_Y_predicted_relu,axis=1)
        test_accuracy_relu.append(getAcc(test_data_Y_predicted_relu,test_data_Y_max))
        train_accuracy_relu.append(getAcc(train_data_Y_predicted_relu,train_data_Y_max))
    #sigmoid
    test_accuracy_sigmoid = []
    train_accuracy_sigmoid = []
    for i in range(2,6):
        weigths_sigmoid,biases_sigmoid = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list[:i],learning_rate,epochs,adaptive_learning_rate=True, activation_function="sigmoid")
        test_data_Y_predicted_sigmoid = predict(weigths_sigmoid,biases_sigmoid,test_data_X,activation_function="sigmoid")
        test_data_Y_predicted_sigmoid = np.argmax(test_data_Y_predicted_sigmoid,axis=1)
        train_data_Y_predicted_sigmoid = predict(weigths_sigmoid,biases_sigmoid,train_data_X,activation_function="sigmoid")
        train_data_Y_predicted_sigmoid = np.argmax(train_data_Y_predicted_sigmoid,axis=1)
        test_accuracy_sigmoid.append(getAcc(test_data_Y_predicted_sigmoid,test_data_Y_max))
        train_accuracy_sigmoid.append(getAcc(train_data_Y_predicted_sigmoid,train_data_Y_max))
    #plotting
    plt.plot([2,3,4,5],test_accuracy_relu,label="test accuracy relu")
    plt.plot([2,3,4,5],train_accuracy_relu,label="train accuracy relu")
    plt.plot([2,3,4,5],test_accuracy_sigmoid,label="test accuracy sigmoid")
    plt.plot([2,3,4,5],train_accuracy_sigmoid,label="train accuracy sigmoid")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_folder_path+"/e_accuracy_plots.png")
    plt.close()
    # report accuracies for ReLU and sigmoid activation functions
    output_file = open(output_folder_path+"/e.txt","w")
    output_file.write("ReLU Activation Function Train accuracy: "+str(train_accuracy_relu)+"\n")
    output_file.write("ReLU Activation Function Test accuracy: "+str(test_accuracy_relu)+"\n")
    output_file.write("Sigmoid Activation Function Train accuracy: "+str(train_accuracy_sigmoid)+"\n")
    output_file.write("Sigmoid Activation Function Test accuracy: "+str(test_accuracy_sigmoid)+"\n")
    output_file.close()

def partf(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    #report the training and test accuracy for BCE loss with relu activation function
    test_data_Y_max = np.argmax(test_data_Y,axis=1)
    train_data_Y_max = np.argmax(train_data_Y,axis=1)
    batch_size = 100
    number_of_classes = 10
    hidden_layer_size_list = [50,50,50,50]
    learning_rate = 0.1
    epochs = 100
    #relu
    weigths_relu,biases_relu = neural_network_construction(train_data_X,train_data_Y,batch_size,number_of_classes,hidden_layer_size_list,learning_rate,epochs,adaptive_learning_rate=True, activation_function="relu",objective_function="cross_entropy")
    test_data_Y_predicted_relu = predict(weigths_relu,biases_relu,test_data_X,activation_function="relu")
    test_data_Y_predicted_relu = np.argmax(test_data_Y_predicted_relu,axis=1)
    train_data_Y_predicted_relu = predict(weigths_relu,biases_relu,train_data_X,activation_function="relu")
    train_data_Y_predicted_relu = np.argmax(train_data_Y_predicted_relu,axis=1)
    test_accuracy_relu = getAcc(test_data_Y_predicted_relu,test_data_Y_max)
    train_accuracy_relu = getAcc(train_data_Y_predicted_relu,train_data_Y_max)
    #report accuracies for BCE loss with relu activation function
    output_file = open(output_folder_path+"/f.txt","w")
    output_file.write("ReLU Activation Function Train accuracy: "+str(train_accuracy_relu)+"\n")
    output_file.write("ReLU Activation Function Test accuracy: "+str(test_accuracy_relu)+"\n")
    output_file.close()

def partg(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path):
    # MLPClassifier from scikit-learn library to implement a neural network with the same architecture
    # report the training and test accuracy for BCE loss with relu activation function
    clf = MLPClassifier(hidden_layer_sizes=(50,50,50,50),activation="relu",solver="sgd",max_iter=100)
    mlpTime = time.time()
    clf.fit(train_data_X,train_data_Y)
    mlpTime = time.time() - mlpTime
    #get accuracies using score
    test_accuracy = clf.score(test_data_X,test_data_Y)
    train_accuracy = clf.score(train_data_X,train_data_Y)
    #report accuracies for BCE loss with relu activation function
    output_file = open(output_folder_path+"/g.txt","w")
    output_file.write("ReLU Activation Function Train accuracy: "+str(train_accuracy)+"\n")
    output_file.write("ReLU Activation Function Test accuracy: "+str(test_accuracy)+"\n")
    output_file.write("Time taken: "+str(mlpTime)+"\n")
    output_file.close()

    return None

def main():
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    question_part = sys.argv[4]
    train_data_X,train_data_Y = get_data(train_data_path)
    test_data_X,test_data_Y = get_data(test_data_path)
    if question_part == 'a':
        parta(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    elif question_part == 'b':
        partb(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    elif question_part == 'c':
        partc(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    elif question_part == 'd':
        partd(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    elif question_part == 'e':
        parte(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path) 
    elif question_part == 'f':
        partf(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    elif question_part == 'g':
        partg(train_data_X,train_data_Y,test_data_X,test_data_Y,output_folder_path)
    

main()
    