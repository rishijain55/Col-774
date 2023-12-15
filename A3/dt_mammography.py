#Code for decision tree using sklearn
from asyncore import write
import sys
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from os.path import join
from sklearn import tree
import csv
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


#Read data for decision tree
def get_data_pure(dataFile):
    file = open(dataFile)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    X=[]
    Y=[]
    for row in csvreader:
            
            if '?' not in row[1:]:
                temp=[]
                for i in range(len(row)):
                    if row[i]=='?':
                        temp.append(np.nan)
                    else:
                        temp.append(float(row[i]))
                X.append(temp[1:-1])
                Y.append(temp[-1])
    # print(type(X))
    file.close()
    # print(data)

    return X, Y,header[1:]

#Read data for decision tree by imputing
def get_data_impute(dataFile,strategy):
    file = open(dataFile)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    X=[]
    Y=[]
    for row in csvreader:
        X.append(row[1:-1])
        for i in range(len(X[-1])):
            if X[-1][i]=='?':
                X[-1][i]=np.nan
        Y.append(row[-1])
    file.close()
    # print(data)
    #impute values by mean values
    # print(X)
    imp = SimpleImputer(missing_values= np.nan, strategy=strategy)
    imp.fit(X)
    X = imp.transform(X)
    return X, Y,header[1:]

def getAcc(yPredic,yTest):
    cor=0
    inc=0
    for i in range(len(yTest)):
        if yPredic[i]==yTest[i]:
            cor+=1
        else:
            inc+=1
    return 100*(cor/(cor+inc))

#import data and implement decision tree
def dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="/1_a.txt",write_mode="w",q_part="1_a"):
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    print(clf)
    #tree using sklearn with validation data
    clf = clf.fit(train_data_X, train_data_Y,)
    y_pred_train = clf.predict(train_data_X)
    y_pred_val = clf.predict(validation_data_X)
    y_pred_test = clf.predict(test_data_X)
    #report accuracy in a.txt and save graph
    with open(outpath+file_name,write_mode) as f:
        f.write("Train Accuracy: "+str(getAcc(y_pred_train,train_data_Y))+"\n")
        f.write("Validation Accuracy: "+str(getAcc(y_pred_val,validation_data_Y))+"\n")
        f.write("Test Accuracy: "+str(getAcc(y_pred_test,test_data_Y))+"\n")
    #save graph
    fig = plt.figure(figsize=(125,100))
    _ = tree.plot_tree(clf, 
                   feature_names=header,  
                   filled=True)
    fig.savefig(outpath+"/decision_tree_simple_Q1q{}.png".format(q_part))

def grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="/1_b.txt",write_mode="w",q_part="1_b"):    
    params = {
        'max_depth':  [ 2,3, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 4, 8, 10,12,14,16,18,19,20],
    }
    
    # clf = GridSearchCV(
    #     estimator=DecisionTreeClassifier(),
    #     param_grid=params,
    # )
    # clf.fit(train_data_X, train_data_Y)
    # print(clf.best_params_)
    # clf = DecisionTreeClassifier(criterion='entropy', splitter='best',**clf.best_params_)
    # clf.fit(train_data_X, train_data_Y)

    best_accuracy = -1
    best_max_depth = -1
    best_min_samples_split = -1
    best_min_samples_leaf = -1
    best_clf = None

    for max_depth in params['max_depth']:
        for min_samples_split in params['min_samples_split']:
            for min_samples_leaf in params['min_samples_leaf']:
                clf = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
                clf.fit(train_data_X, train_data_Y)
                yPredic = clf.predict(validation_data_X)
                acc = getAcc(yPredic,validation_data_Y)
                # print('max_depth:',max_depth,'min_samples_split:',min_samples_split,'min_samples_leaf:',min_samples_leaf,'acc:',acc)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_min_samples_leaf = min_samples_leaf
                    best_clf = clf

    #report accuracy in b.txt and save graph
    with open(outpath+file_name,write_mode) as f:
        f.write("Best max_depth: "+str(best_max_depth)+"\n")
        f.write("Best min_samples_split: "+str(best_min_samples_split)+"\n")
        f.write("Best min_samples_leaf: "+str(best_min_samples_leaf)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_clf.predict(validation_data_X),validation_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")


    # print('best_max_depth:',best_max_depth,'best_min_samples_split:',best_min_samples_split,'best_min_samples_leaf:',best_min_samples_leaf)

    # y_pred_train = best_clf.predict(train_data_X)
    # y_pred_val = best_clf.predict(validation_data_X)
    # y_pred_test = best_clf.predict(test_data_X)
    # print("accuracy by sklearn on training data is: ",getAcc(y_pred_train,train_data_Y))
    # print("accuracy by sklearn on validation data is: ",getAcc(y_pred_val,validation_data_Y))
    # print("accuracy by sklearn on test data is:",getAcc(y_pred_test,test_data_Y))
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    _ = tree.plot_tree(best_clf, 
                   feature_names=header,  
                   filled=True)
    fig.savefig(outpath+"/decision_tree_grid_search_Q1q{}.png".format(q_part))

#implement decision tree and pruning
def dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="/1_c.txt",write_mode="w",q_part="1_c"):
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    path = clf.cost_complexity_pruning_path(train_data_X, train_data_Y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # print(ccp_alphas)
    # print(impurities)
    clfs = []
    y_pred_val = []
    accuracies = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best', ccp_alpha=ccp_alpha)
        clf.fit(train_data_X, train_data_Y)
        y_pred_val.append(clf.predict(validation_data_X))
        accuracies.append(getAcc(y_pred_val[-1],validation_data_Y))
        clfs.append(clf)

    #plot total impurity of leaves vs effective alphas of pruned tree
    fig, ax = plt.subplots()
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q1{}impurity_vs_alpha.png".format(q_part))

    # Plot the number of nodes vs alpha
    node_counts = [clf.tree_.node_count for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("number of nodes")
    ax.set_title("Number of nodes vs alpha for training set")
    ax.plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q1{}nodes_vs_alpha.png".format(q_part))

    # Plot the depth of the tree vs alpha
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("depth of tree")
    ax.set_title("Depth vs alpha for training set")
    ax.plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q1{}depth_vs_alpha.png".format(q_part))

    # Plot the training, validation and test accuracy vs alpha
    train_accuracy = [getAcc(clf.predict(train_data_X),train_data_Y) for clf in clfs]
    val_accuracy = [getAcc(clf.predict(validation_data_X),validation_data_Y) for clf in clfs]
    test_accuracy = [getAcc(clf.predict(test_data_X),test_data_Y) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training set")
    ax.plot(ccp_alphas, train_accuracy, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_accuracy, marker='o', label="test",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, val_accuracy, marker='o', label="validation",
            drawstyle="steps-post")
    ax.legend()
    fig.savefig(outpath+"/Q1{}accuracy_vs_alpha.png".format(q_part))

    #report model with highest accuracy on validation set
    best_clf = clfs[np.argmax(val_accuracy)]
    best_alpha = ccp_alphas[np.argmax(val_accuracy)]
    with open(outpath+file_name,write_mode) as f:
        f.write("Best alpha: "+str(best_alpha)+"\n")
        f.write("Best Depth: "+str(best_clf.tree_.max_depth)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_clf.predict(validation_data_X),validation_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")
    #plot decision tree
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    
    tree.plot_tree(best_clf,
                     feature_names = header,
                        class_names=['0','1'],
                        filled = True)
    fig.savefig(outpath+'/Q1{}_decision_tree_post_pruning.png'.format(q_part))

def random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="/1_d.txt",write_mode="w",q_part="1_d"):
    params = {
        'n_estimators' : [50, 150, 250,1000], 
        'max_features' : [0.1, 0.3, 0.5, 0.9],
        'min_samples_split' : [2, 6, 10,14,18,20,24,26,28,30],
    }

    best_oob_score = -1
    best_n_estimators = -1
    best_max_features = -1
    best_min_samples_split = -1
    best_model = None

    # Use the out-of-bag accuracy (as explained in the class) to tune to the optimal values for these parameters. You should perform a grid search over the space of parameters
    for n_estimator in params['n_estimators']:
        for max_feature in params['max_features']:
            for min_sample_split in params['min_samples_split']:
                # print("n_estimator: ", n_estimator, "max_feature: ", max_feature, "min_sample_split: ", min_sample_split)
                clf = RandomForestClassifier(n_estimators=n_estimator, max_features=max_feature, min_samples_split=min_sample_split, oob_score=True)
                clf.fit(train_data_X, train_data_Y)
                # print("n_estimator: ", n_estimator, "max_feature: ", max_feature, "min_sample_split: ", min_sample_split, "oob_score: ", clf.oob_score_)

                if clf.oob_score_ > best_oob_score:
                    best_oob_score = clf.oob_score_
                    best_n_estimators = n_estimator
                    best_min_samples_split = min_sample_split
                    best_max_features = max_feature
                    best_model = clf
    #report oobscore and accuracy on test set
    with open(outpath+file_name,write_mode) as f:
        f.write("Best n_estimators: "+str(best_n_estimators)+"\n")
        f.write("Best max_features: "+str(best_max_features)+"\n")
        f.write("Best min_samples_split: "+str(best_min_samples_split)+"\n")
        f.write("Best oob_score: "+str(best_oob_score)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_model.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_model.predict(test_data_X),test_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_model.predict(validation_data_X),validation_data_Y))+"\n")
        

    # print("best_oob_score: ", best_oob_score)
    # print("best_n_estimators: ", best_n_estimators)
    # print("best_min_samples_split: ", best_min_samples_split)
    # print("best_max_features: ", best_max_features)
    # print(clf.best_params_)
    # print(clf.best_score_)
    # print(clf.best_estimator_)
    # print(clf.best_estimator_.oob_score_)

    # Use the best parameters to train a random forest classifier on the training data
    # clf = RandomForestClassifier(n_estimators=250, max_features=0.1, min_samples_split=2, oob_score=True, random_state=10)
    # clf = RandomForestClassifier(**clf.best_params_,oob_score=True, random_state=10)
    # clf.fit(train_data_X, train_data_Y)
    
    # Reporting, out-of-bag training, validation and test set accuracies for the optimal set of parameters obtained.
    # print(clf.oob_score_)
    # print("Training Accuracy: ", getAcc(best_model.predict(train_data_X),train_data_Y))
    # print("Validation Accuracy: ", getAcc(best_model.predict(validation_data_X),validation_data_Y))
    # print("Test Accuracy: ", getAcc(best_model.predict(test_data_X),test_data_Y))

def xgboost_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="/1_f.txt",write_mode="w"):
    params = {
        'n_estimators':[10,20,30,40,50],
        'subsample': [0.1,0.2,0.3,0.4,0.5,0.6],
        'max_depth':  [ 4,5, 6,7, 8,9, 10],
    }
    # clf = GridSearchCV(
    #     estimator=XGBClassifier(),
    #     param_grid=params,
    # )
    # clf.fit(train_data_X, train_data_Y)
    # print(clf.best_params_)
    # best_clf = XGBClassifier(**clf.best_params_)
    # best_clf.fit(train_data_X, train_data_Y)

    #hand written grid search
    best_accuracy = -1
    best_n_estimators = -1
    best_subsample = -1
    best_max_depth = -1
    best_clf = None
    for n_estimators in params['n_estimators']:
        for subsample in params['subsample']:
            for max_depth in params['max_depth']:
                clf = XGBClassifier(n_estimators=n_estimators,subsample=subsample,max_depth=max_depth)
                clf.fit(train_data_X, train_data_Y)
                yPredic = clf.predict(validation_data_X)
                accuracy = getAcc(yPredic,validation_data_Y)
                # print("n_estimators: ",n_estimators," subsample: ",subsample," max_depth: ",max_depth," accuracy: ",accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_n_estimators = n_estimators
                    best_subsample = subsample
                    best_max_depth = max_depth
                    best_clf = clf
    
    #report best params
    with open(outpath+file_name,write_mode) as f:
        f.write("Best n_estimators: "+str(best_n_estimators)+"\n")
        f.write("Best subsample: "+str(best_subsample)+"\n")
        f.write("Best max_depth: "+str(best_max_depth)+"\n")
        f.write("Best Validation Accuracy: "+str(best_accuracy)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")
    # print("max_depth:",best_max_depth,"n_estimators:",best_n_estimators,"subsample:",best_subsample)

    # # make prediction on test data
    # pred_test = best_clf.predict(test_data_X)
    # pred_train = best_clf.predict(train_data_X)
    # pred_val = best_clf.predict(validation_data_X)
    # #print accuracy
    # print("Accuracy on train data: ", getAcc(pred_train,train_data_Y))
    # print("Accuracy on validation data: ", getAcc(pred_val,validation_data_Y))
    # print("Accuracy on test data: ", getAcc(pred_test,test_data_Y))

def main():
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    validation_data_path = sys.argv[3]
    output_path = sys.argv[4]
    question_path = sys.argv[5]
    train_data_file = train_data_path
    test_data_file = test_data_path
    validation_data_file = validation_data_path
    train_data_X,train_data_Y,header = get_data_pure(train_data_file)
    test_data_X,test_data_Y,_ = get_data_pure(test_data_file)
    validation_data_X,validation_data_Y,_ = get_data_pure(validation_data_file)

    if question_path=='a':
        dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path)
    elif question_path=='b':
        grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path)
    elif question_path=='c':
        dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path)
    elif question_path=='d':
        random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path)
    elif question_path =='e':
        #obrain data using imputing median
        train_data_X,train_data_Y,header = get_data_impute(train_data_file,"median")
        test_data_X,test_data_Y,_ = get_data_impute(test_data_file,"median")
        validation_data_X,validation_data_Y,_ = get_data_impute(validation_data_file,"median")
        #repeat parts a to d
        with open(output_path+"/1_e.txt","w") as f:
            f.write("median Imputation: \n Part a\n")
        dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_median")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart b\n")
        grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_median")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart c\n")
        dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_median")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart d\n")
        random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a",   "1_e_median")
        #obrain data using imputing mode
        train_data_X,train_data_Y,header = get_data_impute(train_data_file,"most_frequent")
        test_data_X,test_data_Y,_ = get_data_impute(test_data_file,"most_frequent")
        validation_data_X,validation_data_Y,_ = get_data_impute(validation_data_file,"most_frequent")
        #repeat parts a to d
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\n\n\nmode Imputation: \n Part a\n")
        dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_mode")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart b\n")
        grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_mode")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart c\n")
        dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_mode")
        with open(output_path+"/1_e.txt","a") as f:
            f.write("\nPart d\n")
        random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path,"/1_e.txt","a","1_e_mode")
    elif question_path =='f':
        xgboost_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,output_path)



main()