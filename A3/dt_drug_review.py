#drug review with decision tree and countVectorizer

from cmath import inf
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from os.path import join
import csv
from sklearn.tree import DecisionTreeClassifier 
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


#month dictionary type float32
monthDict = {"January" : 1.0, "February" : 2.0, "March" : 3.0, "April" : 4.0, "May" : 5.0, "June" : 6.0, "July" : 7.0, "August" : 8.0, "September" : 9.0, "October" : 10.0, "November" : 11.0, "December" : 12.0}
# get data condition,review,rating,date,usefulCount
def remove_stop_words(data):
        #remove unnecessary characters (commas, full stop) and backslash character constants (carriage-return, linefeed)
        data = data.replace(',', '')
        data = data.replace('.', '')
        data = data.replace('\r', '')
        data = data.replace('\n', '')
        data = data.replace('\\', '')
        data = data.replace(';', '')
        data = data.replace('(', '')
        data = data.replace(')', '')
        data = data.replace(':', '')
        data = data.replace('"', '')
        #remove nltk stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = data.split()
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        if filtered_sentence==[]:
            return ""
        return " ".join(filtered_sentence)

# use scipy.sparse.csr_matrix to store sparse matrix for output of count vectorizer and then stack it with other inputs
def get_data(dataFile,vectorizer,lim_val=inf):
    #using vectorizer to get the data for validation
    file = open(dataFile)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    X_datas=[[],[],[]]
    Dates_val=[]
    Y=[]
    i =0
    # take row[0],row[1],row[3] and row[4] in X and row[2] in Y
    for row in csvreader:
        if i>=lim_val:
            break
        for j in [0,1]:
            X_datas[j].append(remove_stop_words(row[j]))
        X_datas[2].append([int(row[4])])
        Dates_val.append(row[3])
        Y.append(int(row[2]))
        i+=1
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    #Now use countvectorizer for X_datas[0] and X_datas[1] and store sparse form
    for j in [0,1]:
        X_datas[j] = vectorizer.transform(X_datas[j])
        # print(X_datas[j])
    X = X_datas[0]
    X=hstack((X,X_datas[1]))
    #convert dates
    dates_conv=[]
    for j in range(len(Dates_val)):
        temp = Dates_val[j].split()
        dates_conv.append([float(temp[1][:-1]),monthDict[temp[0]],float(temp[2])])
    dates_conv = csr_matrix(dates_conv)
    # convert X_datas[2] to sparse matrix form and then stack it with X
    X_datas[2] = csr_matrix(X_datas[2])
    X=hstack((X,dates_conv))
    X=hstack((X,X_datas[2]))
    
    # print(X.toarray())
    return X, Y,header

def obtain_count_vectorizer(dataFile,lim_val=inf):
    file = open(dataFile)
    csvreader = csv.reader(file)
    next(csvreader)
    X=[]
    lim_val =inf
    i =0
    # take row[0],row[1],row[3] and row[4] in X and row[2] in Y
    for row in csvreader:
        if i>=lim_val:
            break
        for j in [0,1]:
            X.append(remove_stop_words(row[j]))
        i+=1
    vectorizer = CountVectorizer(dtype=np.float32)
    vectorizer.fit(X)
    return vectorizer

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
def dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_a.txt",write_mode="w",q_part="2_a"):
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    print(clf)
    #tree using sklearn with validation data
    #train time
    start_time = time.time()
    clf = clf.fit(train_data_X, train_data_Y,)
    train_time = time.time() - start_time
    y_pred_train = clf.predict(train_data_X)
    y_pred_val = clf.predict(validation_data_X)
    y_pred_test = clf.predict(test_data_X)
    #report accuracy in a.txt and save graph
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Train Accuracy: "+str(getAcc(y_pred_train,train_data_Y))+"\n")
        f.write("Validation Accuracy: "+str(getAcc(y_pred_val,validation_data_Y))+"\n")
        f.write("Test Accuracy: "+str(getAcc(y_pred_test,test_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")
    #return test accuracy,time
    return getAcc(y_pred_test,test_data_Y),train_time

def grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_b.txt",write_mode="w",q_part="2_b"):    
    params = {
        'max_depth':  [115,120, 125,130],
        'min_samples_split': [2,4,6,8],
        'min_samples_leaf': [1,2,10],
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

    #train time
    train_time = 0
    for max_depth in params['max_depth']:
        for min_samples_split in params['min_samples_split']:
            for min_samples_leaf in params['min_samples_leaf']:
                clf = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
                start_time = time.time()
                clf.fit(train_data_X, train_data_Y)
                time_taken = time.time() - start_time
                yPredic = clf.predict(validation_data_X)
                acc = getAcc(yPredic,validation_data_Y)
                # print('max_depth:',max_depth,'min_samples_split:',min_samples_split,'min_samples_leaf:',min_samples_leaf,'acc:',acc)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_min_samples_leaf = min_samples_leaf
                    best_clf = clf
                    train_time = time_taken
    #report accuracy in b.txt and save graph
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Best max_depth: "+str(best_max_depth)+"\n")
        f.write("Best min_samples_split: "+str(best_min_samples_split)+"\n")
        f.write("Best min_samples_leaf: "+str(best_min_samples_leaf)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_clf.predict(validation_data_X),validation_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")


    # print('best_max_depth:',best_max_depth,'best_min_samples_split:',best_min_samples_split,'best_min_samples_leaf:',best_min_samples_leaf)

    # y_pred_train = best_clf.predict(train_data_X)
    # y_pred_val = best_clf.predict(validation_data_X)
    # y_pred_test = best_clf.predict(test_data_X)
    # print("accuracy by sklearn on training data is: ",getAcc(y_pred_train,train_data_Y))
    # print("accuracy by sklearn on validation data is: ",getAcc(y_pred_val,validation_data_Y))
    # print("accuracy by sklearn on test data is:",getAcc(y_pred_test,test_data_Y))
    #return test accuracy,time
    return getAcc(best_clf.predict(test_data_X),test_data_Y),train_time


#implement decision tree and pruning
def dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_c.txt",write_mode="w",q_part="2_c"):
    #train time
    start_time = time.time()
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    path = clf.cost_complexity_pruning_path(train_data_X, train_data_Y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # print(ccp_alphas)
    # print(impurities)
    clfs = []
    y_pred_val = []
    accuracies = []
    #take last 5 ccp alphas
    if len(ccp_alphas) > 5:
        ccp_alphas = ccp_alphas[-5:]
        impurities = impurities[-5:]

    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best', ccp_alpha=ccp_alpha)
        clf.fit(train_data_X, train_data_Y)
        y_pred_val.append(clf.predict(validation_data_X))
        accuracies.append(getAcc(y_pred_val[-1],validation_data_Y))
        clfs.append(clf)
    train_time = time.time() - start_time

    #plot total impurity of leaves vs effective alphas of pruned tree
    fig, ax = plt.subplots()
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q2{}impurity_vs_alpha.png".format(q_part))

    # Plot the number of nodes vs alpha
    node_counts = [clf.tree_.node_count for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("number of nodes")
    ax.set_title("Number of nodes vs alpha for training set")
    ax.plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q2{}nodes_vs_alpha.png".format(q_part))

    # Plot the depth of the tree vs alpha
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("depth of tree")
    ax.set_title("Depth vs alpha for training set")
    ax.plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    fig.savefig(outpath+"/Q2{}depth_vs_alpha.png".format(q_part))

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
    fig.savefig(outpath+"/Q2{}accuracy_vs_alpha.png".format(q_part))

    #report model with highest accuracy on validation set
    best_clf = clfs[np.argmax(val_accuracy)]
    best_alpha = ccp_alphas[np.argmax(val_accuracy)]
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Best alpha: "+str(best_alpha)+"\n")
        f.write("Best Depth: "+str(best_clf.tree_.max_depth)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_clf.predict(validation_data_X),validation_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")

    return getAcc(best_clf.predict(test_data_X),test_data_Y),train_time

def random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_d.txt",write_mode="w",q_part="2_d"):
    params = {
        'n_estimators' : [50,100,150,200,250,300,350,400,450], 
        'max_features' : [0.4,0.5,0.6,0.7,0.8],
        'min_samples_split' : [2,4,6,8,10],
    }

    best_oob_score = -1
    best_n_estimators = -1
    best_max_features = -1
    best_min_samples_split = -1
    best_model = None

    #train time
    train_time = 0
    # Use the out-of-bag accuracy (as explained in the class) to tune to the optimal values for these parameters. You should perform a grid search over the space of parameters
    for n_estimator in params['n_estimators']:
        for max_feature in params['max_features']:
            for min_sample_split in params['min_samples_split']:
                # print("n_estimator: ", n_estimator, "max_feature: ", max_feature, "min_sample_split: ", min_sample_split)
                clf = RandomForestClassifier(n_estimators=n_estimator, max_features=max_feature, min_samples_split=min_sample_split, oob_score=True)
                start_time = time.time()
                clf.fit(train_data_X, train_data_Y)
                time_taken = time.time() - start_time
                print("n_estimator: ", n_estimator, "max_feature: ", max_feature, "min_sample_split: ", min_sample_split, "oob_score: ", clf.oob_score_)

                if clf.oob_score_ > best_oob_score:
                    best_oob_score = clf.oob_score_
                    best_n_estimators = n_estimator
                    best_min_samples_split = min_sample_split
                    best_max_features = max_feature
                    best_model = clf
                    train_time = time_taken
    #report oobscore and accuracy on test set
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Best n_estimators: "+str(best_n_estimators)+"\n")
        f.write("Best max_features: "+str(best_max_features)+"\n")
        f.write("Best min_samples_split: "+str(best_min_samples_split)+"\n")
        f.write("Best oob_score: "+str(best_oob_score)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_model.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_model.predict(test_data_X),test_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(best_model.predict(validation_data_X),validation_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")

    return getAcc(best_model.predict(test_data_X),test_data_Y),train_time
        
def xgboost_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_e.txt",write_mode="w",q_part="2_e"):
    params = {
        'n_estimators':[50,100,150,200,250,300,350,400,450],
        'subsample': [0.4,0.5,0.6,0.7,0.8],
        'max_depth':  [ 40, 50, 60, 70],
    }
    # clf = GridSearchCV(
    #     estimator=XGBClassifier(),
    #     param_grid=params,
    # )
    # clf.fit(train_data_X, train_data_Y)
    # print(clf.best_params_)
    # best_clf = XGBClassifier(**clf.best_params_)
    # best_clf.fit(train_data_X, train_data_Y)

    #train time
    #hand written grid search
    best_accuracy = -1
    best_n_estimators = -1
    best_subsample = -1
    best_max_depth = -1
    best_clf = None
    train_time =0
    for n_estimators in params['n_estimators']:
        for subsample in params['subsample']:
            for max_depth in params['max_depth']:
                clf = XGBClassifier(n_estimators=n_estimators,subsample=subsample,max_depth=max_depth)
                start_time = time.time()
                clf.fit(train_data_X, train_data_Y)
                time_taken = time.time() - start_time
                yPredic = clf.predict(validation_data_X)
                accuracy = getAcc(yPredic,validation_data_Y)
                print("n_estimators: ",n_estimators," subsample: ",subsample," max_depth: ",max_depth," accuracy: ",accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_n_estimators = n_estimators
                    best_subsample = subsample
                    best_max_depth = max_depth
                    best_clf = clf
                    train_time = time_taken
    #report best params
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Best n_estimators: "+str(best_n_estimators)+"\n")
        f.write("Best subsample: "+str(best_subsample)+"\n")
        f.write("Best max_depth: "+str(best_max_depth)+"\n")
        f.write("Best Validation Accuracy: "+str(best_accuracy)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(best_clf.predict(train_data_X),train_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(best_clf.predict(test_data_X),test_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")
    # print("max_depth:",best_max_depth,"n_estimators:",best_n_estimators,"subsample:",best_subsample)

    # # make prediction on test data
    # pred_test = best_clf.predict(test_data_X)
    # pred_train = best_clf.predict(train_data_X)
    # pred_val = best_clf.predict(validation_data_X)
    # #print accuracy
    # print("Accuracy on train data: ", getAcc(pred_train,train_data_Y))
    # print("Accuracy on validation data: ", getAcc(pred_val,validation_data_Y))
    # print("Accuracy on test data: ", getAcc(pred_test,test_data_Y))

    return getAcc(best_clf.predict(test_data_X),test_data_Y),train_time

def light_bgm_train(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="2_f.txt",write_mode="w",q_part="2_f"):
    param_grid = [
    {'n_estimators': [500,1000,2000], 'max_depth' : [40,50,60,70], 'subsample' : [0.4,0.5,0.6,0.7,0.8]}  
    ]
    # manual grid search
    best_accuracy = -1
    best_n_estimators = -1
    best_subsample = -1
    best_max_depth = -1
    best_clf = None
    train_time = 0
    for n_estimators in param_grid[0]['n_estimators']:
        for subsample in param_grid[0]['subsample']:
            for max_depth in param_grid[0]['max_depth']:
                clf = lgb.LGBMClassifier(n_estimators=n_estimators,subsample=subsample,max_depth=max_depth)
                start_time = time.time()
                clf.fit(train_data_X, train_data_Y)
                end_time = time.time()
                yPredic = clf.predict(validation_data_X)
                accuracy = getAcc(yPredic,validation_data_Y)
                print("n_estimators: ",n_estimators," subsample: ",subsample," max_depth: ",max_depth," accuracy: ",accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_n_estimators = n_estimators
                    best_subsample = subsample
                    best_max_depth = max_depth
                    best_clf = clf
                    train_time = end_time - start_time

    # clfs = GridSearchCV(estimator = lgb.LGBMClassifier(),param_grid=param_grid,verbose = 2, n_jobs = 1)
    # clfs = clfs.fit(train_data_X, train_data_Y)
    # clf = clfs.best_estimator_

    # make prediction on test data
    pred_test = best_clf.predict(test_data_X)
    pred_train = best_clf.predict(train_data_X)
    pred_val = best_clf.predict(validation_data_X)
    #print accuracy
    with open(join(outpath,file_name),write_mode) as f:
        f.write("Best n_estimators: "+str(best_n_estimators)+"\n")
        f.write("Best subsample: "+str(best_subsample)+"\n")
        f.write("Best max_depth: "+str(best_max_depth)+"\n")
        f.write("Best Validation Accuracy: "+str(best_accuracy)+"\n")
        f.write("Best Train Accuracy: "+str(getAcc(pred_train,train_data_Y))+"\n")
        f.write("Best Test Accuracy: "+str(getAcc(pred_test,test_data_Y))+"\n")
        f.write("Best Validation Accuracy: "+str(getAcc(pred_val,validation_data_Y))+"\n")
        f.write("Train Time: "+str(train_time)+"\n")

    return getAcc(pred_test,test_data_Y),train_time

def train_on_varying_train_size(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name="g.txt",write_mode="w",q_part="2_g"):
    train_size = [20000,40000,60000,80000,100000,120000,140000,160000]
    accuracy = {'decision_tree':[],'grid_search':[],'pruning':[],'random_forest':[],'xgboost':[],'lightgbm':[]}
    train_time = {'decision_tree':[],'grid_search':[],'pruning':[],'random_forest':[],'xgboost':[],'lightgbm':[]}
    with open(join(outpath,file_name),"w") as f:
        f.write("Training on various sizes of training data\n")

    for size in train_size:
        # vectorizer = obtain_count_vectorizer(train_data_file,size)
        train_data_X_cur, train_data_Y_cur = train_data_X[:size],train_data_Y[:size]
        with open(join(outpath,file_name),"a") as f:
            f.write("\n\n\nTrain size: "+str(size)+"\n")
            f.write("part: a"+"\n")
        acc,t =dec_tree_construction(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['decision_tree'].append(acc)
        train_time['decision_tree'].append(t)
        with open(join(outpath,file_name),"a") as f:
            f.write("part: b"+"\n")
        acc,t = grid_search_dec_tree(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['grid_search'].append(acc)
        train_time['grid_search'].append(t)

        with open(join(outpath,file_name),"a") as f:
            f.write("part: c"+"\n")
        acc,t =dec_tree_pruning(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['pruning'].append(acc)
        train_time['pruning'].append(t)

        with open(join(outpath,file_name),"a") as f:
            f.write("part: d"+"\n")
        acc,t =random_forest_dec_tree(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['random_forest'].append(acc)
        train_time['random_forest'].append(t)

        with open(join(outpath,file_name),"a") as f:
            f.write("part: e"+"\n")
        acc,t =xgboost_dec_tree(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['xgboost'].append(acc)
        train_time['xgboost'].append(t)

        with open(join(outpath,file_name),"a") as f:
            f.write("part: f"+"\n")
        acc,t =light_bgm_train(train_data_X_cur,train_data_Y_cur,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,file_name=file_name,write_mode="a",q_part=q_part)
        accuracy['lightgbm'].append(acc)
        train_time['lightgbm'].append(t)

        #print accuracies
        # print("decision_tree accuracy on size{} is: ".format(size),accuracy['decision_tree'][-1])
        # print("grid_search accuracy on size{} is: ".format(size),accuracy['grid_search'][-1])
        # print("pruning accuracy on size{} is: ".format(size),accuracy['pruning'][-1])
        # print("random_forest accuracy on size{} is: ".format(size),accuracy['random_forest'][-1])
        # print("xgboost accuracy on size{} is: ".format(size),accuracy['xgboost'][-1])
        # print("lightgbm accuracy on size{} is: ".format(size),accuracy['lightgbm'][-1])
        #report accuracy

    # plot all accuracies vs train size and save in diffenet plots
    fig,ax = plt.subplots()
    ax.plot(train_size,accuracy['decision_tree'],label='decision_tree')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Train size for decision tree')
    ax.legend()
    fig.savefig(outpath+'/decison_tree_accuracy_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,accuracy['grid_search'],label='grid_search')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Train size for grid search')
    ax.legend()
    fig.savefig(outpath+'/grid_search_accuracy_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,accuracy['pruning'],label='pruning')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Train size for pruning')
    ax.legend()
    fig.savefig(outpath+'/pruning_accuracy_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,accuracy['random_forest'],label='random_forest')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Train size for random forest')
    ax.legend()
    fig.savefig(outpath+'/random_forest_accuracy_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,accuracy['xgboost'],label='xgboost')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for xgboost')
    ax.legend()
    fig.savefig(outpath+'/xgboost_accuracy_vs_train_size.png')
    plt.close(fig)



    # plot all train times vs train size and save
    fig,ax = plt.subplots()
    ax.plot(train_size,train_time['decision_tree'],label='decision_tree')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for decision tree')
    ax.legend()
    fig.savefig(outpath+'/decison_tree_time_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,train_time['grid_search'],label='grid_search')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for grid search')
    ax.legend()
    fig.savefig(outpath+'/grid_search_time_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,train_time['pruning'],label='pruning')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for pruning')
    ax.legend()
    fig.savefig(outpath+'/pruning_time_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,train_time['random_forest'],label='random_forest')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for random forest')
    ax.legend()
    fig.savefig(outpath+'/random_forest_time_vs_train_size.png')
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.plot(train_size,train_time['xgboost'],label='xgboost')
    ax.set_xlabel('Train size')
    ax.set_ylabel('Train Time')
    ax.set_title('Train Time vs Train size for xgboost')
    ax.legend()
    fig.savefig(outpath+'/xgboost_time_vs_train_size.png')
    plt.close(fig)



def main():
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    validation_data_path = sys.argv[3]
    outpath = sys.argv[4]
    question_part = sys.argv[5]
    train_data_file = train_data_path
    test_data_file = test_data_path
    validation_data_file = validation_data_path
    vectorizer = obtain_count_vectorizer(train_data_file)
    train_data_X, train_data_Y,header = get_data(train_data_file,vectorizer)
    test_data_X, test_data_Y,header = get_data(test_data_file,vectorizer)
    validation_data_X, validation_data_Y,header = get_data(validation_data_file,vectorizer)
    if question_part == "a":
        print("Question a")
        dec_tree_construction(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)
    elif question_part == "b":
        print("Question b")
        grid_search_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)
    elif question_part == "c":
        print("Question c")
        dec_tree_pruning(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)  
    elif question_part == "d":
        print("Question d")
        random_forest_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)  
    elif question_part == "e":
        print("Question e")
        xgboost_dec_tree(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)
    elif question_part == "f":
        print("Question f")
        light_bgm_train(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath)
    elif question_part =="g":
        print("Question g")
        train_on_varying_train_size(train_data_X,train_data_Y,test_data_X,test_data_Y,validation_data_X,validation_data_Y,header,outpath,question_part)

    

main()


