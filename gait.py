
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
import numpy as np 
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import KernelPCA



main = tkinter.Tk()
main.title("Artificial Neural Network Model for Prediction of Knee Osteoarthritis Progression on 3D MRI Data") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, naive_acc, ann_acc # all global variables names define in above lines

def traintest(train):     #method to generate test and train data from dataset
    X = train.drop("condition", axis=1).values
    Y = train['condition']
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    print(Y.unique())
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.delete('1.0', END)
    text.insert(END,"Data read and model generated \n")


def upload(): #function to upload tweeter profile
    global filename
    filename = askopenfilename(initialdir = "")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def applyPCA():
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];
    text.insert(END,"Total Features : "+str(total)+"\n")
    pca = KernelPCA(n_components=6, kernel='linear')
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    text.insert(END,"Features set reduce after applying features PCA concept : "+str((total - X_train.shape[1]))+"\n")
    


def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n\n\n\n")  
    return accuracy        

def runSVM():
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC() 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    
    svm_acc = accuracy_score(y_test, prediction_data)
    svm_acc = svm_acc * 100
    text.insert(END,"\n\nSVM Accuracy : "+str(svm_acc)+"\n")
    precision = precision_score(y_test, prediction_data)
    precision = precision * 100
    text.insert(END,"SVM Precision : "+str(precision)+"\n")
    recall = recall_score(y_test, prediction_data)
    recall = recall * 100
    text.insert(END,"SVM Recall : "+str(recall)+"\n")
#   svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix')
    
               
def runRandomForest():
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=0.9,random_state=None) 
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix')    

def runNaiveBayes():
    global naive_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
#   naive_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Algorithm Accuracy, Classification Report & Confusion Matrix')   
    naive_acc = accuracy_score(y_test, prediction_data)
    naive_acc = naive_acc * 100
    text.insert(END,"\n\nNaive Bayes Accuracy : "+str(naive_acc)+"\n")
    precision = precision_score(y_test, prediction_data)
    precision = precision * 100
    text.insert(END,"Naive Bayes Precision : "+str(precision)+"\n")
    recall = recall_score(y_test, prediction_data)
    recall = recall * 100
    text.insert(END,"Naive Bayes Recall : "+str(recall)+"\n")
    classifier = cls

def runANN():
    global ann_acc
    text.delete('1.0', END)
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=64)
    yhat_classes = model.predict_classes(X_test, verbose=0)
    accuracy = accuracy_score(y_test, yhat_classes)
    ann_acc = accuracy * 100
    text.insert(END,"\n\nANN Accuracy : "+str(ann_acc)+"\n")
    precision = precision_score(y_test, yhat_classes)
    precision = precision * 100
    text.insert(END,"ANN Precision : "+str(precision)+"\n")
    recall = recall_score(y_test, yhat_classes)
    recall = recall * 100
    text.insert(END,"ANN Recall : "+str(recall)+"\n")
    

def runPrediction():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 0:42] 
    text.insert(END,filename+" test file loaded\n");
    y_pred = classifier.predict(test)
    index = 0
    for i in range(len(test)):
        print(y_pred[i]) 
        if str(y_pred[i]) == '1.0':
            text.insert(END,'Test Record No : '+str(index)+' Predicted : Cartilage Change/Progression Predicted\n')
        else:
            text.insert(END,'Test Record No : '+str(index)+' Predicted : No Cartilage Change/Progression Predicted\n')
        index = index + 1    
    

def graph():
    height = [svm_acc,naive_acc,ann_acc]
    bars = ('SVM','Naive Bayes','Proposed ANN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

   
font = ('times', 16, 'bold')
title = Label(main, text='Artificial Neural Network Model for Prediction of Knee Osteoarthritis Progression on 3D MRI Data')
title.config(bg='DarkSeaGreen', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload OAI Knee MRI Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkSeaGreen', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=360,y=100)

extractButton = Button(main, text="Read Dataset & Build Model", command=generateModel)
extractButton.place(x=50,y=150)
extractButton.config(font=font1)
'''
pearsonButton = Button(main, text="Apply PCA", command=applyPCA)
pearsonButton.place(x=320,y=150)
pearsonButton.config(font=font1) 
'''
runsvm = Button(main, text="Run SVM", command=runSVM)
runsvm.place(x=440,y=150)
runsvm.config(font=font1) 
'''
runrandomforest = Button(main, text="Run Random Forest Classifier", command=runRandomForest)
runrandomforest.place(x=640,y=150)
runrandomforest.config(font=font1) 
'''
runnb = Button(main, text="Run Naive Bayes Classifier", command=runNaiveBayes)
runnb.place(x=50,y=200)
runnb.config(font=font1) 

annButton = Button(main, text="Run ANN Model", command=runANN)
annButton.place(x=320,y=200)
annButton.config(font=font1) 

predictButton = Button(main, text="Predict Progression of OA", command=runPrediction)
predictButton.place(x=520,y=200)
predictButton.config(font=font1)

graph = Button(main, text="Accuracy Comparison", command=graph)
graph.place(x=770,y=200)
graph.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='DarkSeaGreen')
main.mainloop()
