from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.models import load_model
import seaborn as sns
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import pickle


precision = []
recall = []
fscore = []
accuracy_list = []

main = tkinter.Tk()
main.title("Multivariate Gait Analysis for Healthy Subjects under Various Walking Conditions")
main.geometry("1000x650")

global filename
global fnn_model, mlp_model
global X_train, y_train, X_test, y_test

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))

def preprocess():
    global dataset
    global X_train, y_train, X_test, y_test ,y_train1,y_test1
    global y_test1,y_train1
    text.delete('1.0', END)
    X = dataset.drop("condition", axis=1).values
    Y = dataset['condition']
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    sns.countplot(x=Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END,"\n\nTotal Records for training : "+str(len(X_train))+"\n")
    text.insert(END,str(X_train))
    text.insert(END,"\n"+str(y_train))
    # Assuming y_train is your original target variable
    y_train1 = to_categorical(y_train, num_classes=3)
    # Assuming y_test is your original test target variable
    y_test1= to_categorical(y_test, num_classes=3)

def feedforwardneuralnetwork():
    
    global X_train, X_test, y_train, y_test
    
    text.delete('1.0', END)
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train1, epochs=10, batch_size=16)
    yhat_classes = model.predict_classes(X_test, verbose=0)
    accuracy = accuracy_score(y_test1, yhat_classes)
    ann_acc = accuracy * 100
    text.insert(END,"\n\nANN Accuracy : "+str(ann_acc)+"\n")
    precision = precision_score(y_test1, yhat_classes)
    precision = precision * 100
    text.insert(END,"ANN Precision : "+str(precision)+"\n")
    recall = recall_score(y_test1, yhat_classes)
    recall = recall * 100
    text.insert(END,"ANN Recall : "+str(recall)+"\n")
    if os.path.exists('model/ann_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
        #training the model
        hist = model.fit(X_train, y_train, batch_size = 32, epochs = 22, validation_split=0.2, callbacks=[model_check_point], verbose=1)
        f = open('model/ann_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        model.load_weights('model/ann_weights.hdf5')

    #model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)  # Adjust epochs and batch size as needed
    text.insert(END,"Total records found in ytest: ",str(X_test.shape)+"\n\n")

    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Accuracy: {accuracy*100:.2f}%')
    print("Shape of y_test1:", y_test.shape)
               
    # Make predictions on the test set
    predictions = model.predict_classes(X_test)
    print("Shape of predictions1:", predictions.shape)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    
    # Display the evaluation metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    
    cm = confusion_matrix(y_test,predictions)
    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ANN Confusion Matrix")
    plt.show()
    report = classification_report(y_test,predictions)
    
    text.insert(END, "ANN Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    
    text.insert(END, "ANN Classification Report:\n")
    text.insert(END, report)

def runMLP():
    global mlp_model
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc1 = accuracy_score(y_test,y_pred) * 100
    p1 = precision_score(y_test,y_pred, average='macro') * 100
    r1 = recall_score(y_test,y_pred, average='macro') * 100
    f1 = f1_score(y_test,y_pred, average='macro') * 100
    print(" Precision: " + str(p1))
    print(" Recall: " + str(r1))
    print( " F1-Score: " + str(f1))
    print(" Accuracy: " + str(acc1))
    precision.append(p1)
    accuracy_list.append(acc1)  # Using the renamed list
    recall.append(r1)
    fscore.append(f1)
    text.insert(END," Model  Accuracy = " + str(acc1) + "\n")
    text.insert(END," Model Precision = " + str(p1) + "\n")
    text.insert(END," Model Recall = " + str(r1) + "\n")
    text.insert(END," Model F1-Score = " + str(f1) + "\n")
    report=classification_report(y_test, y_pred)
    text.insert(END, f"MLP classification_report: {report}\n")
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    text.insert(END, f"MLP confusion_matrix: {cm}\n")
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('MLP Classifier Confusion Matrix')
    plt.show()
def graph():
    # Check if the lists have enough values
    # Create a DataFrame
    df = pd.DataFrame([
        ['MobileNet', 'Precision', precision[0]],
        ['MobileNet', 'Recall', recall[0]],
        ['MobileNet', 'F1 Score', fscore[0]],
        ['MobileNet', 'Accuracy', accuracy_list[0]],
        ['VGG16', 'Precision', precision[1]],
        ['VGG16', 'Recall', recall[1]],
        ['VGG16', 'F1 Score', fscore[1]],
        ['VGG16', 'Accuracy', accuracy_list[1]],
    ], columns=['Algorithms', 'Parameters', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Multivariate Gait Analysis for Healthy Subjects under Various Walking Conditions', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Multivariate Gait Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)


FNNButton = Button(main, text="Feed Forward Neural Network Algorithm", command=feedforwardneuralnetwork)
FNNButton.place(x=700,y=100)
FNNButton.config(font=font1)

MLPButton = Button(main, text="Run MLP Algorithm", command=runMLP)
MLPButton.place(x=480,y=100)
MLPButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph", command=graph)
graphButton.place(x=300,y=200)
graphButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=10,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
