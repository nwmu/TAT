from tkinter import *
import tkinter
from tkinter.filedialog import askopenfilename

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn import svm

import xlsxwriter

import pylab as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
pl.style.use('fivethirtyeight')
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title("T & T")
main.geometry("1300x1200")

class test:
    names=[]
    mean=[]
    sd=[]
    accuracy=[]
    precision=[]
    Sensitivity_recall=[]
    Specificity=[]
    F1_score=[]

    def upload():
        global filename
        text.delete('1.0', END)
        filename = askopenfilename(initialdir = "Dataset")
        pathlabel.config(text=filename)
        text.insert(END,"Dataset loaded\n\n")

    def csv():
        global data
        text.delete('1.0', END)
        data=pd.read_csv(filename)
        text.insert(END,"Top Five rows of dataset\n"+str(data.head())+"\n")
        text.insert(END,"Last Five rows of dataset\n"+str(data.tail()))

    def splitdataset():
        text.delete('1.0', END)
        X = data.iloc[:,:-1] 
        Y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 7)
        text.insert(END,"\nTrain & Test Model Generated\n\n")
        text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
        text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
        text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")
        return X_train, X_test, y_train, y_test

    def MLmodels():
        X_train, X_test, y_train, y_test=test.splitdataset()
        text.delete('1.0', END)
        models=[]	   
        models.append(('KNeighborsClassifier',KNeighborsClassifier(n_neighbors=13)))
        models.append(('LinearSvc',LinearSVC()))
        models.append(('SVC',SVC()))
        models.append(('Random Forest',RandomForestClassifier()))
        models.append(('Decision Tree Classifier',DecisionTreeClassifier()))
        models.append(('Bagging classifier',BaggingClassifier()))
        models.append(('Ada',AdaBoostClassifier()))
        models.append(('MLP',MLPClassifier()))
        models.append(('Navie Bayes ' ,GaussianNB()))
        models.append(('SGD',SGDClassifier()))
        models.append(('QDA',QuadraticDiscriminantAnalysis()))
        results=[]
        predicted_values=[]
        text.insert(END,"Machine and deep Learning Classification Models\n")
        text.insert(END,"Predicted values,Accuracy Scores and S.D values \n\n")
        for name,model in models:
            kfold=KFold(n_splits=10,shuffle=True,random_state=7)
            cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
            model.fit(X_train,y_train)
            predicted=model.predict(X_test)
            predicted_values.append(predicted)
            test.names.append(name)
            text.insert(END,"\n\t\t"+str(name)+" \n\n")
            data = {'actual': y_test,'predicted': predicted}
            df = pd.DataFrame(data)
            confusion_matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'], colnames=['Predicted'])
            text.insert(END, "%s\n confusion matrix :\n %s \n mean : %f \n SD : %f\n " %(name,confusion_matrix,cv_results.mean()*100,cv_results.std(),))
            test.mean.append(cv_results.mean()*100)
            test.sd.append(cv_results.std())
            Accuracy = metrics.accuracy_score(y_test, predicted)
            test.accuracy.append(Accuracy)
            Precision = metrics.precision_score(y_test, predicted)
            test.precision.append(Precision)
            Sensitivity_recall = metrics.recall_score(y_test, predicted)
            test.Sensitivity_recall.append(Sensitivity_recall)
            Specificity = metrics.recall_score(y_test, predicted, pos_label=0)
            test.Specificity.append(Specificity)
            F1_score = metrics.f1_score(y_test, predicted)
            test.F1_score.append(F1_score)
            text.insert(END,"\nAccuracy : %f \nPrecision : %f \nSensitivity_recall : %f \nSpecificity : %f \n F1_score : %f \n" %(Accuracy*100,Precision,Sensitivity_recall,Specificity,F1_score,))
            results.append(Accuracy*100)
        return results

    def graph():
        results=test.MLmodels()
        bars = ('KNN','LinearSvc','SVC','RF','DT','BC','Ada','MLP','GNB','SDG','QDA')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, results)
        plt.xticks(y_pos, bars)
        plt.show()


    def excel():
        dict ={
        'Names' : test.names, 
        'Mean' : test.mean,
        'Standarad Deviation ' : test.sd, 
        'Accuracy ' : test.accuracy ,
        'Precision ': test.precision,
        'Sensitivity': test.Sensitivity_recall,
        'Specificity ': test.Specificity,
        'F1_score':test.F1_score}
        df = pd.DataFrame(dict)
        workbook = xlsxwriter.Workbook('report.xlsx')
        df.to_excel(r'C:\Users\nandu\Desktop\WMU\Adv Stor, Ret, Pro of Big Data\project\report.xlsx',index=False)
        text.insert(END,"report generated\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='Test & Trial Algorithms')
title.config(bg='sky blue', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=test.upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

df = Button(main, text="Reading Data ", command=test.csv)
df.place(x=700,y=200)
df.config(font=font1)

split = Button(main, text="Train_Test_Split ", command=test.splitdataset)
split.place(x=700,y=250)
split.config(font=font1)

ml= Button(main, text="All Classifiers", command=test.MLmodels)
ml.place(x=700,y=300)
ml.config(font=font1) 

graph= Button(main, text="Model Comparison", command=test.graph)
graph.place(x=700,y=350)
graph.config(font=font1)

report = Button(main, text="Generate Report" , command=test.excel )
report.place(x=700,y=400)
report.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='pale goldenrod')
main.mainloop()
