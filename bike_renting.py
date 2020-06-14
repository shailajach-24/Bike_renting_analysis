#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


# In[2]:


root= tk.Tk() 
root.title("Bike renting analysis")
root.geometry("1300x1200")


# In[ ]:


global df_hour, df_day
def upload_data1():
    global df_hour
    df_hour = askopenfilename(initialdir = "Dataset")
    #pathlabel.config(text=train_data)
    text.insert(END,"Dataset loaded\n\n")
    
def upload_data2():
    global df_day
    text.delete('1.0',END)
    df_day = askopenfilename(initialdir = "Dataset")
    #pathlabel1.config(text=test_data)
    text.insert(END,"New Dataset loaded\n\n")
    
def data():
    global df_hour,df_day,df
    text.delete('1.0',END)
    df_hour = pd.read_csv("hour.csv")
    df_day = pd.read_csv("day.csv")
    df_hour.drop('instant',axis=1,inplace=True)
    df=pd.merge(df_day,df_hour,how='left',left_on='dteday',right_on='dteday')
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,df.head())
    text.insert(END,"column names\n\n")
    text.insert(END,df.columns)
    text.insert(END,"Total no. of rows and coulmns\n\n")
    text.insert(END,df.shape)
def statistics():
    text.delete('1.0',END)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,df.head())
    stats=df.describe()
    text.insert(END,"\n\nStatistical Measurements for Data\n\n")
    text.insert(END,stats)
    null=df.isnull().sum()
    text.insert(END,null)   
def train_test():
    text.delete('1.0',END)
    global x,y
    global x_train,x_test,y_train,y_test
    text.delete('1.0',END)
    x=df.drop(['dteday','cnt_y'],axis=1)
    y=df['cnt_y']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
    text.insert(END,"Train and Test model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(df))+"\n")
    text.insert(END,"Training Size : "+str(len(x_train))+"\n")
    text.insert(END,"Test Size : "+str(len(x_test))+"\n")
    return x_train,x_test,y_train,y_test   

def LR():
    text.delete('1.0',END)
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    predictions=lm.predict(x_test)
    res = pd.DataFrame(predictions)
    res.to_csv("prediction_results.csv")
    res['season_x']=x['season_x']
    #res.index = X_test.index # its important for comparison
    res['predictions'] = predictions
    res.to_csv("LRprediction_results.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,res)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    
    
def KNN():
    text.delete('1.0',END)
    regressor = KNeighborsRegressor()
    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)
    res = pd.DataFrame(predictions)
    res.to_csv("KNNprediction_results.csv")
    res['season_x']=x['season_x']
    #res.index = X_test.index # its important for comparison
    res['predictions'] = predictions
    res.to_csv("prediction_results.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,res)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    
def RFT():
    text.delete('1.0',END)
    global new_x_test,new_x_train,y_train
    
    regressor = RandomForestRegressor(max_depth=5,n_estimators = 10, random_state = 0)
    regressor.fit(x_train, y_train)
    features = pd.DataFrame()
    features['Feature'] = x_train.columns
    features['Importance'] = regressor.feature_importances_
    features.sort_values(by=['Importance'], ascending=False, inplace=True)
    features.set_index('Feature', inplace=True)
    text.insert(END,"Selected Important Features Automatically by using *feature_importances_* & *SelectFromModel*\n\n")
    text.insert(END,features[:3])
    selector = SelectFromModel(regressor, prefit=True)
    train_reduced = selector.transform(x_train)
    new_x_train=pd.DataFrame(train_reduced,columns=['registered_y','casual_y'])
    test_reduced = selector.transform(x_test)
    new_x_test=pd.DataFrame(test_reduced,columns=['registered_y','casual_y'])
    #new_reduced=selector.transform(New_data)
    #new_data=pd.DataFrame(new_reduced,columns=['registered_y','casual_y','casual_x'])
    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 10,
              'min_samples_split': 6,
              'max_features': 'sqrt',
              'max_depth': 5}

    rf = RandomForestRegressor(**parameters)
    rf.fit(new_x_train, y_train)
    predictions=rf.predict(new_x_test)
    
    
    
    res = pd.DataFrame(predictions)
    res.to_csv("prediction_results.csv")
    res['season_x']=x['season_x']
    #res.index = X_test.index # its important for comparison
    res['predictions'] = predictions
    res.to_csv("RFTprediction_results.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,res)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    
def lasso():
    # Lasso
    text.delete('1.0',END)
    from sklearn.linear_model import Lasso
    lassoReg = Lasso(alpha=0.1, normalize=True)
    lassoReg.fit(x_train,y_train)
    predictions = lassoReg.predict(x_test)
    res = pd.DataFrame(predictions)
    res.to_csv("lassoprediction_results.csv")
    res['season_x']=x['season_x']
    #res.index = X_test.index # its important for comparison
    res['predictions'] = predictions
    res.to_csv("prediction_results.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,res)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

def input_values():
    text.delete('1.0',END)
    global new_x_train,new_x_test
    global RFT
     
    
    global registered_y#our 2nd input variable
    registered_y = float(entry1.get())

    global casual_y 
    casual_y = float(entry2.get())

    #global casual_x
    #casual_x = float(entry3.get())

    list1=[[registered_y,casual_y]]
    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}
    
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(new_x_train)
    x_test = sc_X.transform(new_x_test)
    rf = RandomForestRegressor(**parameters)
    rf.fit(x_train, y_train)
    Prediction_result  = rf.predict(list1)
    text.insert(END,"New values are predicted from Random Forest Regressor\n\n")
    text.insert(END,"Predicted cnt_y for the New inputs\n\n")
    text.insert(END,Prediction_result)

    
font = ('times', 14, 'bold')
title = Label(root, text='Bike renting Using Machine Learning')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times',13 ,'bold')
button1 = tk.Button (root, text='Upload Data1',width=13,command=upload_data1) 
button1.config(font=font1)
button1.place(x=60,y=100)

button2 = tk.Button (root, text='Upload Data2',width=13,command=upload_data2)
button2.config(font=font1)
button2.place(x=60,y=150)

button3 = tk.Button (root, text='Data',width=13,command=data)  
button3.config(font=font1)
button3.place(x=60,y=200)


button4 = tk.Button (root, text='statistics',width=13,command=statistics)
button4.config(font=font1) 
button4.place(x=60,y=250)

button5 = tk.Button (root, text='Train & Test',width=13,command=train_test)
button5.config(font=font1) 
button5.place(x=60,y=300)

title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)

button6 = tk.Button (root, text='Linear Regression',width=15,bg='pale green',command=LR)
button6.config(font=font1) 
button6.place(x=300,y=100)

button7 = tk.Button (root, text='KNN',width=15,bg='sky blue',command=KNN)
button7.config(font=font1) 
button7.place(x=300,y=150)

button8 = tk.Button (root, text='RFT',width=15,bg='orange',command=RFT)
button8.config(font=font1) 
button8.place(x=300,y=200)

button9 = tk.Button (root, text='Lasso',width=15,bg='violet',command=lasso)
button9.config(font=font1) 
button9.place(x=300,y=250)



title = Label(root, text='Enter Input values for the New Prediction')
title.config(bg='black', fg='white')  
title.config(font=font1)           
title.config(width=40)       
title.place(x=60,y=380)

font3=('times',9,'bold')
title1 = Label(root, text='*You Should enter scaled values between 0 and 1')
 
title1.config(font=font3)           
title1.config(width=40)       
title1.place(x=50,y=415)

def clear1(event):
    entry1.delete(0, tk.END)

font2=('times',10)
entry1 = tk.Entry (root) # create 1st entry box
entry1.config(font=font2)
entry1.place(x=60, y=450,height=30,width=150)
entry1.insert(0,'registered_y')
entry1.bind("<FocusIn>",clear1)

def clear2(event):
    entry2.delete(0, tk.END)

font2=('times',10)
entry2 = tk.Entry (root) # create 1st entry box
entry2.config(font=font2)
entry2.place(x=315, y=450,height=30,width=150)
entry2.insert(0,'casual_y')
entry2.bind("<FocusIn>",clear2)



Prediction = tk.Button (root, text='Prediction',width=15,fg='white',bg='green',command=input_values)
Prediction.config(font=font1) 
Prediction.place(x=180,y=550)



font1 = ('times', 11, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

root.mainloop()


