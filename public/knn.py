import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import pyrebase

config = {
    "apiKey": "AIzaSyD9kyifI27SA7nT4zwHeCewFeiNSZ1dXdI",
    "authDomain": "con-spotter.firebaseapp.com",
    "databaseURL": "https://con-spotter-default-rtdb.firebaseio.com",
    "projectId": "con-spotter",
    "storageBucket": "con-spotter.appspot.com",
    "messagingSenderId": "251669384282",
    "appId": "1:251669384282:web:c807a8af700d1581795c63",
    "measurementId": "G-6YJY0FHLKC"
  }

firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()
storage.child("creditcard.csv").download(filename="creditcard.csv" , path="D:/ConSpotter/public/")

LABELS = ["Normal", "Fraud"]

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(0)

def build_model_train_test(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)

    y_pred = model.predict(x_train)

    print("\n----------Accuracy Scores on Train data------------------------------------")

    print("F1 Score: ", f1_score(y_train,y_pred))
    print("Precision Score: ", precision_score(y_train,y_pred))
    print("Recall Score: ", recall_score(y_train,y_pred))

    print("\n----------Accuracy Scores on Cross validation data------------------------------------")
    y_pred_cv = cross_val_predict(model,x_train,y_train,cv=5)
    print("F1 Score: ", f1_score(y_train,y_pred_cv))
    print("Precision Score: ", precision_score(y_train,y_pred_cv))
    print("Recall Score: ", recall_score(y_train,y_pred_cv))


    print("\n----------Accuracy Scores on Test data------------------------------------")
    y_pred_test = model.predict(x_test)

    print("F1 Score: ", f1_score(y_test,y_pred_test))
    print("Precision Score: ", precision_score(y_test,y_pred_test))
    print("Recall Score: ", recall_score(y_test,y_pred_test))

    #Confusion Matrix
    plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1,2)

    ax1 = plt.subplot(gs[0])
    cnf_matrix = confusion_matrix(y_train,y_pred)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Train Data")

    ax3 = plt.subplot(gs[1])
    cnf_matrix = confusion_matrix(y_test,y_pred_test)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)
    plt.title("Normalized Confusion Matrix - Test Data")



cc_dataset = pd.read_csv("creditcard.csv")
cc_dataset.shape
cc_dataset.head()
cc_dataset.describe()
cc_dataset.isnull().values.any()
cc_dataset['Class'].value_counts()



count_classes = pd.value_counts(cc_dataset['Class'], sort = True)
count_classes.plot(kind = 'bar')
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.savefig('Fd_Nl.png')
storage.child("Fd_Nl.png").put("Fd_Nl.png")


#Splitting the input features and target label into different arrays
X = cc_dataset.iloc[:,0:-1]
Y = cc_dataset.iloc[:,-1]
X.columns

#Train Test split - By default train_test_split does STRATIFIED split based on label (y-value).
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling - Standardizing the scales for all x variables
#PN: We should apply fit_transform() method on train set & only transform() method on test set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



knn_clf = KNeighborsClassifier(n_neighbors=3)
build_model_train_test(knn_clf,x_train,x_test,y_train,y_test)
plt.savefig('knn_clf.png')
storage.child("knn_clf.png").put("knn_clf.png")
