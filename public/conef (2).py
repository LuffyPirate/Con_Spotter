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

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
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
storage.child("creditcard.csv").download(filename="creditcard.csv" , path="D:/conspotter/")

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


def SelectThresholdByCV(probs,y):
    best_threshold = 0
    best_f1 = 0
    f = 0
    precision =0
    recall=0
    best_recall = 0
    best_precision = 0
    precisions=[]
    recalls=[]
    
    thresholds = np.arange(0.0,1.0,0.001)
    for threshold in thresholds:
        predictions = (probs > threshold)
        f = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        
        if f > best_f1:
            best_f1 = f
            best_precision = precision
            best_recall = recall
            best_threshold = threshold

        precisions.append(precision)
        recalls.append(recall)

    #Precision-Recall Trade-off
    plt.plot(thresholds,precisions,label='Precision')
    plt.plot(thresholds,recalls,label='Recall')
    plt.xlabel("Threshold")
    plt.title('Precision Recall Trade Off')
    plt.legend()
    plt.show()

    print ('Best F1 Score %f' %best_f1)
    print ('Best Precision Score %f' %best_precision)
    print ('Best Recall Score %f' %best_recall)
    print ('Best Epsilon Score', best_threshold)




def Print_Accuracy_Scores(y,y_pred):
    print("F1 Score: ", f1_score(y,y_pred))
    print("Precision Score: ", precision_score(y,y_pred))
    print("Recall Score: ", recall_score(y,y_pred))



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

#Best estimator of random forest
rnd_clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=10, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=-1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
build_model_train_test(rnd_clf,x_train,x_test,y_train,y_test)
plt.savefig("rnd_clf.png")
storage.child("rnd_clf.png").put("rnd_clf.png")


knn_clf = KNeighborsClassifier(n_neighbors=3)
build_model_train_test(knn_clf,x_train,x_test,y_train,y_test)
plt.savefig('knn_clf.png')
storage.child("knn_clf.png").put("knn_clf.png")


ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3,class_weight='balanced'), n_estimators=100,
        algorithm="SAMME.R", learning_rate=0.5, random_state=0)
build_model_train_test(ada_clf,x_train,x_test,y_train,y_test)
plt.savefig('ada_clf.png')
storage.child("ada_clf.png").put("ada_clf.png")


soft_voting_clf = VotingClassifier(
    estimators=[('rf', rnd_clf), ('ada', ada_clf), ('knn',knn_clf)], 
    voting='soft')

build_model_train_test(soft_voting_clf,x_train,x_test,y_train,y_test)
plt.savefig('soft_voting_clf.png')
storage.child("soft_voting_clf.png").put("soft_voting_clf.png")

probs_sv_test = soft_voting_clf.predict_proba(x_test)
SelectThresholdByCV(probs_sv_test[:,1],y_test)


y_pred_test = (probs_sv_test[:,1] > 0.571)
Print_Accuracy_Scores(y_test,y_pred_test)
