# import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics, tree
from sklearn.neural_network import MLPClassifier
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  


from sklearn import datasets
import sklearn.ensemble as ske

from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  


def knn(n):




    titanic_data = pd.read_csv('./data/titanic_data.csv')

    titanic_data = titanic_data.drop('Cabin', 1)
    titanic_data = titanic_data.dropna()
    knn = KNeighborsClassifier(n_neighbors=n)
    le = preprocessing.LabelEncoder()
    label=le.fit_transform(titanic_data['Sex'].values)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(titanic_data['Age'].values, titanic_data['Pclass'].values, titanic_data['Fare'].values, label)), titanic_data['Survived'].values, test_size=0.3) # 70% training and 30% test
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    return [knn.score(X_train, y_train), knn.score(X_test, y_test), cross_val_score(knn, X_train, y_train).mean()]


    # fifa_data = pd.read_csv('./data/fifa.csv')
    # fifa_data.dropna()
    # features=list(zip(fifa_data['Finishing'].values, fifa_data['Volleys'].values, fifa_data['Curve'].values, fifa_data['Acceleration'].values, fifa_data['Agility'].values))
    # le = preprocessing.LabelEncoder()
    # label=le.fit_transform(fifa_data['Club'].values)
    # X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3) # 70% training and 30% tes
    # knn = KNeighborsClassifier(n_neighbors=100)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def nn(n):
    # cocao_data = pd.read_csv('./data/flavors_of_cacao.csv')
    # cocao_data = cocao_data.dropna()
    # mlp = MLPClassifier(hidden_layer_sizes=(1, 1),max_iter=500)
    # le = preprocessing.LabelEncoder()
    # specific_bean_origin = le.fit_transform(cocao_data['Specific Bean Origin or Bar Name'].values)
    # company = le.fit_transform(cocao_data['Company (Maker-if known)'].values)
    # cocoa_percent_string = cocao_data['Cocoa Percent'].values
    # cocoa_percent = []
    # for f in cocoa_percent_string:
    #     f = float(f.strip('%')) / 100.0
    #     cocoa_percent.append(f)

    # rating_broken = cocao_data['Rating'].values
    # rating = []
    # for f in rating_broken:
    #     f = float(f)
    #     rating.append(f)

    # features = list(zip(specific_bean_origin, rating, cocoa_percent))

    # X_train, X_test, y_train, y_test = train_test_split(features, company, test_size=0.3)

    # mlp.fit(X_train,y_train)
    # predictions = mlp.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, predictions))

    titanic_data = pd.read_csv('./data/titanic_data.csv')
    titanic_data = titanic_data.drop('Cabin', 1)
    titanic_data = titanic_data.dropna()
    mlp = MLPClassifier(hidden_layer_sizes=(n, n, n, n),max_iter=500)
    le = preprocessing.LabelEncoder()
    label=le.fit_transform(titanic_data['Sex'].values)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(titanic_data['Age'].values, titanic_data['Pclass'].values, titanic_data['Fare'].values, label)), titanic_data['Survived'].values, test_size=0.3) # 70% training and 30% test

    mlp.fit(X_train,y_train)
    # predictions = mlp.predict(X_test)
    # print '{0}'.format(metrics.accuracy_score(y_test, predictions))
    return [mlp.score(X_train, y_train), mlp.score(X_test, y_test), cross_val_score(mlp, X_train, y_train).mean()]

def svm(n):


    titanic_data = pd.read_csv('./data/titanic_data.csv')
    titanic_data = titanic_data.drop('Cabin', 1)
    titanic_data = titanic_data.dropna()
    le = preprocessing.LabelEncoder()
    label=le.fit_transform(titanic_data['Sex'].values)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(titanic_data['Age'].values, titanic_data['Pclass'].values, titanic_data['Fare'].values, label)), titanic_data['Survived'].values, test_size=0.3) # 70% training and 30% test
    svclassifier = SVC(kernel='rbf', degree=n)  
    svclassifier.fit(X_train, y_train)  
    # predictions = svclassifier.predict(X_test)
    # print '{0}'.format(metrics.accuracy_score(y_test, predictions))
    return [svclassifier.score(X_train, y_train), svclassifier.score(X_test, y_test), cross_val_score(svclassifier, X_train, y_train).mean()]
     #"Train Score: {0}".format()
    # print "Test Score: {0}".format(vclassifier.score(X_test, y_test))


    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    # irisdata = pd.read_csv(url, names=colnames)  
    # X = irisdata.drop('Class', axis=1)  
    # y = irisdata['Class']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
    # svclassifier = SVC(kernel='sigmoid', degree=8)  
    # svclassifier.fit(X_train, y_train)  
    # y_pred = svclassifier.predict(X_test)  
    # print(confusion_matrix(y_test, y_pred))  
    # print(classification_report(y_test, y_pred)) 



def decision_tree(n):
    titanic_data = pd.read_csv('./data/titanic_data.csv')
    titanic_data = titanic_data.drop('Cabin', 1)
    titanic_data = titanic_data.dropna()
    le = preprocessing.LabelEncoder()
    label=le.fit_transform(titanic_data['Sex'].values)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(titanic_data['Age'].values, titanic_data['Pclass'].values, titanic_data['Fare'].values, label)), titanic_data['Survived'].values, test_size=0.3) # 70% training and 30% test
    dec_tree = tree.DecisionTreeClassifier(max_depth=n)
    dec_tree.fit(X_train, y_train) 
    # print dec_tree.score(X_train, y_train) 
    # print dec_tree.score(X_test, y_test) 
    return [dec_tree.score(X_train, y_train), dec_tree.score(X_test, y_test), cross_val_score(dec_tree, X_train, y_train).mean()]
    # predictions = dec_tree.predict(X_test)
    # print '{0}'.format(cross_val_score(dec_tree, X_train, y_train).mean())



def decision_tree_boost(n):
    titanic_data = pd.read_csv('./data/titanic_data.csv')
    titanic_data = titanic_data.drop('Cabin', 1)
    titanic_data = titanic_data.dropna()
    le = preprocessing.LabelEncoder()
    label=le.fit_transform(titanic_data['Sex'].values)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(titanic_data['Age'].values, titanic_data['Pclass'].values, titanic_data['Fare'].values, label)), titanic_data['Survived'].values, test_size=0.3) # 70% training and 30% test
    dec_tree = ske.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), n_estimators= n)
    dec_tree.fit(X_train, y_train)  
    predictions = dec_tree.predict(X_test)
    # print '{0}'.format(metrics.accuracy_score(y_test, predictions))

    return [dec_tree.score(X_train, y_train), dec_tree.score(X_test, y_test), cross_val_score(dec_tree, X_train, y_train).mean()]












    # df = pd.read_csv('/Users/justinduan/Documents/MLProj1/train.csv')
    
    # df = df.drop(['Cabin', 'Ticket', 'Embarked'], axis=1)
    # df = df.dropna()
    
    # def preprocess_titanic_df(df):
    #     processed_df = df.copy()
    #     le = preprocessing.LabelEncoder()
    #     processed_df.Sex = le.fit_transform(processed_df.Sex)
    #     processed_df = processed_df.drop(['Name', 'PassengerId'],axis=1)
    
    #     return processed_df
    
    # processed = preprocess_titanic_df(df)
    
    # x = processed.drop(['Survived'], axis=1).values
    # y = processed['Survived'].values
    # X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    
    # model = tree.DecisionTreeClassifier() # edit model with features to change accuracy
    # model.fit(X_train, y_train)
    # model.score(X_test, y_test)
    
    #boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators= 50)
    #boost.fit
    
    # print(model.score(X_train, y_train))

train = []
test = []
cross = []
for x in range(1, 100):
    a, b, c = decision_tree_boost(x)
    # print a
    # print b
    # print c
    train.append(a)
    test.append(b)
    cross.append(c)

for a in train:
    print a

print "---------------------------------"

for a in test:
    print a

print "---------------------------------"

for a in cross:
    print a

print "---------------------------------"



# print np.mean(train)
# print np.mean(test)
# print np.mean(cross)

