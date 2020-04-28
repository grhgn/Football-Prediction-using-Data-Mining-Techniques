import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

def preprocessing(dataset):
    x = dataset[['FTHG','FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
    y = dataset[['FTR']]
    for i, row in y.iterrows():
        if (y.at[i, 'FTR'] == "H"):
            y.at[i, 'FTR'] = 1
        elif (y.at[i, 'FTR'] == "A"):
            y.at[i, 'FTR'] = 2
        elif (y.at[i, 'FTR'] == "D"):
            y.at[i, 'FTR'] = 3
    y = y.astype('int')
    return x, y

def KNN_Classifier(dataset):
    x, y = preprocessing(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
    print("Processing Classification")
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measures:", f1)
    print("")
    return [accuracy, precision, recall, f1]

def DTree_Classifier(dataset):
    x, y = preprocessing(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
    print("Processing Classification")
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measures:", f1)
    print("")
    return [accuracy, precision, recall, f1]

def MLP_Classifier(dataset):
    x, y = preprocessing(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
    print("Processing Classification")
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measures:", f1)
    print("")
    return [accuracy, precision, recall, f1]

def NB_Classifier(dataset):
    x, y = preprocessing(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
    print("Processing Classification")
    classifier = GaussianNB()
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measures:", f1)
    print("")
    return [accuracy, precision, recall, f1]

dataset14 = pd.read_csv('football 13-14.csv')
dataset15 = pd.read_csv('football 14-15.csv')
dataset16 = pd.read_csv('football 15-16.csv')
dataset = [dataset14, dataset15, dataset16]
i = 13
knn = []
dtree = []
mlp = []
nb = []
for data in dataset:
    print("Classifying Dataset ", i, "-", i+1)
    print("KNN Football ", i, "-", i+1)
    knn_temp = KNN_Classifier(data)
    knn.append(knn_temp)
    print("Decision Tree Football ", i, "-", i+1)
    dtree_temp = DTree_Classifier(data)
    dtree.append(dtree_temp)
    print("MLP Football ", i, "-", i+1)
    mlp_temp = MLP_Classifier(data)
    mlp.append(mlp_temp)
    print("NB Football ", i, "-", i+1)
    nb_temp = NB_Classifier(data)
    nb.append(nb_temp)
    i += 1

knn_pandas = pd.DataFrame(knn, columns = ['Accuracy', 'Precision', 'Recall', 'F-measures'])
dtree_pandas = pd.DataFrame(dtree, columns = ['Accuracy', 'Precision', 'Recall', 'F-measures'])
mlp_pandas = pd.DataFrame(mlp, columns = ['Accuracy', 'Precision', 'Recall', 'F-measures'])
nb_pandas = pd.DataFrame(nb, columns = ['Accuracy', 'Precision', 'Recall', 'F-measures'])
print("Rata-Rata KNN =  Accuracy:", knn_pandas["Accuracy"].mean(), ", Precision:", knn_pandas["Precision"].mean(), ", Recall:",
      knn_pandas["Recall"].mean(), ", F-measures:", knn_pandas["F-measures"].mean())
print("Rata-Rata Dtree =  Accuracy:", dtree_pandas["Accuracy"].mean(), ", Precision:", dtree_pandas["Precision"].mean(), ", Recall:",
      dtree_pandas["Recall"].mean(), ", F-measures:", dtree_pandas["F-measures"].mean())
print("Rata-Rata MLP =  Accuracy:", mlp_pandas["Accuracy"].mean(), ", Precision:", mlp_pandas["Precision"].mean(), ", Recall:",
      mlp_pandas["Recall"].mean(), ", F-measures:", mlp_pandas["F-measures"].mean())
print("Rata-Rata NB =  Accuracy:", nb_pandas["Accuracy"].mean(), ", Precision:", nb_pandas["Precision"].mean(), ", Recall:",
      nb_pandas["Recall"].mean(), ", F-measures:", nb_pandas["F-measures"].mean())
print("")

result = pd.concat(dataset, ignore_index=True, sort=False)
print("Classifying Dataset Keseluruhan")
print("KNN Football Keseluruhan")
knn_temp = KNN_Classifier(result)
print("Decision Tree Football Keseluruhan")
dtree_temp = DTree_Classifier(result)
print("MLP Football Keseluruhan")
mlp_temp = MLP_Classifier(result)
print("NB Football Keseluruhan")
nb_temp = NB_Classifier(result)
print("KNN Keseluruhan =  Accuracy:", knn_temp[0], ", Precision:", knn_temp[1], ", Recall:",
      knn_temp[2], ", F-measures:", knn_temp[3])
print("Decision Tree Keseluruhan =  Accuracy:", dtree_temp[0], ", Precision:", dtree_temp[1], ", Recall:",
      dtree_temp[2], ", F-measures:", dtree_temp[3])
print("MLP Keseluruhan =  Accuracy:", mlp_temp[0], ", Precision:", mlp_temp[1], ", Recall:",
      mlp_temp[2], ", F-measures:", mlp_temp[3])
print("NB Keseluruhan =  Accuracy:", nb_temp[0], ", Precision:", nb_temp[1], ", Recall:",
      nb_temp[2], ", F-measures:", nb_temp[3])

