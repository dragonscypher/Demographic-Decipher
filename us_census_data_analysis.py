
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
print("Loading Dataset...")
X, y = load_svmlight_file("a9a")

# Convert to dense
X_dense = X.todense()

# Split dataset for train and test evaluation at the end
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.3, random_state=42)

# SVM with various kernels
kernels = ['linear', 'rbf', 'poly']
C_values = [1]
gammas = [0.1, 0.01]
degrees = [2]
results = []

for kernel in kernels:
    if kernel == 'rbf':
        for gamma in gammas:
            clf = svm.SVC(kernel=kernel, C=1, gamma=gamma, random_state=42)
            y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
            acc = accuracy_score(y_train, y_pred)
            results.append(('SVM with RBF kernel, gamma=' + str(gamma), acc))
    elif kernel == 'poly':
        for degree in degrees:
            clf = svm.SVC(kernel=kernel, C=1, degree=degree, random_state=42)
            y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
            acc = accuracy_score(y_train, y_pred)
            results.append(('SVM with Poly kernel, degree=' + str(degree), acc))
    else:
        clf = svm.SVC(kernel=kernel, C=1, random_state=42)
        y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
        acc = accuracy_score(y_train, y_pred)
        results.append(('SVM with ' + kernel + ' kernel', acc))

# Decision Tree Classifier with Pipeline and GridSearchCV
print("Decision Tree Classifier with Pipeline and GridSearchCV...")
pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
param_grid = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__max_depth': [5, 10, 15, 20]
}
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)
dt_acc = grid_search.best_score_
results.append(('Decision Tree', dt_acc))

# Gaussian Naive Bayes Classifier
print("Gaussian Naive Bayes Classifier...")
gnb_pipeline = make_pipeline(StandardScaler(), GaussianNB())
gnb_scores = cross_val_score(gnb_pipeline, X_train, y_train, cv=10)
gnb_acc = np.mean(gnb_scores)
results.append(('GaussianNB', gnb_acc))

# Results visualization
labels, accs = zip(*results)
plt.figure(figsize=(10, 8))
plt.barh(labels, accs, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Comparison on US Census Data')
plt.show()
