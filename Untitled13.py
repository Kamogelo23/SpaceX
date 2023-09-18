#!/usr/bin/env python
# coding: utf-8

# # **Space X  Falcon 9 First Stage Landing Prediction**

# # Machine learning predictions

# ## Objectives

# Perform exploratory  Data Analysis and determine Training Labels
# 
# *   create a column for the class
# *   Standardize the data
# *   Split into training data and test data
# 
# \-Find best Hyperparameter for SVM, Classification Trees and Logistic Regression
# 
# *   Find the method performs best using test data
# 

# # Lets define libraries and auxilary functions we will need

# In[2]:


get_ipython().system('pip install numpy pandas seaborn')


# ### Here we import all the libraries we will need

# In[3]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# ### Lets load the data frame

# In[9]:


get_ipython().system('pip install requests')
import requests
import io

URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = requests.get(URL1)
data1 = resp1.content  # This will give you the content in bytes
# If you want to convert it to a string:
data1_str = resp1.text
import pandas as pd

df1 = pd.read_csv(io.StringIO(resp1.text))


# In[11]:


df1.tail()


# In[12]:


URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = requests.get(URL2)
data2 = resp2.content  # This will give you the content in bytes
# If you want to convert it to a string:
data1_str = resp1.text
import pandas as pd

X = pd.read_csv(io.StringIO(resp2.text))


# In[14]:


X.head(100)


# In[17]:


Y = df1['Class'].to_numpy()


# In[19]:


# students get this 
transform = preprocessing.StandardScaler()
# Fit and transform the data
X = transform.fit_transform(X)


# #### Split the data into training and testing sets in a split ratio of 80:20

# In[20]:


# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[24]:


Y_test.shape


# #### Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[25]:


# 
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[26]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()


# In[29]:


# Create a GridSearchCV object
logreg_cv = GridSearchCV(lr, parameters, cv=10)

# Fit the GridSearchCV object to the data (assuming X_train and Y_train are your training data)
logreg_cv.fit(X_train, Y_train)

# To get the best parameters:
best_params = logreg_cv.best_params_
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# #### Calculate accuracy on the test data

# In[30]:


accuracy = logreg_cv.score(X_test, Y_test)
print("Accuracy on test data:", accuracy)


# #### Assess the confusion matrix

# In[31]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# Examining the confusion matrix, we see that logistic regression can distinguish between the different classes.  We see that the major problem is false positives.
# 

# In[35]:


# Parameters for GridSearchCV
parameters = {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-3, 3, 5)
}
# Create a support vector machine object
svm = SVC()

# Create a GridSearchCV object
svm_cv = GridSearchCV(svm, parameters, cv=10)

# Fit the GridSearchCV object to the data (assuming X_train and Y_train are your training data)
svm_cv.fit(X_train, Y_train)

# To get the best parameters:
best_params = svm_cv.best_params_
print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# On the test data (Grid  search on the test data)

# In[36]:


# Fit the GridSearchCV object to the data (assuming X_train and Y_train are your training data)
svm_cv.fit(X_test, Y_test)

# To get the best parameters:
best_params = svm_cv.best_params_
print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# #### Plot a confusion matrix

# In[37]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In this instance only one instance was predicted to land when it actually did not land

# In[38]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the Decision Tree Classifier
tree = DecisionTreeClassifier()

# Define the parameters for grid search
parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2 * n for n in range(1, 10)],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

# Create a GridSearchCV object with 10-fold cross-validation (cv = 10)
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to find the best parameters
tree_cv.fit(X_train, Y_train)  #Train data

# Print the best parameters and the corresponding best score
print("Best Parameters:", tree_cv.best_params_)
print("Best Score:", tree_cv.best_score_)


# Now we test the performance on the test set

# In[39]:


# Fit the GridSearchCV object to find the best parameters
tree_cv.fit(X_test, Y_test)  #Test data

# Print the best parameters and the corresponding best score
print("Best Parameters:", tree_cv.best_params_)
print("Best Score:", tree_cv.best_score_)


# In[41]:


# Define the KNN Classifier
KNN = KNeighborsClassifier()

# Define the parameters for grid search
parameters = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# Create a GridSearchCV object with 10-fold cross-validation (cv = 10)
knn_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to find the best parameters
knn_cv.fit(X_train, Y_train)  # Replace X and y with your dataset and target variable

# Print the best parameters and the corresponding best score
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# On the test data

# In[42]:


knn_cv.fit(X_test, Y_test)  # Replace X and y with your dataset and target variable

# Print the best parameters and the corresponding best score
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# #### Plot the confusion matrix

# In[43]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# This confusion matrix for this model has no False negative or False positives

# In[44]:


# Compare
knn_accuracy = knn_cv.score(X_test, Y_test)
tree_accuracy = tree_cv.score(X_test, Y_test) 
logreg_accuracy = logreg_cv.score(X_test, Y_test)
svm_accuracy = svm_cv.score(X_test, Y_test)

best_accuracy = max(tree_accuracy, logreg_accuracy, svm_accuracy, knn_accuracy)

if best_accuracy == logreg_accuracy:
    print("Logistic Regression performs best with accuracy:", logreg_accuracy)
elif best_accuracy == svm_accuracy:
    print("SVM performs best with accuracy:", svm_accuracy)
else:
    print("KNN performs best with accuracy:", knn_accuracy)


# #### The K nearest neighbour is the optimal model in this case
