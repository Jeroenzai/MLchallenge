from os import chdir
chdir("C:/Users/Jeroen/Desktop/ML Challenge")
## Load pacakges
import numpy as np

#visualization
import matplotlib
import matplotlib.pyplot as plt

#Preprocessing
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Models:
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Model testing:
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

## Load datasets:
written_train = np.load("written_train.npy")
spoken_train = np.load("spoken_train.npy")
match_train = np.load("match_train.npy")

###FUNCTIONS

#get summary of spoken data with mean, max min, std, skew and kurtosis of all channels
def summary_spoken(data, shape):
  N_FRAMES = 4
  n_channels = data[0].shape[1]
  
  summarized = np.empty((shape))
  for i in range(len(data)):   
    #creating features
    means = np.mean(data[i], axis = 0)
    maxs = np.max(data[i], axis = 0)
    mins = np.min(data[i], axis = 0)
    stds = np.std(data[i], axis = 0)
    skews = skew(data[i], axis = 0)
    kurts = kurtosis(data[i], axis = 0)
    frames = data[i][:N_FRAMES].reshape(n_channels * N_FRAMES,)
    #building the dataset
    summarized[i] = np.hstack([means, maxs, mins, stds, skews, kurts, frames])
    
  return summarized

#combine the spoken and written data
def combine_data(data_a, data_b):
  return np.hstack([data_a, data_b])

#upsample the observations with true as a label and stack with false observations
def upsample(X, y):
  from sklearn.utils import resample
  #upsample true's and create y_true and stack together
  X_true = X[y]
  X_true_resample = resample(X_true, n_samples = sum(y == False), random_state = 40)
  y_true = np.full((X_true_resample.shape[0], 1), True)
  Xy_true = np.hstack([X_true_resample, y_true])
  
  #subset false and create y_false and stack together
  X_false = X[y != True]
  y_false = np.full((X_false.shape[0],1), False)
  Xy_false = np.hstack([X_false, y_false])
  
  #combine both datasets
  Xy_data = np.vstack([Xy_false, Xy_true])
  
  #Shuffle the concatenated data
  np.random.seed(40)
  np.random.shuffle(Xy_data)
  
  X_data = Xy_data[:,:-1]
  y_data = Xy_data[:,-1]
  
  return X_data, y_data

#standardize summary of spoken data and combine with written data
def standardize_spoken(train, val):
  #n of colums for spoken and written_summary
  N_COL_SPOKEN = spoken_summary_train.shape[1]
  #use this as index
  index = train.shape[1] - N_COL_SPOKEN
  
  #standardize
  zscore = StandardScaler()
  #transform the spoken data
  Z_train_spoken = zscore.fit_transform(train[:,index:])
  Z_val_spoken = zscore.transform(val[:,index:])
  #combine with written data
  X_train = np.hstack([train[:,:index], Z_train_spoken])
  X_val = np.hstack([val[:,:index], Z_val_spoken])
  
  return X_train, X_val
  
### EXPLORATION

#written data
written_train.shape
#explore a row:
written_train[0].shape
written_train[0]
#visualise a number:
plt.imshow(written_train[0].reshape(28, 28), cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")

#spoken data
spoken_train.shape
spoken_train[0]
spoken_train[0].shape
#the matrices have variabele lengths
len(spoken_train[0])
len(spoken_train[1])

#match train
np.mean(match_train == True)


### PREPROCESSING

#summarize spoken data
spoken_summary_train = summary_spoken(spoken_train, (spoken_train.shape[0], 13 * (6 + 4)))

#clean noise
#written_train[written_train < 50] = 0

#binarize (clean noise)
written_train = (written_train > 50).astype(int)
#cleaned noise
plt.imshow(written_train[0].reshape(28, 28), cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")

#combine written and spoken summary data
X_data = combine_data(written_train, spoken_summary_train)

## Split data to create an internal validation set (hold-out method):
X_full_train, X_test, y_full_train, y_test = train_test_split(X_data, match_train, test_size = 0.2, random_state = 40)
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size = 0.2, random_state = 40)

#upsample train data
X_train, y_train = upsample(X_train, y_train)

#Original z-scoring
""" 
Z-score (if not binarized, apply):
zscore = StandardScaler()
X_train = zscore.fit_transform(X_train)
X_val = zscore.transform(X_val)
"""

#Z-scoring of train and validation data except of binarized data
X_train, X_val = standardize_spoken(X_train, X_val)


### MODELLING

#Logistic Regression:
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_val)
f1_score(y_val, y_pred_log)
print(classification_report(y_val, y_pred_log))

#Stochastic Gradient Descent:
sgd = SGDClassifier(random_state = 40, n_jobs = 4)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_val)
f1_score(y_val, y_pred_sgd)
print(classification_report(y_val, y_pred_sgd))

#Perceptron
perceptron = Perceptron(random_state = 40, n_jobs = 4)
perceptron.fit(X_train, y_train)
y_pred_perc = perceptron.predict(X_val)
f1_score(y_val, y_pred_perc)
print(classification_report(y_val, y_pred_perc))

#Naive bayes:
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_val)
f1_score(y_val, y_pred_nb)
print(classification_report(y_val, y_pred_nb))

#KNN:
knn = KNeighborsClassifier(n_jobs = 4)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_val)
f1_score(y_val, y_pred_knn)
print(classification_report(y_val, y_pred_knn))

#Random Forest:
forest = RandomForestClassifier(random_state = 40, n_jobs = 4)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_val)
f1_score(y_val, y_pred_forest)
print(classification_report(y_val, y_pred_forest))

#Multi-layer Perceptron:
mlp = MLPClassifier(random_state = 40)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_val)
f1_score(y_val, y_pred_mlp)
print(classification_report(y_val, y_pred_mlp))

###OPTIMALISATION

#KNN
"""
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
weights = ['uniform', 'distance']
n_neighbors = [1, 5, 25]

print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format("metrics", "weights", "k_neigh", "precision", "recall", "f1_score"))
for metric in metrics:
  for weight in weights:
    for k in n_neighbors:
      knn = KNeighborsClassifier(n_neighbors = k, metric = metric, weights = weight, n_jobs = 4)
      knn.fit(X_train, y_train)
      y_pred_knn = knn.predict(X_val)
      recall = round(recall_score(y_val, y_pred_knn),3)
      precision = round(precision_score(y_val, y_pred_knn),3)
      f1 = round(f1_score(y_val, y_pred_knn),3)
      print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(metric, weight, k, precision, recall, f1))
"""

#RANDOM FOREST
"""
criterion = ['gini', 'entropy']
n_estimators = [5, 100, 1000]
max_depth = [100, 1000, 10000]
min_samples_split = [2, 10, 20]
min_impurity_decrease = [0.0, 0.01, 0.001]

print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format("criterion", "n_estimators", "max_depth", "min_samples", "min_impurity", "precision", "recall", "f1_score"))
for n in n_estimators:
  for crit in criterion:
    for depth in max_depth:
      for min_split in min_samples_split:
        for min_impurity in min_impurity_decrease:
          forest = RandomForestClassifier(n_estimators = n, criterion = crit, max_depth = depth, min_samples_split = min_split, min_impurity_decrease = min_impurity, random_state = 40, n_jobs = 4)
          forest.fit(X_train, y_train)
          y_pred_forest = forest.predict(X_val)
          recall = round(recall_score(y_val, y_pred_forest), 3)
          precision = round(precision_score(y_val, y_pred_forest), 3)          f1 = round(f1_score(y_val, y_pred_forest), 3)
          print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(crit, n, depth, min_split, min_impurity, precision, recall, f1))
"""

#MLP CONSTANT
hidden_layer_sizes = [100, 250, 350, 400, 500]
alpha = [0.00001, 0.005, 0.01, 0.05, 0.1]
activation = ['tanh', 'logistic', 'identity', 'relu']
learning_rate_init = [0.00001, 0.0001, 0.001, 0.01, 0.1]

print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format("layers", "alpha", "activation", "init", "precision", "recall", "f1_score"))
for layer in hidden_layer_sizes:
  for a in alpha:
    for active in activation:
      for init in learning_rate_init:
          mlp = MLPClassifier(hidden_layer_sizes = layer, alpha = a, activation = active, learning_rate_init = init, random_state = 40)
          mlp.fit(X_train, y_train)
          y_pred_mlp = mlp.predict(X_val)
          recall = round(recall_score(y_val, y_pred_mlp),3)
          precision = round(precision_score(y_val, y_pred_mlp),3)
          f1 = round(f1_score(y_val, y_pred_mlp),3)
          print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(layer, a, active, init, recall, precision, f1))
          
#MLP NON-CONSTANT
hidden_layer_sizes = [100, 250, 350, 400, 500]
alpha = [0.00001, 0.005, 0.01, 0.05, 0.1]
activation = ['tanh', 'logistic', 'identity', 'relu']
learning_rate = ['adaptive', 'constant', 'invscaling']

print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format("layers", "alpha", "activation", "rate", "precision", "recall", "f1_score"))
for layer in hidden_layer_sizes:
  for a in alpha:
    for active in activation:
      for rate in learning_rate:
          mlp = MLPClassifier(hidden_layer_sizes = layer, alpha = a, activation = active, learning_rate = rate, random_state = 40)
          mlp.fit(X_train, y_train)
          y_pred_mlp = mlp.predict(X_val)
          recall = round(recall_score(y_val, y_pred_mlp),3)
          precision = round(precision_score(y_val, y_pred_mlp),3)
          f1 = round(f1_score(y_val, y_pred_mlp),3)
          print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(layer, a, active, rate, recall, precision, f1))
          
             
### EVALUATION (INTERNAL) TEST SET

##Preprocessing
#Upsampling
X_full_train, y_full_train = upsample(X_full_train, y_full_train)
#Z-scoring for binarized data
X_full_train, X_test = standardize_spoken(X_full_train, X_test)

#Evaulation on internal test
mlp = MLPClassifier(hidden_layer_sizes = 250, activation = 'tanh', alpha = 0.01, learning_rate_init = 0.0001, random_state = 40)
mlp.fit(X_full_train, y_full_train)
y_pred_mlp = mlp.predict(X_test)
f1_score(y_test, y_pred_mlp)
print(classification_report(y_test, y_pred_mlp))

### PREPERATION CODALAB

#Load in the codalab test data:
written_test = np.load("written_test.npy")
spoken_test = np.load("spoken_test.npy")

#preprocessing of the test data
written_test = (written_test > 50).astype(int)
spoken_summary_coda = summary_spoken(spoken_test, (spoken_test.shape[0], 130))
X_test_coda = combine_data(written_test, spoken_summary_coda)

#preprocessing of X_data
X_data, y_data = upsample(X_data, match_train)

#preprocessing X_data (test) and X_test_data (test)
X_data, X_test_coda = standardize_spoken(X_data, X_test_coda)

#apply optimal model
mlp = MLPClassifier(hidden_layer_sizes = 250, activation = 'tanh', alpha = 0.01, learning_rate_init = 0.0001, random_state = 40)
mlp.fit(X_data, y_data)
y_pred_coda = mlp.predict(X_test_coda)

#inspect the labels
sum(y_pred_coda == True) / len(y_pred_coda)

#save the predicted array values as npy file
np.save("result", y_pred_coda)



