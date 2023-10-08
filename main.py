# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence
import os
import matplotlib.pyplot as plt
import stellargraph as sg

# Load your dataset
file_path = "D:/PROG/Python/ML/iris dataset/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file_path, names=names)

# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

# Convert class labels to one-hot encodings
unique_classes = np.unique(Y)  # Get unique class names
class_to_one_hot = {class_name: i for i, class_name in enumerate(unique_classes)}
Y_one_hot = np.array([class_to_one_hot[class_name] for class_name in Y])

cv = StratifiedKFold(n_splits=10, shuffle=True)

# Create the neural network model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # Input layer with 8 neurons and ReLU activation
model.add(Dense(3, activation='softmax'))  # Output layer with 3 neurons for classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

histories = []
for train_index, test_index in cv.split(X, Y_one_hot):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_one_hot[train_index], Y_one_hot[test_index]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    from tensorflow.keras.utils import to_categorical
    Y_train = to_categorical(Y_train, num_classes=3)  # Assuming you have 3 classes
    Y_test = to_categorical(Y_test, num_classes=3)
    Y_val = to_categorical(Y_val, num_classes=3)

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=10, verbose=2)
    histories.append(history)
    accuracy = model.evaluate(X_test, Y_test)
    print("Accuracy: %.2f%%" % (accuracy[1] * 100))