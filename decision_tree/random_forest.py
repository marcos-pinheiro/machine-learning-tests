import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('./kyphosis.csv')

samples = dataset.drop('Kyphosis', axis=1)
labels  = dataset['Kyphosis']

samples_train, samples_test, labels_train, labels_test = train_test_split(samples, labels, test_size=0.3)

rfc = RandomForestClassifier(n_estimators=200) #Quantidade de arvores na floresta a serem criadas, quanto mais, menos performatico
rfc.fit(samples_train, labels_train)

predictions = dtree.predict(samples_test)
print classification_report(labels_test, predictions)
print confusion_matrix(labels_test, predictions)