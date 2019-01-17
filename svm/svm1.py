import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


cancer = load_breast_cancer()

dataframe_samples = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
dataframe_labels = pd.DataFrame(cancer['target'], columns=['Cancer'])

train_samples, test_samples, train_labels, test_labels = train_test_split(dataframe_samples, np.ravel(dataframe_labels), test_size=0.3, random_state=101)

model = SVC()
model.fit(train_samples, train_labels)
predictions = model.predict(test_samples)

#Pessima predicao
print classification_report(test_labels, predictions)
print confusion_matrix(test_labels, predictions)

print '/n/n'



#GridSearch, vai testar uma seria de combinacoes para encontrar a que melhor se adeque ao modelo do SVM, sem configurar nada
#a saida da predicao sera pessima

#Parametro C Controla o custo de classificacoes erradas, quanto maior, mais penalizara as classificacoes, quanto maior, se tem uma
#alta variancia e baixo vies.

#Parametro gamma, faz mencao ao tipo do kernel, ele altera o comportamento da funcao guassiana. Um baixo gamma e uma funcao gaussiana
#com baixa variancia, e o alto gamma te dara alto vies e baixa variancia.

#Vies e variancia tem q ter um equilibrio para o modelo nunca predizer nada e nao ficar overfitado
grid_params = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}

#Testa com todos os parametros
grid = GridSearchCV(SVC(), grid_params, refit=True, verbose=3)
grid.fit(train_samples, train_labels)

#Exibe os melhores parametros
print grid.best_params_ #Nesse caso o melhor Tradeoff entre vies e variancia foi {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001}

predictions = grid.predict(test_samples)

#Predicao Otima
print classification_report(test_labels, predictions)
print confusion_matrix(test_labels, predictions)