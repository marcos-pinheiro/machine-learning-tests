#Testando com os parametros voltados do grid search e como se fosse para Producao

import numpy as np
import pandas as pd
import keras

#Separacao treino-teste comum
from sklearn.model_selection import train_test_split

#Separacao cruzada usando o wrapper do keras (Usa em conjunto com o skitlearn)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Modelo Sequencial, uma camada apos a outra (Input -> Hiddens... -> Output)
from keras.models import Sequential

#Camada densa porque todos os neuronios da camada se conectarao com os neuronios da camada subsequente (Fully Connect)
from keras.layers import Dense

#Para previnir Overfitting
from keras.layers import Dropout

#Metricas de acertividade
from sklearn.metrics import confusion_matrix, accuracy_score


previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

network = Sequential()
network.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
network.add(Dropout(0.2))
network.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
network.add(Dropout(0.2))
network.add(Dense(units=1, activation='sigmoid'))
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

#No final, apos treinar e testar com os conjuntos, ajustamos os valores da rede para os melhores e usamos 
#tudo como treinamento para ter a rede criada
network.fit(previsores, classes, batch_size=10, epochs=100)


new_value = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

predictations = network.predict(new_value)
print predictations
print (predictations > 0.5)