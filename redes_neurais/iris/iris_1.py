
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrapper.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #Transforma atributor categorico (String) para numerico

dataset = pd.read_csv('./iris.csv')
predictors  = dataset.iloc[:, 0:4]
classes     = dataset.iloc[:, 4]

classes = LabelEncoder().fit_transform(classes) #As redes fazem calculos e precisam ser numeros

#A rede espera uma saida de 3 dimensoes, entao devemos fazer o esquema de false false true false ...
#iris setora    = 1 0 0
#iris virginica = 0 1 0
#iris versicolor= 0 0 1 

 # [
 #   [1, 0, 0], 
 #   [0, 0, 1], 
 # ]
classes_dumy = np_utils.to_categorical(classes)

train_samples, test_samples, train_classes, test_classes = train_test_split(predictors, classes_dumy)

#units=(4+3)/2 = 3,5 >> 4
#Para problemas de classificacao com mais de 2 classes, recomendado softmax.
#Quando a saida e binaria como probabilidade, a sigmoid e a recomendada
#Diferente da sigmoid (que gera uma probabilidade da saida de um neuronio) a softmax gera a probabilidade para cada neuronio de saida, retornamdo o mais provavel

network = Sequential()
network.add(Dense(units=4, activation='relu', input_dim=4))
network.add(Dense(units=4, activation='relu'))
network.add(Dense(units=3, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

network.fit(train_samples, train_classes, batch_size=10, epochs=1000)

#Do result o index 0 e o resultado da loss_function e o index 1 e o score dos testes
result = network.evaluate(test_samples, test_classes)
print result

#Retorna uma probabilidade para cada neuronio. Ex:
#[0.00001, 0.0002, 0.982738] Que se interpreta
#   0         0        1
result = network.predict(test_samples)
print result

result = (result > 0.5)

#Gera uma lista com o numero do index aonde se encontra os maiores valores. Ex:
#para as linhas [0, 0, 1] e [1, 0, 0] ele retornara [2, 0] aonde 2 e 0 sao os index.
#isso para podermos jogar na funcao da confusion matrix
test_classes_transformed = [np.argmax(t) for t in test_classes]
result_transformed = [np.argmax(t) for t in result]

matrix = confusion_matrix(result_transformed, test_classes_transformed)
print matrix