#Salvando os pesos (sinapses da rede)

import numpy as np
import pandas as pd
import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout

def save_network():
    previsores = pd.read_csv('entradas-breast.csv')
    classes = pd.read_csv('saidas-breast.csv')

    network = Sequential()
    network.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
    network.add(Dropout(0.2))
    network.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
    network.add(Dropout(0.2))
    network.add(Dense(units=1, activation='sigmoid'))
    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    network.fit(previsores, classes, batch_size=10, epochs=100)

    #Export Network Structure
    network_structure_serialized = network.to_json()

    with open('./net.json', 'w') as json_file:
        json_file.write(network_structure_serialized)

    #Export Network Weights
    network.save_weights('./weights.h5')

#######################################################################################################################################

#save_network()

#Import Structure
net_file = open('./net.json', 'r')
imported_json = net_file.read()
net_file.close()

#Create Network and load weigths
loaded_network = model_from_json(imported_json)
loaded_network.load_weights('./weights.h5')

new_value = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

predictations = loaded_network.predict(new_value)
print predictations
print (predictations > 0.5)