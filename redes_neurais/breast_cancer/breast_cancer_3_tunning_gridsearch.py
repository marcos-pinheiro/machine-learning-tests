#Usara o GridSearchCV para testar diversos parametros e encontrar os melhores para a nossa rede neural
#Testando quantidade de neuronios, camadas e afins. Com muitos dados roda por horas para encontrar a melhor 
#combinacao

import pandas as pd
import keras

#Separacao treino-teste comum
from sklearn.model_selection import GridSearchCV

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


def criarRedeNeural(optimizer, loss, kernel_initializer, activation, neurons):
    network = Sequential()

    #Camada Oculta 1
    network.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))

    #Ele vai pegar x% de dados da camada de entrada e zerar, para previnir o overfitting. 0.2 = 20%
    network.add(Dropout(0.2))

    #Camada Oculta 2 (Nem sempre a adicao de camadas novas vai melhorar, tem q ir testando)
    network.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))

    #Ele vai pegar x% de dados da camada anterior e zerar, para previnir o overfitting. 0.2 = 20%.
    network.add(Dropout(0.2))

    #Camada de saida, saida binaria com um neuronio
    network.add(Dense(units=1, activation='sigmoid'))

    #Cria a rede neural
    network.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return network

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

network = KerasClassifier(build_fn = criarRedeNeural)

parameters = {
    "batch_size": [10, 30],
    "epochs": [50, 100],
    "optimizer": ['adam', 'sgd'],
    "loss": ['binary_crossentropy', 'hinge'],
    "kernel_initializer": ['random_uniform', 'normal'],
    "activation": ['relu', 'tanh'],
    "neurons" : [16, 8]
}

#ISSO DEMORA HORAS

#cv = K
grid_search = GridSearchCV(estimator = network, param_grid=parameters, scoring='accuracy', cv=5)
grid_search = grid_search.fit(previsores, classes)

better_params = grid_search.better_params_
better_score  = grid_search.best_score_
print better_params
print better_score

#Segundo o video os melhores foram
# parameters = {
#     "batch_size": 10
#     "epochs": 100
#     "optimizer": adam
#     "loss": binary_crossentropy
#     "kernel_initializer": ['random_uniform', 'normal'],
#     "activation": relu
#     "neurons" : 8
# }


# predictations = network.predict(previsores_teste)
# predictations = (predictations > 0.5) #Converte os valores em True/False se maior 0.5

# precision = accuracy_score(classes_teste, predictations)
# print precision

# matrix = confusion_matrix(classes_teste, predictations)
# print matrix

# result = network.evaluate(previsores_teste, classes_teste)
# print result

