#Classificacao Binaria com Redes neurais

import pandas as pd
import keras

#Separacao treino-teste comum
from sklearn.model_selection import train_test_split

#Separacao cruzada usando o wrapper do keras (Usa em conjunto com o skitlearn)
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score

#Modelo Sequencial, uma camada apos a outra (Input -> Hiddens... -> Output)
from keras.models import Sequential

#Camada densa porque todos os neuronios da camada se conectarao com os neuronios da camada subsequente (Fully Connect)
from keras.layers import Dense

#Metricas de acertividade
from sklearn.metrics import confusion_matrix, accuracy_score


previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

#Separacao da base de teste e treinamento de maneira comum
previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size=0.25)

network = Sequential()

#Camada Oculta 1
#units=Quantidade de Neuronios, valor base = (Entradas + Saidas) / 2
#activation=Funcao de ativavao (Funcao RELU)
#input_dim=Definimos quantos neuronios na camada de entrara (criada implicitamente quando setado esse valor) se vai ter
#kernel_initializer=Algoritmos para distribuicao dos pesos
network.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

#Camada Oculta 2 (Nem sempre a adicao de camadas novas vai melhorar, tem q ir testando)
network.add(Dense(units=16, activation='relu'))

#Camada de saida, saida binaria com um neuronio
network.add(Dense(units=1, activation='sigmoid'))

#lr=Learning Rate, quanto maior, mais rapido, quanto menor melhor para se encontrar os minimos globais (Testes para encontrar o melhor valor)
#decay=Taxa de decremento do Learning Rate, nos ajuda no inicio tendo un LearningRate maior e diminuir progressivamente a cada update
#clipvalue=Padrao em todos os otimizadores, server para se ao descer no gradiente se ficar em um ping pong, podendo subir o valor ao inves de descer no gradiente, ele serve para 
#          congelar o valor minimo entontrado. DOC do keras
optimizer = keras.optimizers.Adam(lr=0.0003, decay=0.00001, clipvalue=0.5)


#Cria a rede neural
#optimizer= Otimizador da descida do gradiente, varia de descida comum, descida do gradiente estocastica e adam 
#           que e o mais recomentado, ele e uma descida estocastica do gradiente otimizado que se adapta mt bem a muitos cenarios
#           Melhoria na descida do gradiente estocastica
#loss=Funcao para calculo do erro para depois fazer o backpropagation e treinar a rede, existe varios algoritmos como mean_square_error, mean_absolute_error que sao bastante usados para regressoes
#     e para saidas binarias, tem o binary_crossentropy que sao complexos matematicamentes e oference um bom resultado. Para saidas com mais de 2, tem o category_crossentropy
#metrics=Metricas para saber o score da rede no treinamento, binary_accuracy para binario
#network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])


#batch_size=Para descidas estocasticas do gradiente, ele faz o treinamento de X registros passados e atualiza os pesos, treina mais X e atualiza sucessivamente
#epochs=Epocas de treinamento
network.fit(previsores_treinamento, classes_treinamento, batch_size=20, epochs=100)


##### Visualizando os pesos #####
#0=camada de entrada, 30 neuronios para 16 neuronios, logo 16 pesos para 30 neuronios. 
#  Tera 2 registros, o primeiro pesos q vao da camada de entrara para a camada oculta e o segundo tem os valores dos bias para cada neuronio
# pesos0 = network.layers[0].get_weights() 
# print pesos0

# #A mesma coisa, a primeira camada oculta para a segunda, e a unidade de bias
# pesos1 = network.layers[1].get_weights() 
# print pesos1

# #A mesma coisa, a segunda camada oculta para a camada de saida, e a unidade de bias para o unico neuronio de saida
# pesos2 = network.layers[2].get_weights() 
# print pesos2


predictations = network.predict(previsores_teste)
predictations = (predictations > 0.5) #Converte os valores em True/False se maior 0.5

precision = accuracy_score(classes_teste, predictations)
print precision

matrix = confusion_matrix(classes_teste, predictations)
print matrix

result = network.evaluate(previsores_teste, classes_teste)
print result