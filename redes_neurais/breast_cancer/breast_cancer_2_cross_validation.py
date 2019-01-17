#CROSS VALIDATION, tecnica para treinar e testar a rede aproveitando todos os dados
#E USANDO O DROPOUT, TECNICA PARA PREVINIR O OVERFITTING

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


def criarRedeNeural():
    network = Sequential()

    #Camada Oculta 1
    #units=Quantidade de Neuronios, valor base = (Entradas + Saidas) / 2
    #activation=Funcao de ativavao (Funcao RELU)
    #input_dim=Definimos quantos neuronios na camada de entrara (criada implicitamente quando setado esse valor) se vai ter
    #kernel_initializer=Algoritmos para distribuicao dos pesos
    network.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

    #Ele vai pegar x% de dados da camada de entrada e zerar, para previnir o overfitting. 0.2 = 20%. 
    #Ele pega aleatoriamente. Geralmente nao definir acima de 40% para nao ter underfitting
    network.add(Dropout(0.2))

    #Camada Oculta 2 (Nem sempre a adicao de camadas novas vai melhorar, tem q ir testando)
    network.add(Dense(units=16, activation='relu'))

    #Ele vai pegar x% de dados da camada anterior e zerar, para previnir o overfitting. 0.2 = 20%.
    #Ele pega aleatoriamente. Geralmente nao definir acima de 40% para nao ter underfitting
    network.add(Dropout(0.2))

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
    #loss=Funcao para calculo do erro para depois fazer o backpropagation e treinar a rede, existe varios algoritmos como mean_square_error, mean_absolute_error que sao bastante usados para regressoes
    #     e para saidas binarias, tem o binary_crossentropy que sao complexos matematicamentes e oference um bom resultado. Para saidas com mais de 2, tem o category_crossentropy
    #metrics=Metricas para saber o score da rede no treinamento
    #network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return network


previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

#Separacao da base de teste e treinamento de maneira comum
#previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size=0.25)

#Separacao da base usando K-Fold Cross Validation
#o K representa um numero de partes, e ele divide a base de dados nesse numero e pega a primeira parte
#e o define como teste e as outras treinamento e gera o score na tabela de probabilidade, depois ele pega outro grupo
#como teste e todos os outros incluindo o anterior como treinamento e assim ate todas as partes terem sido usadas no model
#Dessa maneira se aproveita registros que poderiam ser importantes no treinamento e que estao no testes e vice versa.
#Depois se pega todos esses scorings gerados de cada parte e se faz uma media simples para ter um scoring melhor
#Geralmente usam o numero 10 de K e e um algoritmo muito usado pelos cientistas, dessa maneira vc consegue nao ter a falsa impressao de 
#se sua rede acerta muito ou erra muito
network = KerasClassifier(build_fn = criarRedeNeural, epochs=100, batch_size=10) #Rede criada no formato de cross validation

#cv = K (Quantidade de partes que se dividira a base e de repeticoes do treinamento com as partes de teste e treino trocadas)
#X = amostras
#y = classes
#scoring = Como queremos que se volte os resultados
results = cross_val_score(estimator = network, X = previsores, y = classes, cv = 10, scoring='accuracy')
print results
print results.mean() #Media

#Desvio Padrao, vai ver quanto os registros estao desviando da media
#Quanto maior o valor do desvio, maior a change de se ter overfitting na rede, e a rede esteja se adaptando muito aos dados
#e em uma base nova, nao predizer bem, porque esta muito adaptada aos dados de treinamento da rede. 
#Overfitting ele perde a capacidade de generalizacao, vc sob estima ela. A rede fica viciado nos dados
#Underfitting, vc tem um problema complexo e vc subestima ele.
desvio = results.std()
print desvio


# Caracteristicas: Underfitting Ele tera resultados ruims na base de treinamento, 
#e o Overfitting tera otimos resultados na base de treinamento e ruims na de teste
# Para evitar se pode usar o Dropout que zera os valores de uma camada aleatoriamente dentre um percentual definido
#para evitar Overfitting, um vicio da rede nos dados do treinamento

predictations = network.predict(previsores_teste)
predictations = (predictations > 0.5) #Converte os valores em True/False se maior 0.5

precision = accuracy_score(classes_teste, predictations)
print precision

matrix = confusion_matrix(classes_teste, predictations)
print matrix

result = network.evaluate(previsores_teste, classes_teste)
print result

