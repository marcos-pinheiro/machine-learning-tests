import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataframe = pd.read_csv('./Classified Data.csv')

dataframe_dropado = dataframe.drop('TARGET CLASS', axis=1)

#Treinando o pre-processador para que ele descubra a melhor maneira de normalizar os dados entrantes
scaler = StandardScaler()
scaler.fit(dataframe_dropado)

#Executara o desvio padrao em todos para normaliza-los
dataframe_normalizado = scaler.transform(dataframe_dropado)


samples = pd.DataFrame(dataframe_normalizado, columns=dataframe.columns[ :-1])
labels  = dataframe['TARGET CLASS']

samples_train, samples_test, labels_train, labels_test = train_test_split(samples, labels, test_size=0.3)

#Poucos vizinhos e ruim porque um outlier gera muito ruido, 
#Muitos vizinhos fica pesao e abrange as vezes dados que nao deveria
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(samples_train, labels_train)

# Predicao com o K ruim
predictions = knn.predict(samples_test)
print classification_report(labels_test, predictions)
print confusion_matrix(labels_test, predictions)


error_rate = []

#Metodo do cotovelo, testara diversos K e salvara o quanto a media se afasta do esperado na lista de error_rate,
#para tentar encontrar o K com melhores resultados
#Com isso e legal para visualizar a taxa de eficiencia conforme se aumenta ou diminui o K e quando aumenatar o diminuir para de gerar efeitos
#significativos
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(samples_train, labels_train)
    predictions = knn.predict(samples_test)
    error_rate.append(np.mean(predictions != labels_test))

k = 19
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(samples_train, labels_train)
predictions = knn.predict(samples_test)

print classification_report(labels_test, predictions)
print confusion_matrix(labels_test, predictions)