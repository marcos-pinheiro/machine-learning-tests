import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./USA_Housing.csv', usecols=[0,1,2,3,4,5])

samples = dataset[['Avg. Area Income', 
    'Avg. Area House Age', 'Avg. Area Number of Rooms', 
    'Avg. Area Number of Bedrooms', 'Area Population']]
labels = dataset['Price']

#30% dos valores reservados para teste
test_percent = 0.3

#Divide o dataset entre treino e teste e retorna as amostras e labelas de treino e teste. Usa um algoritmo randomico para dividir
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=test_percent, random_state=101)

regression = LinearRegression()
regression.fit(train_samples, train_labels)

# print regression.intercept_
# print regression.coef_

predict = regression.predict(test_samples)
print 'Score: %d%%' % (regression.score(test_samples, test_labels) * 100)