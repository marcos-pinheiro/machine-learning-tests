import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('./Ecommerce_Customers.csv')
#dataset.info()

#valor que queremos descobrir
labels = dataset['Yearly Amount Spent']

#valores que serao usados para descobrir predizer o Yearly Amount Spent
#e que analizando cada atributor como X e o Yearly Amount Spent como Y, tiveram uma linha reta vertical indicando que
#a regressao pode ser uma boa!
samples = dataset[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

#divide todos os dados de amostras e rotulos em conjunto de treino e de teste
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.3, random_state=101)

regression = LinearRegression()
regression.fit(train_samples, train_labels)
predict = regression.predict(test_samples)

print 'Score: %d%%' % (regression.score(test_samples, test_labels) * 100)

#Taxas de erro da predicao em comparacao com o real
print 'Metrica MAE %s' % metrics.mean_absolute_error(test_labels, predict)
print 'Metrica RMSE %s' % np.sqrt(metrics.mean_squared_error(test_labels, predict))