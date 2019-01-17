import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train_dataset = pd.read_csv('./titanic_train.csv', usecols=(1,2,4,5,6,7,11))

#Esse dataset tem datos faltantes para a idade
#Como esses compoe quase 20% dos dados de teste nao apagaremos e vamos substituir por outro valor
#Foi tirado a media de idade pela classe e aparentemente pessoas mais velhas estavam na primeira classe e com isso 3 valores de idade saiu para 3 classes diferentes
# 37 para a primeira classe, 29 para a segunda e 24 para a terceira
# Substituiremos os valores
def arrumar_idade(cols):
    idade  = cols[0]
    classe = cols[1]
    
    if(pd.isnull(idade)):
        if classe == 1:
            return 37
        elif classe == 2:
            return 29
        else:
            return 24
    else:
        return idade

def arrumar_sex(cols):
    if('male' == cols[0]):
        return 1
    else:
        return 0

#axis=1 coluna axis=0 linha
train_dataset['Age'] = train_dataset[['Age', 'Pclass']].apply(arrumar_idade, axis=1)
train_dataset['Sex'] = train_dataset[['Sex']].apply(arrumar_sex, axis=1)
embark = pd.get_dummies(train_dataset['Embarked'], drop_first=True)

#Remove a coluna embarked
del train_dataset['Embarked']

#concatena ao dataset novos dataframes
train_dataset = pd.concat([train_dataset, embark], axis=1)

#amostras de treino e posteriormente rotulos
train_samples, test_samples, train_labels, test_labels = train_test_split(train_dataset.drop('Survived', axis=1), train_dataset['Survived'], test_size=0.3)

regression = LogisticRegression()
regression.fit(train_samples, train_labels)
predictions = regression.predict(test_samples)

print "------------ Test------------"
#Metrica de validacao1
print classification_report(test_labels, predictions)

# ______________________     (FALSE)         (TRUE)
# Valor Correto: FALSE |        TN             FP
# Valor Correto: TRUE  |        FN             TP
print confusion_matrix(test_labels, predictions)