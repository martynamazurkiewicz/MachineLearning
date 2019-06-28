# Import bibliotek
from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt


#Import zbioru danych Twenty Newsgroups - zbiór testujący
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
#Wyświetlenie wszystkich kategorii (target_names)
print(twenty_train.target_names)

#Formatowanie zbioru trenującego oraz wprowadzenie go do algorytmu DTC
clf = Pipeline([('vect', CountVectorizer(max_features=1600, min_df=2, max_df=0.75)),
                ('tfidf', TfidfTransformer()),
                ('clf', DecisionTreeClassifier(criterion="gini"))])

#Import zbioru testującego
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

#Predykcja
clf = clf.fit(twenty_train.data, twenty_train.target)
predicted = clf.predict(twenty_test.data)
print(predicted)

#Sprawdzenie dokładności
print(np.mean(predicted == twenty_test.target))

#Wykresy
count_pred = np.zeros(20)
cat = np.arange(20)
for x in cat:
    for y in predicted:
        if x == y:
            count_pred[x] = count_pred[x]+1
x = twenty_train.target_names
plt.xticks(rotation=90)
plt.scatter(x, count_pred)
plt.show()

count_target = np.zeros(20)
for x in cat:
    for y in twenty_test.target:
        if x == y:
            count_target[x] = count_target[x] + 1

x = twenty_train.target_names
plt.xticks(rotation=90)
plt.scatter(x, count_target)
plt.show()
