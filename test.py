import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('imdb_tr.csv')
vectorizer = CountVectorizer(min_df=1)
corpus = []
answers= []

for document in df.text:
 corpus.append(document)

for polar in df.polarity:
 answers.append(polar)
 
 
X = vectorizer.fit_transform(corpus)
y = answers

print len(vectorizer.get_feature_names())

from sklearn.linear_model import SGDClassifier
#X = [[0., 0., 0.], [1., 0., 1.], [0., 0., 1.], [1., 1., 1.]]
#y = [0, 1, 0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)

test = vectorizer.transform([df.text[1]])

print clf.predict(test)
print df.polarity[1]


#Testing
print clf.score(X,y)

"""

for word in test_vocab: 
 if(word not in unique_word_list): unique_word_list.append(word)

for word in unique_word_list:
 test_word_count.append(test_vocab.count(word))

for index, _ in enumerate(unique_word_list): 
 print unique_word_list[index], test_word_count[index]
"""
 
"""
from sklearn.linear_model import SGDClassifier
X = [[0., 0., 0.], [1., 0., 1.], [0., 0., 1.], [1., 1., 1.]]
y = [0, 1, 0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)

print clf.predict([[0., 1., 0.]])
"""