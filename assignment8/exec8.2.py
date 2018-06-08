import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Part a - Load data
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_train = fetch_20newsgroups(categories = categories, shuffle = True, random_state = 41)

print(len(twenty_train.data), "documents have been loaded.")

# Part b - Generate the bag of words 

count_vect = CountVectorizer(stop_words = 'english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
print("Number of words:", len(count_vect.vocabulary_))
print(count_vect.get_feature_names()[4690])

# There are 35788 words. In order to access the words you can use the dictionary "vocabulary_" build the CountVectorizer. The index of the feature is stored in the vocabulary and it indicates both the position in the feature vector and frequency of the word.

# Part c - Training MultinomialNB classifier

# Use frequencies over occurencies
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['I do not believe in afterdeath', 'The human brain is the largest brain of all vertebrates', 'Machine Learning algorithms are faster on GPUs']

# Extract feature of new doc
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Predict doc category
predicted = clf.predict(X_new_tfidf) 

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                    ])

# Train model with single command
text_clf.fit(twenty_train.data, twenty_train.target)
# Check accuracy of prediction
twenty_test = fetch_20newsgroups(subset = 'test',
                                 categories = categories,
                                 shuffle = True,
                                 random_state = 42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(metrics.classification_report(twenty_test.target,
                                    predicted,
                                    target_names = twenty_test.target_names))

# Part d -
'''
The stop words argument seems to filter out known words which do not add meaning or value to the actual text such as articles or prepositions.

35482 are found using the stop words list, compared to 35788. Moreover the accuracy increases too from 0.88 to 0.90.
'''

# Part e -
'''
The reason for working with frequencies is that larger documents are more likely to have more words compared to smaller documents, even though the topic of discussion may be the same. 
'''
