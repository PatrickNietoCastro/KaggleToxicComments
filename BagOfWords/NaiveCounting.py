import numpy as np
import pandas as pd

import spacy
import re

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from collections import Counter

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

#Multilabel classification task. Check proportion of each label:
#for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
#    print(label, (train[label] == 1.0).sum() / len(train))
    
#train.loc[(train['severe_toxic'] == 1)]
#=> if severe_toxic then toxic

tokenizer = RegexpTokenizer(r'\w+')
stopwords_set = set(stopwords.words('english'))

def text_process(comment_column):
    filtered_rows = []
    for rows in comment_column:
        only_words = tokenizer.tokenize(rows)
        no_stopwords = [word for word in only_words if word.lower() not in stopwords_set]
        filtered_rows.append(' '.join(no_stopwords))
    return filtered_rows

train['filtered_comments'] = text_process(train.comment_text)
#train.filtered_comments

train.index = train['id']
x_train = train['filtered_comments']
y_train = train.iloc[:, 2:]
y_train.drop(['filtered_comments'], axis=1, inplace = True)
y_train['clean'] = [max(1-i,0) for i in y_train.sum(axis=1)]

word_counts = dict()

for kind in y_train.columns:
    word_counts[kind] = Counter()
    comments = x_train[y_train[kind]==1]
    for _, comment in comments.iteritems():
        word_counts[kind].update(comment.split(" "))

def most_common_words(kind, num_words):
    words = word_counts[kind].most_common(num_words)[::-1]
    return words

def probabilities(comment, wordlist):
    prob = 0.0
    for i in [x[0] for x in wordlist]:
        if (i.lower() in comment.lower()):
            prob += 0.05
    return min(1,prob)

def get_kind_probs(kind):
    kind_list = []
    wordlist = most_common_words(kind, 10)
    for words in test.comment_text:
        tmp = probabilities(words, wordlist)
        kind_list.append(tmp)
    return kind_list

bag_of_words_submission = test

for kind in y_train.columns:
    bag_of_words_submission[kind] = get_kind_probs(kind)
    
bag_of_words_submission.drop(['comment_text','clean'], axis=1, inplace = True)
bag_of_words_submission.to_csv('submission.csv', index = False, encoding='utf-8')
