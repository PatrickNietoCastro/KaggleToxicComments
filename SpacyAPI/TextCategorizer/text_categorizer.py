from time import strftime, gmtime

import pandas as pd
import os
import numpy as np

import en_core_web_sm
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt

import thinc.extra.datasets
import random


def load_data(filepath=os.path.join('..', '..', 'Data', 'train.csv'), limit=None, cats_start=2, cats_end=8):
    data_pd = pd.read_csv(filepath_or_buffer=filepath)[:limit]
    """Load data from the IMDB dataset."""
    data_pd.loc[:, 'cats'] = data_pd.apply(axis=1,
                                           func=lambda row: dict([(cat_str, cat_val) for cat_str, cat_val in zip(
                                               (row[list(data_pd.columns[cats_start:cats_end])]).keys(),
                                               [bool(x) for x in
                                                list((row[list(data_pd.columns[cats_start:cats_end])]).values)])]))
    data_pd.loc[:, 'cats_str'] = data_pd.loc[:, 'cats'].apply(
        lambda x: ''.join(map(str, [int(x) for x in list(x.values())])))
    return data_pd


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}



def split_data(data_full=None, split=0.9):
    targets_train = list()
    sources_train = list()
    targets_test = list()
    sources_test = list()
    for cat_str in list(set(list(data_full['cats_str']))):
        print('current category: ', )
        n_train        = max(1, int(split*len(list(data_full[data_full['cats_str'] == cat_str]['cats']))))
        targets_train += list(data_full[data_full['cats_str'] == cat_str]['cats'        ])[:n_train]
        sources_train += list(data_full[data_full['cats_str'] == cat_str]['comment_text'])[n_train:]
        targets_test  += list(data_full[data_full['cats_str'] == cat_str]['cats'        ])[:n_train]
        sources_test  += list(data_full[data_full['cats_str'] == cat_str]['comment_text'])[n_train:]
        print('n_test: {}, n_train: {}'.format(len(list(data_full[data_full['cats_str'] == cat_str]['cats'])) - n_train,
                                               n_train))

    data_train = zip(sources_train, targets_train)
    data_test  = zip(sources_test , targets_test )
    return data_train, data_test


if __name__ == "__main__":
    output_dir  = None
    cats_start  = 2
    cats_end    = 8
    n_iter      = 20
    n_texts     = 1000
    random_seed = 1234567
    np.random.seed(random_seed)
    current_time = strftime("%Y%m%d-%H%M%S", gmtime())
    print('Current time: ', current_time)
    # Load spacy nlp model
    print('Loading data...')
    data_full = load_data(limit=n_texts)
    print('Split into test/train')
    data_train, data_test = split_data(data_full=data_full, split=0.8)
    print('Loading model...')
    nlp = en_core_web_sm.load()
    print('Configuring model')
    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    for custom_label in list(data_full.columns)[cats_start:cats_end]:
        textcat.add_label(custom_label)

    # load the IMDB dataset
    print("Loading IMDB data...")
    # (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
    train_texts , train_cats    = zip(*data_train)
    dev_texts   , dev_cats      = zip(*data_test)
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))
