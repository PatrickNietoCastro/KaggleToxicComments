from time import strftime, gmtime

import pandas as pd
import os
import numpy as np

import en_core_web_sm
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding, decaying
from spacy.pipeline import TextCategorizer
import matplotlib.pyplot as plt


def get_batches(train_data, model_type, batch_step=0.001):
    max_batch_sizes = {'tagger': 32, 'parser': 16, 'ner': 16, 'textcat': 64}
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, batch_step)
    batches = minibatch(train_data, size=batch_size)
    return batches


def load_data(data_pd=None, cats_start=2, cats_end=8, nlp=None):
    #data_pd = pd.read_csv(filepath_or_buffer=filepath, nrows=limit, header=None, skiprows=skip_rows+1, names=["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"])
    """Load data."""
    data_pd.loc[:, 'cats'] = data_pd.loc[:, list(data_pd.columns[cats_start:cats_end])].apply(axis=1,
                                                                                              func=lambda row: dict(
                                                                                                  row))
    data_pd.loc[:, 'cats_str'] = data_pd.loc[:, 'cats'].apply(
        lambda x: ''.join(map(str, [int(x) for x in list(x.values())])))
    data_pd.loc[:, 'comment_text_doc']  = data_pd.loc[:, 'comment_text'].apply(lambda x: nlp(x))
    data_pd.loc[:, 'gold_parse']        = [GoldParse(doc=doc, cats=tags) for doc, tags in zip(data_pd.loc[:, 'comment_text_doc'], data_pd.loc[:, 'cats'])]
    return data_pd


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8     # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label]:
                tp += 1.
            elif score >= 0.5 and gold[label] == 0:
                fp += 1.
            elif score < 0.5 and gold[label] == 0:
                tn += 1
            elif score < 0.5 and gold[label]:
                fn += 1
    precision   = tp / (tp + fp)
    recall      = tp / (tp + fn)
    f_score     = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score, 'tp': int(tp),
            'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}



def split_data(data_full=None, split=0.9, n_texts=None, all_labels=[]):
    targets_train = list()
    sources_train = list()
    targets_test = list()
    sources_test = list()
    for cat_str in list(set(list(data_full['cats_str']))):
        print('current category: ', cat_str)
        if cat_str == '000000':
            if n_texts is None:
                n_texts = data_full.shape[0]
            n_train = min(len(list(data_full[data_full['cats_str'] == cat_str]['cats'])), int(n_texts/7))
        else:
            n_train = max(1, int(split*len(list(data_full[data_full['cats_str'] == cat_str]['cats']))))
        targets_train += list(data_full[data_full['cats_str'] == cat_str]['gold_parse'          ])[:n_train]
        sources_train += list(data_full[data_full['cats_str'] == cat_str]['comment_text_doc'    ])[:n_train]
        targets_test  += list(data_full[data_full['cats_str'] == cat_str]['cats'                ])[n_train:]
        sources_test  += list(data_full[data_full['cats_str'] == cat_str]['comment_text'        ])[n_train:]
        print('n_test: {}, n_train: {}'.format(len(list(data_full[data_full['cats_str'] == cat_str]['gold_parse'])) - n_train,
                                               n_train))
    data_train = zip(sources_train, targets_train)
    data_test  = zip(sources_test , targets_test )
    return data_train, data_test


if __name__ == "__main__":
    """
    Main function of text_categorizer
    """


    save_my_model       = 1
    plot_eval           = 1
    drop_max            = 0.6
    drop_min            = 0.2
    drop_step           = 1e-4
    batch_step          = 1.005
    cats_start          = 2
    cats_end            = 8
    n_iter              = 3
    #n_texts             = 159571  # all the samples from kaggle toxic comments
    n_texts             = 1000
    chunk_size          = 100  # for training performance
    random_seed         = 1234567
    train_test_split    = 0.8
    # Configuration values
    #data_filepath = os.path.join('/dev', 'shm', 'train.csv')
    #plot_dir = os.path.join('/dev', 'shm', 'Plots')
    #model_dir = os.path.join('/dev', 'shm', 'Textcats')

    data_filepath       = os.path.join('/home', 'castro', 'PycharmProjects', 'KaggleToxicComments', 'Data', 'train.csv')
    plot_dir            = os.path.join('/home', 'castro', 'PycharmProjects', 'KaggleToxicComments', 'SpacyAPI', 'Plots')
    model_dir           = os.path.join('/home','castro','PycharmProjects','KaggleToxicComments','SpacyAPI','Textcats')
    model_name          = 'spacy_categorizer'
    #textcat_file        = 'spacy_categorizer20180301-223202'
    textcat_file        = None
    current_time        = strftime("%Y%m%d-%H%M%S", gmtime())

    np.random.seed(random_seed)
    print('Current time: ', current_time)
    # Load spacy nlp model
    print('Loading model...')
    nlp_model = en_core_web_sm.load()

    print('Configuring model')
    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy

    if textcat_file:
        print('  load existing textcat from file', textcat_file)
        textcat = TextCategorizer(nlp_model.vocab)
        textcat.from_disk(os.path.join(model_dir, textcat_file))
        nlp_model.add_pipe(textcat, last=True)
    else:
        if 'textcat' not in nlp_model.pipe_names:
            print('  create textcat from scratch')
            textcat = nlp_model.create_pipe('textcat')
            nlp_model.add_pipe(textcat, last=True)
            # add label to text classifier
            for custom_label in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:  # Enter custom labels here <---------------------
                textcat.add_label(custom_label)

        # otherwise, get it, so we can add labels to it
        else:
            print('model already contains textcat!')
            textcat = nlp_model.get_pipe('textcat')

    optimizer = textcat.begin_training()
    dropout = decaying(drop_max, drop_min, drop_step)

    # store for evaluation
    precisions = list()
    recalls = list()
    f_scores = list()
    loss_vals = list()
    data_pds = pd.read_csv(nrows=n_texts, filepath_or_buffer=data_filepath, chunksize=chunk_size, header=0,
                           names=["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult",
                                 "identity_hate"])

    for data_pd in data_pds:
        print('Loading data...')

        data_full = load_data(data_pd=data_pd, nlp=nlp_model, cats_start=cats_start, cats_end=cats_end)
        print('Split into test/train')
        data_train, data_test = split_data(data_full=data_full,
                                           split=train_test_split,
                                           n_texts=data_full.shape[0])

        train_texts , train_cats    = zip(*data_train)
        val_texts   , val_cats      = zip(*data_test)

        print("Using {} examples ({} training, {} evaluation)"
              .format(chunk_size, len(train_texts), len(val_texts)))

        data_train      = list(zip(train_texts, train_cats))
        # begin training
        # get names of other pipes to disable them during training
        other_pipes         = [pipe for pipe in nlp_model.pipe_names if pipe != 'textcat']
        with nlp_model.disable_pipes(*other_pipes):
            print("Training the model {} epochs...".format(str(n_iter)))
            print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F', 'tp', 'tn', 'fp', 'fn'))
            for i in range(n_iter):
                losses  = {}
                # batch up the examples using spaCy's minibatch
                batches = get_batches(train_data=data_train, model_type='textcat', batch_step=batch_step)
                for batch in batches:
                    drop = next(dropout)
                    texts, annotations  = zip(*batch)
                    nlp_model.update(docs=list(texts), golds=list(annotations), sgd=optimizer, losses=losses, drop=drop)
                with nlp_model.use_params(optimizer.averages):
                    # evaluate on the validation data split off in load_data()
                    scores = evaluate(nlp_model.tokenizer, textcat, val_texts, val_cats)
                precisions  .append(scores['textcat_p'])
                recalls     .append(scores['textcat_r'])
                f_scores    .append(scores['textcat_f'])
                loss_vals   .append(losses['textcat'  ])
                print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:5d}\t{5:5d}\t{6:5d}\t{7:5d}'  # print a simple table
                      .format(losses['textcat'  ], scores['textcat_p'   ],
                              scores['textcat_r'], scores['textcat_f'   ],
                              scores['tp'       ], scores['tn'          ],
                              scores['fp'       ], scores['fn'          ]))

    evaluation_frame = pd.DataFrame.from_dict({'precision': precisions,
                                               'recall':    recalls,
                                               'f_score':   f_scores,
                                               'loss_val':  loss_vals})

    if plot_eval:
        plt.figure()
        ax = evaluation_frame[['precision', 'recall', 'f_score']].plot(title='p = tp/(tp+fp), r = tp/(tp+fn)',
                                                                       grid=True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        plt.savefig(os.path.join(plot_dir, 'stochastic_eval-' + current_time))
        plt.clf()
        plt.figure()
        ax = evaluation_frame['loss_val'].plot(title='?-loss of nlp_model',
                                               grid=True,
                                               logy=True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Amount of ?-loss')
        plt.savefig(os.path.join(plot_dir, 'textcat_loss-' + current_time))
        plt.clf()

    if save_my_model:
        with nlp_model.use_params(optimizer.averages):
            textcat.to_disk(os.path.join('..', model_dir, model_name + current_time))

    current_time        = strftime("%Y%m%d-%H%M%S", gmtime())
    print('Current time: ', current_time)
    print('Spacy is fun!')
