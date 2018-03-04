from time import strftime, gmtime

import pandas as pd
import os
import numpy as np

import en_core_web_sm
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding, decaying
from spacy.pipeline import TextCategorizer
import matplotlib.pyplot as plt



def load_data(filepath=None, limit=None, cats_start_idx=2, cats_end_idx=8, nlp=None):
    """Load data from filepath."""
    data_pd = pd.read_csv(filepath_or_buffer=filepath)[:limit]
    return data_pd


def create_textcat(nlp_model=None, model_dir=None, textcat_filename=None, custom_labels=None):
    # Add textcat to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy

    if textcat_filename:
        print('[INFO]  load existing textcat from ', os.path.join(model_dir, textcat_filename))
        textcat = TextCategorizer(nlp_model.vocab)
        textcat.from_disk(os.path.join(model_dir, textcat_filename))
        nlp_model.add_pipe(textcat, last=True)
    else:
        if 'textcat' not in nlp_model.pipe_names:
            print('[INFO]  create textcat from scratch')
            textcat = nlp_model.create_pipe('textcat')
            nlp_model.add_pipe(textcat, last=True)
            # add label to text classifier
            for custom_label in list(custom_labels):
                textcat.add_label(custom_label)
        # otherwise, get it, so we can add labels to it
        else:
            print('[INFO] load existing textcat from model-directory')
            textcat = nlp_model.get_pipe('textcat')
    return textcat


if __name__ == "__main__":

    current_time        = strftime("%Y%m%d-%H%M%S", gmtime())
    print('Current time: ', current_time)
    # Configuration values
    data_filepath       = os.path.join('/home', 'castro', 'PycharmProjects', 'KaggleToxicComments', 'Data', 'test.csv')
    plot_dir            = os.path.join('/home', 'castro', 'PycharmProjects', 'KaggleToxicComments', 'SpacyAPI', 'Plots')
    model_dir           = os.path.join('/home','castro','PycharmProjects','KaggleToxicComments','SpacyAPI','Textcats')
    model_load_name     = 'spacy_categorizer20180301-223202'

    n_texts             = None
    custom_labels       = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    # Start
    # Load spacy nlp model
    print('Loading model...')
    nlp_model = en_core_web_sm.load()
    print('Loading data...')
    data_full = load_data(filepath=data_filepath, limit=n_texts, nlp=nlp_model)
    print('Evaluating {} samples'.format(str(data_full.shape[0])))
    print('Configuring model')
    # create the text classifier
    textcat = create_textcat(nlp_model=nlp_model,
                             model_dir=model_dir,
                             textcat_filename=model_load_name,
                             custom_labels=custom_labels)
    my_tokenizer = nlp_model.tokenizer
    print('make predictions')
    data_full.loc[:, 'comment_doc'     ] = data_full.loc[:, 'comment_text'].apply(lambda x: my_tokenizer(str(x)))
    data_full.loc[:, 'comment_doc'     ] = data_full.loc[:, 'comment_doc' ].apply(lambda x: textcat(x))
    data_full.loc[:, 'prediction'      ] = data_full.loc[:, 'comment_doc' ].apply(lambda x: x.cats.items())
    print('create submission')
    for label in custom_labels:
        print(str(label))
        data_full.loc[:, label] = data_full.loc[:, 'prediction'].apply(lambda x: dict(x)[label])
    submission = data_full.loc[:, ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    print('save submission to', str(os.path.join('/home',
                                                 'castro',
                                                 'PycharmProjects',
                                                 'KaggleToxicComments',
                                                 'Submissions',
                                                 'submission+'+current_time+'.csv')))
    submission.to_csv(path_or_buf=os.path.join('/home',
                                               'castro',
                                               'PycharmProjects',
                                               'KaggleToxicComments',
                                               'Submissions',
                                               'submission+'+current_time+'.csv'),
                      index=False)

    current_time        = strftime("%Y%m%d-%H%M%S", gmtime())
    print('Current time: ', current_time)
    print('spacy predictions are fun!')
