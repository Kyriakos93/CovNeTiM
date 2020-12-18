# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
import argparse
import os
import sys
import scipy.sparse
import pickle

from gensim import matutils, models
from gensim.corpora import Dictionary
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

#
# # Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_stopwords(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
#
#
# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]
#
#
# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]
#
#
# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out
#
#
# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Define the following (default) parameters before start topic modeling

# Pickled DataFrame of Document-Term Matrix to Load
pkl_file = 'pickles/dtm_stop_hd_content_FINAL_CyprusMail.pkl'
pkl_dictionary = 'pickles/cv_stop_hd_content_FINAL_CyprusMail.pkl'

# Number of topics
topics_num = 10

# Number of iterations
passes_num = 200

print(
    '___________           .__            _____             .___     .__                \n' +
    '\__    ___/___ ______ |__| ____     /     \   ____   __| _/____ |  |   ___________\n' +
    '  |    | /  _ \\\____ \|  |/ ___\   /  \ /  \ /  _ \ / __ |/ __ \|  | _/ __ \_  __ \\\n' +
    '  |    |(  <_> )  |_> >  \  \___  /    Y    (  <_> ) /_/ \  ___/|  |_\  ___/|  | \/\n' +
    '  |____| \____/|   __/|__|\___  > \____|__  /\____/\____ |\___  >____/\___  >__|   \n' +
    '               |__|           \/          \/            \/    \/          \/       \n')

print("Welcome to RISE TAG Topic Modeller\n")

parser = argparse.ArgumentParser()
parser.add_argument("-file", help="Pickled file to execute LDA analysis and extract topics", type=str)
parser.add_argument("-dictionary", help="Pickled dictionary file to execute LDA analysis and extract topics", type=str)
parser.add_argument("-topics", help="Number of topics to extract", type=int)
parser.add_argument("-passes", help="Number of LDA passes", type=int)
parser.add_argument("-optimize", default=False, help="Calculate perplexity and coherence scores to find the optimal "
                                                     "number of topics per given dataset",
                    dest='isOptEnabled', action='store_true')
parser.add_argument("-start", help="Starting number of topics to start the optimization process", type=int)
parser.add_argument("-step", help="A step value for increasing the number of topics during the optimization process",
                    type=int)
parser.add_argument("-limit", help="Limit number of topics to stop the optimization process",
                    type=int)
args = parser.parse_args()

if args.file:
    pkl_file = args.file
if args.dictionary:
    pkl_dictionary = args.dictionary
if args.topics:
    topics_num = args.topics
if args.passes:
    passes_num = args.passes

print('Executing Topic Modeler for ' + str(topics_num) + ' topics in ' + str(
    passes_num) + ' passes on ' + pkl_file + ' ..\n')

# Load pickled DataFrame (Document-Term Matrix)
print('■ Loading pickled DataFrame of Document-Term Matrix..', end='')
data = pd.read_pickle(pkl_file)
print('Done')

# One of the required inputs is a term-document matrix
print('■ Transposing the DataFrame into Term-Document Matrix..', end='')
tdm = data.transpose()
print('Done')
# print(tdm.head())

# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
print('■ Converting Term-Document Matrix to gensim format (from df -> sparse matrix -> gensim corpus)..', end='')
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
print('Done')

# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
print(
    '■ Loading pickled dictionary of all terms and their respective location in the T-D Matrix(' + pkl_dictionary + ')..',
    end='')
cv = pickle.load(open(pkl_dictionary, "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
print('Done')


# Todo: Debug vocabulary here
# print('Vocabulary\n')
# for k in id2word.items():
#     print(k)
# exit()

def run_lda(corpus, id2word, topics_num, passes_num, alpha, enable_messages=True):
    # Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
    # we need to specify two other parameters as well - the number of topics and the number of passes
    #    Note for the LDA parameters: Use the following if you want to run LDA with fixed seed(s) or compute other stuff:
    #    random_state=100, update_every=1, chunksize=100, per_word_topics=True
    if enable_messages:
        print('■ Latent Dirichlet Allocation (LDA) analysis is loading. Please wait..', end='')
    lda_model = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=topics_num, passes=passes_num, alpha=alpha)

    return lda_model


def compute_percoh_values(corpus, dictionary, passes, alpha, limit, start=2, step=1):
    """
    Compute perplexity and u_mass coherence for various number of topics

    Parameters:
    ----------
    corpus : Gensim corpus
    dictionary : Gensim dictionary
    passes : Number of passes for the LDA execution
    alpha : Alpha hyper-parameter value for the LDA execution
    limit : Max num of topics
    start : Starting value of topics to compute for the optimization process
    step : Step value to increase the number of topics during the optimization process

    Returns:
    -------
    model_list : List of LDA topic models
    perplexity_values : Perplexity values corresponding to the LDA model with respective number of topics
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    perplexity_values = []
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('■■ Running LDA for ' + str(num_topics) + ' topics in ' + str(passes_num) + ' passes..', end='')
        lda_model = run_lda(corpus, dictionary.id2token, num_topics, passes, alpha, enable_messages=False)
        model_list.append(lda_model)

        # Compute Perplexity
        perplexity_score = lda_model.log_perplexity(corpus)  # a measure of how good the model is. lower the better.
        perplexity_values.append(perplexity_score)

        # Compute Coherence Score

        # coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=d, coherence='c_v')
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coherence_score = coherence_model_lda.get_coherence()

        coherence_values.append(coherence_score)

        print('Done')

    return model_list, perplexity_values, coherence_values


def save_score(values, save_as):
    with open('output/' + save_as + '.txt', 'w') as f:
        for item in values:
            f.write("%s\n" % item)
        f.close()
    return True


if args.isOptEnabled:
    print('■ LDA Optimization Process is starting..')

    start_from = 2
    step = 1
    limit = 20

    if args.start:
        start_from = args.start
    if args.step:
        step = args.step
    if args.limit:
        limit = args.limit

    # Loop for increasing the number of topics and calculating the scores
    # cur_num_of_topics = start_from
    # while cur_num_of_topics <= limit:
    #     lda = run_lda(corpus, id2word, cur_num_of_topics, passes_num, 'auto')
    #

    # Construct Dictionary
    word2id = dict((k, v) for k, v in cv.vocabulary_.items())
    d = corpora.Dictionary()
    d.id2token = id2word
    d.token2id = word2id
    #
    #     cur_num_of_topics += step
    model_list, perplexity_values, coherence_values = compute_percoh_values(corpus=corpus, dictionary=d,
                                                                            passes=passes_num, alpha='auto',
                                                                            limit=limit, start=start_from, step=step)

    print('■■ Generating plots for perplexity and coherence scores..', end='')

    # Show graph for Coherence Score
    x = range(start_from, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    # plt.show()
    plt.savefig('output/coh_' + os.path.basename(pkl_file).replace('.pkl', '') + '.png')
    plt.close()

    # Show graph for Perplexity Score
    x = range(start_from, limit, step)
    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    # plt.show()
    plt.savefig('output/per_' + os.path.basename(pkl_file).replace('.pkl', '') + '.png')
    plt.close()

    print('Done')

    print('■■ Saving list of number of topics values to disk..', end='')
    num_of_topics_values = []
    for num in range(start_from, limit, step):
        num_of_topics_values.append(num)

    save_score(values=num_of_topics_values, save_as='num_of_topics_values')
    print('Done')

    print('■■ Saving list of perplexity values to disk..', end='')
    save_score(values=perplexity_values, save_as='perplexity_values')
    print('Done')

    print('■■ Saving list of coherence values to disk..', end='')
    save_score(values=coherence_values, save_as='coherence_values')
    print('Done')

    print('Optimization process finished. Exiting..')

    exit()
else:
    lda = run_lda(corpus, id2word, topics_num, passes_num, 'auto')

    print('Done')
    print('\n** ' + str(topics_num) + ' topics found in ' + str(passes_num) + ' passes:\n')
    print(lda.print_topics())

    # Compute Perplexity
    print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    word2id = dict((k, v) for k, v in cv.vocabulary_.items())
    # Debugging:
    # print(word2id)
    d = corpora.Dictionary()
    d.id2token = id2word
    d.token2id = word2id

    # coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=d, coherence='c_v')
    coherence_model_lda = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # EXTRA STEP: Identify the topic

    # # Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
    # add_stop_words = []
    # stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    # cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
    # data_cvna = cvna.fit_transform(data.Content)
    # data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names(), index=data.Headline)
    #
    # # Let's take a look at which topics each content contains
    # corpus_transformed = lda[corpus]
    # list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))
