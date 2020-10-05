from pathlib import Path

import gensim
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

# Declare the filename of the target data to generate the document-term matrix
filepath = 'input/hd_content_FINAL_CyprusMail.csv'
filename = Path(filepath).stem

# Document contents lists
docs = []

# Read CSV File
df = pd.read_csv(filepath)

# Iterate DataFrame to gather the headlines and their content
for index, row in df.iterrows():
    if row['Content']:
        docs.append(str(row['Content']).encode(encoding='UTF-8', errors='strict'))

def vect2gensim(vectorizer, dtmatrix):
     # transform sparse matrix into gensim corpus and dictionary
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)


# compute vector space with sklearn
vect = CountVectorizer(min_df=1, ngram_range=(1, 1), max_features=30000)
corpus_vect = vect.fit_transform(docs)

# transport to gensim
(corpus_vect_gensim, gensim_dict) = vect2gensim(vect, corpus_vect)

dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                                    id2word=dict((id, word) for word, id in vect.vocabulary_.items()))

pd.to_pickle(vect,'pickles/vocab_'+filename+'.pkl')
print(dictionary)
