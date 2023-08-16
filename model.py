import bz2

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("Movies.csv", lineterminator="\n")
df['Overview'] = df['Overview'].fillna('')


# This function is used for preprocessing the genre feature in the dataset.
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def sigmoid(matrix):
    return sigmoid_kernel(matrix, matrix)


def cos(matrix):
    return cosine_similarity(matrix, matrix)


# This function uses cosine_similarity function to train the model based on overview of the movie.
def overview():
    tfidf = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', ngram_range=(1, 3),
                            token_pattern=r'\w{1,}', stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Overview'])
    return cos(tfidf_matrix)


# This function uses sigmoid_kernel function to train the model based on the genre of the movie.
def genre():
    count = CountVectorizer(stop_words='english')
    df['Genre'] = df['Genre'].apply(clean_data)
    count_matrix = count.fit_transform(df['Genre'])
    return sigmoid(count_matrix)


cos1 = overview()
sig1 = genre()

models = [cos1, sig1]
pickle.dump(models, open("model.pkl", 'wb'))
