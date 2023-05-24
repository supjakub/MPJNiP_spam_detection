import gensim.models
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, linear_model, metrics
from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from textblob import Word
from tqdm import tqdm
import os
import csv

def get_dataset(filename):
    df = pd.read_csv(f"datasets/{filename}")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.columns = ['label', 'text', 'class']
    return df

def tokenize(dataset):
    dataset['text'] = dataset['text'].apply(lambda x: word_tokenize(x))

def delete_stopwords(dataset):
    stop_words = set(stopwords.words('english'))
    for i, document in enumerate(dataset['text']):
        dataset['text'][i] = [x for x in document if (x not in stop_words and x not in punctuation)]


def stem(dataset, method):
    if not method:
        return
    elif method == 'porter':
        stemmer = PorterStemmer()
    elif method == 'snowball':
        stemmer = SnowballStemmer()
    for i in range(len(dataset['text']) - 1):
        document_tmp = []
        for token in dataset['text'][i]:
            document_tmp.append(stemmer.stem(token))
        dataset['text'][i] = document_tmp

def lemmatize(dataset, method=None):
    if not method:
        return
    elif method == 'wordnet':
        lemmatizer = WordNetLemmatizer()
        for i in range(len(dataset['text']) - 1):
            document_tmp = []
            for token in dataset['text'][i]:
                document_tmp.append(lemmatizer.lemmatize(token))
            dataset['text'][i] = document_tmp
    elif method == 'textblob':
        for i in range(len(dataset['text']) - 1):
            document_tmp = []
            for token in dataset['text'][i]:
                document_tmp.append(Word(token).lemmatize())
            dataset['text'][i] = document_tmp

def vectorize_w2v(tokens, model):
    vectors = [model.wv[token] if token in model.wv else np.zeros(model.vector_size) for token in tokens]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    vectors = np.array(vectors)
    return vectors.mean(axis=0)

def vectorize_count(dataset):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([" ".join(doc) for doc in dataset])
    return vectors.toarray()


def log_results(experiment_params, metrics, filename='results.csv'):
    results_df = pd.DataFrame([experiment_params], columns=['vec_method', 'stem', 'lem', 'vec_size', 'window', 'min_count'])
    for metric_name, metric_value in metrics.items():
        results_df[metric_name] = np.mean(metric_value)
    if not os.path.isfile(filename):
        results_df.to_csv(filename, sep=',', index=False)
    else:
        results_df.to_csv(filename, mode='a', header=False, index=False, sep=',')

def experiment(dataset, vec_size, win, min, stem_method, lem_method, vec_method):
    balanced_accuracy = []
    f1_score = []
    precision = []
    recall = []
    delete_stopwords(dataset)
    stem(dataset, method=stem_method)
    lemmatize(dataset, method=lem_method)

    vectorizer = None
    if vec_method == 'count':
        vectorizer = CountVectorizer()

    rkf = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    for train_index, test_index in rkf.split(X=dataset['text'], y=dataset['class']):
        X_train = dataset['text'][train_index]
        X_test = dataset['text'][test_index]
        y_train = dataset['class'][train_index]
        y_test = dataset['class'][test_index]

        if vec_method == 'word2vec':
            w2v_model = gensim.models.Word2Vec(X_train, vector_size=vec_size, window=win, min_count=2)
            X_train = np.array([vectorize_w2v(text, w2v_model) for text in X_train])
            X_test = np.array([vectorize_w2v(text, w2v_model) for text in X_test])
        elif vec_method == 'count':
            X_train = vectorizer.fit_transform([" ".join(doc) for doc in X_train]).toarray()
            X_test = vectorizer.transform([" ".join(doc) for doc in X_test]).toarray()

        clf = linear_model.LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        balanced_accuracy.append(metrics.balanced_accuracy_score(y_test, y_pred))
        f1_score.append(metrics.f1_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))

    log_results({'vec_method': vec_method, 'stem': stem_method, 'lem': lem_method, 'vec_size': vec_size, 'window': win, 'min_count': min},
                {'balanced_accuracy': balanced_accuracy, 'f1_score': f1_score, 'precision': precision, 'recall': recall})

def main():
    dataset = get_dataset("spam_ham_dataset.csv")
    tokenize(dataset)

    stem_methods = [None, 'porter', 'snowball']
    lem_methods = [None, 'wordnet', 'textblob']
    vectorization_methods = ['word2vec', 'count']

    total_experiments = len(stem_methods) * len(lem_methods) * len(vectorization_methods)

    pbar = tqdm(total=total_experiments, ncols=100)

    # Run experiments
    for stem_method in stem_methods:
        for lem_method in lem_methods:
            for vec_method in vectorization_methods:
                print(f"Running experiment with stem_method={stem_method}, lem_method={lem_method}, vec_method={vec_method}")
                experiment(dataset, vec_size=100, win=5, min=2, stem_method=stem_method, lem_method=lem_method, vec_method=vec_method)
                pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    main()