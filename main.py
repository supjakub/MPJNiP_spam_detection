import gensim.models
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn as sk
from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from textblob import Word


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
        dataset['text'][i] = [x for x in dataset['text'][i] if (x not in stop_words and x not in punctuation)]


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


def vectorize(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    vectors = np.array(vectors)
    return vectors.mean(axis=0)


def log_metrics(vec_size, win, min, stem, lem, balanced_accuracy, f1_score, precision, recall):
    f = open('metrics.log', 'a')
    f.write('vec_size=' + str(vec_size) +';window=' + str(win) + ';min_count=' + str(min) + ';stem=' + stem + ';lem=' + lem + '\n')
    f.write('balanced_accuracy\n' + str(balanced_accuracy) + '\n')
    f.write('f1_score\n' + str(f1_score) + '\n')
    f.write('precision\n' + str(precision) + '\n')
    f.write('recall\n' + str(recall) + '\n')
    f.close()


def experiment(dataset, vec_size, win, min, stem_method, lem_method):
    balanced_accuracy = []
    f1_score = []
    precision = []
    recall = []
    delete_stopwords(dataset)
    stem(dataset, method=stem_method)
    lemmatize(dataset, method=lem_method)
    f = open('metrics.log', 'a')
    rkf = sk.model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    for train_index, test_index in rkf.split(X=dataset['text'], y=dataset['class']):
        X_train = dataset['text'][train_index]
        X_test = dataset['text'][test_index]
        y_train = dataset['class'][train_index]
        y_test = dataset['class'][test_index]
        w2v_model = gensim.models.Word2Vec(X_train, vector_size=vec_size, window=win, min_count=2)
        words = set(w2v_model.wv.index_to_key)
        X_train = np.array([vectorize(text, w2v_model) for text in X_train])
        X_test = np.array([vectorize(text, w2v_model) for text in X_test])

        clf = sk.linear_model.LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        balanced_accuracy.append(sk.metrics.balanced_accuracy_score(y_test, y_pred))
        f1_score.append(sk.metrics.f1_score(y_test, y_pred))
        precision.append(sk.metrics.precision_score(y_test, y_pred))
        recall.append(sk.metrics.recall_score(y_test, y_pred))

    log_metrics(vec_size, win, min, stem_method, lem_method, balanced_accuracy, f1_score, precision, recall)


def main():
    dataset = get_dataset("spam_ham_dataset.csv")
    tokenize(dataset)
    experiment(dataset, vec_size=100, win=5, min=2, stem_method='porter', lem_method='snowball')

if __name__ == "__main__":
    main()