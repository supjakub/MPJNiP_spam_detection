import gensim.models
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn as sk
from string import punctuation
import numpy as np

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

def vectorize(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    vectors = np.array(vectors)
    return vectors.mean(axis=0)

def experiment(dataset, vec_size, win, min):
    delete_stopwords(dataset)
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
        print('Balanced accuracy', sk.metrics.balanced_accuracy_score(y_test, y_pred))
        print('F-1 score', sk.metrics.f1_score(y_test, y_pred))
        print('Precision', sk.metrics.precision_score(y_test, y_pred))
        print('Recall', sk.metrics.recall_score(y_test, y_pred))
def main():
    dataset = get_dataset("spam_ham_dataset.csv")
    tokenize(dataset)
    experiment(dataset, vec_size=100, win=5, min=2)

if __name__ == "__main__":
    main()