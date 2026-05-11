import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.sparse import hstack, csr_matrix


def getBowEmbeddings(texts, max_features=5000):

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
    )

    X_bow = vectorizer.fit_transform(texts)

    return X_bow, vectorizer

def getTfidfEmbeddings(texts, max_features=5000):

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer


def getDoc2vecEmbeddings(tokenized_texts, vector_size=300, epochs=60):
    documents = [
        TaggedDocument(words=text, tags=[str(i)])
        for i, text in enumerate(tokenized_texts)
    ]

    model = Doc2Vec(
        vector_size=vector_size,
        min_count=1,
        workers=4,
        epochs=epochs
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    X_doc2vec = np.array([model.dv[str(i)] for i in range(len(tokenized_texts))])
    return X_doc2vec


def getWord2vecEmbeddings(tokenized_texts, vector_size=300, window=5, min_count=1, workers=4):

    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )

    X_w2v = []
    for tokens in tokenized_texts:
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vecs) > 0:
            X_w2v.append(np.mean(vecs, axis=0))
        else:
            X_w2v.append(np.zeros(vector_size))

    X_w2v = np.array(X_w2v)
    return X_w2v


def getJoinedEmbeddings(X_tfidf, X_doc2vec=None, X_word2vec=None):

    matrices = [X_tfidf]

    if X_doc2vec is not None:
        matrices.append(csr_matrix(X_doc2vec))

    if X_word2vec is not None:
        matrices.append(csr_matrix(X_word2vec))

    X_joined = hstack(matrices)
    return X_joined