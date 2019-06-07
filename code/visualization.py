from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE


def tsne_plot_cluster(senses, vectors, top_k: int = 30, png_path: str = None):
    """
    Print senses in clusters.
    :param senses: senses to print.
    :param vectors: embeddings.
    :param top_k: print top k most similar senses for each cluster.
    :param png_path: path to png.
    :return:
    """
    embedding_clusters, word_clusters = [], []
    for sense in senses:
        embeddings, words = [], []
        for similar_sense, _ in vectors.most_similar(sense, topn=top_k):
            words.append(similar_sense)
            embeddings.append(vectors[similar_sense])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model = TSNE(
        perplexity=15, n_components=2, init="pca", n_iter=3500, random_state=32
    )
    embedding_clusters_2d = np.array(
        tsne_model.fit_transform(embedding_clusters.reshape(n * m, k))
    ).reshape(n, m, 2)

    _tsne_plot_similar_words(senses, embedding_clusters_2d, word_clusters, 0.7, png_path)


def _tsne_plot_similar_words(
    labels: List[str],
    embedding_clusters: List[str],
    word_clusters: List[List],
    alpha: float,
    png_path=None,
):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(
        labels, embedding_clusters, word_clusters, colors
    ):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=alpha, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend(loc=4)
    plt.grid(True)
    if png_path:
        plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.show()


def plot(path_embeddings: str, png_path: str):
    """
    Plot senses from the given embeddings file.
    :param path_embeddings: embeddings file path.
    :param png_path: where to save the png.
    :return:
    """
    print("Load embeddings")
    vectors = KeyedVectors.load_word2vec_format(path_embeddings, binary=False)
    tsne_plot_cluster(
        [
            "bank_bn:00008363n",
            "bank_bn:00008364n",
            "number_bn:00058286n",
            "number_bn:00001079n",
            "plant_bn:00046568n",
            "plant_bn:00035324n",
        ],
        vectors,
        png_path=png_path,
    )


def word_similarity(words: List[str], path_embeddings: str, top_k: int = 6):
    """
    Return the top k similar words for each word in input.
    :param words: words in input.
    :param path_embeddings: embeddings file path.
    :param top_k: number of similar words to retrieve.
    :return:
    """
    vectors = KeyedVectors.load_word2vec_format(path_embeddings, binary=False)
    return [vectors.most_similar(word, topn=top_k) for word in words]
