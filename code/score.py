import argparse
from collections import defaultdict
from typing import Dict, List

from gensim.models import KeyedVectors
from scipy.stats import spearmanr

import utils


def get_gold_score(path: str) -> Dict:
    """
    Retrieve the human scores for pair of words.
    :param path: path to test file.
    :return: a dictionary (w1, w2) -> human score.
    """
    with open(path) as file:
        next(file)
        tokens = (l.strip().split() for l in file)
        return {(w1.lower(), w2.lower()): score for w1, w2, score in tokens}


def compute_cosine(w1: str, w2: str, embeddings, senses_dict: Dict) -> float:
    """
    Computes cos similarity between two words.
    :param w1: word 1.
    :param w2: word 2.
    :param embeddings: sense embeddings.
    :param senses_dict: a dictionary from word to senses.
    :return: the cos similarity score.
    """
    score = -1.0
    if not senses_dict.get(w1) or not senses_dict.get(w2):
        return score
    for synset1 in senses_dict[w1]:
        for synset2 in senses_dict[w2]:
            sense1 = w1 + "_" + synset1
            sense2 = w2 + "_" + synset2
            if sense1 in embeddings.vocab and sense2 in embeddings.vocab:
                cos = embeddings.similarity(sense1, sense2)
                score = max(score, cos)
    return score


def compute_score(
    dict_gold: Dict, senses_dict: Dict, embeddings
) -> (List[float], List[float]):
    """
    Compute the cosine similarity between each pair of words in the dictionary in input.
    :param dict_gold: dictionary containing human scores for pair of words.
    :param senses_dict: a dictionary from word to senses.
    :param embeddings: sense embeddings.
    :return:
    """
    scores_gold, scores_predicted = [], []
    for k, v in dict_gold.items():
        scores_gold.append(v)
        scores_predicted.append(
            compute_cosine(k[0].lower(), k[1].lower(), embeddings, senses_dict)
        )
    return scores_gold, scores_predicted


def filter_missing(dictionary: Dict, senses_dict: Dict) -> Dict:
    """
    Filter a dictionary.
    :param dictionary: Dictionary to filter.
    :param senses_dict:
    :return: a dictionary filtered.
    """
    return {
        (k[0], k[1]): v
        for k, v in dictionary.items()
        if k[0] in senses_dict and k[1] in senses_dict
    }


def build_sense_map(embeddings):
    """
    Build a dictionary from word to senses, given the embeddings vocab.
    :param embeddings: path to the embeddings vector file.
    :return: the dictionary word -> sense.
    """
    senses_dict = defaultdict(set)
    senses = (s.lower().rpartition("_") for s in embeddings.vocab if "_bn:" in s)
    for lemma, _, synset in senses:
        senses_dict[lemma].add(synset)
    return senses_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to the embeddings", dest="input")
    parser.add_argument(help="path to the test file", dest="test")
    parser.add_argument("--map", help="path to the word to senses map", dest="map")

    return parser.parse_args()


def main(embeddings: str, test_path: str, senses_path: str):
    print("Load embeddings")
    vectors = KeyedVectors.load_word2vec_format(embeddings, binary=False)
    print("Load word -> synsets dict")
    senses_map = (
        utils.read_dictionary(senses_path) if senses_path else build_sense_map(vectors)
    )
    print("Load gold scores")
    dict_gold = get_gold_score(test_path)
    filtered_gold = filter_missing(dict_gold, senses_map)
    print("Missing words:", len(dict_gold.keys()) - len(filtered_gold.keys()))
    scores_gold, scores_predicted = compute_score(dict_gold, senses_map, vectors)
    print(spearmanr(scores_gold, scores_predicted))


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.test, args.map)
