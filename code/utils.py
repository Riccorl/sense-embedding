from collections import defaultdict
from itertools import chain
from typing import List, Set, Dict


def read_dataset(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def write_dataset(filename: str, lines: List[str]):
    """
    Writes a list of string in a file.
    :param filename: path where to save the file.
    :param lines: list of strings to serilize.
    :return:
    """
    with open(filename, "w", encoding="utf8") as file:
        file.writelines(line + "\n" for line in lines)


def read_dictionary(filename: str) -> Dict:
    """
    Open a dictionary from file, in the format key -> value
    :param filename: file to read.
    :return: a dictionary.
    """
    with open(filename) as file:
        return {k: v for k, *v in (l.split() for l in file)}


def write_dictionary(filename: str, dictionary: Dict):
    """
    Writes a dictionary as a file.
    :param filename: file where to save the dictionary.
    :param dictionary: dictionary to serialize.
    :return:
    """
    with open(filename, mode="w") as file:
        for k, *v in dictionary.items():
            file.write(k + "\t" + "\t".join(v[0]) + "\n")


def clean_embeddings(path_input: str, path_output: str):
    """
    Clean embeddings by removing non lemma_synset vectors.
    :param path_input: path to original embeddings.
    :param path_output: path to cleaned embeddings.
    :return:
    """
    old_emb = read_dataset(path_input)
    filtered = [vector for vector in old_emb if "_bn:" in vector]
    write_dataset(path_output, [str(len(filtered)) + " " + "400"] + filtered)


def split_dataset(filename: str, n_split: int):
    """
    Split a large text file in smaller files.
    :param filename: file to split.
    :param n_split: number of parts to split.
    :return:
    """
    sew = read_dataset(filename)
    batch = len(sew) // n_split
    for i in range(0, len(sew), batch):
        j = i + batch
        filename_batch = str(filename).split(".")[0] + "_" + str(n_split) + ".txt"
        print("Writing", filename_batch)
        write_dataset(filename_batch, sew[i:j])
        n_split += 1


def compute_word_sysnet_map(paths: List[str], mapping) -> Dict[str, Set]:
    """
    Produce a dictionary word -> synsets.
    :param paths: path of the input file.
    :param mapping: mapping file from bn to wn.
    :return: a dictionary of word and synsets.
    """
    word_synset_map = defaultdict(set)
    for path in paths:
        with open(path) as file:
            # flat list of words
            words = chain.from_iterable(line.strip().split() for line in file)
            # filter senses from words
            senses = (s.lower().rpartition("_") for s in words if "_bn:" in s)
            for lemma, _, synset in senses:
                if synset in mapping:
                    word_synset_map[lemma].add(synset)

    return word_synset_map


def timer(start: float, end: float) -> str:
    """
    Timer function. Compute execution time from strart to end (end - start).
    :param start: start time
    :param end: end time
    :return: end - start
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
