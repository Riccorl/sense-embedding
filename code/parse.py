import argparse
from typing import List

import constants as const
import utils
from preprocess import eurosense, sew


def parse_es(path_input: str, path_output: str, check_synset: bool = False):
    """
    Parse EuroSense in a single txt file.
    :param path_input: raw EuroSense path.
    :param path_output: where to save the parsed file.
    :param check_synset: if True, check if the synset is correct for the given lemma.
    :return:
    """
    eurosense.main(path_input, path_output, check_synset)


def parse_sew(path_input: str, path_output: str):
    sew.main(path_input, path_output)


def make_dict(paths: List[str], path_dict: str):
    """
    Write a dictionary from word to senses, from the given input files.
    :param paths: files to read.
    :param path_dict: where to save the dictionary.
    :return:
    """
    bnwn_map = utils.read_dictionary(const.BN2WN_MAP)
    word_synset_map = utils.compute_word_sysnet_map(paths, bnwn_map)
    utils.write_dictionary(path_dict, word_synset_map)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        help="corpus name to parse, es=EruoSense, sew=SEW", dest="corpus"
    )
    parser.add_argument("-i", help="path of the corpus", required=True, dest="input")
    parser.add_argument(
        "-o", help="path where to save the parsed file", required=True, dest="output"
    )
    parser.add_argument("-m", help="path where to save the model file", dest="model")
    parser.add_argument(
        "--check-synset",
        help="chek if synset is correct, works with EuroSense only.",
        dest="check_synset",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.check_synset)
    if args.corpus == "es":
        parse_es(args.input, args.output, args.check_synset)
    elif args.corpus == "sew":
        parse_sew(args.input, args.output)
    else:
        print("Option not available")
