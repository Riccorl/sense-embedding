import argparse
import logging
import multiprocessing
from typing import List

import gensim
from gensim.models import Word2Vec, FastText

import utils
from sentence_loader import SentenceLoader


def train_w2v(
    sentences: List[str],
    save_embeddings: str,
    model: gensim.models = Word2Vec,
    min_count: int = 3,
    iter: int = 5,
    size: int = 400,
    save_model: str = None,
):
    # Logs to monitor gensim
    logging.basicConfig(
        format="%(levelname)s - %(asctime)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    print(sentences)
    loader = SentenceLoader(sentences)
    w2v_model = model(
        sentences=loader,
        size=size,
        window=5,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        hs=1,
        sample=1e-3,
        iter=iter,
    )

    print("Saving vectors...")
    w2v_model.wv.save_word2vec_format(save_embeddings, binary=False)
    file, _, ext = save_embeddings.rpartition(".")
    utils.clean_embeddings(save_embeddings, file + "_clean." + ext, size)

    if save_model:
        print("Saving model...")
        w2v_model.save(save_model)
    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", help="paths to the corpora", dest="input")
    parser.add_argument(
        "-o",
        help="path where to save the embeddings file",
        required=True,
        dest="output",
    )
    parser.add_argument(
        "-m",
        help="model implementation, w2v=Word2Vec, ft=FastText",
        dest="model",
        default="w2v",
    )
    parser.add_argument(
        "--model_path", help="path where to save the model file", dest="save_model"
    )
    parser.add_argument(
        "--min-count",
        help="ignores all words with total frequency lower than this",
        dest="min_count",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--iter",
        help="number of iterations over the corpus",
        dest="iter",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--size",
        help="dimensionality of the feature vectors",
        dest="size",
        default=400,
        type=int,
    )

    return parser.parse_args()


def main(
    sentences: List[str],
    save_embeddings: str,
    model_type: str,
    min_count: int = 3,
    iter: int = 5,
    size: int = 400,
    save_model: str = None,
):
    if model_type == "w2v":
        print("Word2Vec model")
        model = Word2Vec
    elif model_type == "ft":
        print("FastText model")
        model = FastText
    else:
        print(
            "Model implementation not recognized. Use 'w2v' for Word2Vec or 'ft' for FastText."
        )
        return

    train_w2v(
        sentences=sentences,
        save_embeddings=save_embeddings,
        model=model,
        min_count=min_count,
        iter=iter,
        size=size,
        save_model=save_model,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        sentences=args.input,
        save_embeddings=args.output,
        model_type=args.model,
        min_count=args.min_count,
        iter=args.iter,
        size=args.size,
        save_model=args.save_model,
    )
