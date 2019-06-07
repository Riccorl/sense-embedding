import xml
from pathlib import Path
from typing import Dict

from lxml import etree
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import constants as const
import utils


def preprocess_sew(input_folder: Path, path_output: str, bn_wn_map: Dict[str, str]):
    """
    Preprocess SEW dataset and writes it in a single text file.
    :param input_folder: folder where SEW is placed.
    :param path_output: file to write SEW.
    :param bn_wn_map: mapping file from bn synsets to wn.
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    pathlist = input_folder.glob("**/*.xml")
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    with open(path_output, mode="w", encoding="utf8") as out:
        for path in tqdm(pathlist, total=4320838):
            try:
                # because path is object not string
                root = xml.etree.cElementTree.parse(str(path), parser).getroot()
                if root.xpath("//wikiArticle")[0].attrib["language"] == "EN":
                    out.write(_extract_annotations(root, lemmatizer, bn_wn_map) + "\n")
            except etree.XMLSyntaxError:
                continue


def _extract_annotations(
    elem: etree.Element, lemmatizer: WordNetLemmatizer, bn_wn_map: Dict[str, str]
) -> str:
    """
    Extract annotatiions and replace words with senses.
    :param elem: node of the tree.
    :return: the string with words replaced with sense
    """
    text = elem.xpath("//text")[0].text
    # if not text, return empty string
    if not text:
        return ""

    for annotation in elem.xpath("//annotation"):
        if bn_wn_map.get(annotation.xpath("babelNetID")[0].text):
            text = _replace_sense(annotation, text, lemmatizer)

    return text


def _replace_sense(
    annotation: etree.Element, text: str, lemmatizer: WordNetLemmatizer
) -> str:
    """
    replae the word with the sense.
    :param annotation: annotation node of the xml file.
    :param text: original sentence.
    :return: sentence in which words are replaced with senses.
    """
    text = text.replace("\n", "")
    # extract synset
    synset = annotation.xpath("babelNetID")[0].text
    # extract anchor
    anchor = annotation.xpath("mention")[0].text
    # extract lemma and concatenate with _
    lemma = (
        lemmatizer.lemmatize(anchor, pos=synset[-1])
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )
    return text.replace(anchor + " ", lemma + "_" + synset + " ", 1)


def main(path_input: str, path_output: str):
    # read bn to wn mapping file
    bnwn_map = utils.read_dictionary(const.BN2WN_MAP)
    preprocess_sew(Path(path_input), path_output, bnwn_map)
