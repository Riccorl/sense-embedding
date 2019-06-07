from copy import deepcopy
from typing import Dict

from lxml import etree
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import constants as const
import utils


def fast_iter(parser, func):
    """
    Iter over an xml file, remove from memory the nodes already seen.
    :param parser: lxml iterparse.
    :param func: function to apply to every node.
    :return:
    """
    for event, elem in tqdm(parser):
        func(elem)
        elem.clear()
        # eliminate now-empty references from the root node
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del parser


def filter_eurosense(path: str, out_path: str, lang="en"):
    """
    Parse the xml file and writes only the lines with tag lang.
    :param path: path of the xml file.
    :param out_path: path of the new xml file.
    :param lang: language to keep.
    :return:
    """
    with open(out_path, mode="w") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<corpus source="europarl">\n')
        parser = etree.iterparse(
            path, events=("end",), tag="sentence", remove_blank_text=True
        )
        fast_iter(
            parser,
            lambda elem: out.write(
                etree.tostring(
                    _filter_node(elem, lang), encoding="unicode", pretty_print=True
                )
            ),
        )
        out.write("</corpus>")


def _filter_node(elem: etree.Element, lang="en") -> str:
    """
    Create a new node tree with only the selected language.
    :param elem: node to filter.
    :param lang: language to keep.
    :return:
    """
    # Create a root node
    sentence = etree.Element("sentence", attrib={"id": elem.attrib["id"]})
    # Copy only the english text
    sentence.append(deepcopy(elem.find('text/[@lang="en"]')))
    # create annotations child
    annotations = etree.SubElement(sentence, "annotations")
    # filter the english ones
    filtered = elem.findall('annotations/annotation/[@lang="{}"]'.format(lang))
    # copy them in the new tree
    annotations.extend(deepcopy(a) for a in filtered)

    return sentence


def write_sentences(
    path: str, out_path: str, bn_wn_map: Dict[str, str], check_synset: bool = False
):
    """
    Produce a file of sentences with senses.
    :param path: path of input file.
    :param out_path: path of output file.
    :param bn_wn_map: mapping file from bn synsets to wn.
    :param check_synset:
    :return:
    """
    with open(out_path, mode="w", encoding="utf-8") as out:
        parser = etree.iterparse(
            path, events=("end",), tag="sentence", remove_blank_text=True
        )
        fast_iter(
            parser,
            lambda elem: out.write(
                _extract_annotations(elem, bn_wn_map, check_synset) + "\n"
            ),
        )


def _extract_annotations(
    elem: etree.Element, bn_wn_map: Dict[str, str], check_synset: bool = False
) -> str:
    """
    Extract annotatiions and replace words with senses.
    :param elem: node of the tree.
    :param bn_wn_map: mapping file from bn synsets to wn.
    :param check_synset:
    :return: the string with words replaced with sense
    """
    text = elem.xpath("//text")[0].text
    # if not text, return empty string
    if not text:
        return ""

    for annotation in elem.xpath(".//annotation"):
        if _is_valid_synset(annotation, bn_wn_map, check_synset):
            text = _replace_sense(annotation, text)

    return text


def _replace_sense(annotation: etree.Element, text: str) -> str:
    """
    replace words with senses.
    :param annotation: annotation node of the xml file.
    :param text: original sentence.
    :return: sentence in which words are replaced with senses.
    """
    # extract anchor
    anchor = annotation.attrib["anchor"]
    # extract lemma
    lemma = annotation.attrib["lemma"].replace(" ", "_").replace("-", "_").lower()
    # extract synset
    synset = annotation.text

    return text.replace(anchor + " ", lemma + "_" + synset + " ", 1)


def _is_valid_synset(
    annotation: etree.Element, bn_wn_map: Dict[str, str], check_synset: bool = False
) -> bool:
    """
    Check if a sentence is valid or not.
    :param annotation: annotation node of the xml file.
    :param bn_wn_map: mapping file from bn synsets to wn.
    :param check_synset: chek if synset is correct.
    :return: True if the annotation is valid, false otherwise.
    """
    # if not in the mapping file, skip
    if not bn_wn_map.get(annotation.text):
        return False

    if check_synset:
        # retrieve wordnet synset
        offset = bn_wn_map[annotation.text][0]
        lemmas_bn = annotation.attrib["lemma"].lower().split()
        lemmas_wn = wn.synset_from_pos_and_offset(
            offset[-1], int(offset[:-1])
        ).lemma_names()

        # check if the given lemma is somehow correct for the given synset.
        lwn = "|".join(set(l.lower() for l in lemmas_wn))
        return any(lbn for lbn in lemmas_bn if lbn in lwn)
    else:
        return True


def main(path_input: str, path_output: str, check_synset: bool = False):
    # read bn to wn mapping file
    bnwn_map = utils.read_dictionary(const.BN2WN_MAP)
    # write a file with only sentences, each annotated word is replaced with the sense
    write_sentences(path_input, path_output, bnwn_map, check_synset)
    # compute a dictionary -> senses
