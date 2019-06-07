import re
import string
from typing import List

from nltk.corpus import stopwords


class SentenceLoader(object):
    """Iterate over a sentence file from disk."""

    def __init__(self, filenames, complete: bool = True):
        self.filenames = filenames
        self.stop = set(stopwords.words("english")) | set(string.punctuation)
        self.html_regex = re.compile(r"&\w+;")
        self.complete = complete

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, mode="r", encoding="utf8") as file:
                for line in file:
                    if self.complete:
                        yield self.complete_clean(line)
                    else:
                        self.naive_clean(line)

    def naive_clean(self, line: str) -> List[str]:
        """
        Normalize in lowercase the string in input and remove stop words.
        :param line:
        :return:
        """
        return [word for word in line.lower().split() if word not in self.stop]

    def complete_clean(self, line: str) -> List[str]:
        """
        Remove punctuations and unescaped html character from string
        and normilize it in lowercase.
        :param line: string to clean
        :return: lowercase string without punctuations.
        """
        # ugly but faster than regex
        words_clean = (
            word.replace("-", "")
            .replace("`", "")
            .replace('"', "")
            .replace("'", "")
            .replace("’", "")
            .replace("–", "")
            for word in line.lower().split()
        )
        return [
            word
            for word in words_clean
            if word and word not in self.stop and "&" not in word
        ]
