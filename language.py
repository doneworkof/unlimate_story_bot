from string import punctuation
import re
import nltk
import json

#nltk.download('popular')

lemmatizer = nltk.stem.WordNetLemmatizer()

class LanguageProcessor:
    def __init__(self):
        with open('langutils.json', 'r', encoding='utf-8') as f:
            langutils = json.load(f)
        self.usless, self.abbreviations, self.slang = [
            langutils['usless'], langutils['abbreviations'], langutils['slang']
        ]

    def _replace(self, s, selector, replacement):
        while match := re.search(selector, s):
            s_l = list(s)
            start, end = match.span()
            s_l[start:end] = replacement
            s = ''.join(s_l)
        return s
    
    def remove_short_verbs(self, s):
        return self._replace(s, "in'(?![A-Za-z0-9])", 'ing')

    def remove_short_alpha(self, s):
        return self._replace(s, "s'(?![A-Za-z0-9])", "s's")
    
    def remove_slang(self, s):
        for sl in self.slang:
            s = self._replace(s, f'(^| ){sl}( |$)', f' {self.slang[sl]} ')
        return s

    def remove_abbreviations(self, s):
        for abbr in self.abbreviations:
            s = self._replace(s, abbr, f' {self.abbreviations[abbr]}')
        return s

    def is_question(self, s):
        match = re.search('(\.|\!|\?)*$', s)
        if not match:
            return False
        end_punct = match.group()
        return '?' in end_punct
    
    def extract(self, s, with_punct=False):
        is_question = self.is_question(s)

        processing = [
            self.remove_short_verbs,
            self.remove_slang,
            self.remove_short_alpha,
            self.remove_abbreviations,
            lambda x: x.replace('’', "'")
        ]

        for pr in processing:
            s = pr(s)

        words = nltk.word_tokenize(s)
        tags = nltk.pos_tag(words)
        extracted = []

        for word, tag in tags:
            if word in self.usless:
                continue
            elif word[0] in punctuation and not with_punct:
                continue
            elif word == "'s":
                word = '<alpha>'
            elif tag == 'JJR':
                to_add = [
                    'more', lemmatizer.lemmatize(
                        word, 'a'
                    )
                ]
            elif tag == 'JJS':
                lemmatized = lemmatizer.lemmatize(
                    word, 'a'
                )
                if lemmatized == word:
                    to_add = [word]
                else:
                    to_add = ['most', lemmatized]
            elif tag[:2] == 'NN' and tag[-1] == 'S':
                to_add = [
                    lemmatizer.lemmatize(word)
                ]
            elif tag in ['VBP', 'VBZ', 'VBG']:
                to_add = [
                    lemmatizer.lemmatize(word, 'v')
                ]
            elif tag in ['VBD']:
                to_add = [
                    'did', lemmatizer.lemmatize(word, 'v')
                ]
            else:
                to_add = [word]

            extracted += to_add

        return extracted, is_question