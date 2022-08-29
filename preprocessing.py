from string import ascii_lowercase
from language import LanguageProcessor
from deep_translator import GoogleTranslator
from tqdm import tqdm
import numpy as np
import os
import pickle
import json
from gensim.models import Word2Vec

rus_to_en = GoogleTranslator(source='ru', target='en')
en_to_rus = GoogleTranslator(source='en', target='ru')
lp = LanguageProcessor()

allowed_punct = ',.?!:'
cyrillic_lowercase = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
allowed_charset = ascii_lowercase + \
    allowed_punct + cyrillic_lowercase


def safe_translate(text, alpha=1000, to_en=True):
    output = ''
    n = len(text) // alpha
    for i in range(n + 1):
        part = text[i * alpha:(i + 1) * alpha]
        if to_en:
            output += rus_to_en.translate(part)
        else:
            output += en_to_rus.translate(part)
    return output


class TextUnit:
    def __init__(self, text):
        if len(text) < 10:
            self.empty = True
            return
        self.text = ''.join(
            ch if ch in allowed_charset else ' '
            for ch in text.lower())
        for ch in self.text:
            if ch in cyrillic_lowercase:
                self.text = safe_translate(self.text)
                break
        self.extracted, _ = lp.extract(self.text, with_punct=True)
        self.empty = False

    def data_for_w2v(self, context):
        sentences = []
        current = []
        for token in self.extracted:
            current.append(token)
            for p in ['?', '.', '!']:
                if p in token:
                    break
            else:
                continue
            if current:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        for i in range(len(sentences)):
            sentences[i] = ['<empty>'] * (context - 1) + sentences[i]
        return sentences

    def sliding_window(self, vocab, context):
        encoded = np.array([vocab['<empty>']] * (context - 1) + [
            vocab[token] for token in self.extracted])
        x = np.lib.stride_tricks.sliding_window_view(encoded, context)[:-1]
        y = encoded[context:]
        return x, y


def extract_text_units(path, wa_data=False):
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()

    if wa_data:
        chunks = text.splitlines()
        for ch in [']', ':']:
            chunks = [
                chunk[chunk.index(ch) + 2:]
                for chunk in chunks]
    else:
        text = text.replace('\n', ' ')
        chunks = text.split('$split$')        

    units = []

    for chunk in tqdm(chunks):
        unit = TextUnit(chunk)
        if not unit.empty:
            units.append(unit)

    return units


folder = input('Input folder: ')
output = input('Output folder: ')
context, vectorize, vector_size = 5, False, 100
units = []

print('Analysing given files...')

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if file.endswith('.txt'):
        print('Found text file!')
        units += extract_text_units(
            path, wa_data=file.startswith('wa_'))
    elif file == 'config.json':
        print('Found config!')
        with open(path, 'r') as f:
            d = json.load(f)
        if 'context' in d:
            context = d['context']
        if 'vectorize' in d:
            vectorize = d['vectorize']
        if 'vector_size' in d:
            vector_size = d['vector_size']

print('Extracted', len(units), 'text units.')
print('Collecing sentences...')

sentences = []

for u in units:
    sentences += u.data_for_w2v(context)

print('Collected', len(sentences), 'sentences.')
print('Training word2vec model...')

model = Word2Vec(
    sentences=sentences,
    min_count=1,
    workers=2,
    window=5,
    vector_size=vector_size
)

print('Constructing X and Y...')

x, y = [], []

for unit in tqdm(units):
    x_sub, y_sub = unit.sliding_window(
            model.wv.key_to_index, context)
    if vectorize:
        x_sub = np.array([
            model.wv[i] for i in x_sub
        ])
    x.append(x_sub)
    y.append(y_sub)

x = np.concatenate(x)
y = np.concatenate(y)

print('Making directory...')

os.mkdir(output)

print('Saving word2vec model...')

model.save(f'output\\word2vec.model')

print('Saving X and Y...')

for lbl, obj in zip('xy', [x, y]):
    with open(f'{output}\\{lbl}.pickle', 'wb') as f:
        pickle.dump(obj, f)

print('Done!')