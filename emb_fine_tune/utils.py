import glob
import os
import pickle
import re
from argparse import Namespace
from collections import Counter

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

stopWords = set(stopwords.words('english'))

PUNCTUATION = {
    'sep'   : u'\u200b' + "/-'´′‘…—−–",
    'keep'  : "&",
    'remove': '?!.,，"#$%\'()*+-/:;<=>@[\\]^_`{|}~“”’™•°'
}


def clean_text(x):
    x = x.lower()

    for p in PUNCTUATION['sep']:
        x = x.replace(p, " ")
    for p in PUNCTUATION['keep']:
        x = x.replace(p, " %s " % p)
    for p in PUNCTUATION['remove']:
        x = x.replace(p, "")

    return x


def get_all_words(data_source: str, emb: dict = None) -> list:
    """source: path to a directory containg files, or to a file containing paths"""

    all_words = []

    if os.path.isdir(data_source):
        listing = sorted(glob.glob('%s/**/*.txt' % data_source, recursive=True))
    else:  # regular file
        listing = [l.strip() for l in open(data_source).readlines()]

    if emb is not None:
        word_predicate = lambda w: re.match(r'[\w]+', w) and w not in stopWords and w in emb
    else:
        word_predicate = lambda w: re.match(r'[\w]+', w) and w not in stopWords

    for file in tqdm(listing, desc="Get all words"):
        # TODO: use something smarter (spacy / nltk)
        lines = [clean_text(l.strip().lower()) for l in open(file, "rt").readlines()]
        all_words += [w for line in lines for w in line.split() if word_predicate(w)]

    return all_words


def get_all_words_python_so(pickle_file: str, min_thr=1, emb: dict = None) -> list:
    all_words = []
    counter = Counter()
    question_words = pickle.load(open(pickle_file, 'rb'))

    if emb is not None:
        word_predicate = lambda w: w in emb
    else:
        word_predicate = lambda w: True

    for sent in tqdm(question_words, desc="Question words processing"):
        for word in sent:
            if word_predicate(word):
                counter[word] += 1
                all_words.append(word)

    return [word for word in all_words if counter[word] >= min_thr]


def create_vocab_counter(words):
    vocab = Counter()
    for w in tqdm(words, desc="Creating vocab counter"):
        vocab[w] += 1

    # print("len(vocab) = %d" % len(vocab))
    return vocab


def tf_idf(data_source: str, vocab_size: int, min_df=0.1, max_df=1.0):
    # source: path to a directory containg files, or to a file containing paths

    if os.path.isdir(data_source):
        files = [os.path.join(data_source, file) for file in os.listdir(data_source)]
    else:  # singular file
        files = [l.strip() for l in open(data_source, "rt").readlines()]

    tfidf = TfidfVectorizer(
        input="filename",
        decode_error="ignore",
        analyzer="word",
        stop_words="english",
        token_pattern='[a-zA-Z]+',
        max_features=vocab_size,
        min_df=min_df,
        max_df=max_df
    )

    tfidf.fit(files)

    return tfidf.get_feature_names()


### Embedding loaders

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_pt_glove(emb_file: str) -> dict:
    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        print(" * load glove from pickle")
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        emb = pickle.load(open(emb_file, "rb"))
    else:
        print(" * load glove from txt, dumping to pickle")
        emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding='latin'))
        pickle.dump(emb, open(emb_file + ".pickle", "wb"))

    return emb


def load_ft_glove(vocab_file, pt_emb_file, ft_emb_file, ft_factor=0.5, pt_factor=0.5):
    print(" * load vocab from [%s]" % vocab_file)
    print(" * load pre-trained glove from [%s]" % pt_emb_file)
    print(" * load fine-tuned glove from [%s]" % ft_emb_file)

    glove_emb = load_pt_glove(pt_emb_file)

    ft_glove_emb_arr = pickle.load(open(ft_emb_file, "rb"))
    vocab = pickle.load(open(vocab_file, "rb"))

    ft_glove_emb = {w: ft_glove_emb_arr[i] for w, i in vocab.items()}

    for w in tqdm(ft_glove_emb, desc="Mixing embeddings (ft %.2f, pt %.2f)" % (ft_factor, pt_factor)):
        if w not in glove_emb:
            glove_emb[w] = ft_glove_emb[w]
        else:
            glove_emb[w] = ft_factor * ft_glove_emb[w] + pt_factor * glove_emb[w]

    return glove_emb


def load_word2vec(emb_file: str):
    return KeyedVectors.load_word2vec_format(emb_file, binary=True)


def load_wiki(emb_file: str) -> dict:
    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        print(" * load wiki from pickle")
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        emb = pickle.load(open(emb_file, "rb"))
    else:
        print(" * load wiki from txt, dumping to pickle")
        emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file) if len(o) > 100)
        pickle.dump(emb, open(emb_file + ".pickle", "wb"))

    return emb


def load_paragram(emb_file: str) -> dict:
    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        print(" * load paragram from pickle")
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        emb = pickle.load(open(emb_file, "rb"))
    else:
        print(" * load paragram from txt, dumping to pickle")
        emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding="utf8", errors='ignore') if len(o) > 100)
        pickle.dump(emb, open(emb_file + ".pickle", "wb"))

    return emb


def dump_cfg(file: str, cfg: Namespace) -> None:
    cfg = sorted(vars(cfg).items(), key=lambda x: x[0])
    fp = open(file, "wt")
    for k, v in cfg:
        fp.write("%32s: %s\n" % (k, v))
    fp.close()
