import os
import pickle

import numpy as np
from gensim.models import KeyedVectors
from tqdm.auto import tqdm

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
