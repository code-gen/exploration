import os
import pickle

import numpy as np
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


def load_pt_glove(emb_file: str) -> dict:
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        print(" * load glove from pickle")
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        glove_emb = pickle.load(open(emb_file, "rb"))
    else:
        print(" * load glove from txt, dumping to pickle")
        glove_emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding='latin'))
        pickle.dump(glove_emb, open(emb_file + ".pickle", "wb"))

    return glove_emb


def load_ft_glove(vocab_file, pt_emb_file, ft_emb_file):
    print(" * load vocab from [%s]" % vocab_file)
    print(" * load pre-trained glove from [%s]" % pt_emb_file)
    print(" * load fine-tuned glove from [%s]" % ft_emb_file)

    glove_emb = load_pt_glove(pt_emb_file)

    ft_glove_emb_arr = pickle.load(open(ft_emb_file, "rb"))
    vocab = pickle.load(open(vocab_file, "rb"))

    ft_glove_emb = {w: ft_glove_emb_arr[i] for w, i in vocab.items()}

    for w in tqdm(ft_glove_emb, desc="Mixing embeddings"):
        if w not in glove_emb:
            glove_emb[w] = ft_glove_emb[w]
        else:
            glove_emb[w] = 0.5 * ft_glove_emb[w] + 0.5 * glove_emb[w]

    return glove_emb
