"""
This is a standalone script that performs fine-tuning for glove embeddings
given a vocabulary constructed from a corpus of text (e.g. pydoc).

File names are standardized, each experiment being identified by:
    name, vocab_size, window_size, num_ft_iter
However, the path to the pre-trained embeddings file must be given as arg, it's usually glove.840B.300d.txt

The loading will look for a pickle dump (i.e. glove.840B.300d.txt.pickle);
if it doesn't exist, the txt file will be loaded then a pickled version will be dumped.
"""

import argparse
import os
import pickle
import re
from collections import Counter

import numpy as np
from mittens.tf_mittens import Mittens
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from utils import clean_text, load_pt_glove

stopWords = set(stopwords.words('english'))

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-root_dir", type=str, default="./data-out", help="Root directory used to store files")
arg_parser.add_argument("-data_source", type=str, help="Path to dir with files OR file listing")
arg_parser.add_argument("-name", type=str, help="Name for data source (e.g. pydoc)")
arg_parser.add_argument("-num_ft_iter", type=int, default=1000, help="Number of fine-tuning iterations")
arg_parser.add_argument("-pt_emb_file", type=str, help="Pre-trained embeddings file")
arg_parser.add_argument("-vocab_size", type=int, default=20000)
arg_parser.add_argument("-window_size", type=int, default=5)

args = arg_parser.parse_args()

VOCAB_FILE = os.path.join(
    args.root_dir, '%s-vocab-%d-window-%d.vocab' % (args.name, args.vocab_size, args.window_size)
)
CO_OCCURR_MAT_FILE = os.path.join(
    args.root_dir, '%s-vocab-%d-window-%d.mat' % (args.name, args.vocab_size, args.window_size)
)
FT_EMB_FILE = os.path.join(
    args.root_dir, "%s-glove-fine-tuned-vocab-%d-window-%d-iter-%d" % (
        args.name, args.vocab_size, args.window_size, args.num_ft_iter
    )
)


def do_fine_tune() -> None:
    print(" * loading vocab [%s]" % VOCAB_FILE)
    vocab = pickle.load(open(VOCAB_FILE, "rb"))

    print(" * loading co-occurrence [%s]" % CO_OCCURR_MAT_FILE)
    co_occurrence = pickle.load(open(CO_OCCURR_MAT_FILE, "rb"))

    print(" * loading pre-trained glove [%s]" % args.pt_emb_file)
    emb_glove = load_pt_glove(args.pt_emb_file)

    dim = len(emb_glove['the'])
    print(" * word vec dim: %d" % dim)

    print(" * training for %d iterations" % args.num_ft_iter)
    mittens_model = Mittens(n=dim, max_iter=args.num_ft_iter)
    new_emb_glove = mittens_model.fit(
        co_occurrence,
        vocab=list(vocab.keys()),
        initial_embedding_dict=emb_glove
    )

    print(" * fine-tuning done; saving to file [%s]" % FT_EMB_FILE)
    pickle.dump(new_emb_glove, open(FT_EMB_FILE, 'wb'))


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


def get_all_words(data_source: str):
    # source: path to a directory containg files, or to a file containing paths

    all_words = []

    if os.path.isdir(data_source):
        listing = os.listdir(data_source)
        path_of = lambda x: os.path.join(data_source, x)
    else:  # regular file
        listing = [l.strip() for l in open(data_source).readlines()]
        path_of = lambda x: x

    for l in tqdm(listing, desc="Get all words"):
        file_contents = [clean_text(l.strip().lower()) for l in open(path_of(l), "rt").readlines()]

        for line in file_contents:
            for w in line.split():
                if re.match(r'[\w]+', w) and w not in stopWords:
                    all_words.append(w)

    return all_words


def create_vocab_counter(words):
    vocab = Counter()
    for w in tqdm(words):
        vocab[w] += 1

    # print("len(vocab) = %d" % len(vocab))
    return vocab


def create_vocab_and_cooccurrence_matrix(data_source, vocab_size=20000, window_size=5) -> None:
    # top_words, top_freqs = zip(*vocab.most_common(thr))
    # top_words = set(top_words)

    # print("Applying TF-IDF on [%s]" % data_source)
    # top_words = set(tf_idf(data_source=data_source, vocab_size=vocab_size, min_df=0.1))
    # print("len(top_words) = ", len(top_words))

    all_words = get_all_words(data_source)
    print("len(all_words) = %d" % len(all_words))

    top_words, _ = zip(*create_vocab_counter(all_words).most_common()[:vocab_size])
    top_words = set(top_words)

    word2idx = {w: i for i, w in enumerate(top_words)}
    co_occur_mat = np.zeros((vocab_size, vocab_size), dtype=np.uint16)

    for i in tqdm(range(len(all_words)), desc='Constructing co-occurrence matrix'):
        if all_words[i] not in top_words: continue

        # window-search
        for j in range(max(i - window_size, 0), min(i + window_size, len(all_words))):
            if i == j or all_words[j] not in top_words: continue
            co_occur_mat[word2idx[all_words[i]], word2idx[all_words[j]]] += 1

    pickle.dump(word2idx, open(VOCAB_FILE, "wb"))
    pickle.dump(co_occur_mat, open(CO_OCCURR_MAT_FILE, "wb"))


def main():
    print("Vocabulary: [%s]" % VOCAB_FILE)
    print("Co-occurrence matrix: [%s]" % CO_OCCURR_MAT_FILE)

    if any([not os.path.isfile(VOCAB_FILE), not os.path.isfile(CO_OCCURR_MAT_FILE)]):
        print(" * preprocessing")
        create_vocab_and_cooccurrence_matrix(args.data_source, args.vocab_size, args.window_size)

    print(" * fine-tuning")
    do_fine_tune()


if __name__ == '__main__':
    main()
