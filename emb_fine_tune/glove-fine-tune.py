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
from typing import Tuple

import numpy as np
from mittens.tf_mittens import Mittens
from tqdm.auto import tqdm

from utils import create_vocab_counter, get_all_words_python_so, load_pt_glove, dump_cfg

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-root_dir", type=str, default="../embeddings", help="Root directory used to store files")
arg_parser.add_argument("-data_source", type=str, help="Path to dir with files OR file listing")
arg_parser.add_argument("-name", type=str, help="Name for data source (e.g. pydoc)")
arg_parser.add_argument("-num_ft_iter", type=int, default=1000, help="Number of fine-tuning iterations")
arg_parser.add_argument("-pt_emb_file", type=str, help="Pre-trained embeddings file")
arg_parser.add_argument("-vocab_size", type=int, default=20000)
arg_parser.add_argument("-window_size", type=int, default=5)

args = arg_parser.parse_args()


def do_fine_tune(vocab_file: str, co_occurr_mat_file: str, pt_emb_file: str, num_ft_iter: str, out_ft_emb_file: str) -> None:
    print(" * loading vocab [%s]" % vocab_file)
    vocab = pickle.load(open(vocab_file, "rb"))

    print(" * loading co-occurrence matrix [%s]" % co_occurr_mat_file)
    co_occurrence = pickle.load(open(co_occurr_mat_file, "rb"))

    print(" * loading pre-trained glove [%s]" % pt_emb_file)
    emb_glove = load_pt_glove(pt_emb_file)

    dim = len(emb_glove['the'])
    print(" * word vec dim: %d" % dim)

    print(" * training for %d iterations" % num_ft_iter)
    mittens_model = Mittens(n=dim, max_iter=num_ft_iter, mittens=0.1)
    new_emb_glove = mittens_model.fit(
        co_occurrence,
        vocab=list(vocab.keys()),
        initial_embedding_dict=emb_glove
    )

    print(" * fine-tuning done; saving to file [%s]" % out_ft_emb_file)
    pickle.dump(new_emb_glove, open(out_ft_emb_file, 'wb'))


def do_preprocess(data_source, pt_emb_file, vocab_size=20000, window_size=5) -> Tuple[str, str, str]:
    # make dir for current experiment
    out_dir = os.path.join(args.root_dir, "%s" % args.name)
    os.makedirs(out_dir, exist_ok=False)

    # top_words, top_freqs = zip(*vocab.most_common(thr))
    # top_words = set(top_words)

    # print("Applying TF-IDF on [%s]" % data_source)
    # top_words = set(tf_idf(data_source=data_source, vocab_size=vocab_size, min_df=0.1))
    # print("len(top_words) = ", len(top_words))

    # all_words = get_all_words(data_source, emb=load_pt_glove(pt_emb_file))
    all_words = get_all_words_python_so(data_source, min_thr=100, emb=load_pt_glove(pt_emb_file))
    print(" * len(all_words) = %d" % len(all_words))

    top_words, _ = zip(*create_vocab_counter(all_words).most_common()[:vocab_size])
    print(" * len(top_words) = %d (vocab_size = %d)" % (len(top_words), vocab_size))

    word2idx = {w: i for i, w in enumerate(top_words)}

    n = min(vocab_size, len(top_words))
    co_occurr_mat = np.zeros((n, n), dtype=np.uint16)
    print(" * co-occurr-mat shape =", co_occurr_mat.shape)

    top_words = set(top_words)

    for i in tqdm(range(len(all_words)), desc='Constructing co-occurrence matrix'):
        if all_words[i] not in top_words: continue

        # window-search
        for j in range(max(i - window_size, 0), min(i + window_size, len(all_words))):
            if i == j or all_words[j] not in top_words: continue
            co_occurr_mat[word2idx[all_words[i]], word2idx[all_words[j]]] += 1

    # dump data
    args.vocab_size = len(co_occurr_mat)

    vocab_file = os.path.join(out_dir, '%s.vocab' % args.name)
    co_occur_mat_file = os.path.join(out_dir, '%s.mat' % args.name)
    ft_emb_file = os.path.join(out_dir, "%s.emb" % args.name)

    pickle.dump(word2idx, open(vocab_file, "wb"))
    pickle.dump(co_occurr_mat, open(co_occur_mat_file, "wb"))

    # dump args
    dump_cfg(os.path.join(out_dir, "config.txt"), cfg=args)

    return vocab_file, co_occur_mat_file, ft_emb_file


def main():
    print(" * preprocessing")
    vocab_file, co_occur_mat_file, ft_emb_file = do_preprocess(
        args.data_source, args.pt_emb_file, args.vocab_size, args.window_size
    )

    print(" * fine-tuning")
    do_fine_tune(
        vocab_file, co_occur_mat_file, args.pt_emb_file, args.num_ft_iter, ft_emb_file
    )


if __name__ == '__main__':
    main()
