"""
This is a standalone script that performs fine-tuning for glove embeddings
given a vocabulary constructed from a corpus of text (e.g. pydoc).

The loading will look for a pickle dump (e.g. glove.840B.300d.txt.pickle);
if it doesn't exist, the txt file will be loaded then a pickled version will be dumped.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
from mittens.tf_mittens import Mittens
from tqdm.auto import tqdm

from utils import create_vocab_counter, dump_cfg, get_all_words, get_all_words_python_so, load_pt_glove

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-root_dir", type=str, default="../embeddings", help="Root directory used to store files")
arg_parser.add_argument("-data_source", type=str, help="Path to dir with files OR file listing")
arg_parser.add_argument("-exp_name", type=str, help="Name for current experiment")
arg_parser.add_argument("-pt_emb_file", type=str, help="Pre-trained embeddings file")

# fine-tune settings
arg_parser.add_argument("-num_ft_iter", type=int, default=1000, help="Number of fine-tuning iterations")
arg_parser.add_argument("-vocab_size", type=int, default=20000, help="Number of words to consider (at most!)")
arg_parser.add_argument("-window_size", type=int, default=5)
arg_parser.add_argument("-only_in_emb", action="store_true", help="If true, only use words that already exist in the pre-trained embeddings")
arg_parser.add_argument("-min_freq", type=int, default=1, help="Consider words with frequency >= min_freq")
arg_parser.add_argument("-mu", type=float, default=0.1, help="Regularization factor (mu from mittens paper)")

args = arg_parser.parse_args()


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(" * loading pre-trained glove [%s]" % args.pt_emb_file)
    pt_emb = load_pt_glove(args.pt_emb_file)

    args.emb_dim = len(pt_emb['the'])

    out_dir = os.path.join(args.root_dir, "%s-%s" % (timestamp, args.exp_name))
    print(" * make dir for current experiment [%s]" % out_dir)
    os.makedirs(out_dir, exist_ok=False)

    print(" * get all words from data source")
    # all_words = get_all_words(args.data_source, emb=pt_emb if args.only_in_emb else None)
    all_words = get_all_words_python_so(args.data_source, min_freq=args.min_freq, emb=pt_emb if args.only_in_emb else None)
    print(" * len(all_words) = %d" % len(all_words))

    print(" * get top %d words from all words" % args.vocab_size)
    top_words, _ = zip(*create_vocab_counter(all_words).most_common()[:args.vocab_size])
    top_words_set = set(top_words)
    print(" * len(top_words) = %d (vocab_size = %d)" % (len(top_words), args.vocab_size))

    word2idx = {w: i for i, w in enumerate(top_words)}

    n = min(args.vocab_size, len(top_words))
    co_occurr_mat = np.zeros((n, n), dtype=np.uint16)
    print(" * co_occurr_mat shape =", co_occurr_mat.shape)

    for i in tqdm(range(len(all_words)), desc='Constructing co-occurrence matrix'):
        if all_words[i] not in top_words_set:
            continue

        # window-search
        for j in range(max(i - args.window_size, 0), min(i + args.window_size, len(all_words))):
            if i == j or all_words[j] not in top_words_set:
                continue
            co_occurr_mat[word2idx[all_words[i]], word2idx[all_words[j]]] += 1

    # dump data
    args.vocab_size = len(co_occurr_mat)

    vocab_file = os.path.join(out_dir, '%s.vocab' % args.exp_name)
    co_occur_mat_file = os.path.join(out_dir, '%s.mat' % args.exp_name)
    ft_emb_file = os.path.join(out_dir, "%s.emb" % args.exp_name)

    print(" * dumping vocab to [%s]" % vocab_file)
    pickle.dump(word2idx, open(vocab_file, "wb"))
    print(" * dumping co_occurr_mat to [%s]" % co_occur_mat_file)
    pickle.dump(co_occurr_mat, open(co_occur_mat_file, "wb"))

    print(" * dump args")
    dump_cfg(os.path.join(out_dir, "config.txt"), cfg=args)

    # FINETUNE

    print(" * fine-tuning for %d iterations" % args.num_ft_iter)
    mittens_model = Mittens(n=args.emb_dim, max_iter=args.num_ft_iter, mittens=args.mu)
    ft_emb = mittens_model.fit(
        X=co_occurr_mat,
        vocab=top_words,
        initial_embedding_dict=pt_emb
    )

    print(" * fine-tuning done; saving to file [%s]" % ft_emb_file)
    pickle.dump(ft_emb, open(ft_emb_file, 'wb'))


if __name__ == '__main__':
    main()
