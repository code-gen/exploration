"""
Check the quality of fine-tuned embeddings.
"""

import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from utils import load_ft_glove, load_pt_glove

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-vocab_file", type=str, help="Vocabulary file")
arg_parser.add_argument("-pt_emb_file", type=str, help="Pre-trained embeddings file")
arg_parser.add_argument("-ft_emb_file", type=str, help="Fine-tuned embeddings file")
arg_parser.add_argument("-ft_factor", type=float, default=0.5, help="How much to account for the fine-tuned vector")
arg_parser.add_argument("-pt_factor", type=float, default=0.5, help="How much to account for the pre-trained vector")
arg_parser.add_argument("-n", type=int, default=5, help="Number of words to show")
arg_parser.add_argument("-plot", action="store_true", help="Whether to plot the top-n sim scores side-by-side")

args = arg_parser.parse_args()

assert all([0 <= args.ft_factor <= 1, np.isclose(1 - args.ft_factor, args.pt_factor)])


def closest_to(emb: dict, w, n=1):
    xs = []

    for w_ in tqdm(emb, desc="Getting %d closest to %s" % (n, w)):
        if w == w_: continue
        cos = np.dot(emb[w], emb[w_]) / (np.linalg.norm(emb[w]) * np.linalg.norm(emb[w_]))
        xs += [(w_, cos)]

    return [(x, sim) for x, sim in sorted(xs, key=lambda x: x[1], reverse=True)[:n]]


def try_closest(name, emb, word, n=20):
    print(name)
    try:
        pprint(closest_to(emb, word, n))
    except KeyError:
        print("%s not in %s" % (word, name))
    print()


def try_closest_both(pt, ft, word, n=20, plot=False):
    try:
        cpt = closest_to(pt, word, n)
        cft = closest_to(ft, word, n)
        pprint(["pt: %20s (%.5f)    ft: %20s (%.5f)" % (p, ps, f, fs) for ((p, ps), (f, fs)) in zip(cpt, cft)])

        if plot:
            plt.suptitle(word)
            plt.subplot(1, 2, 1)
            plt.bar(range(n), [x[1] for x in cpt])
            plt.xticks(range(n), [x[0] for x in cpt], rotation=45)
            plt.title('pre-trained')
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.bar(range(n), [x[1] for x in cft])
            plt.xticks(range(n), [x[0] for x in cft], rotation=45)
            plt.title('fine-tuned')
            plt.tight_layout()

            plt.show()
    except KeyError:
        print("%s not in emb" % word)
    print()


def main():
    print(" * loading fine-tuned glove [%s]" % args.ft_emb_file)
    ft_emb = load_ft_glove(args.vocab_file, args.pt_emb_file, args.ft_emb_file, args.ft_factor, args.pt_factor)

    print(" * loading pre-trained glove [%s]" % args.pt_emb_file)
    pt_emb = load_pt_glove(args.pt_emb_file)

    while True:
        word = input("> enter word: ").strip()
        try_closest_both(pt_emb, ft_emb, word, n=args.n, plot=args.plot)


if __name__ == '__main__':
    main()
