"""
Check the quality of fine-tuned embeddings.
"""

import argparse
from pprint import pprint

import numpy as np
from tqdm.auto import tqdm

from utils import load_ft_glove, load_pt_glove

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-vocab_file", type=str, help="Vocabulary file")
arg_parser.add_argument("-pt_emb_file", type=str, help="Pre-trained embeddings file")
arg_parser.add_argument("-ft_emb_file", type=str, help="Fine-tuned embeddings file")

args = arg_parser.parse_args()


def closest_to(emb: dict, w, n=1):
    xs = []

    for w_ in tqdm(emb, desc="Getting %d closest to %s" % (n, w)):
        if w == w_: continue
        xs += [(w_, np.dot(emb[w], emb[w_]) / (np.linalg.norm(emb[w]) * np.linalg.norm(emb[w_])))]

    return [x for x, _ in sorted(xs, key=lambda x: -x[1])[:n]]


def main():
    print(" * loading fine-tuned glove [%s]" % args.ft_emb_file)
    # ft_emb = load_ft_glove(args.vocab_file, args.pt_emb_file, args.ft_emb_file)

    print(" * loading pre-trained glove [%s]" % args.pt_emb_file)
    pt_emb = load_pt_glove(args.pt_emb_file)

    while True:
        word = input("> enter word: ").strip()

        try:
            pprint(closest_to(pt_emb, word, n=20))
        except KeyError:
            print("%s not in emb" % word)

        print()

        # print("fine-tuned emb")
        # try:
        #     pprint(closest_to(ft_emb, word, n=20))
        # except KeyError:
        #     print("%s not in ft_emb" % word, "\n")


if __name__ == '__main__':
    main()
