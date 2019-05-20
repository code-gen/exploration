"""
This script will generate train / dev / test splits for the Django datasets.
Pre-processing also includes sketch generation and input sanitization.
"""

import argparse
import os
import random
from tokenize import TokenError
from typing import List, Tuple

from tqdm.auto import tqdm

from common import clean_text, print_skipped, replace_strings, split_accessor, write_to_file
from sketch_generation import Sketch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-in_dir', type=str, required=True)
arg_parser.add_argument('-out_dir', type=str, required=True)
arg_parser.add_argument('-dev_split', type=float, default=0.05)
arg_parser.add_argument('-test_split', type=float, default=0.1)
args = arg_parser.parse_args()

CODE_FILE = os.path.join(args.in_dir, 'all.code')
ANNO_FILE = os.path.join(args.in_dir, 'all.anno')

OUT_TRAIN_FILE = os.path.join(args.out_dir, 'train.json')
OUT_TEST_FILE = os.path.join(args.out_dir, 'test.json')
OUT_DEV_FILE = os.path.join(args.out_dir, 'dev.json')


def preprocess(data: List[Tuple[str, str]], verbose=False) -> Tuple[List[dict], List]:
    skipped = []
    out_data = []

    for raw_code, raw_anno in tqdm(data, desc='Processing'):
        code = replace_strings(raw_code)
        anno = replace_strings(raw_anno)

        try:
            sketch = Sketch(code, verbose).generate()
        except TokenError:
            skipped.append((raw_code, raw_anno))
            continue

        code_tokens = code.split()
        label = split_accessor([clean_text(x) for x in anno.split()])

        if len(sketch) != len(code_tokens):
            sketch = str(sketch).split()

            if verbose:
                for j in range(min(len(sketch), len(code_tokens))):
                    print("[%s] [%s]" % (sketch[j], code_tokens[j]))
                for k in range(j, len(sketch)):
                    print("[%s] [N/A]" % sketch[k])
                for k in range(j, len(code)):
                    print("[N/A] [%s]" % code_tokens[j])

            skipped.append((raw_code, raw_anno))
            continue

        out_data.append({
            'token': code_tokens,
            'src'  : label,
            'type' : str(sketch).split(),
        })

    print(" * done, skipped %d faulty examples" % len(skipped))
    return out_data, skipped


if __name__ == '__main__':
    code_lines = [l.strip() for l in open(CODE_FILE, "rt").readlines()]
    anno_lines = [l.strip() for l in open(ANNO_FILE, "rt").readlines()]

    all_data = list(zip(code_lines, anno_lines))
    random.shuffle(all_data)

    ds = int(args.dev_split * len(all_data))
    ts = int(args.test_split * len(all_data))

    dev_split, test_split, train_split = all_data[:ds], all_data[ds:][:ts], all_data[(ds + ts):]
    assert len(dev_split) + len(test_split) + len(train_split) == len(all_data)

    train_data, train_skipped = preprocess(train_split, verbose=False)
    dev_data, dev_skipped = preprocess(dev_split, verbose=False)
    test_data, test_skipped = preprocess(test_split, verbose=False)

    print_skipped('train', train_skipped)
    print_skipped('test', test_skipped)
    print_skipped('dev', dev_skipped)

    write_to_file(OUT_TRAIN_FILE, train_data)
    write_to_file(OUT_TEST_FILE, test_data)
    write_to_file(OUT_DEV_FILE, dev_data)
