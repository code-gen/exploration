import argparse
import os
import random
from tokenize import TokenError

import pandas as pd
from tqdm.auto import tqdm

from common import clean_text, write_to_file
from sketch_generation import Sketch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-in_dir', type=str, required=True)
arg_parser.add_argument('-out_dir', type=str, required=True)
arg_parser.add_argument('-split', type=float, default=0.1, help="Percentage of training example to hold-out for validation.")
args = arg_parser.parse_args()

TRAIN_FILE = os.path.join(args.in_dir, 'conala-train.json')
TEST_FILE = os.path.join(args.in_dir, 'conala-test.json')

OUT_TRAIN_FILE = os.path.join(args.out_dir, 'train.json')
OUT_TEST_FILE = os.path.join(args.out_dir, 'test.json')
OUT_DEV_FILE = os.path.join(args.out_dir, 'dev.json')


def preprocess(df, verbose=False):
    out_data = []
    skipped = []

    for i, ex in tqdm(df.iterrows(), total=len(df)):
        label = ex['rewritten_intent'] if ex['rewritten_intent'] is not None else ex['intent']
        label = [x.strip() for x in list(map(clean_text, label.split())) if x.strip() != '']

        code = ex['snippet']
        code_tokens = code.split()

        # TODO: needs splitting

        try:
            sketch = Sketch(code, verbose).generate()
        except TokenError:
            skipped.append(ex)
            continue

        if len(sketch) != len(code_tokens):
            sketch = str(sketch).split()

            if verbose:
                for j in range(min(len(sketch), len(code_tokens))):
                    print("[%s] [%s]" % (sketch[j], code_tokens[j]))
                for k in range(j, len(sketch)):
                    print("[%s] [N/A]" % sketch[k])
                for k in range(j, len(code)):
                    print("[N/A] [%s]" % code_tokens[j])

            skipped.append(ex)
            continue

        out_data.append({'token': code_tokens, 'src': label, 'type': sketch})

    return out_data


if __name__ == '__main__':
    train_data = preprocess(df=pd.read_json(TRAIN_FILE))

    random.shuffle(train_data)
    n = int(args.split * len(train_data))
    dev_data, train_data = train_data[:n], train_data[n:]

    test_data = preprocess(df=pd.read_json(TEST_FILE))

    write_to_file(OUT_TRAIN_FILE, train_data)
    write_to_file(OUT_DEV_FILE, train_data)
    write_to_file(OUT_TRAIN_FILE, train_data)
