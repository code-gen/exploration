import argparse
import io
import os
import random
import token
from tokenize import TokenError, tokenize

import pandas as pd
from tqdm.auto import tqdm

from common import replace_strings, write_to_file, split_accessor, clean_text
from sketch_generation import Sketch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-in_dir', type=str, required=True)
arg_parser.add_argument('-out_dir', type=str, required=True)
arg_parser.add_argument('-split', type=float, default=0.05, help="Percentage of training example to hold-out for validation.")
args = arg_parser.parse_args()

TRAIN_FILE = os.path.join(args.in_dir, 'conala-train.json')
TEST_FILE = os.path.join(args.in_dir, 'conala-test.json')

OUT_TRAIN_FILE = os.path.join(args.out_dir, 'train.json')
OUT_TEST_FILE = os.path.join(args.out_dir, 'test.json')
OUT_DEV_FILE = os.path.join(args.out_dir, 'dev.json')


def get_intent(ex):
    return ex['rewritten_intent'] if ex['rewritten_intent'] is not None else ex['intent']


def simple_tokenize(code):
    ignore_types = ['ENCODING', 'NEWLINE', 'ENDMARKER', 'ERRORTOKEN']
    tokens = list(tokenize(io.BytesIO(code.encode('utf-8')).readline))

    return [t.string.strip() for t in tokens if token.tok_name[t.type] not in ignore_types and t.string.strip() != '']


def preprocess(df, verbose=False):
    out_data = []
    skipped = []

    for i, ex in tqdm(df.iterrows(), total=len(df)):
        raw_label = get_intent(ex)
        raw_code = ex['snippet']

        # label = replace_strings(raw_label, fmt='_LIT:%d_')
        label_tokens = split_accessor([clean_text(x, lower=True) for x in raw_label.split()])

        # code = simple_tokenize(replace_strings(raw_code, fmt='"_LIT:%d_"'))
        code = simple_tokenize(raw_code)

        # TODO - hackish
        code_tokens = []
        for tok in code:
            if tok[0] == tok[-1] == '"':
                code_tokens.append('%s %s %s' % (tok[0], tok[1:-1], tok[-1]))
            else:
                code_tokens.append(tok)
        # TODO ---

        try:
            sketch = Sketch(' '.join(code_tokens), verbose).generate()
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

        out_data.append({
            'token': code_tokens,
            'src'  : label_tokens,
            'type' : str(sketch).split(),
        })

    return out_data


if __name__ == '__main__':
    train_data = preprocess(df=pd.read_json(TRAIN_FILE))

    random.shuffle(train_data)
    n = int(args.split * len(train_data))
    dev_data, train_data = train_data[:n], train_data[n:]

    test_data = preprocess(df=pd.read_json(TEST_FILE))

    write_to_file(OUT_TRAIN_FILE, train_data)
    write_to_file(OUT_TEST_FILE, test_data)
    write_to_file(OUT_DEV_FILE, dev_data)
