import json
import re
from typing import Any, List
import itertools

PUNCTUATION = {
    'keep'  : "&,.",
    'remove': u'\u200b' + '?!`™•°'
}


def clean_text(x):
    # x = x.lower()

    for p in PUNCTUATION['keep']:
        x = x.replace(p, "%s" % p)
    for p in PUNCTUATION['remove']:
        x = x.replace(p, '')

    if x[-1] in ['.', ',']:
        x = x[:-1]

    return x


def replace_strings(data):
    regex = re.compile(r'(\"{3}(?:[^\"\\]|\\.)*\"{3})|(\'{3}(?:[^\'\\]|\\.)*\'{3})|(\"(?:[^\"\\]|\\.)*\")|(\'(?:[^\'\\]|\\.)*\')')

    for i, x in enumerate(regex.findall(data)):
        m = [a for a in x if len(a)]
        assert len(m) == 1
        r = m[0]
        data = data.replace(r, '"_STR:%d_"' % i)

    return data


def intersperse(x, xs):
    return list(itertools.chain.from_iterable(zip(xs, [x] * len(xs))))[:-1]


def split_accessor(data):
    if not isinstance(data, list):
        data = data.split()

    out = []
    regex = re.compile(r'([^\s\.]+)(\.[^\s\.]+)+')

    for x in data:
        out += [x]
        if regex.match(x):
            out += ['['] + intersperse(".", x.split(".")) + [']']

    return out


def write_to_file(filename: str, data: List[dict]) -> None:
    with open(filename, 'wt') as fp:
        for d in data:
            fp.write(json.dumps(d) + '\n')


def print_skipped(name: str, skipped: List[Any]) -> None:
    if not skipped:
        print(" * no skipped examples for %s" % name)
        return

    print(" * skipped examples for %s" % name)

    if isinstance(skipped[0], tuple):
        for i, (code, anno) in enumerate(skipped, start=1):
            print('%d>\ncode: %s\nanno: %s\n-----' % (i, code, anno))

    if isinstance(skipped[0], dict):
        for i, ex in enumerate(skipped, start=1):
            print('%d>\nintent: %s\nrw-intent: %s\ncode: %s\n-----' % (i, ex['intent'], ex['rewritten_intent'], ex['snippet']))
