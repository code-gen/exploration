import json
import re
from typing import Any, List
import itertools

PUNCTUATION = {
    'keep'  : "&,.",
    'remove': u'\u200b' + '?!`™•°'
}


def clean_text(x, lower=False):
    if lower:
        x = x.lower()

    for p in PUNCTUATION['keep']:
        x = x.replace(p, "%s" % p)
    for p in PUNCTUATION['remove']:
        x = x.replace(p, '')

    if x != '' and x[-1] in ['.', ',']:
        x = x[:-1]

    return x


def unquote(s: str, once=False) -> str:
    qs = ['`', '\'', '"']

    if all([s[0] == s[-1], s[0] in qs, s[-1] in qs]):
        return s[1:-1] if once else unquote(s[1:-1])
    else:
        return s


def replace_strings(data, fmt, return_replaced_dict=False):
    regex = re.compile(
        r'(\"{3}(?:[^\"\\]|\\.)*\"{3})|'
        r'(\'{3}(?:[^\'\\]|\\.)*\'{3})|'
        r'(\`{3}(?:[^\`\\]|\\.)*\`{3})|'
        r'(\"(?:[^\"\\]|\\.)*\")|'
        r'(\'(?:[^\'\\]|\\.)*\')|'
        r'(\`(?:[^\`\\]|\\.)*\`)'
    )

    rpl = {}

    for i, x in enumerate(regex.findall(data)):
        m = [a for a in x if len(a)]
        assert len(m) == 1
        r = m[0]
        data = data.replace(r, fmt % i)
        rpl[fmt % i] = r

    if return_replaced_dict:
        return data, rpl
    else:
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
