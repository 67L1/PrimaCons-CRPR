"""Evaluate all four axes for raw JEV and CPRC; requires only NumPy."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(values):
    return float(np.mean(values))


def tv(p, q):
    return float(np.abs(np.asarray(p) - q).sum() / 2)


def normalized(values):
    values = np.asarray(values, dtype=float)
    if values.sum() <= 0:
        raise ValueError('cannot normalize zero Noul mass')
    return values / values.sum()


def axis_metrics(axis, rows, method):
    gaps = defaultdict(list)
    correct = defaultdict(list)
    conflicts = []
    for row in rows:
        v = {key: value[method] for key, value in row['variants'].items()}
        if axis == 'D1':
            p = dict(positive_noul=v['positive_noul']['noul'],
                     negative_noul=1-v['negative_noul']['noul'],
                     choice=v['choice']['probabilities']['hallucinated'])
            for a, b in [('positive_noul', 'negative_noul'), ('positive_noul', 'choice'),
                         ('negative_noul', 'choice')]:
                gaps[f'{a}_vs_{b}'].append(abs(p[a]-p[b]))
            for key, prob in p.items():
                correct[key].append((prob > .5) == bool(row['gold']))
        elif axis == 'D2':
            for level in ('coarse', 'medium', 'fine'):
                cp = v[f'{level}_choice']['probabilities']
                keys = list(cp)
                if level == 'coarse':
                    n = v['coarse_noul']['noul']
                    nq = [n if key == 'unsafe' else 1-n for key in keys]
                    gold = {'unsafe' if row['coarse_gold'] else 'safe'}
                else:
                    nq = normalized([v[f'{level}_noul::{key}']['noul'] for key in keys])
                    gold = {key for key, flag in row[f'{level}_gold'].items() if flag}
                gaps[level].append(tv([cp[key] for key in keys], nq))
                if gold:
                    correct[f'{level}_choice'].append(max(cp, key=cp.get) in gold)
        elif axis == 'D3':
            cp = v['choice']['probabilities']
            abstain = 1-v['noul']['noul']
            selected = max(cp, key=cp.get)
            choice_abstain = selected == 'Unable to answer'
            noul_abstain = abstain > .5
            gaps['abstention_probability_gap'].append(abs(cp['Unable to answer']-abstain))
            conflicts.append(choice_abstain != noul_abstain)
            correct['choice'].append(choice_abstain if row['should_abstain']
                                     else selected == row['correct_choice_key'])
            correct['noul'].append(noul_abstain == bool(row['should_abstain']))
        elif axis == 'D4':
            rp = v['raw']['probabilities']
            dp = v['described']['probabilities']
            keys = list(rp)
            nq = normalized([v[f'noul::{key}']['noul'] for key in keys])
            rv = [rp[key] for key in keys]
            dv = [dp[key] for key in keys]
            gaps['raw_vs_described'].append(tv(rv, dv))
            gaps['raw_vs_noul'].append(tv(rv, nq))
            gaps['described_vs_noul'].append(tv(dv, nq))
            for name, p in [('raw', rp), ('described', dp)]:
                correct[name].append(max(p, key=p.get) == row['gold'])
        else:
            raise ValueError(f'unknown axis: {axis}')
    distance_key = 'tv' if axis in ('D2', 'D4') else 'gaps'
    result = {'n': len(rows), distance_key: {key: mean(value) for key, value in gaps.items()}}
    result['accuracy_pct'] = {key: 100*mean(value) for key, value in correct.items()}
    result['accuracy_n_scored'] = {key: len(value) for key, value in correct.items()}
    if axis == 'D3':
        result['conflict_pct'] = 100*mean(conflicts)
        result['accuracy_gap_pp'] = abs(result['accuracy_pct']['choice']-result['accuracy_pct']['noul'])
    return result


def evaluate(path):
    groups = defaultdict(list)
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            groups[row['axis']].append(row)
    if set(groups) != {'D1', 'D2', 'D3', 'D4'}:
        raise ValueError('expected results for all four axes')
    return {method: {axis: axis_metrics(axis, groups[axis], method)
                     for axis in ('D1', 'D2', 'D3', 'D4')}
            for method in ('raw', 'calibrated')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = evaluate(args.input)
    text = json.dumps(report, indent=2, allow_nan=False)
    Path(args.output).write_text(text + '\n', encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()
