"""Run every record and request in a benchmark split, in file order."""
import argparse
import json
from pathlib import Path

import calibrator as cal
import benchmark as bench
from online import CalibratedJev
from primaconsis import config, features, project_probability
from runtime import forward_with_hidden


def run(cfg, split, output):
    model = CalibratedJev(cfg)
    with Path(output).open('x', encoding='utf8') as dest:
        for axis in ('D1', 'D2', 'D3', 'D4'):
            source = cfg['paths']['dataset'] / f'axis{axis[1:]}' / f'{split}.jsonl'
            with source.open(encoding='utf8') as stream:
                for index, line in enumerate(stream, 1):
                    row = json.loads(line)
                    variants = {}
                    for view, request in bench.requests_for(row).items():
                        _, hidden, probs, options = forward_with_hidden(model.backbone, row['state'], request)
                        keys = [key for key, _ in options]
                        if request['type'] == 'choice':
                            raw = dict(type='choice', probabilities=dict(zip(keys, probs.tolist())),
                                       choice=keys[int(probs.argmax())])
                        else:
                            raw = dict(type='noul', noul=float(probs[keys.index('true')]))
                        # Historical checkpoints were fitted in benchmark semantic space.
                        vector, _, _, _ = project_probability(axis, view, request, raw, row)
                        h = model.pca.transform(hidden['penultimate'][None])[0]
                        x, logits = features(h, vector, request['type'], view, cfg['feature_recipe'])
                        if len(x) != model.spec['input_dim']:
                            raise ValueError('feature recipe does not match checkpoint')
                        item = dict(features=x, raw_logits=logits, raw_vector=vector)
                        p = cal.predict([item], model.calibrator)[0]['calibrated_probability']
                        if request['type'] == 'noul':
                            reverse = (axis == 'D1' and view == 'negative_noul') or axis == 'D3'
                            calibrated = dict(type='noul', noul=float(1-p[0] if reverse else p[0]))
                        else:
                            if axis == 'D1':
                                p = [p[0] if key == 'hallucinated' else 1-p[0] for key in keys]
                            probabilities = dict(zip(keys, map(float, p)))
                            calibrated = dict(type='choice', probabilities=probabilities,
                                              choice=max(probabilities, key=probabilities.get))
                        variants[view] = dict(raw=raw, calibrated=calibrated)
                    record = dict(row, axis=axis, variants=variants)
                    dest.write(json.dumps(record, allow_nan=False) + '\n')
                    dest.flush()
                    if index % 100 == 0:
                        print(f'{axis}: {index} records', flush=True)
            print(f'{axis}: complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['calibration', 'test'], default='test')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(config(args.config), args.split, args.output)


if __name__ == '__main__':
    main()
