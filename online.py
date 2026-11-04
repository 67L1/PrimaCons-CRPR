"""Resident CPRC sidecar: one backbone forward per decision, no feature files."""
import pickle
import calibrator as cal
from runtime import forward_with_hidden


class CalibratedJev:
    def __init__(self, cfg):
        from jevk5 import JevK5
        self.cfg = cfg
        self.backbone = JevK5(str(cfg['paths']['backbone']), graphs=False)
        bundle = cfg['paths']['calibrator']
        with (bundle / 'pca.pkl').open('rb') as f:
            self.pca = pickle.load(f)
        if isinstance(self.pca, dict):
            self.pca = self.pca['penultimate_64']
        self.calibrator, self.spec = cal.load_model(bundle / 'calibrator.pt')
        self.calibrator.to(self.backbone.device).eval()

    def decide(self, state, request, *, complement=False,
               binary_choice_key=None):
        """Return native-space calibrated probabilities for this request only.

        complement selects a binary Noul's trained semantic orientation.
        binary_choice_key uses the binary head for a two-option Choice.
        Neither argument supplies a peer prediction or a Gold label.
        """
        from primaconsis import features
        if request['type'] not in ('choice', 'noul'):
            raise ValueError('CPRC supports Choice and Noul')
        if complement and request['type'] != 'noul':
            raise ValueError('complement is only valid for Noul')
        if request['type'] == 'choice' and not 2 <= len(request['criteria']) <= 12:
            raise ValueError('released calibrators support 2 to 12 options')
        _, hidden, probs, options = forward_with_hidden(self.backbone, state, request)
        keys = [key for key, _ in options]
        if request['type'] == 'noul':
            raw_native = float(probs[keys.index('true')])
            raw = [1 - raw_native if complement else raw_native]
        elif binary_choice_key is not None:
            if len(keys) != 2 or binary_choice_key not in keys:
                raise ValueError('binary_choice_key requires a matching two-option Choice')
            raw = [float(probs[keys.index(binary_choice_key)])]
        else:
            raw = probs.tolist()
        h = self.pca.transform(hidden['penultimate'][None])[0]
        descriptions = list(request.get('criteria', {}).values())
        description_length = sum(map(len, descriptions)) / len(descriptions) if descriptions else 0
        x, logits = features(h, raw, request['type'], '', self.cfg['feature_recipe'],
                             description_length=description_length)
        if len(x) != self.spec['input_dim']:
            raise ValueError('PCA/feature recipe does not match checkpoint')
        own = dict(features=x, raw_logits=logits, raw_vector=raw)
        calibrated = cal.predict([own], self.calibrator)[0]['calibrated_probability']
        if request['type'] == 'noul':
            p = 1 - calibrated[0] if complement else calibrated[0]
            return dict(type='noul', noul=float(p), selected=bool(p > .5),
                        confidence=float(max(p, 1-p)), raw_noul=raw_native)
        if binary_choice_key is not None:
            calibrated = [calibrated[0] if k == binary_choice_key else 1-calibrated[0]
                          for k in keys]
        result = {k: float(v) for k, v in zip(keys, calibrated)}
        selected = max(result, key=result.get)
        return dict(type='choice', choice=selected, probabilities=result,
                    confidence=result[selected], raw_probabilities=dict(zip(keys, probs.tolist())))
