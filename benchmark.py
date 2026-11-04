from definitions import LABELS
EPS=1e-12
import numpy as np
def ece(confidence, correct, bins=15):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        raise ValueError("ECE received no samples")
    value = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (confidence >= lo) & ((confidence < hi) | ((i == bins - 1) & (confidence <= hi)))
        if mask.any():
            value += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return value

def js(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.clip(a, EPS, 1); b = np.clip(b, EPS, 1)
    a /= a.sum(); b /= b.sum()
    m = (a + b) / 2
    return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))

def tv(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).sum() / 2)

def norm(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"invalid probability vector: {values}")
    return values / values.sum()

def probability(response):
    if response.get("type") == "choice":
        p = response["probabilities"]
        if not p:
            raise ValueError("empty Choice probabilities")
        values = np.asarray(list(p.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or abs(values.sum() - 1) > 1e-5:
            raise ValueError("invalid Choice probabilities")
        return p
    p = float(response["noul"])
    if not 0 <= p <= 1:
        raise ValueError("invalid Noul probability")
    return p

def requests_for(row):
    specs = row.get("request_specs")
    if not specs:
        raise ValueError(f"missing request_specs: {row.get('sample_id')}")
    if "coarse_choice" not in specs:
        return specs
    flat = {
        "coarse_choice": specs["coarse_choice"],
        "medium_choice": specs["medium_choice"],
        "fine_choice": specs["fine_choice"],
        "coarse_noul": specs["coarse_noul"],
    }
    flat.update({f"medium_noul::{name}": request for name, request in specs["medium_noul"].items()})
    flat.update({f"fine_noul::{name}": request for name, request in specs["fine_noul"].items()})
    return flat

def binary_metrics(probs, gold):
    probs = np.asarray(probs, dtype=float); gold = np.asarray(gold, dtype=int)
    pred = probs > .5
    confidence = np.maximum(probs, 1 - probs)
    correct = pred == gold
    return {"n": len(gold), "accuracy": float(correct.mean()), "ece": ece(confidence, correct), "brier": float(np.mean((probs - gold) ** 2))}

def d1_metrics(rows):
    plus, minus, choice = [], [], []
    plus_correct, minus_correct, choice_correct = [], [], []
    for row in rows:
        v = row["variants"]
        p = float(v["positive_noul"]["raw_response"]["noul"])
        n = 1 - float(v["negative_noul"]["raw_response"]["noul"])
        c = float(v["choice"]["raw_response"]["probabilities"]["hallucinated"])
        gold = bool(row["gold"])
        plus.append(p); minus.append(n); choice.append(c)
        plus_correct.append((p > .5) == gold); minus_correct.append((n > .5) == gold); choice_correct.append((c > .5) == gold)
    gaps = {"positive_vs_negative": np.abs(np.asarray(plus)-minus), "positive_vs_choice": np.abs(np.asarray(plus)-choice), "negative_vs_choice": np.abs(np.asarray(minus)-choice)}
    return {"n": len(rows), "gaps": {k: {"mean": float(v.mean()), "median": float(np.median(v))} for k,v in gaps.items()}, "accuracy": {"positive_noul": float(np.mean(plus_correct)), "negative_noul": float(np.mean(minus_correct)), "choice": float(np.mean(choice_correct))}, "ece": {"positive_noul": ece(np.maximum(plus,1-np.asarray(plus)),plus_correct), "negative_noul": ece(np.maximum(minus,1-np.asarray(minus)),minus_correct), "choice": ece(np.maximum(choice,1-np.asarray(choice)),choice_correct)}}

def d2_metrics(rows):
    medium = list(rows[0]["variants"]["medium_choice"]["raw_response"]["probabilities"])
    fine = list(rows[0]["variants"]["fine_choice"]["raw_response"]["probabilities"])
    coarse_tv, medium_tv, fine_tv = [], [], []
    coarse_conf, coarse_correct = [], []
    medium_conf, medium_hit, fine_conf, fine_hit = [], [], [], []
    for row in rows:
        v = row["variants"]
        cc = v["coarse_choice"]["raw_response"]["probabilities"]
        cn = float(v["coarse_noul"]["raw_response"]["noul"])
        coarse_tv.append(tv([cc["safe"],cc["unsafe"]],[1-cn,cn]))
        coarse_pred = max(cc, key=cc.get) == "unsafe"
        coarse_gold = bool(row["coarse_gold"])
        coarse_conf.append(max(cc.values())); coarse_correct.append(coarse_pred == coarse_gold)
        mc = v["medium_choice"]["raw_response"]["probabilities"]
        mn = norm([v[f"medium_noul::{name}"]["raw_response"]["noul"] for name in medium])
        medium_tv.append(tv([mc[name] for name in medium], mn))
        positive = {name for name, flag in row["medium_gold"].items() if flag}
        top = max(mc, key=mc.get); medium_conf.append(mc[top]); medium_hit.append(top in positive)
        fc = v["fine_choice"]["raw_response"]["probabilities"]
        fn = norm([v[f"fine_noul::{name}"]["raw_response"]["noul"] for name in fine])
        fine_tv.append(tv([fc[name] for name in fine], fn))
        positive = {name for name, flag in row["fine_gold"].items() if flag}
        top = max(fc, key=fc.get); fine_conf.append(fc[top]); fine_hit.append(top in positive)
    return {"n": len(rows), "tv": {"coarse": {"mean": float(np.mean(coarse_tv)), "median": float(np.median(coarse_tv))}, "medium": {"mean": float(np.mean(medium_tv)), "median": float(np.median(medium_tv))}, "fine": {"mean": float(np.mean(fine_tv)), "median": float(np.median(fine_tv))}}, "choice_accuracy": {"coarse": float(np.mean(coarse_correct)), "medium_hit_at_1": float(np.mean(medium_hit)), "fine_hit_at_1": float(np.mean(fine_hit))}, "choice_ece": {"coarse": ece(coarse_conf,coarse_correct), "medium_hit_at_1": ece(medium_conf,medium_hit), "fine_hit_at_1": ece(fine_conf,fine_hit)}}

def d3_metrics(rows):
    gaps=[]; conflicts=[]; choice_correct=[]; noul_correct=[]; choice_conf=[]; noul_conf=[]
    by_abstention = {True: {"choice_correct": [], "noul_correct": [], "gap": [], "conflict": []},
                     False: {"choice_correct": [], "noul_correct": [], "gap": [], "conflict": []}}
    for row in rows:
        v=row["variants"]; cp=v["choice"]["raw_response"]["probabilities"]; np_=float(v["noul"]["raw_response"]["noul"])
        abstain_prob=float(cp["Unable to answer"]); noul_abstain=1-np_
        gaps.append(abs(abstain_prob-noul_abstain))
        choice_abstain=v["choice"]["raw_response"]["choice"] == "Unable to answer"; noul_abstain_label=noul_abstain>.5
        conflict = choice_abstain != noul_abstain_label
        conflicts.append(conflict)
        if row["should_abstain"]:
            choice_ok=choice_abstain; noul_ok=noul_abstain_label
        else:
            choice_ok=v["choice"]["raw_response"]["choice"] == row["correct_choice_key"]; noul_ok=not noul_abstain_label
        choice_correct.append(choice_ok); noul_correct.append(noul_ok)
        group = by_abstention[bool(row["should_abstain"])]
        group["choice_correct"].append(choice_ok); group["noul_correct"].append(noul_ok)
        group["gap"].append(gaps[-1]); group["conflict"].append(conflict)
        choice_conf.append(max(cp.values())); noul_conf.append(max(np_,1-np_))
    strata = {"should_abstain" if key else "answerable" : {
        "n": len(value["gap"]),
        "abstention_probability_gap_mean": float(np.mean(value["gap"])),
        "choice_noul_conflict_rate": float(np.mean(value["conflict"])),
        "accuracy": {"choice": float(np.mean(value["choice_correct"])), "noul": float(np.mean(value["noul_correct"]))},
    } for key, value in by_abstention.items()}
    return {"n":len(rows),"abstention_probability_gap":{"mean":float(np.mean(gaps)),"median":float(np.median(gaps))},"choice_noul_conflict_rate":float(np.mean(conflicts)),"accuracy":{"choice":float(np.mean(choice_correct)),"noul":float(np.mean(noul_correct))},"accuracy_gap":float(abs(np.mean(choice_correct)-np.mean(noul_correct))),"ece":{"choice":ece(choice_conf,choice_correct),"noul":ece(noul_conf,noul_correct)},"by_should_abstain":strata}

def d4_metrics(rows):
    raw_desc=[]; raw_noul=[]; desc_noul=[]; raw_tv=[]; raw_noul_tv=[]; desc_tv=[]; choice_correct={"raw":[],"described":[]}; choice_conf={"raw":[],"described":[]}; migrations=[]
    for row in rows:
        v=row["variants"]; rp=v["raw"]["raw_response"]["probabilities"]; dp=v["described"]["raw_response"]["probabilities"]
        nq=norm([v[f"noul::{label}"]["raw_response"]["noul"] for label in LABELS])
        rv=np.asarray([rp[x] for x in LABELS]); dv=np.asarray([dp[x] for x in LABELS])
        raw_desc.append(js(rv,dv)); raw_noul.append(js(rv,nq)); desc_noul.append(js(dv,nq)); raw_tv.append(tv(rv,dv)); raw_noul_tv.append(tv(rv,nq)); desc_tv.append(tv(dv,nq))
        for name,p in [("raw",rp),("described",dp)]:
            selected=max(p,key=p.get); choice_correct[name].append(selected==row["gold"]); choice_conf[name].append(max(p.values()))
        migrations.append(max(rp,key=rp.get)!=max(dp,key=dp.get))
    return {"n":len(rows),"js":{"raw_vs_described":float(np.mean(raw_desc)),"raw_vs_noul":float(np.mean(raw_noul)),"described_vs_noul":float(np.mean(desc_noul))},"tv":{"raw_vs_described":float(np.mean(raw_tv)),"raw_vs_noul":float(np.mean(raw_noul_tv)),"described_vs_noul":float(np.mean(desc_tv))},"choice_accuracy":{k:float(np.mean(v)) for k,v in choice_correct.items()},"choice_ece":{k:ece(choice_conf[k],choice_correct[k]) for k in choice_correct},"label_migration_rate":float(np.mean(migrations))}