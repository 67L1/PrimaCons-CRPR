import os,json,itertools
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from definitions import *
class ResidualProbe(nn.Module):
    def __init__(self, delta_max, independent=False, input_dim=77, option_count=4):
        super().__init__()
        self.input_dim = input_dim
        self.option_count = option_count
        self.binary_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh())
        self.multi_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh())
        self.binary_head = nn.Linear(64, 1)
        self.multi_head = nn.Linear(64, option_count)
        self.binary_projection = nn.Linear(64, 32, bias=False)
        self.multi_projection = nn.Linear(64, 32, bias=False)
        self.binary_gold = nn.Embedding(2, 32)
        self.multi_gold = nn.Embedding(4, 32)
        self.delta_max = delta_max
        self.independent = independent
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        for head in (self.binary_head, self.multi_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, x, raw, sizes):
        x = (x-self.feature_mean)/self.feature_scale
        binary = sizes == 1
        logits = torch.zeros_like(raw)
        z = x.new_zeros((len(x), 32))
        for mask, encoder, head, projection in [(binary, self.binary_encoder, self.binary_head, self.binary_projection), (~binary, self.multi_encoder, self.multi_head, self.multi_projection)]:
            if not mask.any():
                continue
            h = encoder(x[mask]); residual = head(h)
            if self.independent:
                output = residual
            else:
                delta = self.delta_max * torch.tanh(residual)
                if not bool(binary[mask][0]):
                    valid = torch.arange(self.option_count, device=x.device)[None] < sizes[mask, None]
                    delta = delta-(delta*valid).sum(1, keepdim=True)/sizes[mask, None]
                output = raw[mask, :residual.shape[1]] + delta
            logits[mask, :output.shape[1]] = output
            z[mask] = F.normalize(projection(h), dim=1)
        valid = torch.arange(self.option_count, device=x.device)[None] < sizes[:, None]
        multi_logits = logits.masked_fill(~valid, -1e9)
        binary_p = torch.cat([torch.sigmoid(logits[:, :1]), torch.zeros_like(logits[:, 1:])], 1)
        p = torch.where(binary[:, None], binary_p, torch.softmax(multi_logits, 1))
        return logits, p, z

def tensors(rows, include_target=True):
    x = torch.tensor(np.asarray([r['features'] for r in rows]), dtype=torch.float32)
    raw = torch.zeros(len(rows), 12)
    sizes = torch.tensor([len(r['raw_vector']) for r in rows])
    for i, r in enumerate(rows):
        raw[i, :sizes[i]] = torch.tensor(r['raw_logits'])
    target = torch.tensor([r['target'] for r in rows]) if include_target else None
    return x, raw, sizes, target

def group_maps(rows):
    by = defaultdict(dict)
    for i, r in enumerate(rows): by[r['group_id']][r['view_id']] = i
    d1, d2, d3, d4 = [], [], [], []
    for views in by.values():
        exp = rows[next(iter(views.values()))]['experiment']
        if exp == 'D1': d1.append([views[v] for v in ['positive_noul', 'negative_noul', 'choice']])
        if exp == 'D3': d3.append([views[v] for v in ['choice', 'noul']])
        if exp == 'D4': d4.append([views[v] for v in ['raw', 'described'] + ['noul::'+c for c in LABELS]])
        if exp == 'D2' and {'coarse_choice', 'medium_choice', 'fine_choice'} <= views.keys() and ('coarse_noul' in views or 'coarse' in views):
            d2.append({
                'coarse_choice': views['coarse_choice'], 'coarse_noul': views.get('coarse_noul', views.get('coarse')),
                'medium_choice': views['medium_choice'], 'fine_choice': views['fine_choice'],
                'medium_noul': [views.get('medium_noul::'+m, views.get('medium::'+m)) for m, _, _ in MEDIUM],
                'fine_noul': [views.get('fine_noul::'+c, views.get('fine::'+c)) for c in FINE],
            })
        elif exp == 'D2':
            d2.append([views['coarse']] + [views['medium::'+m] for m, a, b in MEDIUM] + [views['fine::'+c] for c in FINE])
    return {'D1':torch.tensor(d1,dtype=torch.long).reshape(-1,3), 'D2':d2,
            'D3':torch.tensor(d3,dtype=torch.long).reshape(-1,2), 'D4':torch.tensor(d4,dtype=torch.long).reshape(-1,6)}

def js(p, q):
    m = (p+q)*.5
    return .5*(p*(p.clamp_min(EPS).log()-m.clamp_min(EPS).log())).sum(-1)+.5*(q*(q.clamp_min(EPS).log()-m.clamp_min(EPS).log())).sum(-1)

def stopped_js(p, q):
    return .5*js(p.detach(), q).mean()+.5*js(q.detach(), p).mean()

def bernoulli(p):
    return torch.stack([1-p,p], -1)

def smooth_max(values, temperature=.05):
    """A bounded smooth maximum: it never exceeds the largest input."""
    x=torch.stack(values)
    m=x.max()
    return m + temperature*torch.log(torch.exp((x-m)/temperature).mean())

def consistency(p, maps, targets=None, aligned=False):
    terms = {}
    d = maps['D1']
    if len(d):
        # All three cache views are already in hallucination probability space.
        d1=torch.stack([p[d[:,0],0],p[d[:,1],0],p[d[:,2],0]],dim=1)
        pair=[]
        for a,b in itertools.combinations(range(3),2):
            pair.append((d1[:,a]-d1[:,b]).abs().mean() if aligned else stopped_js(bernoulli(d1[:,a]),bernoulli(d1[:,b])))
        terms['D1']=pair
    d = maps['D3']
    if len(d):
        # Current D3 cache stores the abstention probability directly.
        abstain_c=p[d[:,0],2]; abstain_n=p[d[:,1],0]
        if aligned:
            gap=(abstain_c-abstain_n).abs().mean()
            conflict=(abstain_c*(1-abstain_n)+(1-abstain_c)*abstain_n).mean()
            if targets is not None:
                gold_n=targets[d[:,1]].float()
                choice_gold=targets[d[:,0]]
                choice_acc=p[d[:,0]].gather(1,choice_gold[:,None]).mean()
                abstain_prob=p[d[:,1],0]
                noul_acc=(gold_n*abstain_prob+(1-gold_n)*(1-abstain_prob)).mean()
                terms['D3']=[gap,conflict,(choice_acc-noul_acc).abs()]
            else: terms['D3']=[gap,conflict]
        else: terms['D3']=[stopped_js(bernoulli(abstain_c),bernoulli(abstain_n))]
    d = maps['D4']
    if len(d):
        n = p[d[:,2:],0]; n = n/n.sum(1,keepdim=True)
        pair=[]
        for j in (0,1):
            c=p[d[:,j], :4]
            pair.append(.5*(c-n).abs().sum(1).mean() if aligned else stopped_js(c,n))
        if aligned: pair.append(.5*(p[d[:,0],:4]-p[d[:,1],:4]).abs().sum(1).mean())
        terms['D4']=pair
    d = maps['D2']
    if len(d) and isinstance(d[0], dict):
        coarse_c = torch.stack([p[m['coarse_choice'], :2] for m in d])
        coarse_n = torch.stack([p[m['coarse_noul'], 0] for m in d])
        medium_c = torch.stack([p[m['medium_choice'], :6] for m in d])
        medium_n = torch.stack([p[m['medium_noul'], 0] for m in d])
        fine_c = torch.stack([p[m['fine_choice'], :12] for m in d])
        fine_n = torch.stack([p[m['fine_noul'], 0] for m in d])
        medium_n = medium_n / medium_n.sum(1, keepdim=True).clamp_min(EPS)
        fine_n = fine_n / fine_n.sum(1, keepdim=True).clamp_min(EPS)
        terms['D2']=[
            .5 * torch.abs(coarse_c - torch.stack([1-coarse_n, coarse_n], 1)).sum(1).mean(),
            .5 * torch.abs(medium_c - medium_n).sum(1).mean(),
            .5 * torch.abs(fine_c - fine_n).sum(1).mean(),
        ]
    elif len(d):
        f = p[d[:,7:],0]; m = p[d[:,1:7],0]; c = p[d[:,0],0]
        fine_parent = 1 - torch.prod(1 - f.reshape(-1,6,2), dim=-1)
        fine_root = 1 - torch.prod(1 - f, dim=-1)
        medium_root = 1 - torch.prod(1 - m, dim=-1)
        eps = 1e-6
        def soft_set_f1_disagreement(left, right):
            overlap = (left * right).sum(-1)
            return (1 - (2 * overlap + eps) / (left.sum(-1) + right.sum(-1) + eps)).mean()
        terms['D2']=[
            soft_set_f1_disagreement(m, fine_parent),
            soft_set_f1_disagreement(c[:,None], fine_root[:,None]),
            soft_set_f1_disagreement(c[:,None], medium_root[:,None]),
        ]
    if not terms: return p.sum()*0
    per_domain=[smooth_max(v) for v in terms.values()]
    return torch.stack(per_domain).mean()

def triplet_loss(z, maps, target, model):
    terms = []; diagnostics = []
    def triple(a,b,y,multi=False):
        if len(a)==0: return
        anchors = model.multi_gold if multi else model.binary_gold
        k = 4 if multi else 2
        gold = F.normalize(anchors(y),dim=-1)
        negative = F.normalize(anchors((y+1)%k),dim=-1)
        losses = [F.triplet_margin_loss(a,b,negative,margin=.2), F.triplet_margin_loss(a,gold,negative,margin=.2), F.triplet_margin_loss(b,gold,negative,margin=.2)]
        terms.append(torch.stack(losses).mean())
        pos=(a*b).sum(-1); neg=(a*negative).sum(-1)
        diagnostics.append(torch.stack([pos.mean(),neg.mean(),((pos-neg)>.2).float().mean()]))
    d = maps['D1']
    if len(d):
        for j in (0,1): triple(z[d[:,2]],z[d[:,j]],target[d[:,2]])
    d = maps['D3']
    if len(d): triple(z[d[:,0]],z[d[:,1]],target[d[:,1]])
    d = maps['D4']
    if len(d):
        canonical=F.normalize(z[d[:,2:]].mean(1),dim=-1)
        for j in (0,1): triple(z[d[:,j]],canonical,target[d[:,j]],True)
    d = maps['D2']
    if len(d):
        fine=F.normalize(z[d[:,7:]].mean(1),dim=-1)
        triple(z[d[:,0]],fine,target[d[:,0]])
        for j in range(6):
            child=F.normalize(z[d[:,7+2*j:9+2*j]].mean(1),dim=-1)
            triple(z[d[:,1+j]],child,target[d[:,1+j]])
    loss=torch.stack(terms).mean() if terms else z.sum()*0
    return loss, torch.stack(diagnostics).mean(0) if diagnostics else z.new_zeros(3)

def dynamic_weights(rows,gamma):
    scored=np.asarray([not (len(r['raw_vector'])>1 and 'gold_set' in r and not r['gold_set']) for r in rows])
    raw_gold=np.asarray([
        sum(r['raw_vector'][i] for i in r.get('gold_set', [r['target']])) if len(r['raw_vector']) > 1
        else r['raw_vector'][0] if r['target'] else 1-r['raw_vector'][0]
        for r in rows
    ])
    weights=np.clip((1-raw_gold)**gamma,.5,2)
    weights[~scored]=0
    by=defaultdict(list)
    for i,r in enumerate(rows): by[r['group_id']].append(i)
    balance=np.zeros(len(rows))
    group_counts=Counter(rows[idx[0]]['experiment'] for idx in by.values())
    for idx in by.values():
        active=np.asarray(idx,dtype=int)[scored[idx]]
        if len(active):
            weights[active]/=weights[active].mean()
            balance[active]=1/(len(active)*group_counts[rows[idx[0]]['experiment']])
    balance/=balance.sum()
    return torch.tensor(weights,dtype=torch.float32),torch.tensor(balance,dtype=torch.float32),raw_gold

def prepare_objective(rows,spec,device):
    x,raw,sizes,y=tensors(rows)
    x,raw,sizes,y=(v.to(device) if v is not None else None for v in (x,raw,sizes,y))
    maps=group_maps(rows)
    maps={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in maps.items()}
    weights,balance,_=dynamic_weights(rows,spec['gamma'])
    return x,raw,sizes,y,maps,weights.to(device),balance.to(device)

def multiclass_gold_loss(logits, rows, option_count):
    valid = torch.arange(option_count, device=logits.device)[None] < torch.tensor(
        [len(row['raw_vector']) for row in rows], device=logits.device)[:, None]
    logp = F.log_softmax(logits.masked_fill(~valid, -1e9), dim=1)
    if any('gold_set' in row for row in rows):
        gold = torch.zeros_like(logp, dtype=torch.bool)
        has_gold = torch.zeros(len(rows), dtype=torch.bool, device=logits.device)
        for i, row in enumerate(rows):
            indices = row.get('gold_set', [int(row['target'])])
            if indices:
                gold[i, indices] = True
                has_gold[i] = True
        losses = -torch.logsumexp(logp.masked_fill(~gold, -1e9), dim=1)
        return torch.where(has_gold, losses, torch.zeros_like(losses))
    targets = torch.tensor([int(row['target']) for row in rows], device=logits.device)
    return F.nll_loss(logp, targets, reduction='none')

def objective_value(model, rows, spec, device, prepared=None):
    x,raw,sizes,y,maps,weights,balance=prepared or prepare_objective(rows,spec,device)
    logits,p,z=model(x,raw,sizes)
    binary=sizes==1
    per=torch.zeros(len(rows),device=device)
    per[binary]=F.binary_cross_entropy_with_logits(logits[binary,0],y[binary].float(),reduction='none')
    if (~binary).any():
        multi_rows=[r for r in rows if len(r['raw_vector'])>1]
        per[~binary]=multiclass_gold_loss(logits[~binary], multi_rows, model.option_count)
    d3=maps['D3']
    if len(d3):
        choice,noul=d3[:,0],d3[:,1]
        abstain_logit=logits[choice,2]-torch.logsumexp(logits[choice,:2],dim=1)
        per[choice]+=F.binary_cross_entropy_with_logits(abstain_logit,y[noul].float(),reduction='none')
    acc=(per*weights*balance).sum()
    cons=consistency(p,maps,y,aligned=spec.get('consistency_loss')=='aligned') if spec['alpha'] else p.sum()*0
    trip,_=triplet_loss(z,maps,y,model) if spec['beta'] else (p.sum()*0,p.new_zeros(3))
    return (spec['alpha']*cons if spec.get('no_gold',False) else acc+spec['alpha']*cons+spec['beta']*trip).detach()

def train(rows,spec,epochs=120,eval_rows=None,patience=5,progress=None,initial_model=None,start_epoch=0):
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get('TRC_THREADS','4')))
    device=torch.device('cuda' if os.environ.get('TRC_TRAIN_DEVICE','cpu')=='cuda' and torch.cuda.is_available() else 'cpu')
    x,raw,sizes,y=tensors(rows)
    x,raw,sizes,y=(v.to(device) if v is not None else None for v in (x,raw,sizes,y))
    option_count=int(spec.get('option_count', max(4,int(sizes.max()))))
    if initial_model is None:
        model=ResidualProbe(spec['delta_max'],spec['method']=='Independent',x.shape[1],option_count)
        model=model.to(device)
        model.feature_mean.copy_(x.mean(0)); model.feature_scale.copy_(x.std(0).clamp_min(.01))
    else:
        model=initial_model.to(device)
        if model.input_dim != x.shape[1]: raise ValueError(f'checkpoint input_dim={model.input_dim} != data={x.shape[1]}')
    maps=group_maps(rows)
    maps={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in maps.items()}
    weights,balance,raw_gold=dynamic_weights(rows,spec['gamma'])
    weights=weights.to(device); balance=balance.to(device)
    optim=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
    eval_data=prepare_objective(eval_rows,spec,device) if eval_rows is not None else None
    best_state=None; best_eval=float('inf'); stale=0; epochs_ran=start_epoch
    if eval_rows is not None:
        model.eval()
        with torch.no_grad(): best_eval=float(objective_value(model,eval_rows,spec,device,eval_data).cpu())
        model.train()
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for epoch in range(start_epoch,epochs):
        epochs_ran=epoch+1
        logits,p,z=model(x,raw,sizes)
        binary=sizes==1
        per=torch.zeros(len(rows),device=device)
        per[binary]=F.binary_cross_entropy_with_logits(logits[binary,0],y[binary].float(),reduction='none')
        if (~binary).any():
            multi_rows=[r for r in rows if len(r['raw_vector'])>1]
            per[~binary]=multiclass_gold_loss(logits[~binary], multi_rows, model.option_count)
        # D3 Choice and Noul have different native labels, but share the
        # should-abstain gold. Add that binary supervision to the Choice head.
        d3=maps['D3']
        if len(d3):
            choice, noul = d3[:,0], d3[:,1]
            abstain_logit=logits[choice,2]-torch.logsumexp(logits[choice,:2],dim=1)
            per[choice] += F.binary_cross_entropy_with_logits(
                abstain_logit, y[noul].float(), reduction='none')
        acc=(per*weights*balance).sum()
        cons=consistency(p,maps,y,aligned=spec.get('consistency_loss')=='aligned') if spec['alpha'] else p.sum()*0
        trip,diag=triplet_loss(z,maps,y,model) if spec['beta'] else (p.sum()*0,p.new_zeros(3))
        if spec.get('no_gold', False):
            loss=spec['alpha']*cons
        else:
            loss=acc+spec['alpha']*cons+spec['beta']*trip
        assert torch.isfinite(loss), 'nonfinite loss'
        optim.zero_grad(); loss.backward(); optim.step()
        if eval_rows is not None:
            model.eval()
            with torch.no_grad(): eval_loss=float(objective_value(model,eval_rows,spec,device,eval_data).cpu())
            model.train()
            if eval_loss < best_eval-1e-8:
                best_eval=eval_loss; stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale+=1
                if stale>=patience: break
            if progress and (epochs_ran%10==0 or stale>=patience):
                progress({'epoch':epochs_ran,'train_loss':float(loss.detach().cpu()),'audit_loss':eval_loss,'best_audit_loss':best_eval,'stale_epochs':stale})
    if best_state is not None: model.load_state_dict(best_state)
    weights_cpu=weights.detach().cpu().numpy()
    corr=None if np.std(weights_cpu)<1e-12 or np.std(1-raw_gold)<1e-12 else float(np.corrcoef(weights_cpu,1-raw_gold)[0,1])
    diag=diag.detach()
    summary={'accuracy_loss':float(acc.detach()),'consistency_loss':float(cons.detach()),'triplet_loss':float(trip.detach()),'mean_w_choice':float(weights[[r['primitive_type']=='choice' for r in rows]].mean().detach().cpu()),'mean_w_noul':float(weights[[r['primitive_type']=='noul' for r in rows]].mean().detach().cpu()),'weight_correlation_with_raw_error':corr,'positive_cosine_similarity':float(diag[0].cpu()),'negative_cosine_similarity':float(diag[1].cpu()),'triplet_margin_accuracy':float(diag[2].cpu()),'epochs_requested':epochs,'epochs_ran':epochs_ran,'early_stopped':bool(best_state is not None and epochs_ran<epochs),'best_eval_loss':None if best_state is None else best_eval,'early_stopping_patience':patience}
    return model,summary

def predict(rows,model):
    model.eval(); output=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(rows),8192):
            batch=rows[start:start+8192]; x,raw,sizes,_=tensors(batch,include_target=False)
            x,raw,sizes=x.to(device),raw.to(device),sizes.to(device)
            logits,p,z=model(x,raw,sizes)
            for i,r in enumerate(batch):
                n=int(sizes[i]); d=(logits[i,:n]-raw[i,:n]).detach().cpu().numpy()
                output.append({**r,'calibrated_probability':p[i,:n].detach().cpu().tolist(),'residual':d.tolist(),'representation':z[i].detach().cpu().tolist()})
    return output

def raw_predictions(rows):
    return [{**r,'calibrated_probability':r['raw_vector'],'residual':[0.]*len(r['raw_vector'])} for r in rows]

def metrics(rows):
    p=[np.asarray(r['calibrated_probability']) for r in rows]
    y=np.asarray([r['target'] for r in rows])
    pred=np.asarray([int(a.argmax()) if len(a)>1 else int(a[0]>.5) for a in p])
    conf=np.asarray([a.max() if len(a)>1 else max(a[0],1-a[0]) for a in p])
    scored=[]; correctness=[]; brier=[]; nll=[]; true_sets=[]; pred_sets=[]
    for i,(r,a) in enumerate(zip(rows,p)):
        if len(a)==1:
            target=int(r['target'])
            scored.append(i); correctness.append(int(pred[i]==target))
            q=float(a[0] if target else 1-a[0])
            nll.append(-np.log(max(q,EPS)))
            brier.append((a[0]-target)**2)
        else:
            gold=list(r.get('gold_set',[int(r['target'])]))
            if 'gold_set' in r and not gold:
                continue
            scored.append(i); correctness.append(int(pred[i] in gold))
            truth=np.zeros(len(a),dtype=float)
            truth[gold]=1.0/len(gold)
            q=float(a[gold].sum())
            nll.append(-np.log(max(q,EPS)))
            brier.append(float(np.square(a-truth).sum()))
            true_sets.append(np.isin(np.arange(len(a)),gold).astype(int))
            pred_sets.append(np.eye(len(a),dtype=int)[pred[i]])
    if not scored:
        raise ValueError('no gold-scored rows')
    scored=np.asarray(scored); correct=np.asarray(correctness,dtype=bool); conf=conf[scored]
    bins=np.minimum((conf*15).astype(int),14)
    ece=sum(float((bins==i).mean()*abs(correct[bins==i].mean()-conf[bins==i].mean())) for i in range(15) if (bins==i).any())
    by_size=defaultdict(list)
    for i,r in enumerate(rows):
        by_size[len(r['raw_vector'])].append(i)
    macro_f1_by_size={}
    for size,indices in by_size.items():
        if size == 1:
            yy=np.asarray([rows[i]['target'] for i in indices])
            ppred=pred[indices]
            macro_f1_by_size[str(size)]=float(f1_score(yy,ppred,average='binary',zero_division=0))
        else:
            pairs=[]
            for i in indices:
                r=rows[i]; gold=list(r.get('gold_set',[int(r['target'])]))
                if not gold: continue
                pairs.append((np.isin(np.arange(size),gold).astype(int),np.eye(size,dtype=int)[pred[i]]))
            if pairs:
                yy=np.asarray([pair[0] for pair in pairs]); ppred=np.asarray([pair[1] for pair in pairs])
                macro_f1_by_size[str(size)]=float(f1_score(yy,ppred,average='macro',zero_division=0))
    macro_f1=next(iter(macro_f1_by_size.values())) if len(macro_f1_by_size)==1 else None
    residual=np.concatenate([np.abs(r['residual']) for r in rows])
    result={'n':len(rows),'n_scored':len(scored),'accuracy':float(correct.mean()),'macro_f1':macro_f1,'macro_f1_by_option_count':macro_f1_by_size,'ece':ece,'brier':float(np.mean(brier)),'nll':float(np.mean(nll)),'mean_absolute_logit_residual':float(residual.mean()),'p95_absolute_logit_residual':float(np.percentile(residual,95))}
    movement=np.concatenate([np.abs(np.asarray(r['calibrated_probability'])-np.asarray(r['raw_vector'])) for r in rows])
    result['mean_absolute_probability_movement']=float(movement.mean())
    result['p95_absolute_probability_movement']=float(np.percentile(movement,95))
    if all(r['experiment']=='D3' for r in rows):
        sem=[(a[2]>.5 if len(a)>1 else a[0]>.5)==bool(r['semantic_gold']) for r,a in zip(rows,p)]
        result['abstention_probability_binary_accuracy']=float(np.mean(sem))
    return result

def distances(rows):
    maps=group_maps(rows); sizes=torch.tensor([len(r['raw_vector']) for r in rows])
    p=torch.zeros(len(rows),12)
    for i,r in enumerate(rows): p[i,:sizes[i]]=torch.tensor(r['calibrated_probability'])
    targets=torch.tensor([r['target'] for r in rows],dtype=torch.long)
    result={}
    def stats(v):
        v=v.numpy(); return {'n':len(v),'mean':float(v.mean()),'median':float(np.median(v))}
    d=maps['D1']
    if len(d):
        result['D1']={f'{a}_vs_{b}':stats((p[d[:,a],0]-p[d[:,b],0]).abs()) for a,b in itertools.combinations(range(3),2)}
    d=maps['D3']
    if len(d):
        choice_abstain=p[d[:,0]].argmax(1)==2
        noul_abstain=p[d[:,1],0]>.5
        gold_abstain=targets[d[:,1]].bool()
        choice_correct=p[d[:,0]].argmax(1)==targets[d[:,0]]
        noul_correct=noul_abstain==gold_abstain
        result['D3']={'probability_gap':stats((p[d[:,0],2]-p[d[:,1],0]).abs()),
                      'choice_noul_label_conflict_rate':float((choice_abstain!=noul_abstain).float().mean()),
                      'accuracy_gap':float((choice_correct.float().mean()-noul_correct.float().mean()).abs())}
    d=maps['D4']
    if len(d):
        n=p[d[:,2:],0]; n=n/n.sum(1,keepdim=True)
        c0=p[d[:,0],:4]; c1=p[d[:,1],:4]
        result['D4']={'raw_vs_described_tv':stats((c0-c1).abs().sum(1)/2),
                      'raw_vs_noul_tv':stats((c0-n).abs().sum(1)/2),
                      'described_vs_noul_tv':stats((c1-n).abs().sum(1)/2)}
    d=maps['D2']
    if len(d) and isinstance(d[0], dict):
        coarse_c=torch.stack([p[m['coarse_choice'],:2] for m in d])
        coarse_n=torch.stack([p[m['coarse_noul'],0] for m in d])
        medium_c=torch.stack([p[m['medium_choice'],:6] for m in d])
        medium_n=torch.stack([p[m['medium_noul'],0] for m in d])
        fine_c=torch.stack([p[m['fine_choice'],:12] for m in d])
        fine_n=torch.stack([p[m['fine_noul'],0] for m in d])
        medium_n=medium_n/medium_n.sum(1,keepdim=True).clamp_min(EPS)
        fine_n=fine_n/fine_n.sum(1,keepdim=True).clamp_min(EPS)
        result['D2']={
            'coarse_tv':stats(.5*(coarse_c-torch.stack([1-coarse_n,coarse_n],1)).abs().sum(1)),
            'medium_tv':stats(.5*(medium_c-medium_n).abs().sum(1)),
            'fine_tv':stats(.5*(fine_c-fine_n).abs().sum(1)),
        }
    elif len(d):
        fine=p[d[:,7:],0]>.5; medium=p[d[:,1:7],0]>.5; coarse=p[d[:,0],0]>.5
        fine_parent=fine.reshape(-1,6,2).any(-1)
        fine_root=fine.any(-1); medium_root=medium.any(-1)
        def set_f1_disagreement(left,right):
            overlap=(left&right).sum(-1).float()
            left_n=left.sum(-1); right_n=right.sum(-1)
            denom=(left_n+right_n).float()
            f1=torch.where(denom>0,2*overlap/denom.clamp_min(1),((left_n==0)&(right_n==0)).float())
            return stats(1-f1)
        result['D2']={
            'medium_fine_set_f1_disagreement':set_f1_disagreement(medium,fine_parent),
            'coarse_fine_set_f1_disagreement':set_f1_disagreement(coarse[:,None],fine_root[:,None]),
            'coarse_medium_set_f1_disagreement':set_f1_disagreement(coarse[:,None],medium_root[:,None]),
        }
    return result

def save_model(path,model,spec):
    torch.save({'state_dict':model.state_dict(),'spec':spec},path)

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    spec=saved['spec']; model=ResidualProbe(spec['delta_max'],spec['method']=='Independent',spec.get('input_dim',77),spec.get('option_count',4)); model.load_state_dict(saved['state_dict']); return model,spec

def stratified_metrics(rows):
    groups=defaultdict(list)
    for row in rows:
        for key in dict.fromkeys([row['experiment']+':'+row['primitive_type'], row['experiment']+':'+row['view_id']]): groups[key].append(row)
    return {k:metrics(v) for k,v in groups.items()}