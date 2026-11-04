"""PrimaConsis: benchmark, source-only training and single-view inference."""
import argparse, hashlib, json, os, pickle
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
import calibrator as cal
import benchmark as bench
from definitions import *


def read(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def append(f,row):
    f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()


def config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf8'))
    base=Path(path).resolve().parent
    for key,value in cfg['paths'].items():
        if Path(value).is_absolute(): raise ValueError('config paths must be relative')
        cfg['paths'][key]=base/value
    return cfg


def rows(cfg,split,axes):
    for axis in axes:
        for r in read(cfg['paths']['dataset']/f'axis{axis[1:]}'/f'{split}.jsonl'):
            yield axis,r


def project_probability(axis,view,request,response,r):
    if request['type']=='choice' and axis!='D1':
        keys=list(request['criteria']); p=[response['probabilities'][k] for k in keys]
        if axis=='D2':
            gold=[int(r['coarse_gold'])] if view=='coarse_choice' else [i for i,k in enumerate(keys) if r['medium_gold' if view=='medium_choice' else 'fine_gold'][k]]
        else: gold=[keys.index(r['correct_choice_key'] if axis=='D3' else r['gold'])]
        return p,gold[0] if gold else 0,gold,keys
    q=float(response['probabilities']['hallucinated']) if request['type']=='choice' else float(response['noul'])
    if axis=='D1':
        q=1-q if view=='negative_noul' else q; y=int(r['gold'])
    elif axis=='D3': q=1-q; y=int(r['should_abstain'])
    elif axis=='D4': y=int(r['gold']==view.split('::')[1])
    else:
        y=int(r['coarse_gold']) if view=='coarse_noul' else int(r['medium_gold' if view.startswith('medium') else 'fine_gold'][view.split('::')[1]])
    return [q],y,[y],['false','true']


def features(h,p,primitive,view,recipe,description_length=None):
    p=np.asarray(p); multi=len(p)>1
    logits=np.log(p.clip(EPS,1)) if multi else np.log(p.clip(EPS,1-EPS)/(1-p).clip(EPS,1-EPS))
    pp=np.zeros(12);ll=np.zeros(12);pp[:len(p)]=p;ll[:len(p)]=logits
    dist=p if multi else np.array([p[0],1-p[0]])
    desc= np.mean([len(s) for s in ['international news, politics, diplomacy','sports events, teams, athletes','business, finance, markets','science, technology, internet']]) if recipe=='4b' and view=='described' else 0
    if description_length is not None:
        desc = description_length if recipe == '4b' else 0
    count=len(p) if multi or recipe=='4b' else 2
    x=np.r_[h,pp,ll,-(dist*np.log(dist.clip(EPS,1))).sum(),np.sort(dist)[-1]-np.sort(dist)[-2],count,int(primitive=='choice'),desc]
    return x.tolist(),logits.tolist()


def collect(cfg,args):
    from jevk5 import JevK5
    from runtime import forward_with_hidden
    model=JevK5(str(cfg['paths']['backbone']),graphs=False)
    out=cfg['paths']['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/f'{args.split}_extracted.jsonl'
    if path.exists(): raise FileExistsError(path)
    items=list(rows(cfg,args.split,args.axes))
    if args.limit: items=[x for a in args.axes for x in [x for x in items if x[0]==a][:args.limit]]
    with path.open('w',encoding='utf8') as f:
        for axis,r in items:
            group=axis+'::'+str(r.get('prompt_group_id',r.get('group',r['sample_id'])))
            for view,request in bench.requests_for(r).items():
                ids,h,probs,options=forward_with_hidden(model,r['state'],request)
                response=model.decide(r['state'],request)
                expected=[response['probabilities'][k] for k,_ in options] if request['type']=='choice' else [response['noul'],1-response['noul']]
                # Runtime Noul option ordering is checked against explicit option keys.
                if request['type']=='noul':
                    expected=[response['noul'] if k == 'true' else 1-response['noul'] for k,_ in options]
                if np.max(np.abs(np.array(expected)-probs))>1e-5: raise ValueError('official readout parity failed: '+view)
                p,y,gold,keys=project_probability(axis,view,request,response,r)
                append(f,dict(experiment=axis,sample_id=r['sample_id'],group_id=axis+'::'+r['sample_id'],family_id=group,split=args.split,state_hash=hashlib.sha256(r['state'].encode()).hexdigest(),view_id=view,primitive_type=request['type'],raw_vector=p,target=y,semantic_gold=int(r['should_abstain']) if axis=='D3' else y,gold_set=gold,options=keys,hidden=h['penultimate'].tolist()))
            print(axis,r['sample_id'],flush=True)


def transform(records,pca,recipe):
    result=[]
    for r in records:
        h=pca.transform(np.asarray(r['hidden'])[None])[0]
        x,l=features(h,r['raw_vector'],r['primitive_type'],r['view_id'],recipe)
        result.append({k:v for k,v in dict(r,features=x,raw_logits=l).items() if k!='hidden'})
    return result


def report(records):
    return dict(performance=cal.stratified_metrics(records),consistency=cal.distances(records))


def train(cfg):
    sources=cfg['training']['source_axes']
    data=[r for r in read(cfg['paths']['output']/'calibration_extracted.jsonl') if r['experiment'] in sources]
    for r in data:
        fold=int(hashlib.sha256(r['family_id'].encode()).hexdigest()[:8],16)%10
        r['partition']='train' if fold<6 else 'select' if fold<8 else 'audit'
    fit=[r for r in data if r['partition']=='train']
    pca=PCA(n_components=cfg['training']['pca_dim'],random_state=SEED,svd_solver='randomized').fit(np.array([r['hidden'] for r in fit]))
    data=transform(data,pca,cfg['feature_recipe'])
    train_rows=[r for r in data if r['partition']=='train'];select=[r for r in data if r['partition']=='select']
    if not train_rows or not select: raise ValueError('empty calibration partition')
    spec=dict(method='TRC-Cal',alpha=cfg['training']['alpha'],gamma=cfg['training']['gamma'],delta_max=1.,beta=0.,consistency_loss='aligned',input_dim=len(train_rows[0]['features']),option_count=12)
    out=cfg['paths']['trained'];out.mkdir(parents=True,exist_ok=True)
    with (out/'progress.jsonl').open('w',encoding='utf8') as f:
        model,summary=cal.train(train_rows,spec,epochs=cfg['training']['max_epochs'],eval_rows=select,patience=cfg['training']['patience'],progress=lambda r:append(f,r))
    cal.save_model(out/'calibrator.pt',model,spec)
    with (out/'pca.pkl').open('wb') as f:pickle.dump(pca,f)
    dump(out/'training.json',dict(sources=sources,summary=summary,config={k:v for k,v in cfg.items() if k!='paths'},fit_ids=sorted({r['group_id'] for r in fit})))


def evaluate(cfg,args):
    source=read(cfg['paths']['output']/f'{args.split}_extracted.jsonl')
    bundle=cfg['paths']['trained'] if args.trained else cfg['paths']['calibrator']
    with (bundle/'pca.pkl').open('rb') as f:pca=pickle.load(f)
    if isinstance(pca,dict): pca=pca['penultimate_64']
    data=transform(source,pca,cfg['feature_recipe']);model,_=cal.load_model(bundle/'calibrator.pt')
    pred=cal.predict(data,model)
    dump(cfg['paths']['output']/f'{args.split}_report.json',dict(raw=report(cal.raw_predictions(data)),calibrated=report(pred)))
    with (cfg['paths']['output']/f'{args.split}_predictions.jsonl').open('w',encoding='utf8') as f:
        for r in pred:append(f,{k:v for k,v in r.items() if k!='features'})


def decide(cfg,args):
    from online import CalibratedJev
    model=CalibratedJev(cfg)
    with Path(args.input).open(encoding='utf8') as source, Path(args.output).open('w',encoding='utf8') as output:
        for line in source:
            if not line.strip(): continue
            item=json.loads(line)
            result=model.decide(item['state'],item['request'],
                complement=item.get('complement',False),
                binary_choice_key=item.get('binary_choice_key'))
            append(output,result)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','train','evaluate','decide']);p.add_argument('--config',default='config.yaml');p.add_argument('--split',choices=['calibration','test'],default='test');p.add_argument('--axes',nargs='+',default=['D1','D2','D3','D4']);p.add_argument('--limit',type=int,default=0);p.add_argument('--trained',action='store_true');p.add_argument('--input');p.add_argument('--output');a=p.parse_args();cfg=config(a.config)
    if a.command=='train': train(cfg)
    else:globals()[a.command](cfg,a)
if __name__=='__main__':main()
