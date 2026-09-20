"""Recompute a fixed, saved epoch without reading any scientific summary."""
import argparse
import torch
from src import drive_cifar_c_0920 as D

def audit(path):
    a=D.load_pt(path);ident=a['identity'];assert ident['source_sha256']==D.sources()
    X,cifar=D.inputs(ident['seeds'],D.setup());assert D.tree_hash(X)==ident['X_sha256'] and cifar.sha256==ident['data_sha256']
    e=D.Engine(ident['seeds'],X,state=a['start'],graph=True);e.train_epoch(a['order'])
    assert D.tree_hash(e.state())==D.tree_hash(a['end']),'audit replay differs'
    return dict(status='PASS',path=str(path),sha256=D.sha(path))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',required=True);a=p.parse_args()
    with D.exclusive():print(audit(a.src))
