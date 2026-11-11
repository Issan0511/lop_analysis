"""Synthetic audit fixtures only: no dataset/model execution or GPU access."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from . import audit as Q


torch.set_num_threads(1)


class FileAnchor:
    """Fixture anchor still hashes every referenced source byte."""
    def provenance(self, p, resolver):
        for path, value in p['sources'].items(): resolver.verify(path, value)


def fixture_source(root):
    """One full-shape synthetic source: fake values, real serialized structure."""
    seed, act, ln = 0, 'LR', 'off'
    initial, labels, orders = Q.mlp_streams(seed)
    initial = [p.clone() for p in initial]
    initial_snapshot = dict(params_and_adam=(initial, [None]*4), torch_rng=torch.Generator().manual_seed(0).get_state())
    snapshots = {}
    for task in (1,10):
        states = [dict(step=torch.tensor(float(task*4000)), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p)) for p in initial]
        snapshots[task] = dict(params_and_adam=([p.clone() for p in initial], states), torch_rng=initial_snapshot['torch_rng'].clone())
    data_root = root/'data'
    old = data_root/'stage1_ctrl_1111/results/stage1_ctrl_1111/raw/states_adam_lr0.001_LR_s0.npz'
    old.parent.mkdir(parents=True)
    subset = torch.randperm(60000, generator=Q.private_generator('pmnist_0905|rl_subset|0'))[:1200]
    arrays = {'subset':subset.numpy(), 'y':np.stack([labels[t].numpy() for t in range(1,12)])}
    for i,p in enumerate(initial):
        arrays[f'init{i}'] = p.numpy()
        arrays[f'end{i}'] = np.stack([p.numpy()]*10)
    np.savez_compressed(old, **arrays)
    mnist = root/'fake_mnist.gz'
    header = bytes((0,0,8,3))+b''.join(v.to_bytes(4,'big') for v in (60000,28,28))
    with gzip.open(mnist,'wb') as f:
        f.write(header)
        block = bytes(60000*784)
        f.write(block)
    inputs = torch.zeros(1200,784)
    source = dict(version=1, complete=True, study='A', model='mlp', activation=act, ln=ln, seed=seed,
        n_examples=1200, tasks=[1,10], history_steps=4000, initial=initial_snapshot, snapshots=snapshots,
        inputs=inputs, subset=subset, labels=copy.deepcopy(labels), orders=copy.deepcopy(orders),
        initial_common_sha256=Q.digest(initial), initial_state_sha256=Q.digest(initial_snapshot),
        inputs_sha256=Q.digest(inputs), subset_sha256=Q.digest(subset),
        labels_sha256={t:Q.digest(labels[t]) for t in labels}, batch_sha256={t:Q.digest(orders[t]) for t in orders},
        snapshot_sha256={t:Q.digest(snapshots[t]) for t in snapshots},
        history=[dict(task=t,acc=.5,ce=1.) for t in range(1,11)],
        replay_checks=[dict(task=t,parameters_bit_exact=True,labels_bit_exact=True,maxabs=0.) for t in range(1,11)],
        deterministic_repeat=dict(updates_per_repeat=3,repeats=2,exact=True,state_and_rng_restored=True),
        provenance=dict(sources={str(old):Q.A.sha256(old),str(mnist):Q.A.sha256(mnist)}))
    path = root/'source_mlp_LR_off_s0.pt'
    return source, path, data_root, mnist


def sidecar(source, source_hash):
    projection = {k:v for k,v in source.items() if k not in ('initial','snapshots','inputs','subset','labels','orders')}
    return json.loads(json.dumps(projection|dict(source_sha256=source_hash,serialization_exact=True)))


class ResolverTests(unittest.TestCase):
    def test_relocation_exact_bytes_and_original_removed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'work/results/raw/source.pt'; backup=root/'archive/results/raw/source.pt'
            source.parent.mkdir(parents=True); backup.parent.mkdir(parents=True)
            source.write_bytes(b'synthetic source'); backup.write_bytes(source.read_bytes())
            digest=Q.A.sha256(source)
            doc=dict(source_root=str(root/'work'),backup_root=str(root/'archive'),files=[
                dict(source=str(source),backup=str(backup),bytes=source.stat().st_size,sha256=digest)])
            manifest=root/'map.json'; manifest.write_text(json.dumps(doc))
            resolver=Q.Resolver([manifest]); resolver.audit_manifest()
            self.assertEqual(resolver.verify(source,digest),source)
            source.unlink()
            self.assertEqual(resolver.verify(source,digest),backup)
            backup.write_bytes(b'corrupt same size')
            with self.assertRaisesRegex(ValueError,'Corrupt archive'):
                resolver.resolve(source)

    def test_scope_misroute_conflict_missing_and_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'work/a.pt'; backup=root/'archive/a.pt'
            source.parent.mkdir(); backup.parent.mkdir(); source.write_bytes(b'abc'); backup.write_bytes(b'abc')
            item=dict(source=str(source),backup=str(backup),bytes=3,sha256=Q.A.sha256(source))
            base=dict(source_root=str(root/'work'),backup_root=str(root/'archive'),files=[item])
            path=root/'map.json'
            for key,value,phrase in [('source',str(root/'outside/a.pt'),'outside'),('backup',str(root/'archive/wrong.pt'),'exact relative')]:
                doc=copy.deepcopy(base); doc['files'][0][key]=value; path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError,phrase): Q.Resolver([path])
            path.write_text(json.dumps(base)); resolver=Q.Resolver([path])
            source.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'Original/archive disagreement'): resolver.resolve(source)
            source.unlink(); backup.unlink()
            with self.assertRaisesRegex(ValueError,'Missing archive backup'): resolver.resolve(source)
            with self.assertRaisesRegex(ValueError,'No exact archive mapping'): Q.Resolver().resolve(source)
            doc=copy.deepcopy(base); doc['files'].append(item|dict(sha256='f'*64)); path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'Conflicting archive'): Q.Resolver([path])

    def test_source_digest_cannot_be_spoofed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.pt'; p.write_bytes(b'known bytes'); resolver=Q.Resolver()
            resolver.verify(p,Q.A.sha256(p))
            with self.assertRaisesRegex(ValueError,'Source bytes do not match'): resolver.verify(p,'a'*64)
            p.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError,'Conflicting source SHA'): resolver.verify(p,Q.A.sha256(p))


class CommitTests(unittest.TestCase):
    def test_external_commit_inventory_and_spec_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            paths=list(Q.EXEC_FILES)+['analysis/joudaki_vit_battle_0919/model.py','specs/spec_elu_specificity_1112.md']
            for path in paths:
                p=root/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture '+path+'\n')
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
            git('init','-q'); git('add','.')
            git('-c','user.name=Audit Fixture','-c','user.email=fixture@example.invalid','commit','-qm','synthetic anchor')
            anchor=Q.CommitAnchor(root,git('rev-parse','HEAD').decode().strip())
            source=root/'input.bin'; source.write_bytes(b'input')
            prov=dict(execution_git_hash=anchor.commit,git_hash=anchor.commit,code_hashes=anchor.code,
                      code_sha256=anchor.code_digest,spec_sha256=anchor.spec,sources={str(source):Q.A.sha256(source)})
            anchor.provenance(prov,Q.Resolver())
            for field,value in [('execution_git_hash','0'*40),('code_sha256','0'*64),('spec_sha256','0'*64),('code_hashes',{})]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    anchor.provenance(prov|{field:value},Q.Resolver())
            # A later checkout edit does not silently redefine the frozen commit.
            (root/Q.EXEC_FILES[0]).write_text('new implementation')
            anchor.provenance(prov,Q.Resolver())


class MLPSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.source,cls.path,cls.data_root,cls.mnist=fixture_source(cls.root)

    @classmethod
    def tearDownClass(cls):
        Q.mnist_images.cache_clear(); cls.temp.cleanup()

    def audit(self, source, path=None, meta_mutation=None):
        path=path or self.path
        torch.save(source,path); value=Q.A.sha256(path)
        meta=sidecar(source,value)
        if meta_mutation: meta_mutation(meta)
        return Q.audit_mlp_source(source,meta,path,value,Q.Resolver(),FileAnchor(),self.data_root,self.mnist)

    def test_valid_full_shape_serialized_source(self):
        info=self.audit(self.source)
        self.assertEqual(set(info['fingerprints']),{1,10})
        self.assertEqual(info['initial_fingerprints']['body_adam'],Q.EMPTY)

    def test_changed_labels_and_batches_rejected_even_with_updated_hash(self):
        for field,hash_field in [('labels','labels_sha256'),('orders','batch_sha256')]:
            s=copy.deepcopy(self.source); s[field][11].flatten()[0]=(int(s[field][11].flatten()[0])+1)%10
            s[hash_field][11]=Q.digest(s[field][11])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'Wrong canonical'):
                self.audit(s)

    def test_changed_init_input_source_identity_and_sidecar_rejected(self):
        s=copy.deepcopy(self.source); s['initial']['params_and_adam'][0][0][0,0]+=.1
        s['initial_common_sha256']=Q.digest(s['initial']['params_and_adam'][0]); s['initial_state_sha256']=Q.digest(s['initial'])
        with self.assertRaisesRegex(ValueError,'Initial common weights'): self.audit(s)
        s=copy.deepcopy(self.source); s['inputs'][0,0]=.1; s['inputs_sha256']=Q.digest(s['inputs'])
        with self.assertRaisesRegex(ValueError,'Saved inputs differ'): self.audit(s)
        with self.assertRaisesRegex(ValueError,'Spoofed MLP source filename'):
            self.audit(self.source,self.root/'source_mlp_R_off_s0.pt')
        with self.assertRaisesRegex(ValueError,'sidecar does not match'):
            self.audit(self.source,meta_mutation=lambda m:m.update(seed=1))

    def test_wrong_snapshot_or_Adam_step_rejected(self):
        s=copy.deepcopy(self.source); s['snapshots'][10]['params_and_adam'][1][0]['step']=torch.tensor(100.)
        s['snapshot_sha256'][10]=Q.digest(s['snapshots'][10])
        with self.assertRaisesRegex(ValueError,'Wrong history Adam step'): self.audit(s)
        s=copy.deepcopy(self.source); s['snapshots'][10]['params_and_adam'][0][0][0,0]+=.1
        s['snapshot_sha256'][10]=Q.digest(s['snapshots'][10])
        with self.assertRaisesRegex(ValueError,'No-LN saved weights differ'): self.audit(s)

    def test_common_condition_match_is_strict(self):
        groups={}
        Q.same_group(groups,0,{'labels':'aaa'},'stream mismatch')
        Q.same_group(groups,0,{'labels':'aaa'},'stream mismatch')
        with self.assertRaisesRegex(ValueError,'stream mismatch'):
            Q.same_group(groups,0,{'labels':'bbb'},'stream mismatch')

    def test_MLP_fork_checks_are_bound_to_source_tensors(self):
        info=self.audit(self.source); s=info['source']; before=info['fingerprints'][10]
        geometry=dict(P_exact_column_multisets=True,O_orthogonal_maxabs=0.,O_ones_maxabs=0.,O_gram_relative_error=0.)
        for arm in ('S1','S01','P1','P01','O1','O01'):
            geometry[arm+'_centered_norm_relative_error']=0.; geometry[arm+'_mean_logits_maxabs']=0.
        checks=dict(before=before,after=before|{'head_adam':Q.EMPTY},source_state_exact=True,
            head_adam_cleared=True,body_adam_cleared=False,fresh_body=False,
            batch_sha256=s['batch_sha256'][11],labels_sha256=s['labels_sha256'][11],
            canonical_permutation_sha256=Q.digest(Q.permutation(0,'A',10).tolist()),geometry=geometry,
            mean_logits=dict(maxabs=0.,tolerance=2e-5,preserved=True),legacy=[])
        r=dict(activation='LR',ln='off',seed=0,arm='S1',t_end=10,source_id='mlp_LR_off_s0_t10',
            source_sha256=info['hash'],snapshot_sha256=s['snapshot_sha256'][10],
            initial_state_sha256=s['initial_state_sha256'],source_provenance=s['provenance'],checks=checks)
        Q.audit_mlp_curve(r,info,Q.Resolver(),self.data_root)
        changed=copy.deepcopy(r); changed['checks']['before']['body']='f'*64
        with self.assertRaisesRegex(ValueError,'before fingerprint'):
            Q.audit_mlp_curve(changed,info,Q.Resolver(),self.data_root)
        changed=copy.deepcopy(r); changed['checks']['labels_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'different task stream'):
            Q.audit_mlp_curve(changed,info,Q.Resolver(),self.data_root)
        changed=copy.deepcopy(r); changed['checks']['after']['body_adam']=Q.EMPTY
        with self.assertRaisesRegex(ValueError,'retained/cleared body Adam'):
            Q.audit_mlp_curve(changed,info,Q.Resolver(),self.data_root)


class ViTChecksTests(unittest.TestCase):
    def test_fingerprint_joins_actual_model_and_Adam_bytes(self):
        model={'cls_token':torch.ones(2),
            'layers.block_0.layers.mlp.layers.act._extra_state':{'rng':None},
            'layers.out.weight':torch.zeros(200,384),
            'layers.block_1.layers.mlp.layers.act._extra_state':{'rng':None},
            'layers.out.bias':torch.zeros(200)}
        adam={'task':30,'optimizer':{'param_groups':[{'params':[0,1,2]}],
            'state':{0:{'step':torch.tensor(150000.),'exp_avg':torch.ones(2),'exp_avg_sq':torch.ones(2)}}}}
        fp=Q.vit_state_fingerprints(model,adam)
        self.assertEqual(fp['head_adam'],Q.EMPTY)
        self.assertEqual(fp['body'],Q.digest([torch.ones(2)]))
        changed=copy.deepcopy(adam); changed['optimizer']['state'][0]['exp_avg'][0]+=1
        self.assertNotEqual(Q.vit_state_fingerprints(model,changed)['body_adam'],fp['body_adam'])
        with self.assertRaisesRegex(ValueError,'wrong task'):
            Q.vit_state_fingerprints(model,adam|{'task':100})
        malformed=copy.deepcopy(model); malformed['layers.block_0.layers.mlp.layers.act._extra_state']={'rng':'unexpected'}
        with self.assertRaisesRegex(ValueError,'Unexpected ViT nonparameter'):
            Q.vit_state_fingerprints(malformed,adam)
        badorder=copy.deepcopy(adam); badorder['optimizer']['param_groups'][0]['params']=[1,0,2]
        with self.assertRaisesRegex(ValueError,'order/shape mismatch'):
            Q.vit_state_fingerprints(model,badorder)

    def test_private_streams_fixed_shape_and_nonmutation(self):
        torch.manual_seed(99); before=torch.get_rng_state().clone()
        one=Q.vit_streams(10252010); two=Q.vit_streams(10252011)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(one['data_plan']['task'],31)
        self.assertEqual(len(one['data_plan']['classes']),10)
        self.assertEqual(one['data_plan']['first'],250)
        self.assertNotEqual(one['labels_new_sha256'],one['labels_old_sha256'])
        self.assertNotEqual(one['batch_order_sha256'],two['batch_order_sha256'])
        p=Q.permutation(10252010,'B',30)
        self.assertEqual(tuple(p.shape),(10,384))
        self.assertTrue(torch.equal(p.sort(0).values,torch.arange(10)[:,None].expand(10,384)))

    def test_geometry_flags_do_not_replace_residual_checks(self):
        g=dict(P_exact_column_multisets=True,O_orthogonal_maxabs=0.,O_ones_maxabs=0.,O_gram_relative_error=0.)
        for arm in ('S1','S01','P1','P01','O1','O01'):
            g[arm+'_centered_norm_relative_error']=0.;g[arm+'_mean_logits_maxabs']=0.
        Q.check_geometry(g)
        with self.assertRaisesRegex(ValueError,'norm invariant'):
            Q.check_geometry(g|{'P01_centered_norm_relative_error':1.})
        with self.assertRaisesRegex(ValueError,'column multisets'):
            Q.check_geometry(g|{'P_exact_column_multisets':False})

    def test_fork_memory_zerohead_and_eager_guards(self):
        fp=dict(body='1'*64,head='2'*64,body_adam='3'*64,head_adam='4'*64)
        meta=dict(study='B',model='vit',activation='R',source_activation='R',source_arm='S',ln='off',
            seed=10252010,t_end=30,source_id='source',source_sha256='a'*64,checks={'source':'fixed'},
            provenance={'args':{'engine':'eager'}},head_bank={'checks':{'geometry':'fixed'}})
        fork=dict(before=fp,after=fp|{'head_adam':Q.EMPTY},activation='R',body_unchanged_by_head_policy=True,
            head_adam_empty=True,all_adam_empty=False,source_or_fresh='learned_source',inactive_head_rows='unchanged')
        r={k:v for k,v in meta.items() if k not in ('checks','head_bank')}
        r.update(arm='S1',prepared_fork_hashes=fork,checks=dict(source=meta['checks'],head_bank=meta['head_bank']['checks'],
            fork=fork,warmup=dict(optimizer_steps=0,state_and_rng_restored=True,batch_sizes=[]),
            selected_engine={'logits':{'passed':True,'maxabs':0.,'relative_l2':0.}}))
        info=dict(meta=meta,fingerprints=fp,zero_head_hash='0'*64)
        Q.audit_vit_curve(r,info)
        changed=copy.deepcopy(r); changed['checks']['fork']['after']['body_adam']=Q.EMPTY
        with self.assertRaisesRegex(ValueError,'body memory'): Q.audit_vit_curve(changed,info)
        changed=copy.deepcopy(r); changed['checks']['warmup']['state_and_rng_restored']=False
        with self.assertRaisesRegex(ValueError,'warmup changed'): Q.audit_vit_curve(changed,info)
        changed=copy.deepcopy(r); changed['checks']['selected_engine']['logits']['maxabs']=.1
        with self.assertRaisesRegex(ValueError,'not exact'): Q.audit_vit_curve(changed,info)
        changed=copy.deepcopy(r); changed_meta=copy.deepcopy(meta)
        changed['provenance']['args']['engine']='compile'; changed_meta['provenance']['args']['engine']='compile'
        with self.assertRaisesRegex(ValueError,'unanimous native-eager'):
            Q.audit_vit_curve(changed,info|{'meta':changed_meta})


if __name__ == '__main__': unittest.main(verbosity=2)
