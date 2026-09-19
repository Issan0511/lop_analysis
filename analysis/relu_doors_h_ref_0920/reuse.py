from common import *
import common

def main():
 torch.set_num_threads(2);E.H.setup('cuda');common.CIFAR=E.RC.Cifar10()
 prov=provenance([Path(__file__),ROOT/'src/pmnist_0905.py',ROOT/'src/pmnist_rlcifar_0907.py'])
 assert not (CHECK/'S-reuse.json').exists()
 write(CHECK/'reuse_start.json',prov)
 old=frozen('ab983d9802f273af5135e7a3aa3bef55bedc46cf','relu_doors_0919')
 details={};count=0;positive=True;mutants={}
 for arm in ('C','CH'):
  path=ROOT/'results/relu_doors_0919'/arm
  assert sha(path/'per_task.csv')==HASHES[arm]
  legacy=df(path);legacy=legacy[legacy.task<=2].reset_index(drop=True)
  assert len(legacy)==20 and not legacy.duplicated(['seed','task']).any()
  pp=json.loads((path/'provenance.json').read_text())
  assert pp['data_sha256']==common.CIFAR.sha256
  print('START frozen',arm,flush=True)
  a=run(old,arm,'reuse_frozen_'+arm,epochs=400,seeds=list(range(10)),snapshots=False)
  print('START current',arm,flush=True)
  b=run(E,arm,'reuse_current_'+arm,epochs=400,seeds=list(range(10)),snapshots=False)
  for tag,xx,yy in [('registered',df(a),legacy),('current',df(b),df(a))]:
   ok,n=equal_rows(xx,yy);details[arm+'_'+tag]={'pass':ok,'count':n};positive&=ok;count+=n
  ok=equal_state(state(a),state(b));positive&=ok;details[arm+'_state']=ok;count+=sum(q.numel() for q in state(a)['P'])
  for dest in (a,b):
   got=json.loads((dest/'provenance.json').read_text())
   for k in ('data_sha256','subset_sha256','slots','R','seeds','conds','lr','epochs_per_task','door_beta','doors'):
    assert got[k]==pp[k],(arm,k)
  bad=legacy.copy();bad.loc[0,'online_acc']=np.nextafter(bad.loc[0,'online_acc'],np.inf)
  mutants[arm+'_csv_ulp']=not equal_rows(df(a),bad)[0]
  bad=legacy.copy();bad['seed']=(bad.seed+1)%10;mutants[arm+'_seed']=not equal_rows(df(a),bad)[0]
  # The same selection/config qualification rejects wrong conditions and C->H;
  # never probe H at the main seeds just to create a negative control.
  def qualifies(config):return all(config[k]==pp[k] for k in ('conds','doors','slots','seeds'))
  for label,key,value in [('cond','conds',['std']),('C_to_H','doors',E.DOORS['H'])]:
   bad=dict(pp);bad[key]=value;mutants[arm+'_'+label]=not qualifies(bad)
 if not part('S-reuse',positive,mutants,count,details,prov):raise SystemExit(2)
if __name__=='__main__':main()
