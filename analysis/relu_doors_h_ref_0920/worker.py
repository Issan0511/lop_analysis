from common import *
import common
if __name__=='__main__':
 cfg=json.loads(Path(sys.argv[1]).read_text());torch.set_num_threads(2);E.H.setup('cuda');common.CIFAR=E.RC.Cifar10()
 mod=module(Path(cfg.pop('source'))) if 'source' in cfg else E
 run(mod,**cfg)
