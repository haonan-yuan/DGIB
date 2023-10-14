import sys
sys.path.append("..")

from DGIB.config import args
from DGIB.utils.mutils import *
from DGIB.utils.data_util import *
from DGIB.utils.util import init_logger

import warnings
import networkx as nx


# load data
args, data = load_data(args)

# pre-logs
log_dir = args.log_dir
init_logger(prepare_dir(log_dir) + 'log_' + args.dataset + '.txt')
info_dict = get_arg_dict(args)

# Runner
from DGIB.runner import Runner
from DGIB.model import DGIBNN

model = DGIBNN(args=args).to(args.device)
runner = Runner(args, model, data)

results = []

if args.mode == 'train':
    results = runner.run()
elif args.mode == 'eval' and args.attack == 'random':
    results = runner.re_run()
elif args.mode == 'eval' and args.attack == 'evasive':
    results = runner.re_run_evasive()
elif args.mode == 'eval' and args.attack == 'poisoning':
    results = runner.re_run_poisoning()

# post-logs
measure_dict = results
info_dict.update(measure_dict)
filename = 'info_' + args.dataset + '.json'
json.dump(info_dict, open(osp.join(log_dir, filename), 'w'))