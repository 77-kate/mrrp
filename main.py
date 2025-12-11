import torch 
from config.opts import get_config
from train_and_eval import Trainer
from utils.utils import *
import os 
from utils.print_utils import *
import json 
from datetime import datetime

if __name__ == '__main__':
    opts, parser = get_config()
    torch.set_default_dtype(torch.float32)
    save_dir_root = opts.save_dir
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # logger = build_logging(os.path.join(opts.save_dir, f"{timestamp}_log.log"))
    logger = build_logging(os.path.join(opts.save_dir, "log.log"))

    printer = logger.info
    print_log_message('Arguments', printer)
    printer(json.dumps(vars(opts), indent=4, sort_keys=True))

    for seed in range(5):
        opts.seed = seed+1
        print_log_message(f'Data Fold {opts.seed}', printer)
        opts.save_dir = os.path.join(save_dir_root, str(opts.seed))
        os.makedirs(opts.save_dir, exist_ok=True)
        trainer = Trainer(opts=opts, printer=printer)
        trainer.run()