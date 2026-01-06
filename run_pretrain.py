import argparse
import torch
from exp.exp_pretrain import Exp_All_Task as Exp_All_Task_SSL
import random
import numpy as np
# import wandb
from utils.ddp import is_main_process, init_distributed_mode
from pprint import pprint
from pprint import pformat
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RmGPT Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='Base',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False,
                        default='Pretain_all', help='model id')
    parser.add_argument('--model', type=str, required=False, default='RmGPT',
                        help='model name')
    parser.add_argument('--label_type', type=str, default='local',
                        help='label type')
    parser.add_argument('--project_name', type=str, default='RmGPT2_pretrain',
                        help='wandb project name')
    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--pretrain_data_config_path', type=str,
                        default='data_provider/data_config/main_result/multi_task_pretrain.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument('--cls_mode', type=int, default=-1,
                        help='class split mode for NLN-EMP')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes for classification datasets')
    parser.add_argument('--num_channels', type=int, default=None,
                        help='channel count for NLN-EMP')

    #device
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    # ddp
    parser.add_argument('--ddp',action="store_true", default=False,help='whether to use ddp')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='data loader num workers')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--pretrain_epochs', type=int,
                        default=1, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=768,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.00000005, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='optimizer learning rate')
    parser.add_argument('--beta2', type=float,
                        default=0.999, help='optimizer beta2')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='optimizer weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='eps for optimizer')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--wandb_debug', type=str,
                        default='online', help='disabled or online')
    parser.add_argument('--clip_grad', type=float, default=3, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument("--memory_check", action="store_true", default=False)
    parser.add_argument("--large_model", action="store_true", default=True)

    
    #Pretrain Loss
    parser.add_argument('--teacher_temp', type=float, default=0.05,
                        help='teacher_temp')
    parser.add_argument('--student_temp', type=float, default=0.25,
                        help='student_temp')
    parser.add_argument('--cls_weight', type=float, default=0.4,
                        help='cls_weight')
    parser.add_argument('--teacher_momentum', type=float, default=0.85,                         
                        help='teacher_momentum')

    #Model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=5,
                        help='num of encoder layers')
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--input_len", type=int, default=2048)
    parser.add_argument("--mode_debug", action="store_true", default=False)
    parser.add_argument("--train_quantizer", action="store_true", default=False)
    ##MoE setting
    parser.add_argument('--expert_num', type=int, default=16,
                        help='num of shared experts')
    parser.add_argument('--activated_expert', type=int, default=8,
                        help='activated experts')
    



    args = parser.parse_args()
    if args.num_classes is None:
        cls_to_num_class_mapping = {
            -1: 2,
            0: 21,
            1: 21,
            3: 12,
            11: 6,
            12: 7,
            13: 8,
        }
        args.num_classes = cls_to_num_class_mapping.get(args.cls_mode, 2)
    if args.num_channels is None:
        args.num_channels = 5
        if isinstance(args.data, str) and args.data.lower() == 'electric':
            args.num_channels = 6

    if args.ddp:
        init_distributed_mode(args)
    # 如果是主进程

    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)
    exp_name = '{}_{}_{}_hd{}_el{}_en{}_at{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.d_model,
            args.e_layers,
            args.expert_num,
            args.n_heads
            )

    if is_main_process():
        # wandb.login(key='7c05777e3b354a6cc5956d04e1b3a1cda2a16be0')
        # wandb.init(
        #     name=exp_name,
        #     # set the wandb project where this run will be logged
        #     project=args.project_name,
        #     # track hyperparameters and run metadata
        #     config=args,
        #     mode=args.wandb_debug,
        
        # )
        pprint('Args in experiment:')
        print(pformat(vars(args), indent=4))    
    Exp = Exp_All_Task_SSL

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_hd{}_el{}_en{}_at{}_it{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.d_model,
                args.e_layers,
                args.expert_num,
                args.n_heads,
                ii)

            exp = Exp(args)  # set experiments
            if is_main_process():
                pprint('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            torch.cuda.empty_cache()
