import argparse
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
# import wandb
from utils.ddp import is_main_process, init_distributed_mode
from pprint import pprint
from pprint import pformat
import re
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RmGPT supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='sft_wo_cwru',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                         default=0, help='status')
    parser.add_argument('--model_id', type=str, required=False,
                        default='full_lora', help='model id')
    parser.add_argument('--model', type=str, required=False, default='RmGPT',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/data_config/baseline/ROT.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct',
                        default=None, help='subsample percent',type=str)
    #device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # ddp
    parser.add_argument('--ddp',type=bool,default=False,help='whether to use ddp')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=16,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=False)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--lora_transform', type=bool, default=False, help='whether to use lora transform')
    parser.add_argument('--efficiency_tuning', type=bool, default=False, help='whether to use efficiency_tuning')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, 
                        default=10, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float, 
                        default=1e-6, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='lora_tuning', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=5.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')


    parser.add_argument('--checkpoints', type=str, default='checkpoints/lr3e-3_decay1',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='disabled or online')
    parser.add_argument('--project_name', type=str,
                        default='RmGPTv2-sft', help='wandb project name')
    
    # tokenzier settings
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=1024,
                        help='codebook size')
    parser.add_argument("--tokenizer_path", type=str, default='checkpoints/Pretrain_Token_pretrain_Tokenizer_cs1024_it0/pretrain_checkpoint.pth')
    
    
    # model settings
    parser.add_argument('--d_model', type=int, default=256,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=5,
                        help='num of encoder layers')
    parser.add_argument("--input_len", type=int, default=2048)
    parser.add_argument("--mode_debug", type=bool, default=False)
    parser.add_argument('--fix_seed', type=int, default=2024, help='seed')

    #MoE setting
    parser.add_argument('--expert_num', type=int, default=8,
                        help='num of shared experts')
    parser.add_argument('--activated_expert', type=int, default=4,
                        help='activated experts')
    

    # task related settings
    # classification task
    parser.add_argument('--label_type', type=str,default='local',help='local or global')
    
    # visualization analysis
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Run visualization analysis after training/testing')
    parser.add_argument('--visualize_output_dir', type=str, default='visualization_results',
                       help='Output directory for visualization results')
    parser.add_argument('--extract_attention', action='store_true', default=True,
                       help='Extract and visualize attention weights')
    
    # classes num mode
    parser.add_argument('--cls_mode', type=int, default=0,
                        help='classes num mode')


    args = parser.parse_args()

    # 专门针对NLN-EMP的映射 
    if not hasattr(args, 'num_classes'):
        cls_to_num_class_mapping = {
            0: 21,
            1: 21,
            3: 12,
            11: 6,
            12: 7,
            13: 8
        }
        args.num_classes = cls_to_num_class_mapping[args.cls_mode]
    if not hasattr(args, 'num_channels'):
        args.num_channels = 5
        if args.data == "Electric" or args.data == "electric":
            args.num_channels = 6

    if args.pretrained_weight is not None:
        checkpoint_dir = os.path.dirname(args.pretrained_weight)
        folder_name = os.path.basename(checkpoint_dir)
        pattern = r'hd(\d+)_el(\d+)_en(\d+)_at(\d+)'
        match = re.search(pattern, folder_name)
        if match:
            args.d_model = int(match.group(1))
            args.e_layers = int(match.group(2))
            # 如果需要使用 expert_num，可在后续逻辑中添加它到 args 中
            args.expert_num = int(match.group(3))
            args.activated_expert = args.expert_num//2
            args.n_heads = int(match.group(4))
            print(f'从预训练文件夹 {folder_name} 中解析配置: d_model={args.d_model}, e_layers={args.e_layers}, expert_num={args.expert_num}, n_heads={args.n_heads}')

            
    if args.subsample_pct is None:
        pass
    elif args.subsample_pct[0] == '0':
        args.subsample_pct = float(args.subsample_pct)

    if args.ddp:
        init_distributed_mode(args)
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
    
    # if is_main_process():
    #     wandb.init(
    #         name=exp_name,
    #         # set the wandb project where this run will be logged
    #         project=args.project_name,
    #         # track hyperparameters and run metadata
    #         config=args,
    #         mode=args.debug,
    #     )

    
    if is_main_process():
        pprint('Args in experiment:')
        print(pformat(vars(args), indent=4))

    Exp = Exp_All_Task_SUP
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
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
    else:
        ii = 0
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
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load_pretrain=True)
        torch.cuda.empty_cache()
        
        # Run visualization analysis if requested
        if args.visualize:
            from visualize_analysis import VisualizationAnalyzer
            checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print('>>>>>>>Running visualization analysis<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                analyzer = VisualizationAnalyzer(args, checkpoint_path=checkpoint_path)
                analyzer.run_full_analysis(
                    output_dir=args.visualize_output_dir,
                    extract_attention=args.extract_attention
                )
            else:
                print(f'Warning: Checkpoint not found at {checkpoint_path}. Skipping visualization.')
