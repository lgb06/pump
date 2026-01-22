"""
RmGPT
"""
#指定目录
import sys
sys.path.append('/dataWYL/WYL/PHM-Large-Model/')
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import fft   
from torch.quasirandom import SobolEngine
from models.layers.basic_component import BasicBlock,LearnablePositionalEmbedding
from models.layers.MoE import DeepSeekMoE
from models.layers.vq import ResidualTokenizer
from models.layers.Tokenizer import PatchFFT_Tokenizer
from models.layers.Pretrain_head import DINOHead
# from models.layers.FSQ import FSQ
def initialize_high_dimensional_space(num_class, dimension):
    # 创建一个 Sobol 生成器
    sobol = SobolEngine(dimension=dimension, scramble=True)
    # 生成低差异序列点
    points = sobol.draw(num_class)  # 生成 num_class 个点
    # 将 [0, 1] 范围内的点转换为更标准的范围 (-1, 1) 适合初始化
    points = points * 2 - 1
    
    return points #M x D

def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows

    

class Model(nn.Module):
    """
    RmGPT
    """
    def __init__(self, args, configs_list, pretrain=False):
        super().__init__()

        pretrain_dropout_rate = args.pretrain_dropout if hasattr(args, 'pretrain_dropout') else 0.1
        self.pretrain_dropout = nn.Dropout(pretrain_dropout_rate)
        # Tokens settings
        self.num_task = len(configs_list)
        if args.task_name == 'DG':
            self.global_token = nn.Parameter(initialize_high_dimensional_space(3, args.d_model))
            self.global_token = self.global_token.unsqueeze(0).unsqueeze(1)  
            self.DG = True
        else:
            self.category_head = nn.ModuleDict({})
            # self.category_token = nn.ParameterDict({})
            self.category_token = None
            self.rul_head = nn.ModuleDict({})  
            self.DG = False
            for i in range(self.num_task):
                task_data_name = configs_list[i][0]
                if 'classification'  in configs_list[i][1]['task_name']:
                    if self.category_token is None:
                        shared_category_token = initialize_high_dimensional_space(args.num_classes,args.d_model) #[M,D]
                        shared_category_token = shared_category_token.unsqueeze(0).unsqueeze(1)  # 现在形状是[1, 1, M, D]
                        # if "NLN-EMP" in args.task_data_config_path or "NLNEMP" in args.task_data_config_path:
                        #     shared_category_token.repeat(1,args.num_channels,1,1)   # [1, C, M, D]  未将 repeat 的结果赋给 category_token, 不生效
                        # else:
                        #     shared_category_token.repeat(1,configs_list[i][1]['enc_in'],1,1)   # [1, C, M, D]   未将 repeat 的结果赋给 category_token, 不生效
                        self.category_token = nn.Parameter(shared_category_token)  # Shared across classification tasks  ！！！还是[1, 1, M, D]
                    # # args.num_classes 针对NLN-EMP
                    # category_token = initialize_high_dimensional_space(args.num_classes,args.d_model) #[M,D]
                    # # category_token = initialize_high_dimensional_space(configs_list[i][1]['num_class'],args.d_model) #[M,D]       configs_list[i][1]['num_class']目的是为了多个数据集混合训

                    # print(f"category_token.shape:{category_token.shape}")
                   
                    # category_token = category_token.unsqueeze(0).unsqueeze(1)  # 现在形状是 [1, 1, M, D]
                    
                    # if "NLN-EMP" in args.task_data_config_path or "NLNEMP" in args.task_data_config_path:
                    #     category_token.repeat(1,args.num_channels,1,1)   # [1, C, M, D]
                    # else:
                    #     category_token.repeat(1,configs_list[i][1]['enc_in'],1,1)   # [1, C, M, D]
                    
                    # self.category_token[task_data_name]= nn.Parameter(category_token)
                if 'RUL' in configs_list[i][1]['task_name']:
                    self.rul_head[task_data_name] = nn.Linear(args.d_model*configs_list[i][1]['enc_in'],1)
        self.configs_list = configs_list 
        self.d_model = args.d_model
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)
        #Tokenizer 
        self.tokeninzer = PatchFFT_Tokenizer(args)
        self.cls_head = nn.Linear(args.d_model, args.d_model)
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, 
                        expert_num=args.expert_num,activated_expert_num=args.activated_expert) for l in range(args.e_layers)]
        )
        self.debug = args.mode_debug

        ###
        # self.do_fft = False if args.data == "electric" or args.data == "Electric" else True
        ###
        self.do_fft = True

    def get_input_tokens(self, x,x_mark,task_id):
        if self.do_fft:
            # #取出第一个维度
            #根据x_mark得到对应的condition token
            x,seq_x = self.tokeninzer(x)
            #复制到每个变量和batch

        else:
            print("--------Electric, not to do fft------------")
            # --- 修改后的不经过FFT的代码： ---
            # 1. 维度转换 (B, L, M) -> (B, M, L)
            x_in = x.permute(0, 2, 1)

            # 2. Padding 处理 (保持与原 Tokenizer 一致的逻辑)
            patch_len = self.tokeninzer.patch_len
            stride = self.tokeninzer.stride
            remainder = x_in.shape[-1] % patch_len
            if remainder != 0:
                padding = patch_len - remainder
                x_in = F.pad(x_in, (0, padding))
            else:
                padding = 0
            # 此时 x_in shape: (B, M, S), S 是 padding 后的长度

            # 3. 生成 Patch 表征 (x)
            # 使用 unfold 切分 patch，形状变为 (B, M, Patch数量, Patch长度)
            x_patches = x_in.unfold(dimension=-1, size=patch_len, step=stride)
            # 使用线性层投影，将 Patch长度 映射为 d_model
            # Shape 变化: (B, M, N, P) -> (B, M, N, D)
            x = self.tokeninzer.patch_project(x_patches)

            # 4. 生成全局序列表征 (seq_x) - *不使用 FFT*
            # 直接使用经过 padding 的原始序列 (时域数据) 进行投影
            # 原逻辑中 FFT 拼接后的长度等于 padding 后的长度 S，所以可以直接复用 sequence_project
            # Shape 变化: (B, M, S) -> (B, M, D) -> (B, M, 1, D)
            seq_x = self.tokeninzer.sequence_project(x_in).unsqueeze(2)


        return x,seq_x

    def mark2token(self, x_mark):
        x_mark = x_mark.unfold(
            dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x,pretrain=False):
        #找到中间层数
        middle_layer = len(self.blocks)//2
        attn_mask = None
        if pretrain:
            for block in self.blocks[:middle_layer]:
                x = block(x, attn_mask=attn_mask)
                x_middle = x
            for block in self.blocks[middle_layer:]:
                x = block(x,  attn_mask=attn_mask)
            return x, x_middle
        else:
            for block in self.blocks:
                x = block(x,attn_mask=attn_mask)
            return x
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, task_id=None, task_name=None, enable_mask=None, debug = False):
        #X_enc [B,L,D]
        x_mark_enc = x_mark_enc.to(torch.int)
        if 'classification' in task_name:
            dec_out = self.classification(x_enc, x_mark_enc, task_id, debug)
            return dec_out  # logits ([B, N]) or (logits, debug_tokens, ..., ...) in debug mode
        if 'pretrain' in task_name:
            dec_out = self.pretraining(x_enc, x_mark_enc, task_id,
                                       enable_mask=enable_mask)
            return dec_out #结构化输出
        if 'RUL' in task_name:
            dec_out = self.RUL(x_enc, x_mark_enc, task_id)
            return dec_out
        if 'get_feature' in task_name:
            dec_out = self.get_feature(x_enc, x_mark_enc, task_id)
            return dec_out
        return None

    def classification(self, x, x_mark, task_id, debug = False):
        #变换得到输入的tokens
        # print(f"def classification: x.shape:{x.shape}")
        signal_tokens, cls_tokens = self.get_input_tokens(x,x_mark, task_id)
        x = torch.cat((cls_tokens,signal_tokens), dim=2)
        x = x + self.position_embedding(x)
        #输入到backbone
        x = self.backbone(x)
        B, V, L, C = x.shape 
        cls_token  = x[:, :, :1,:].reshape(B,V,C)
        if self.DG:
            category_token = self.global_token
        else:
            # category_token = self.category_token[self.configs_list[task_id][0]] #[1,V,M,D]
            category_token = self.category_token  # Shared across all classification tasks
            if category_token is None:
                raise ValueError("Shared category_token is not initialized.")
        cls_token_projected = self.cls_head(cls_token)

        # 将 cls_token 从 [B, V, C] 调整为 [B, V, 1, D]
        cls_token_reshaped = cls_token_projected.unsqueeze(2)  # [B, V, 1, D]         [B，C， 1，d_model] 

        # 将 category_token 从 [1, 1！！, M, D] 广播到 [B, 1！！, M, D]    ## 其实就是[B，1！！， num_classes，d_model] 
        category_token_expanded = category_token.repeat(B, 1, 1, 1)

        # 计算余弦相似度，在最后一个维度(D)上
        # 输入F.cosine_similarity中时，向量被广播到共同形状 [B, V, M, D] ，category_token_expanded也从[B, 1！！, M, D]被广播到[B, V, M, D]  
        # category_token_expanded 在 V 维是 1，会被广播到 V；cls_token_reshaped 在 M 维是 1，会被广播到 M
        # 这会得到形状为 [B, V, M] 的张量，cosine_similarity的参数keepdim默认是 False，不保留dim=3维度
        similarity = F.cosine_similarity(cls_token_reshaped, category_token_expanded, dim=3)

        # 对V通道求均值，得到 [B, M]
        similarity = similarity.squeeze(2).mean(dim=1)  # squeeze(2) 只能压缩维度大小为 1 的维度。如果 M 不等于 1，PyTorch 会忽略这个操作

        # 在类别维度M上应用log_softmax
        # category_vector = F.log_softmax(similarity, dim=1)  # [B, M]
        category_vector = similarity    
        # exp/exp_sup.py 中的 _select_criterion 和 train_classification。nn.CrossEntropyLoss 在 PyTorch 内部的计算公式是 LogSoftmax + NLLLoss。也就是说，你的代码实际上执行了：LogSoftmax( LogSoftmax( similarity ) )

        if debug:
            debug_cls_token = cls_token.detach()  # cls_token0: [B, V, D] 
            debug_cls_token_projected = cls_token_projected.detach()  # cls_token_projected: [B, V, D]
            # debug_category_token = category_token.squeeze(0).squeeze(0).detach()  # [M, D]
            debug_category_token = category_token.squeeze(0).repeat(cls_token.shape[1], 1, 1).permute(1, 0, 2) # category_token: [1, 1, M, D]->[M, V, D]
            # Return logits together with raw tokens for visualization
            return category_vector, debug_cls_token, debug_cls_token_projected, debug_category_token
        return category_vector

    def get_feature(self, x, x_mark, task_id):
        #变换得到输入的tokens
        signal_tokens, cls_tokens = self.get_input_tokens(x,x_mark, task_id)
        # signal_tokens = self.pretrain_dropout(signal_tokens)
        x = torch.cat((cls_tokens,signal_tokens), dim=2)
        x = x + self.position_embedding(x)
        #输入到backbone
        x = self.backbone(x)
        B, V, L, C = x.shape 
        cls_token  = x[:, :, :1].reshape(B, V, C).mean(dim=1,keepdim=False)
        cls_token = cls_token.reshape(B,C)
        return cls_token

    def RUL(self, x, x_mark, task_id):
        #变换得到输入的tokens
        signal_tokens, cls_tokens =self.get_input_tokens(x,x_mark, task_id)
        x = torch.cat((cls_tokens,signal_tokens), dim=2)
        x = x + self.position_embedding(x)
        #输入到backbone
        x = self.backbone(x)
        B, V, L, C = x.shape 
        cls_token  = x[:, :, :1].reshape(B, V, C)
        cls_token = cls_token.reshape(B,V*C)
        if self.DG:
            rul_head = self.global_head
        else:
            rul_head = self.rul_head[self.configs_list[task_id][0]]
        rul = rul_head(cls_token)
        return rul


    def pretraining(self, x, x_mark, task_id, enable_mask=True):
        ###
        # Pretraining task
        #input: x, x_mark, task_id
        #x: [B, L, M]
        #task_id: task_id
        ###

        signal_tokens, cls_tokens = self.get_input_tokens(x,x_mark, task_id)
        B,V,L,C = signal_tokens.shape
        # cls_tokens = self.pretrain_dropout(cls_tokens)
        x = torch.cat((cls_tokens,signal_tokens), dim=2)
        x = x + self.position_embedding(x)
        #得到backbone的mask输出
        x,_ = self.backbone(x,pretrain=True)
        #得到cls的mask输出
        cls_last = x[:, :, :1,:].reshape(B,V,C)
        seq_middle = x[:, :, 1:,:].reshape(B,V*L,C)
        # 处理cls_last
        cls_flat = cls_last.reshape(-1, C)# [B*V, C]
        cls_last = cls_flat.reshape(B, V, 1, C)  # [B*V, C]
        # cls_last = cls_processed.reshape(B, V, 1, C)  # 恢复原始形状 [B, V, 1, C]
        
        # 处理seq_middle
        seq_flat = seq_middle.reshape(-1, C)  # [B*V*L, C]
        seq_middle = seq_flat.reshape(B, V, L, C)  # [B*V*L, C]
        # seq_middle = seq_processed.reshape(B, V, L, C)  # 恢复原始形状 [B, V, L, C]
        return cls_last,seq_middle


def test_model():
    import yaml
    import argparse

    def read_task_data_config(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        task_dataset_config = config.get('task_dataset', {})
        return task_dataset_config

    def get_task_data_config_list(task_data_config, default_batch_size=32):
        task_data_config_list = []

        for task_name, task_config in task_data_config.items():
            task_config['max_batch'] = default_batch_size
            task_data_config_list.append([task_name, task_config])

        return task_data_config_list
    #读取args
    parser = argparse.ArgumentParser(description='RmGPT Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='status') 
    parser.add_argument('--model_id', type=str, required=False,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='RmGPT',
                        help='model name')
    parser.add_argument('--label_type', type=str, default='local',
                        help='label type')
    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/data_config/main_result/multi_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # pretrain
    parser.add_argument('--right_prob', type=float,
                        default=1.0, help='right mask prob')
    parser.add_argument('--min_mask_ratio', type=float,
                        default=0.5, help='min right mask prob')
    parser.add_argument('--max_mask_ratio', type=float,
                        default=0.8, help='max right mask prob')
    parser.add_argument('--min_keep_ratio', type=float, default=None,
                        help='min crop ratio for various length in pretraining')

    #device
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    # ddp
    parser.add_argument('--ddp',type=bool,default=False,help='whether to use ddp')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=32,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='optimizer learning rate')
    parser.add_argument('--beta2', type=float,
                        default=0.999, help='optimizer beta2')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='eps for optimizer')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--debug', type=str,
                        default='disabled', help='disabled')
    parser.add_argument('--clip_grad', type=float, default=None, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # tokenzier settings
    parser.add_argument('--tokenizer_path', type=str, default=None,   
                        help='tokenizer path')
    parser.add_argument('--codebook_size', type=int, default=1024,
                        help='codebook size')
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)


    # model settings

    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--input_len", type=int, default=2048)
    parser.add_argument("--mode_debug", type=bool, default=False)
    parser.add_argument("--train_quantizer", type=bool, default=True)
    parser.add_argument('--expert_num', type=int, default=None,
                        help='num of shared experts')
    parser.add_argument('--activated_expert', type=int, default=4,
                        help='activated experts')

    args = parser.parse_args()
    # 创建一个Model实例
    config = read_task_data_config(args.task_data_config_path)
    config_list = get_task_data_config_list(config)
    model = Model(args,config_list,pretrain=True)

    

    # 创建一些假的输入数据
    x_enc = torch.randn(10, 2048, 3)  # 假设有10个样本，每个样本有50个时间步，每个时间步有100个特征
    x_mark_enc = torch.randint(3, (10,1))
    x_dec = torch.randn(10, 2048, 3)
    x_mark_dec = torch.randn(10, 2048, 3)
    mask = torch.ones(10, 2048, dtype=torch.bool)  # 假设所有的数据都是有效的
    task_id = torch.tensor([0])  # 假设任务ID为0
    task_name = 'pretrain'  # 假设任务名称为'pretrain'
    enable_mask = True  # 假设启用掩码

    # 调用forward方法
    dec_out = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, task_id, task_name, enable_mask)

    # 打印输出
    print(dec_out)

# 在脚本的最后调用测试函数
if __name__ == '__main__':
    test_model()

