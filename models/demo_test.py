import torch
import torch.nn as nn

def initialize_high_dimensional_space(num_classes, d_model):
    # 只是为了测试 shape：返回 [M, D]
    return torch.randn(num_classes, d_model)

class DummyArgs:
    def __init__(self, num_classes=5, d_model=16, num_channels=7, task_data_config_path="dummy.yaml"):
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_channels = num_channels
        self.task_data_config_path = task_data_config_path

class DummyObj(nn.Module):
    def __init__(self, num_task, configs_list, args):
        super().__init__()
        self.num_task = num_task
        self.args = args

        self.category_token = None
        self.rul_head = nn.ModuleDict({})
        self.DG = False

        for i in range(self.num_task):
            task_data_name = configs_list[i][0]
            task_cfg = configs_list[i][1]

            if "classification" in task_cfg["task_name"]:
                if self.category_token is None:
                    shared_category_token = initialize_high_dimensional_space(
                        args.num_classes, args.d_model
                    )  # [M, D]

                    shared_category_token = shared_category_token.unsqueeze(0).unsqueeze(1)  # [1, 1, M, D]

                    # 注意：repeat 不会原地修改！你这里没接返回值，所以 shape 不变
                    if ("NLN-EMP" in args.task_data_config_path) or ("NLNEMP" in args.task_data_config_path):
                        shared_category_token.repeat(1, args.num_channels, 1, 1)  # 结果被丢掉
                    else:
                        shared_category_token.repeat(1, task_cfg["enc_in"], 1, 1)  # 结果被丢掉

                    self.category_token = nn.Parameter(shared_category_token)

def run_case(task_data_config_path):
    args = DummyArgs(
        num_classes=5,
        d_model=16,
        num_channels=7,
        task_data_config_path=task_data_config_path
    )

    configs_list = [
        ("dataset_A", {"task_name": "classification", "enc_in": 3}),
        ("dataset_B", {"task_name": "long_term_forecast", "enc_in": 8}),
        ("dataset_C", {"task_name": "classification", "enc_in": 10}),
    ]

    obj = DummyObj(num_task=len(configs_list), configs_list=configs_list, args=args)
    print(f"task_data_config_path={task_data_config_path}")
    print("category_token.shape =", tuple(obj.category_token.shape))
    print("-" * 60)

if __name__ == "__main__":
    # 情况1：不是 NLNEMP 分支
    run_case("some_config.yaml")

    # 情况2：触发 NLNEMP 分支
    run_case("NLNEMP_config.yaml")