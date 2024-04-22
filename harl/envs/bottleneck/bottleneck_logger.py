from harl.common.base_logger import BaseLogger

# TODO： 统计以下指标：
# 1. 平均速度
# 2. 每30个episode的done的原因
# 3. 每30个episode的平均长度

class BottleneckLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]
