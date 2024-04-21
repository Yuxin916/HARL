from harl.common.base_logger import BaseLogger

# TODO： 统计以下指标：
# 1. 平均速度
# 3. episode 长度

class BottleneckLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]
