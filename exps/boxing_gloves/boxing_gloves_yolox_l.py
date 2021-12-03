import os

from exps.basketball.basketball_yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.basketball_detection_dir = "2022"
