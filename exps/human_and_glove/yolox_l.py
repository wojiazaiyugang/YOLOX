from exps.boxing_gloves.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.exp_name = "human_and_glove"
        self.basketball_detection_dir = "2023"
