from exps.basketball.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.exp_name = "boxing_glove"
        self.voc_dir_suffix = "2022"
