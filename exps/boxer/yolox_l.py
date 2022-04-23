from exps.boxing_gloves.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.exp_name = "boxer_detection"
        self.voc_dir_suffix = "2023"

