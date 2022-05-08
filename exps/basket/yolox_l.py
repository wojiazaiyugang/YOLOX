from exps.boxing_gloves.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.exp_name = "basket_yolox_l"
        self.voc_dir_suffix = "2024"

