from exps.basketball.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "basketball_yolox_l"
        self.depth = 0.33
        self.width = 0.50

