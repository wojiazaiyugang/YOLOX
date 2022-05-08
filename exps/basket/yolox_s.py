from exps.basket.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "basket_yolox_s"
        self.depth = 0.33
        self.width = 0.50
