from exps.boxing_gloves.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.exp_name = "human_transfer"
        self.basketball_detection_dir = "2023"

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model

