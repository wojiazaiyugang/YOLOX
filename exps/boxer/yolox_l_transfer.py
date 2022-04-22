from exps.boxing_gloves.yolox_l import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.exp_name = "boxer_detection_transfer"
        self.voc_dir_suffix = "2023"

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model

