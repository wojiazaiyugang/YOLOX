from exps.boxer.yolox_l import Exp as MyExp


class Exp(MyExp):

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model

