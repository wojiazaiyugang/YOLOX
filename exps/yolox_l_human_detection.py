import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/mnt/nfs-storage/yujiannan/data/crowdhuman_coco/Images"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1

