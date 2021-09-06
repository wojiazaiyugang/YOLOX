import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/senseport0/data/train_data"
        self.train_ann = "train_21.7.19.json"
        self.val_ann = "train_21.7.19.json"

        self.num_classes = 1

        self.max_epoch = 10
        self.data_num_workers = 1
        self.eval_interval = 1

