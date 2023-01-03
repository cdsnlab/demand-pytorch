from config.base_config import BaseConfig

class DeepSTN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        self.batch_size = 16 
        self.c = 3
        self.p = 4
        self.t = 4
        self.channel = 2
        self.heigh = 21
        self.width = 12
        self.RP_N = 2
        self.PoI_N = 9
        self.PT_F = 6
        self.T_feat = 28