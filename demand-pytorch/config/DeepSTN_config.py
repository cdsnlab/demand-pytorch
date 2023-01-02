from config.base_config import BaseConfig

class DeepSTN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        self.input_dim = 2
        self.hidden_dim = [64 for _ in range(5)] + [2]
        self.num_cells = 6
        self.kernel_size = [[3, 3] for _ in range(6)]
        self.bias = True 
        self.batch_size = 16 
        self.c = 3
        self.p = 4
        self.t = 4
        self.channel = 2
        self.heigh = 21
        self.width = 12