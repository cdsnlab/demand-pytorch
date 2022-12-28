from config.base_config import BaseConfig

class STMGCN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        self.M = 1
        self.input_dim = 2
        self.num_nodes = 32 * 32
        self.gcn_hidden_dim = 64
        self.lstm_hidden_dim = 64
        self.lstm_num_layers = 3
        self.gconv_use_bias = True 
        self.sta_kernel_config = {'kernel_type':'chebyshev', 'K': 2}
        self.batch_size = 16 