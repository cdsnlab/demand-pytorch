from config.base_config import BaseConfig

class STMGCN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        self.M = 1
        self.input_dim = 2
        self.num_nodes = 32 * 32
        self.gcn_hidden_dim = 128
        self.lstm_hidden_dim = 128
        self.lstm_num_layers = 6
        self.gconv_use_bias = True 
        self.sta_kernel_config = {'kernel_type':'chebyshev', 'K': 2}
        self.batch_size = 16

        self.loss = 'MaskedMAE'
        self.lr = 8e-4
        self.scheduler_args = {
			"milestones": [20, 50, 90],
            "gamma": 0.5
        }
        self.null_value = 0.0
        self.total_epoch = 120