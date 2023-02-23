from config.base_config import BaseConfig

class DMVSTNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        self.input_dim = 2
        self.output_dim = 2
        self.cnn_hidden = 64
        self.lstm_feature = 196
        self.lstm_hidden = 512 
        self.lstm_layers = 6

        self.num_his = 7
        self.batch_size = 64

        self.train_norm = 299.0
        self.test_norm = 307.0

        self.scheduler = 'MultiStepLR'
        self.loss = 'DMVSTNetLoss'
        self.lr = 1e-3
        self.scheduler_args = {
			"milestones": [100],
            "gamma": 0.5
        }
        self.null_value = 0.0
        self.total_epoch = 200