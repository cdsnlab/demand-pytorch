from config.base_config import BaseConfig

class ST_MetaNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size = 4
        self.input_size = 2
        self.hidden_size = 64
        self.output_size = 2
        self.geo_feature_size = 32