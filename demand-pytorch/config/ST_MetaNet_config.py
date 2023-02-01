from config.base_config import BaseConfig

class ST_MetaNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size = 4
        # self.rnn_hiddens = [64, 64]
        # self.graph = config.graph
        # self.graph_type = config.graph_type
        # self.input_dim = config.input_dim
        # self.output_dim = config.output_dim
        # self.use_sampling = config.use_sampling
        # self.cl_decay_steps = config.cl_decay_steps
        # self.geo_hiddens = config.geo_hiddens
    