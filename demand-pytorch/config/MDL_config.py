from config.base_config import BaseConfig

class MDL_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)


        self.batch_size = 16
        self.lr = 5e-4
        self.total_epoch = 200
        height = 10
        width = 20
        node_channels = 2
        edge_channels = 2 * height * width
        self.T = 48
        self.len_closeness = 3
        self.len_period = 2
        self.len_trend = 1
        self.node_conf = (self.len_closeness, node_channels, height, width)
        self.node_tconf = (self.len_trend, node_channels, height, width)
        self.node_pconf = (self.len_period, node_channels, height, width)
        self.edge_conf = (self.len_closeness, edge_channels, height, width)
        self.edge_tconf = (self.len_trend, edge_channels, height, width)
        self.edge_pconf = (self.len_period, edge_channels, height, width)
        self.external_dim = 28
        self.embed_dim = 64
        self.bridge = 'concat'
        self.loss = 'MaskedRMSE'
        self.metrics = ['MaskedMAE']