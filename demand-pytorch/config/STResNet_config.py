from config.base_config import BaseConfig

class STResNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size=32
        self.T = 48
        self.len_closeness=3
        self.len_period=3
        self.len_trend=3
        self.PeriodInterval = 1
        self.TrendInterval = 7
        self.external_dim=28
        self.map_heigh=32
        self.map_width=32
        self.nb_flow=2
        self.nb_residual_unit=12
        self.loss = 'MaskedRMSE'
        self.metrics = ['MaskedRMSE']
        self.lr = 5e-5
        self.total_epoch = 100 
        self.scheduler_args = {
			"milestones": [50],
            "gamma": 0.5
        }