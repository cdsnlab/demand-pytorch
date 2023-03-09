from config.base_config import BaseConfig

class STSSL_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size=32
        self.T = 48
        self.loss = 'MaskedRMSE'
        self.metrics = ['MaskedMAE']
        self.lr = 1e-4
        self.scheduler_args = {
			"milestones": [20],
            "gamma": 0.2
        }
        self.total_epoch = 40 
        self.num_nodes = 1024
        self.d_input = 1                  # means inflow and outflow
        self.d_output = 2                 # means inflow and outflow
        self.d_model = 64
        self.dropout = 0.1
        self.percent = 0.1                # augumentation percentage  
        self.shm_temp = 0.5               # temperature for loss of spatial heterogeneity modeling 
        self.nmb_prototype = 50           # number of cluster 
        self.yita = 0.5                   # balance for inflow loss and outflow loss, $yita * inflow + (1 - yita) * outflow$
        self.early_stop = True
        self.early_stop_patience = 15
        self.batch_sizegrad_norm = True
        self.max_grad_norm = 5
        self.use_dwa = True         # whether to use dwa for loss balance
        self.temp =  2               # tempurature parameter in dwa, a larger T means more similer weights
        self.input_length = 35 # 8+9*3
