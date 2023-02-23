class BaseConfig: 
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Device
        self.device = device 

        # Data
        self.format = 'h5'
        self.test_batch_size = 1
        self.num_his = 12 
        self.num_pred = 3
        self.offset = 12
        self.dataset_dir = dataset_dir 
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio 
        self.test_ratio = test_ratio

        # Temporal features
        self.use_tod = True 
        self.use_dow = True 

        # Train 
        self.optimizer = 'Adam'
        self.loss = 'RMSE'
        self.metrics = ['RMSE', 'MaskedMAPE']
        self.scheduler = 'MultiStepLR'
        self.lr = 1e-3
        self.scheduler_args = {
			"milestones": [20, 30, 40, 50],
            "gamma": 0.5
        }
        self.null_value = 0.0
        self.total_epoch = 200
        self.valid_every_epoch = 4 # Validate epoch