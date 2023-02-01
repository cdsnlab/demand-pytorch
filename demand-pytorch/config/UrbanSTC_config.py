from config.base_config import BaseConfig

class UrbanSTC_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.reg_in_channels = 1
        self.reg_base_channels = 128
        self.reg_margin = 1e-4

        self.tc_in_channels = 1
        self.tc_base_channels = 128
        self.tc_margin = 15.0

        self.inf_in_channels = 1
        self.inf_base_channels = 128       
        
        self.batch_size = 16
        self.test_batch_size = 8
        
        self.reg_epoch = 100
        self.tc_epoch = 200
        self.inf_epoch = 200
        
        self.do_reg = True
        self.do_tc = True
        self.do_inf = True
        
        
        
