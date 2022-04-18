from torch import nn



class _MultiScaleMetric(nn.Module):

    def __init__(self, domain_num, in_channel_list, d_bs):
        super(_MultiScaleMetric, self).__init__()
        self.domain_num = domain_num
        self.in_channel_list = in_channel_list

        self.d_bs = d_bs
        

    def forward(self):
        pass
    
    
    

class MultiScaleBatchNormalization(_MultiScaleMetric):
    
    def __init__(self, domain_num, in_channel_list, d_bs):
        super(MultiScaleBatchNormalization, self).__init__(domain_num, in_channel_list, d_bs)
        for in_channel in self.in_channel_list:
            self.nls = []
            self.nls.append(nn.ModuleList({k: nn.BatchNorm2d(in_channel) for k in range(self.domain_num)}))

        self.domain_mean = [[] for i in range(len(self.in_channel_list))]
        self.domain_var = [[] for i in range(len(self.in_channel_list))]

    def forward(self, x):
        feats = []
        for i, feature in enumerate(x):
            for j in range(self.domain_num):
                feat = self.nls[i][j](feature[j: j + self.d_bs])
                feats.append(feat)
        return feats


    def get_domain_distribution(self):
        for i, scale in enumerate(self.nls):
            for j, nl in enumerate(scale):
                self.domain_mean[i][j] = nl.running_mean.numpy()
                self.domain_var[i][j] = nl.running_var.numpy()








    









