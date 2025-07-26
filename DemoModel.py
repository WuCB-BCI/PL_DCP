import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Function
from torch.utils.data import DataLoader
import time
from utils import split_domain, partition_into_three, PackedDataset
from Pairwise_Learning import Pairwise_learning

class FeatureExtractor(nn.Module):
    """
    Feature Extractor
    """

    def __init__(self, fea_len, hidden_1, hidden_2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(fea_len, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #         x=F.leaky_relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        #         x=F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class DomainDisentangler(nn.Module):
    """
    Domain Feature Disentangler

    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params


class ClassDisentangler(nn.Module):
    """
    Class Feature Disentangler
    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params




class CommonDiscriminator(nn.Module):
    """
    Common Discriminator
    """

    def __init__(self, hidden_1, hidden_2, class_num):
        super(CommonDiscriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_2)
        self.fc2 = nn.Linear(hidden_2, class_num)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #         x=F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params




class ReverseLayerF(Function):
    """
    gradient Reverse layer
    Sample：
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)    # alpha1=0.1
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')

    Usage scenarios
    When extracting domain features, it is placed before the class discriminator to make
    the features exhibit the characteristic of class invariance
    When extracting class features, it is placed before the domain discriminator to
    make the features exhibit the characteristic of domain invariance
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DemoModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        # Feature Extractor
        self.fea_extractor = FeatureExtractor(parameter["fea_len"], parameter["fea_h1"], parameter["fea_h2"])

        # domain FeatureExtractor
        self.domain_fea = DomainDisentangler(parameter["fea_h2"], parameter["domain_fea_h1"],
                                             parameter["domain_fea_h2"])
        # class FeatureExtractor
        self.class_fea = ClassDisentangler(parameter["fea_h2"], parameter["class_fea_h1"], parameter["class_fea_h2"])

        # Domain Discriminator
        self.domain_disc = CommonDiscriminator(parameter["domain_fea_h2"], parameter["domain_dis_h2"],
                                               parameter["domain_num"])
        # Class Discriminator
        self.class_disc = CommonDiscriminator(parameter["class_fea_h2"], parameter["class_dis_h2"],
                                              parameter["class_num"])

    def forward(self, s_raw):
        """

        Args:
            s_raw: Source data
        Returns:
            dict:{
            "class_fea":class feature
             "domain_fea": domain feature
            }
        """
        self.train()

        s_data = s_raw
        s_data = s_data.to(dtype=torch.float32)  # trans data format

        s_fea = self.fea_extractor(s_data)
        s_class_fea = self.class_fea(s_fea)
        s_domain_fea = self.domain_fea(s_fea)

        # Distinguish classes by class features
        s_c = self.class_disc(s_class_fea)
        # GRL, Distinguish Domain by class features
        s_cd = ReverseLayerF.apply(s_class_fea, 0.1)
        s_cd = self.domain_disc(s_cd)

        s_d = self.domain_disc(s_domain_fea)
        s_dc = ReverseLayerF.apply(s_domain_fea, 0.1)
        s_dc = self.class_disc(s_dc)

        result = {"class_fea": s_class_fea, "domain_fea": s_domain_fea, "s_c": s_c, "s_d": s_d,
                  "s_cd": s_cd, "s_dc": s_dc}

        return result

    def get_features(self, data):
        """
        obtain the class feature and domain feature
        Args:
            data

        Returns:
            class_fea:tensor(num,fea_len) class feature
            domain_fea:tensor(num,fea_len) domain feature

        """
        fea = self.fea_extractor(data)
        class_fea = self.class_fea(fea)
        domain_fea = self.domain_fea(fea)
        return class_fea, domain_fea

    def get_prototype(self, source_zip):
        """
        computing the domain prototype and class prototype
        Args:
            num: sample number
            flag: subject ID in Test

        Returns:
            domain_prototype_matrix：tensor(14,64)
            class_prototype_matrix：tensor(14,3,64)

        """

        samples_num = source_zip[0].size()[0]
        subject_num = int(source_zip[2].size()[1])
        one_trail_samples = int(samples_num / subject_num)
        all_selecet_list = []
        for i in range(0, samples_num, one_trail_samples):
            list_1 = source_zip[0][i:i + one_trail_samples]
            list_2 = source_zip[1][i:i + one_trail_samples]
            datalist = [list_1, list_2]
            all_selecet_list.append(datalist)

        domain_prototype_list = []
        class_prototype_list = []

        for x in all_selecet_list:  #Get the domain prototype and class prototype
            data = x[0]

            c_fea, d_fea = self.get_features(data)
            d_prototype = torch.mean(d_fea, dim=0)
            labels = x[1]  # (100,3)
            c_prototype = torch.matmul(labels.t(), c_fea)  # （3,100）*（100,64）

            domain_prototype_list.append(d_prototype)
            class_prototype_list.append(c_prototype)

            domain_prototype_matrix = torch.stack(domain_prototype_list)
            class_prototype_matrix = torch.stack(class_prototype_list)

        return domain_prototype_matrix, class_prototype_matrix

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fea_extractor.parameters(), "lr_mult": 1},
            {"params": self.class_fea.parameters(), "lr_mult": 1},
            {"params": self.class_disc.parameters(), "lr_mult": 1},
            {"params": self.domain_fea.parameters(), "lr_mult": 1},
            {"params": self.domain_disc.parameters(), "lr_mult": 1},
        ]
        return params


if __name__=='__main__':
    parameter = {
        "fea_len": 310,
        "fea_h1": 64,
        "fea_h2": 64,
        "domain_fea_h1": 64,
        "domain_fea_h2": 64,
        "class_fea_h1": 64,
        "class_fea_h2": 64,
        "domain_dis_h2": 64,
        "domain_num": 14,
        "class_dis_h2": 64,
        "class_num": 3
    }
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    train_dataset, test_dataset = split_domain(1)
    train_dataset = PackedDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    source_zip = partition_into_three(train_dataset.datalist)
    model = DemoModel(parameter).to("cuda")
    cluster_centers, class_prototype_matrix,d_fea_std = model.prototype_init(source_zip,4)
    # Generate random numbers to test code availability
    s_c_fea = torch.randn(256,64).cuda()
    s_c_label = torch.randn(256,3).cuda()
    s_d_fea = torch.randn(256,64).cuda()
    s_d_label = torch.randn(256,3).cuda()
    source_data_iter = enumerate(train_dataloader)
    _, s_data_list = next(source_data_iter)

    model.prototype_update(cluster_centers, d_fea_std,s_d_fea,s_d_label,s_c_fea,s_c_label)
