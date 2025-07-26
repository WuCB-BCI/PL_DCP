import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
from torch.utils.data import DataLoader
from utils import split_domain, partition_into_three, PackedDataset


class Pairwise_learning(nn.Module):
    def __init__(self, parameter):
        super().__init__()

        self.max_iter = parameter["max_iter"]
        self.upper_threshold = parameter["upper_threshold"]
        self.lower_threshold = parameter["lower_threshold"]

        self.threshold = parameter["upper_threshold"]
        self.cluster_label = np.zeros(parameter["num_of_class"])
        self.num_of_class = parameter["num_of_class"]

        # Bilinear transformation matrix
        self.V1 = nn.parameter.Parameter(torch.ones((32, 64)), requires_grad=True)
        self.U1 = nn.parameter.Parameter(torch.ones((32, 64)), requires_grad=True)

    def domain_similarity_distance(self, features, domain_prototype):
        """
        Calculate the similarity between the features of the sample domain
        and the prototypes of each domain
        Args:
            features: domain feature
            domain_prototype:

        Returns:
            d_labels：tensor(num,)  The most similar domain numbers

        """
        store_mat = torch.matmul(self.V1, domain_prototype.T)
        predict = torch.matmul(torch.matmul(self.U1, features.T).T, store_mat)
        predict_d = F.softmax(predict, dim=1)
        # 判断属于哪个域，贴上标签
        d_labels = torch.argmax(predict_d, dim=1)
        return d_labels

    def domain_select(self, features, domain_prototype, num):
        '''
        Args:
            features:
            domain_prototype:
            num:

        Returns:

        '''
        store_mat = torch.matmul(self.V1, domain_prototype.T)
        predict = torch.matmul(torch.matmul(self.U1, features.T).T, store_mat)
        predict_d = F.softmax(predict, dim=1)
        d_similarity, indices = torch.topk(predict_d, num, dim=1, largest=True, sorted=True)

        return d_similarity, indices

    def class_similarity_distance(self, features, class_prototype):
        """


        Args:
            features: tensor(num,64) feature
            class_prototype: tensor(num,3,64)

        Returns:

        """
        classify_list = []
        for i in range(0, features.shape[0]):
            class_similarity = torch.matmul(class_prototype[i], features[i].T)
            classify_list.append(class_similarity)
        result = torch.stack(classify_list)
        result = F.softmax(result, dim=1)
        return result

    def predict_class(self,c_features,d_features,domain_p,class_p):
        """
        Classification
        Args:
            c_features: class feature
            d_features: domain feature
            domain_p: domain prototype
            class_p: class prototype

        Returns:
            c_predict：
        """
        d_predict = self.domain_similarity_distance(d_features, domain_p)
        select_c_prototype = class_p[d_predict]
        c_predict = self.class_similarity_distance(c_features, select_c_prototype)
        return c_predict

    def predict_by_multi(self, c_fea, d_fea, domain_p, class_p, num):
        '''
        Classification
        Args:
            c_fea:
            d_fea:
            domain_p:
            class_p:
            num:

        Returns:
        predict:(batchsize,class_num)
        '''

        d_sim, indices = self.domain_select(d_fea, domain_p, num)
        batchsize = d_sim.shape[0]

        select_p = class_p[indices]  # shape (batchsize, num, 3, 64)
        c_fea_expanded = c_fea.unsqueeze(2).unsqueeze(1)  #  (batchsize, 1, 1, 64)
        result = torch.matmul(select_p, c_fea_expanded).squeeze(-1)  #  (batchsize, num, 3)
        result = result.softmax(dim=2)


        d_sim_expanded = d_sim.unsqueeze(-1)  # (batchsize, num, 1)
        predict_score = d_sim_expanded * result  # (batchsize, num, 3)
        predict = torch.sum(predict_score, dim=1)  #  (batchsize, 3)
        predict = F.softmax(predict, dim=1)

        return predict


    def get_cos_similarity_distance(self, features):
        """
        The similarity between element pairs
        Args:
            features:

        Returns:
            cos_dist_matrix: tensor(batch_size,batch_size)

        """
        # (batchsize,num_class)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batchsize,num_class)
        features = features / features_norm
        # (batchsize,batchsize)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """
        Args:
            cos_dist_matrix: 由get_cos_similarity_distance()

        Returns:
            sim_matrix： tensor(batch_size,batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar, dissimilar)
        return sim_matrix

    def update_threshold(self, epoch: int):
        """
        upper_threshold and lower_threshold
        Args:
            epoch:

        Returns:

        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
        #        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold - eta
            self.lower_threshold = self.lower_threshold + eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold = (self.upper_threshold + self.lower_threshold) / 2

    def compute_indicator(self, cos_dist_matrix):
        """

        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        selected = torch.tensor(1, dtype=dtype, device=device)
        not_selected = torch.tensor(0, dtype=dtype, device=device)
        w2 = torch.where(cos_dist_matrix < self.lower_threshold, selected, not_selected)
        w1 = torch.where(cos_dist_matrix > self.upper_threshold, selected, not_selected)
        w = w1 + w2
        nb_selected = torch.sum(w)
        return w, nb_selected

    def target_domain_evaluation(self,model , test_features, test_labels,source_zip):
        """
        target_domain_evaluation
        Args:
            test_features:
            test_labels:

        Returns:
            acc：
            nmi：

        """
        self.eval()
        model.eval()
        # get the feature of target data
        test_class_features,test_domain_features = model.get_features(test_features)
        domain_p, class_p = model.get_prototype(source_zip)
        t_c_predict = self.predict_class(test_class_features, test_domain_features, domain_p, class_p)
        test_cluster = np.argmax(t_c_predict.cpu().detach().numpy(), axis=1)
        #
        test_labels = np.argmax(test_labels.cpu().detach().numpy(), axis=1)
        test_predict = np.zeros_like(test_labels)
        for i in range(len(self.cluster_label)):
            cluster_index = np.where(test_cluster == i)[0]
            test_predict[cluster_index] = self.cluster_label[i]
        #       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
        acc = np.sum(test_predict == test_labels) / len(test_predict)
        nmi = metrics.normalized_mutual_info_score(test_predict, test_labels)
        return acc, nmi

    def target_domain_evaluation_predict(self,model , test_features, test_labels,source_zip):
        """

        """
        self.eval()
        model.eval()

        test_class_features,test_domain_features = model.get_features(test_features)
        domain_p, class_p = model.get_prototype(source_zip)
        t_c_predict = self.predict_class(test_class_features, test_domain_features, domain_p, class_p)
        test_cluster = np.argmax(t_c_predict.cpu().detach().numpy(), axis=1)
        test_labels = np.argmax(test_labels.cpu().detach().numpy(), axis=1)
        test_predict = np.zeros_like(test_labels)
        for i in range(len(self.cluster_label)):
            cluster_index = np.where(test_cluster == i)[0]
            test_predict[cluster_index] = self.cluster_label[i]
        #       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
        acc = np.sum(test_predict == test_labels) / len(test_predict)
        nmi = metrics.normalized_mutual_info_score(test_predict, test_labels)
        return acc, nmi, test_predict


    def target_domain_evaluation_mul(self, model, test_features, test_labels, source_zip, num):
        """
        Args:

        Returns:
            acc
            nmi

        """
        self.eval()
        model.eval()

        test_class_features, test_domain_features = model.get_features(test_features)
        domain_p, class_p = model.get_prototype(source_zip)
        t_c_predict = self.predict_by_multi(test_class_features, test_domain_features, domain_p, class_p, num)
        test_cluster = np.argmax(t_c_predict.cpu().detach().numpy(), axis=1)
        test_labels = np.argmax(test_labels.cpu().detach().numpy(), axis=1)
        test_predict = np.zeros_like(test_labels)
        for i in range(len(self.cluster_label)):
            cluster_index = np.where(test_cluster == i)[0]
            test_predict[cluster_index] = self.cluster_label[i]
        #       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
        acc = np.sum(test_predict == test_labels) / len(test_predict)
        nmi = metrics.normalized_mutual_info_score(test_predict, test_labels)
        return acc, nmi

    def cluster_label_update(self, model, source_features, source_labels,source_zip):

        self.eval()
        model.eval()
        source_class_features,source_domain_features = model.get_features(source_features)
        domain_p, class_p = model.get_prototype(source_zip)
        s_c_predict = self.predict_class(source_class_features, source_domain_features, domain_p, class_p)

        source_cluster = np.argmax(s_c_predict.cpu().detach().numpy(), axis=1)
        # the sample label
        source_labels = np.argmax(source_labels.cpu().detach().numpy(), axis=1)
        for i in range(len(self.cluster_label)):
            samples_in_cluster_index = np.where(source_cluster == i)[0]
            label_for_samples = source_labels[samples_in_cluster_index]
            # If there are no samples in this cluster, set the cluster label to 0
            if len(label_for_samples) == 0:
                self.cluster_label[i] = 0
            # Count the real labels that appear most frequently in this cluster
            # and update the labels of this cluster
            else:
                label_for_current_cluster = np.argmax(np.bincount(label_for_samples))
                self.cluster_label[i] = label_for_current_cluster

        source_predict = np.zeros_like(source_labels)

        for i in range(len(self.cluster_label)):
            cluster_index = np.where(source_cluster == i)[0]
            source_predict[cluster_index] = self.cluster_label[i]
        acc = np.sum(source_predict == source_labels) / len(source_predict)
        nmi = metrics.normalized_mutual_info_score(source_predict, source_labels)
        return acc, nmi

    def cluster_label_update_mul(self, model, source_features, source_labels, source_zip, num):

        self.eval()
        model.eval()
        source_class_features, source_domain_features = model.get_features(source_features)
        domain_p, class_p = model.get_prototype(source_zip)
        s_c_predict = self.predict_by_multi(source_class_features, source_domain_features, domain_p, class_p, num)
        source_cluster = np.argmax(s_c_predict.cpu().detach().numpy(), axis=1)
        source_labels = np.argmax(source_labels.cpu().detach().numpy(), axis=1)
        for i in range(len(self.cluster_label)):
            samples_in_cluster_index = np.where(source_cluster == i)[0]
            label_for_samples = source_labels[samples_in_cluster_index]
            if len(label_for_samples) == 0:
                self.cluster_label[i] = 0

            else:
                label_for_current_cluster = np.argmax(np.bincount(label_for_samples))
                self.cluster_label[i] = label_for_current_cluster

        source_predict = np.zeros_like(source_labels)

        for i in range(len(self.cluster_label)):
            cluster_index = np.where(source_cluster == i)[0]
            source_predict[cluster_index] = self.cluster_label[i]
        acc = np.sum(source_predict == source_labels) / len(source_predict)
        nmi = metrics.normalized_mutual_info_score(source_predict, source_labels)
        return acc, nmi

    def get_parameters(self) -> List[Dict]:
        params = [
            # {"params": self.U1, "lr_mult": 1},
            # {"params": self.V1, "lr_mult": 1},

        ]
        return params

