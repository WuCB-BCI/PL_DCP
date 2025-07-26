import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io as scio
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from DataProcess import get_one_subject_dataset, \
    get_one_subject_dataset_IV, get_one_subject_dataset_V, \
    get_one_subject_dataset_all


def get_dataset(test_id, session):
    """
    Get Dataset
    Args:
        test_id:subject ID
        session: session 1 or all_session

    Returns:
        target_set:  { 'feature':ndarray(3394,310)   ;   'label':ndarray(3394,3) }
        source_set:  { 'feature':ndarray(47516,310)  ;  'label':ndarray(47516,3) }
    """

    ## dataloading function, you should modify this function according to your environment setting.
    path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)

    feature_list = []
    label_list = []
    ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)
        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]
        feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            # one_Hot
            if label[i] == 0:
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 1:
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 2:
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
        label_list.append(one_hot_label_mat.astype('float32'))
    target_feature, target_label = feature_list[test_id], label_list[test_id]
    del feature_list[test_id]
    del label_list[test_id]
    source_feature, source_label = np.vstack(feature_list), np.vstack(label_list)
    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set


def data_pipeline(rawdata, labels, domain):
    """
    This function wraps all the data of a single domain

    Args:
        rawdata: Original data  nparray(num,data)
        labels: class labels  nparray(num,)
        domain: domain labels

    Returns:
        all_data_list:

    """
    data_num = labels.shape[0]
    all_data_list = []
    for i in range(0, data_num):
        new_datalist = [rawdata[i], labels[i]]
        domain = torch.tensor(domain)
        domain_one_hot = F.one_hot(domain, num_classes=14)
        new_datalist.append(domain_one_hot.numpy())
        all_data_list.append(new_datalist)

    return all_data_list


def data_pipeline_v(rawdata, labels, domain):
    """
    This function wraps all the data of a single domain

    Args:
        rawdata: Original data  nparray(num,data)
        labels: class labels  nparray(num,)
        domain: domain labels

    Returns:
        all_data_list:

    """
    data_num = labels.shape[0]
    all_data_list = []
    for i in range(0, data_num):
        new_datalist = [rawdata[i], labels[i]]
        domain = torch.tensor(domain)
        domain_one_hot = F.one_hot(domain, num_classes=15)
        new_datalist.append(domain_one_hot.numpy())
        all_data_list.append(new_datalist)

    return all_data_list


def random_split_domains(source_num):
    """
    The data set is divided into source domain data set and target domain data set

    Args:
        source_num:

    Returns:
        source_dataset： list(num)
        target_dataset： list(num)
    """
    import random
    numberlist = list(range(1, 16))
    random_elements = random.sample(numberlist, source_num)  # 随机挑选指定数量的被试编号，储存在列表里
    source_dataset = []
    target_dataset = []
    domain_num = len(random_elements)
    for i in range(1, 16):
        if i in random_elements:  #
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline_v(features, labels, i)
            source_dataset = source_dataset + result
        else:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline_v(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain(test_num):
    """
    SEED
    Args:
        test_num:

    Returns:
        source_dataset： list(num)
        target_dataset： list(num)
    """
    source_dataset = []
    target_dataset = []

    domain_idx = 0
    for i in range(1, 16):

        if i != test_num:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain_IV(test_num):
    """
    seed_IV
    Args:
        test_num:

    Returns:
        source_dataset： list(num)
        target_dataset： list(num)
    """
    source_dataset = []
    target_dataset = []
    domain_idx = 0
    for i in range(1, 16):  #

        if i != test_num:
            features, labels = get_one_subject_dataset_IV(i, 1)
            result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_IV(i, 1)
            result = data_pipeline(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain_V(test_num):
    """
    seed_V
    Args:
        test_num:

    Returns:
        source_dataset： list(num)
        target_dataset： list(num)
    """
    source_dataset = []
    target_dataset = []

    domain_idx = 0
    for i in range(1, 17):
        if i != test_num:
            features, labels = get_one_subject_dataset_V(i, 1)
            result = data_pipeline_v(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_V(i, 1)
            result = data_pipeline_v(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def partition_into_three(all_list):
    """
    Data format conversion to [source data, class label, domain label]
    such as： all_list[0] the 0th trial
           |---- all_list[0][0]:original
           |---- all_list[0][1]:class labels
           |---- all_list[0][2]:domain babels

    Args:
        all_list:

    Returns:
        (rawdata, c_labels, d_labels) tensor

    """
    list_1 = [sublist[0] for sublist in all_list]
    list_2 = [sublist[1] for sublist in all_list]
    list_3 = [sublist[2] for sublist in all_list]
    new_data_list = [list_1, list_2, list_3]

    rawdata = np.array(new_data_list[0])
    rawdata = torch.tensor(rawdata, dtype=torch.float32)
    c_labels = np.array(new_data_list[1])
    c_labels = torch.tensor(c_labels, dtype=torch.float32)
    d_labels = np.array(new_data_list[2])
    d_labels = torch.tensor(d_labels, dtype=torch.float32)

    return rawdata.cuda(), c_labels.cuda(), d_labels.cuda()


class PackedDataset(Dataset):
    """
    Wrap the processed datalist into a Dataset object

    """

    def __init__(self, all_datalist):
        self.datalist = all_datalist

    def __getitem__(self, index):
        return self.datalist[index]

    def __len__(self):
        return len(self.datalist)


def split_domain_all_session(test_num, class_num):
    """
    Get data for all sessions
    Args:
        test_num:
        class_num: 3：seed  4:seed-iv 5:seed-v

    Returns:
        source_dataset, target_dataset
    """
    source_dataset = []
    target_dataset = []

    domain_idx = 0
    if class_num == 5:
        subject_num = 17
    else:
        subject_num = 16
    for i in range(1, subject_num):

        if i != test_num:
            features, labels = get_one_subject_dataset_all(i, class_num)
            if subject_num ==17:
                result=data_pipeline_v(features, labels, domain_idx)
            else:
                result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_all(i, class_num)
            if subject_num ==17:
                result=data_pipeline_v(features, labels, 0)
            else:
                result = data_pipeline(features, labels, 0)

            target_dataset = target_dataset + result

    return source_dataset, target_dataset


if __name__ == "__main__":
    # test code availability
    target, source = get_one_subject_dataset_all(1, 4)
    # train_dataset, test_dataset = split_domain(1)
    print(target.shape, source.shape)

