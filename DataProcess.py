import os
import numpy as np
from scipy import io as scio
from sklearn import preprocessing




def get_one_subject_dataset(test_id, session):
    """
    SEED
    Args:
        test_id:
        session:

    Returns:
        featres:  nparray(3394,310)
        labels:  nparray(3394,3)   one-hot
    """
    path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    info = os.listdir(path)[test_id - 1]

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
    # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
    one_hot_label_mat = np.zeros((len(label), 3))
    for i in range(len(label)):

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
    feature = min_max_scaler.fit_transform(feature)

    return feature, one_hot_label_mat



def get_one_subject_dataset_IV(test_id, session):
    """
    seed IV
    Args:
        test_id:
        session:

    Returns:
        featres:
        labels:
    """
    path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de_IV'
    os.chdir(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    info = os.listdir(path)[test_id - 1]

    domain = os.path.abspath(path)
    info_ = os.path.join(domain, info)
    if session == 1:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    elif session == 2:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    else:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
    one_hot_label_mat = np.zeros((len(label), 4))
    for i in range(len(label)):

        if label[i] == 0:
            one_hot_label = [1, 0, 0, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 1:
            one_hot_label = [0, 1, 0, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 2:
            one_hot_label = [0, 0, 1, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 3:
            one_hot_label = [0, 0, 0, 1]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
            one_hot_label_mat[i, :] = one_hot_label
    feature = min_max_scaler.fit_transform(feature)
    return feature, one_hot_label_mat


def get_one_subject_dataset_V(test_id, session):
    """
    seed V

    Args:
        test_id:
        session:

    Returns:
        featres:  nparray(681,310)
        labels: nparray(681,5)   one-hot
    """

    path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de_V'
    os.chdir(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    info = os.listdir(path)[test_id - 1]

    domain = os.path.abspath(path)
    info_ = os.path.join(domain, info)
    if session == 1:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    elif session == 2:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    else:
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
    # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
    one_hot_label_mat = np.zeros((len(label), 5))
    for i in range(len(label)):

        if label[i] == 0:
            one_hot_label = [1, 0, 0, 0, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 1:
            one_hot_label = [0, 1, 0, 0, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 2:
            one_hot_label = [0, 0, 1, 0, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 3:
            one_hot_label = [0, 0, 0, 1, 0]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
            one_hot_label_mat[i, :] = one_hot_label
        if label[i] == 4:
            one_hot_label = [0, 0, 0, 0, 1]
            one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
            one_hot_label_mat[i, :] = one_hot_label
    feature = min_max_scaler.fit_transform(feature)
    return feature, one_hot_label_mat


def get_one_subject_dataset_all(test_id, class_num):
    """
    Obtain all_session data in one_subject

    Args:
        test_id:
        session

    Returns:
        featres:  nparray(3394,310)
        labels:  nparray(3394,3)   one-hot
    """
    all_data = []
    all_label = []
    if class_num == 3:
        for session in range(1, 4):

            path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de'
            os.chdir(path)
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            info = os.listdir(path)[test_id - 1]

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
            # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
            one_hot_label_mat = np.zeros((len(label), 3))
            for i in range(len(label)):

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
            feature = min_max_scaler.fit_transform(feature)
            all_data.append(feature)
            all_label.append(one_hot_label_mat)
        feature, one_hot_label_mat = np.vstack(all_data), np.vstack(all_label)
        return feature, one_hot_label_mat

    if class_num == 4:
        all_data = []
        all_label = []
        for session in range(1, 4):
            path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de_IV'
            os.chdir(path)
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

            info = os.listdir(path)[test_id - 1]

            domain = os.path.abspath(path)
            info_ = os.path.join(domain, info)
            feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
            one_hot_label_mat = np.zeros((len(label), 4))
            for i in range(len(label)):

                if label[i] == 0:
                    one_hot_label = [1, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 1:
                    one_hot_label = [0, 1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 2:
                    one_hot_label = [0, 0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 3:
                    one_hot_label = [0, 0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
            feature = min_max_scaler.fit_transform(feature)
            all_data.append(feature)
            all_label.append(one_hot_label_mat)
        feature, one_hot_label_mat = np.vstack(all_data), np.vstack(all_label)
        return feature, one_hot_label_mat

    if class_num == 5:
        all_data = []
        all_label = []
        for session in range(1, 4):
            path = 'H:\\UnSeen_Target_PL_DCP\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de_V'
            os.chdir(path)
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            info = os.listdir(path)[test_id - 1]

            domain = os.path.abspath(path)
            info_ = os.path.join(domain, info)
            feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            # feature_list.append(min_max_scaler.fit_transform(feature).astype('float32'))
            one_hot_label_mat = np.zeros((len(label), 5))
            for i in range(len(label)):
                if label[i] == 0:
                    one_hot_label = [1, 0, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 1:
                    one_hot_label = [0, 1, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 2:
                    one_hot_label = [0, 0, 1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 3:
                    one_hot_label = [0, 0, 0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 4:
                    one_hot_label = [0, 0, 0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
            feature = min_max_scaler.fit_transform(feature)
            all_data.append(feature)
            all_label.append(one_hot_label_mat)
        feature, one_hot_label_mat = np.vstack(all_data), np.vstack(all_label)
        return feature, one_hot_label_mat


