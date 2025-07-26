from datetime import time
from typing import Optional

from torch.nn import init
from torch.optim import Optimizer
from DemoModel import *
from Pairwise_Learning import Pairwise_learning
from utils import *
import random
import time
import warnings
warnings.filterwarnings("ignore")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print("use or no：", torch.cuda.is_available())  #
print("number of GPU：", torch.cuda.device_count())  #
print("Version of CUDA：", torch.version.cuda)  #
print("index of GPU：", torch.cuda.current_device())  #
print("Name of GPU：", torch.cuda.get_device_name(0))  #


def setup_seed(seed):  # setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weigth_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        #        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()


class StepwiseLR_GRL:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def get_generated_source(disentangler_model, pl_model, labels_s):
    """
    Obtain the similarity relationship of the source domain labels
    Args:
        disentangler_model:
        pl_model:
        labels_s:

    Returns:
        sim_matrix： the similarity relationship of the source domain labels

    """
    with torch.no_grad():
        disentangler_model.eval()
        pl_model.eval()
        sim_matrix = pl_model.get_cos_similarity_distance(labels_s)
        return sim_matrix


def train(disentangler_model, pl_model, source_data_loader, epoch, max_epoch,
          lr_scheduler, source_zip,batch_size):
    disentangler_model.train()
    pl_model.train()

    t = 3394 * 14 // batch_size
    one_epoch_loss = 0
    one_epoch_dann_loss = 0

    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    source_data_iter = enumerate(source_data_loader)
    tt = time.time()
    for i in range(t):
        disentangler_model.train()
        pl_model.train()
        _, s_data_list = next(source_data_iter)
        s_data = s_data_list[0].to(device="cuda", dtype=torch.float32)

        # one-hot labels
        s_c_labels = s_data_list[1].to(device="cuda", dtype=torch.float32)
        s_d_labels = s_data_list[2].to(device="cuda", dtype=torch.float32)
        # domain prototype and class prototype
        domain_p, class_p = disentangler_model.get_prototype(source_zip)
        s_result = disentangler_model(s_data)
        rij_s = get_generated_source(disentangler_model, pl_model, s_c_labels)

        s_c_features = s_result["class_fea"]
        s_d_features = s_result["domain_fea"]

        # predict
        s_c_predict = pl_model.predict_class(s_c_features, s_d_features, domain_p, class_p)
        features_pairwise_s = pl_model.get_cos_similarity_distance(s_c_predict)
        # Soft regularization
        R = torch.matmul(domain_p.T, domain_p) - torch.eye(64).cuda()
        eta = 0.00001
        source_bce = -(torch.log(features_pairwise_s + eta) * rij_s) - (1 - rij_s) * torch.log(
            1 - features_pairwise_s + eta)
        # source domain supervised loss
        supervised_pairwise_loss = torch.mean(source_bce) + 0.01*torch.norm(R)

        # loss of domain discriminators and class discriminators
        dann_loss = (bce_loss(s_result["s_c"], s_c_labels) + bce_loss(s_result["s_dc"], s_c_labels)
                     + bce_loss(s_result["s_d"], s_d_labels) + bce_loss(s_result["s_cd"], s_d_labels))
        loss = supervised_pairwise_loss + dann_loss

        optimizer.zero_grad()
        loss.backward()

        one_epoch_loss = one_epoch_loss + loss.item()
        one_epoch_dann_loss = one_epoch_dann_loss + dann_loss.item()
        optimizer.step()
    # every epoch cost
    print(f'coast:{time.time() - tt:.4f}s')
    lr_scheduler.step()
    avg_epoch_loss = one_epoch_loss / t
    avg_epoch_dann_loss = one_epoch_dann_loss / t
    return avg_epoch_loss, avg_epoch_dann_loss


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

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

    parameter_pl = {
        "max_iter": 100,
        "num_of_class": 3,
        "delta": 2
    }
    savepath = 'H:\\UnSeen_Target_PL_DCP\\PL-DCP\\' + \
               time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    max_iter = 100
    learning_rata = 1e-3
    weight_c = 1e-5
    big_list = []
    for big in range(1):
        all_sub_best = []
        for subject in range(1, 16):
            # random seed
            setup_seed(114514 + big)
            # Data reading
            train_dataset, test_dataset = split_domain(subject)  #
            train_dataset = PackedDataset(train_dataset)
            test_dataset = PackedDataset(test_dataset)
            batch_size = 3394
            # data iterator
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

            source_zip = partition_into_three(train_dataset.datalist)
            source_rawdata = source_zip[0]
            source_clabels = source_zip[1]
            source_dlabels = source_zip[2]
            target_rawdata, target_clabels, _ = partition_into_three(test_dataset.datalist)

            model = DemoModel(parameter).to("cuda")
            pl_model = Pairwise_learning(parameter_pl).to("cuda")
            model.apply(weigth_init)
            pl_model.apply(weigth_init)
            optimizer = torch.optim.RMSprop(model.get_parameters() + pl_model.get_parameters(), lr=1e-3,
                                            weight_decay=1e-5)
            lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=0.001, gamma=10, decay_rate=0.75, max_iter=max_iter)

            loss_list = []
            dann_loss_list = []
            target_acc_list = np.zeros(max_iter)
            target_nmi_list = np.zeros(max_iter)
            source_acc_list = np.zeros(max_iter)
            source_nmi_list = np.zeros(max_iter)
            best_acc = 0.
            savepath = savepath + '_' +str(subject)
            os.makedirs(savepath)
            for epoch in range(0, max_iter):
                #Train Model
                one_epoch_loss, one_epoch_dann_loss = train(model, pl_model, train_dataloader,
                                                            epoch, max_iter, lr_scheduler, source_zip,batch_size)

                # Update the clustering labels
                source_acc, source_nmi = pl_model.cluster_label_update(model, source_rawdata, source_clabels,
                                                                       source_zip)
                print('epoch:', epoch, ' loss: ', one_epoch_loss, ' dann loss:', one_epoch_dann_loss, ' source_acc:',
                      source_acc)

                # target_Data_evaluation
                target_acc, target_nmi = pl_model.target_domain_evaluation(model, target_rawdata,
                                                                           target_clabels, source_zip)
                target_acc_list[epoch] = target_acc
                source_acc_list[epoch] = source_acc
                target_nmi_list[epoch] = target_nmi
                source_nmi_list[epoch] = source_nmi
                print('target_acc:', target_acc, '  traget_NMI:', target_nmi)
                loss_list.append(one_epoch_loss)
                dann_loss_list.append(one_epoch_dann_loss)
                # save the best weight
                if target_acc > best_acc:
                    weight_name1 = savepath + '/a_weight_init_' + str(target_acc) + '.pth'
                    weight_name2 = savepath + '/b_weight_init_' + str(target_acc) + '.pth'
                    torch.save(model, weight_name1)
                    torch.save(pl_model, weight_name2)
                    best_nmi = target_nmi
                    best_acc = target_acc
            print('best acc:', best_acc, ' best nmi:', best_nmi)
            all_sub_best.append(best_acc)
            loss_list = np.array(loss_list)
            dann_loss_list = np.array(dann_loss_list)
            np.save(savepath + '/loss_curve_' + str(subject) + '.npy',
                    loss_list)
            np.save(savepath + '/dannloss_curve_' + str(subject) + '.npy',
                    dann_loss_list)
            np.save(savepath + '/best_acc_list_' + str(subject) + '.npy',
                    all_sub_best)

            print('-------------------subject ', subject, '   over------------------------')
            print(all_sub_best)
        big_list.append(all_sub_best)

