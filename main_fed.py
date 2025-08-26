import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import sensing_data_dict
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import *
from models.Fed import FedAvg
from models.test import test_img
from data_set import MyDataset, MyDataset_train
# from main_fed import inverse_local_ep
# from cvx_main import *

import logging
import time

import os

args = args_parser()
log_name = 'Fed-{}-{}-m7-user-{}-round-BS-{}-{}-{}-{}.log'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), args.num_users,
                                                            args.epochs, args.rule, args.eta, args.T, args.E)
# 创建一个logger
logger = logging.getLogger('CNN')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件

fh = logging.FileHandler('./log/{}'.format(log_name))

fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

# logger.info('fed_learn_mnist_cnn_100_iid_v2')
logger.info('fed_isac_cnn')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    # args, unparsed = args_parser.parse_known_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.7723, 0.8303, 0.9284), (0.3916, 0.3057, 0.1893)),
    ])



    net_glob = ResNet.ResNet10().to(args.device)
    # net_glob = CNNCifar(args).to(args.device)

    model_root = './save/models/models_10_m7.pth'
    if os.path.exists(model_root) is False:
        torch.save(net_glob.state_dict(), model_root)
    net_glob.load_state_dict(torch.load(model_root))

    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params: {}'.format(net_total_params))
    print(net_glob)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()



    logger.info(args)
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    batch_size_init = args.local_bs

    loss_train_init = 6

    root_radar_1 = './data/spect/THREE_RADAR_3000/radar_1/'
    root_radar_2 = './data/spect/THREE_RADAR_3000/radar_2/'
    root_radar_3 = './data/spect/THREE_RADAR_3000/radar_3/'


    for tmp_round in range(1, args.epochs + 1):

        num = [int(x // 7) for x in [504., 548., 534., 529., 516., 545., 543., 474., 483., 524.]]
        # num, p_s, eta = Solve_main()

        # root_radar_1 = './data/spect/THREE_RADAR_3000/radar_1/'
        # dataset_train_1 = MyDataset(txt=root_radar_1 + 'train_1_m7.txt', transform=data_transform)
        dataset_train_1 = MyDataset_train(txt=root_radar_1 + 'train_1_m7.txt', transform=data_transform, num = num[0])

        # root_radar_2 = './data/spect/THREE_RADAR_3000/radar_2/'
        # dataset_train_2 = MyDataset(txt=root_radar_2 + 'train_1_m7.txt', transform=data_transform)
        dataset_train_2 = MyDataset_train(txt=root_radar_2 + 'train_1_m7.txt', transform=data_transform, num = num[1])

        # root_radar_3 = './data/spect/THREE_RADAR_3000/radar_3/'
        # dataset_train_3 = MyDataset(txt=root_radar_3 + 'train_1_m7.txt', transform=data_transform)
        dataset_train_3 = MyDataset_train(txt=root_radar_3 + 'train_1_m7.txt', transform=data_transform, num = num[2])

        # dataset_train_4 = MyDataset(txt=root_radar_1 + 'train_2_m7.txt', transform=data_transform)
        dataset_train_4 = MyDataset_train(txt=root_radar_1 + 'train_2_m7.txt', transform=data_transform, num = num[3])

        # dataset_train_5 = MyDataset(txt=root_radar_2 + 'train_2_m7.txt', transform=data_transform)
        dataset_train_5 = MyDataset_train(txt=root_radar_2 + 'train_2_m7.txt', transform=data_transform, num = num[4])

        # dataset_train_6 = MyDataset(txt=root_radar_3 + 'train_2_m7.txt', transform=data_transform)
        dataset_train_6 = MyDataset_train(txt=root_radar_3 + 'train_2_m7.txt', transform=data_transform, num = num[5])



        dataset_test = MyDataset(txt='./data/spect/THREE_RADAR_3000/' + 'test_m7.txt', transform=data_transform)
        dataset_train = [dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5,
                        dataset_train_6]
        dict_users = sensing_data_dict(dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5,
                                       dataset_train_6)


        local_steps = args.local_ep
        p_s = [0.0114042,  0.00234581, 0.02999672, 0.02999662, 0.02979823, 0.02999697,
               0.02999684, 0.02728937, 0.00758947, 0.01024858]



        batch_size = 128
        
        # logger.info('Batch size: {}'.format(batch_size))


        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        for idx in range(6): ################
            local = LocalUpdate(args=args, batch_size=batch_size, dataset=dataset_train[idx], idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), local_steps=local_steps, p_s=p_s[idx])
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)
        

        n_var = 10e-6
        eta = args.eta
        aircomp_noise = np.random.normal(0, n_var/(eta ** 0.5))
        for key in w_glob.keys():
            w_glob[key] = w_glob[key] - aircomp_noise * 0.1
        # logger.info('eta: {}'.format(eta))
        # logger.info('var: {}'.format(args.var))

        

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        logger.info('Epoch: {}'.format(tmp_round))
        logger.info('Train loss: {:.4f}'.format(loss_avg))

        # testing
        net_glob.eval()
        acc_test_1, loss_train_1 = test_img(net_glob, dataset_test, args)
        logger.info("average test acc: {:.2f}%".format(acc_test_1))
        # print("Test on dataset from Radar 2: {:.3f}%, training loss: {:.6f}".format(acc_test_2, loss_train_2))
