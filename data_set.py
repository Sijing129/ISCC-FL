import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random


def gen_txt(dir, train):
    folder_level_list = os.listdir(dir)
    folder_level_list.sort()
    for folder_level in folder_level_list:
        for folder_label in os.listdir(dir + folder_level):
            for file in os.listdir(os.path.join(dir, folder_level, folder_label)):
                name = os.path.join(dir, folder_level, folder_label, file) + ' ' + str(int(folder_label) - 1) + '\n'
                train.write(name)
    train.close()


# def gen_txt_origin(dir, train, test):
#     folder_label_list = os.listdir(dir)
#     folder_label_list.sort()
#     for folder_label in folder_label_list[0:5]:
#         file_list = os.listdir(os.path.join(dir, folder_label))
#         file_list.sort()
#         label_idx = int(folder_label[-1])-1
#         for idx in range(0, 130):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             train.write(name)
#         for idx in range(130, 230):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             test.write(name)
#     train.close()
#     test.close()

def gen_txt_origin(dir, train, test, num_class):
    folder_label_list = os.listdir(dir)
    folder_label_list.sort()
    for folder_label in folder_label_list[0:num_class]:
        file_list = os.listdir(os.path.join(dir, folder_label))
        file_list.sort()
        label_idx = int(folder_label[-1]) - 1
        for idx in range(0, 600):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            train.write(name)
        for idx in range(1200, 1800):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            test.write(name)
    train.close()
    test.close()


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class MyDataset_train(Dataset):
    def __init__(self, txt, transform, num, loader=default_loader):
        super(MyDataset_train, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()

        imgs = []
        for i in range(7):
            content = contents[i * 600: (i + 1) * 600]
            content = random.sample(content, num)
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content]
            # print(imgs)

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    

# class MyDataset_train(Dataset):
#     def __init__(self, txt, transform, num, loader=default_loader):
#         super(MyDataset_train, self).__init__()
#         fh = open(txt, 'r')
#         imgs = []
#         line_count = 0
#         for line in fh:
#             if 0 <= line_count % 600 <= num:
#                 line = line.strip('\n')
#                 words = line.split()
#                 imgs.append((words[0], int(words[1])))
#                 line_count += 1
#             else:
#                 line_count += 1
#         self.imgs = imgs
#         self.transform = transform
#         self.loader = loader
#         self.num = num

#     def __getitem__(self, item):
#         fn, label = self.imgs[item]
#         img = self.loader(fn)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label

#     def __len__(self):
#         return len(self.imgs)
    

class CenDataset(Dataset):
    def __init__(self, data, labels):
        super(CenDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        img = self.data[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    gen_txt_flag = True
    if gen_txt_flag:
        # # Data generated at the beginning and 20% accuracy
        # dir_train = './data/spect/radar_1/fig_train/'
        # train = open('./data/spect/radar_1/train.txt', 'w')
        # gen_txt(dir_train, train)
        #
        # dir_test = './data/spect/radar_1/fig_test/'
        # test = open('./data/spect/radar_1/test.txt', 'w')
        # gen_txt(dir_test, test)

        dir = './data/spect/THREE_RADAR_3000/radar_3/'
        train_1 = open(dir + 'train_1_m7.txt', 'w')
        train_2 = open(dir + 'test_m7.txt', 'w')
        num_class = 7
        gen_txt_origin(dir, train_1, train_2, num_class)
