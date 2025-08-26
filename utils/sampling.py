
# def sensing_data_dict(dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6):
#     all_idx_1 = [i for i in range(len(dataset_1))]
#     all_idx_2 = [i for i in range(len(dataset_2))]
#     all_idx_3 = [i for i in range(len(dataset_3))]
#     all_idx_4 = [i for i in range(len(dataset_4))]
#     all_idx_5 = [i for i in range(len(dataset_5))]
#     all_idx_6 = [i for i in range(len(dataset_6))]
#     dict_users = {0: set(all_idx_1), 1: set(all_idx_2), 2: set(all_idx_3), 3: set(all_idx_4), 4: set(all_idx_5),
#                   5: set(all_idx_6)}
#     print(len(dataset_1))
#     return dict_users


# def sensing_data_dict(dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8):
#     all_idx_1 = [i for i in range(len(dataset_1))]
#     all_idx_2 = [i for i in range(len(dataset_2))]
#     all_idx_3 = [i for i in range(len(dataset_3))]
#     all_idx_4 = [i for i in range(len(dataset_4))]
#     all_idx_5 = [i for i in range(len(dataset_5))]
#     all_idx_6 = [i for i in range(len(dataset_6))]
#     all_idx_7 = [i for i in range(len(dataset_7))]
#     all_idx_8 = [i for i in range(len(dataset_8))]
#     dict_users = {0: set(all_idx_1), 1: set(all_idx_2), 2: set(all_idx_3), 3: set(all_idx_4), 4: set(all_idx_5),
#                   5: set(all_idx_6), 6: set(all_idx_7), 7: set(all_idx_8)}
#     print(len(dataset_1))
#     return dict_users


def sensing_data_dict(*datasets):
    dict_users = {i: set(range(len(dataset))) for i, dataset in enumerate(datasets)}
    print(len(datasets[0]))  # 假设至少有一个数据集，打印第一个数据集的长度
    return dict_users