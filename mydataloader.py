# -*-coding:utf-8-*-
import word2vec
from gensim.models import KeyedVectors
import csv
import torch
import numpy as np

def loadvocabulary():
    model = KeyedVectors.load_word2vec_format('./model/snips_phrase2vec.bin')
    print( model.key_to_index)
    # mapping = {'鐪': 0, 'NP': 1, 'VP': 2, 'PP': 3, 'SQ': 4, 'QP': 5, 'ADJP': 6, 'SBARQ': 7, 'ADVP': 8, 'SBAR': 9, 'NP-TMP': 10, 'S': 11, 'FRAG': 12, 'UCP': 13, 'SINV': 14, 'PRN': 15, 'LST': 16, '[SEP]': 17, '[CLS]': 18}
    max_length = 0
    origin_data = []
    with open('./data/res2id.txt', 'r', encoding='utf-8')as f:
        read = f.readlines()
        for info in read:
            info = info.replace('\n', '')
            a = info.split(',')
            temp = [a[0].split(' '), a[1]]
            origin_data.append(temp)
            if len(temp[0]) > max_length:
                max_length = len(temp[0])
    for info in origin_data:
        need_add = max_length - len(info[0])
        if need_add == 0:
            continue
        # for i in range(len(info)):
        #     info[i] = mapping[info[i]]
        for j in range(need_add):
            info[0].append(0)
    text_save_1('./data/train.txt', origin_data)


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def text_save_1(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i][0]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + ',' + str(data[i][1]) + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")
    # print(model.get_id('NP'))


# class TrainLoader:
#
#     def __init__(self):
#         pass
#
#     def get_batch(self):

# loadvocabulary()

def str_to_num(str):
    str = str.split(' ')
    num = []
    for i in str:
        num.append(int(i))
    return num



def train_to_tensor():
    with open('./data/train.txt', 'r', encoding='utf-8')as f:
        batch = []
        ids = []
        # ids.append()
        labels = []
        read = f.readlines()
        count = 0
        for info in read:
            count += 1
            row = info.split(',')
            idnum = str_to_num(row[0])
            id = idnum
            label = int(row[1].replace('\n', ''))
            ids.append(id)
            labels.append(label)

            if count % 64 != 0:
                continue
            else:
                ids = torch.tensor(ids)
                labels = torch.tensor(labels)
                tuple = (ids, labels)
                batch.append(tuple)
                count = 0
                ids = []
                labels = []
    return batch


batch = train_to_tensor()
# batch = np.array(batch)
# a = torch.tensor([])
# batch = torch.tensor(batch)
print(batch)

# str = str_to_num('2 1 0')
# print(str)


