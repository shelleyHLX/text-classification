import _pickle as pkl
import numpy as np
from multiprocessing import Queue

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, datapath, batch_size=128, bucket_sz=1000, shuffle=False, sample_balance=False, id2weight=None):

        with open(datapath, 'rb') as fd:
            data = pkl.load(fd)
            # print(data[555])
            # print('data.keys()')
            # exit(0)
        '''data==> [(labe, doc),]'''
        """
        (9, [
        [17, 29090, 9, 160, 43, 18, 29, 6, 8, 14, 93, 1339, 7], 
        [49, 12, 54, 82, 17018, 6000, 13, 18, 29, 7], 
        [5, 165, 17828, 13207, 12, 9, 61, 225, 474, 6, 8, 17, 135, 49, 12, 9, 160, 10, 6000, 13, 26328, 7], 
        [5, 281, 6, 10, 247, 6, 1164, 51, 106, 14, 12, 38, 2543, 2572, 7], 
        [18, 1212, 11, 5, 867, 10, 5, 29, 7], 
        [2441, 98, 14959, 10, 684, 117, 2410, 11, 99, 61, 659, 18, 29, 211, 2715, 2309, 8, 320, 9, 58, 8, 1153, 1214, 10, 5, 17828, 13207, 7], 
        [45, 172, 115, 2504, 6, 14, 12, 9, 82, 29, 6, 8, 17, 71, 524, 14, 7], 
        [14, 12, 35, 51, 3491, 1919, 10, 247, 21, 19, 9, 5660, 10, 9, 684, 474, 14, 12, 71, 9, 82, 23, 737, 7]])
        (0, [
        [18, 29, 12, 35, 326, 338, 7], 
        [46, 89, 4993, 59, 50, 181, 7], 
        [69, 35, 120, 285, 7], 
        [9, 1331, 1636, 10, 70, 7], 
        [17, 1524, 10225, 2351, 144, 192, 29798, 3573, 11, 425, 8, 120, 18, 7], 
        [639, 1642, 12, 9, 1418, 19, 5, 965, 7],
        [514, 9, 184, 64, 349, 10, 18, 3289, 10, 857, 7], 
        [350, 35, 326, 274, 70, 7]])
        (9, [
        [18, 23, 12, 55, 275, 602, 42, 33, 3626, 210, 2445, 7], 
        [47, 33, 3605, 6, 3725, 9906, 6, 427, 17217, 8, 261, 528, 20, 8576, 6, 13, 98, 797, 261, 9, 3338, 2568, 365, 7], 
        [21, 47, 62, 34, 144, 15, 18, 3338, 341, 5, 221, 22, 109, 7], 
        [17, 27, 34, 1150, 81, 10, 18, 29, 106, 17, 214, 35, 9, 205, 435, 10, 18, 248, 10, 110, 6, 106, 740, 13, 18, 248, 10, 110, 49, 12, 87, 443, 8, 9, 3085, 141, 59, 12, 936, 6, 21, 132, 2903, 573, 7], 
        [17, 27, 878, 112, 58, 18, 29, 27, 7], 
        [96, 10, 45, 106, 47, 93, 34, 8410, 20, 9, 64, 297, 11, 1875, 11, 5, 812, 8, 2147, 137, 19, 653, 19, 695, 7], 
        [49, 27, 126, 766, 871, 137, 8, 14, 71, 27, 1509, 6, 57, 425, 11, 317, 1034, 13, 68, 2273, 6, 14, 131, 102, 1676, 7], 
        [50, 14, 16, 2903, 326, 826, 184, 23, 7]])


        """
        example_num = len(data)  # 67426
        '''shape(example_num) '''
        doc_sz = np.array([len(doc) for _, doc in data], dtype=np.int32) #  67426 8 8 10

        if shuffle:
            self.tidx = np.argsort(doc_sz)
        else:
            self.tidx = np.arange(example_num)

        self.num_example = example_num
        self.shuffle = shuffle
        self.bucket_sz = bucket_sz
        self.batch_sz = batch_size
        self.data = data

        self.sample_balance = sample_balance
        self.id2weight = id2weight

    def __iter__(self):
        if self.bucket_sz < self.batch_sz:
            self.bucket_sz = self.batch_sz
        if self.bucket_sz > self.num_example:
            self.bucket_sz = self.num_example
        self.startpoint = 0
        return self

    def __next__(self):
        if self.startpoint >= self.num_example:
            raise StopIteration

        if self.shuffle:
            bucket_start = np.random.randint(0, self.num_example)
            bucket_end = (bucket_start + self.bucket_sz) % self.num_example
            if bucket_end - bucket_start < self.bucket_sz:
                candidate = np.concatenate([self.tidx[bucket_start:], self.tidx[:bucket_end]])
            else:
                candidate = self.tidx[bucket_start: bucket_end]
            candidate_p = None
            if self.sample_balance and self.id2weight:
                candidate_label = [self.data[c][0] for c in candidate]
                candidate_p = np.array([self.id2weight[l] for l in candidate_label])
                candidate_p = candidate_p/np.sum(candidate_p)
            target_idx = np.random.choice(candidate, size=self.batch_sz, p=candidate_p)
        else:
            target_idx = self.tidx[self.startpoint:self.startpoint+self.batch_sz]
        self.startpoint += self.batch_sz

        labels = []
        data_x = []
        for idx in target_idx:
            l, d = self.data[idx]
            labels.append(l)
            data_x.append(d)
        return labels, data_x


def preparedata(dataset: list, q: Queue, max_snt_num: int, max_wd_num: int, class_freq: dict):
    for labels, data_x in dataset:
        example_weight = np.array([class_freq[i] for i in labels])    #(b_sz)
        data_batch, sNum, wNum = paddata(data_x, max_snt_num=max_snt_num, max_wd_num=max_wd_num)
        labels = np.array(labels)
        q.put((data_batch, labels, sNum, wNum, example_weight))
    q.put(None)


def paddata(data_x: list, max_snt_num: int, max_wd_num: int):
    '''

    :param data_x: (b_sz, snt_num, wd_num)
    :param max_snt_num:
    :param max_wd_num:
    :return:
    '''

    b_sz = len(data_x)

    snt_num = np.array([len(doc) for doc in data_x], dtype=np.int32)
    snt_sz = min(np.max(snt_num), max_snt_num)

    wd_num = [[len(sent) for sent in doc] for doc in data_x]
    wd_sz = min(max(map(max, wd_num)), max_wd_num)

    b = np.zeros(shape=[b_sz, snt_sz, wd_sz], dtype=np.int32)  # == PAD

    sNum = snt_num
    wNum = np.zeros(shape=[b_sz, snt_sz], dtype=np.int32)

    for i, document in enumerate(data_x):
        for j, sentence in enumerate(document):
            if j >= snt_sz:
                continue
            wNum[i, j] = wd_num[i][j]
            for k, word in enumerate(sentence):
                if k >= wd_sz:
                    continue
                b[i, j, k] = word

    return b, sNum, wNum