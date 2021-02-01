# coding: utf-8
import numpy as np
# a = [0, 0, 1, 0, 1]
# b = [0, 0, 1, 0, 1]
#
# # print(not(a==b))
# print([i for i, x in enumerate(a) if x !=0])
a = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
b = [True, False, True]
# b = [[1, 0], [1], [0]]
for i in range(len(a)):
    topic_index = int(b[i])
    a[i, topic_index] = 1
print(a)


