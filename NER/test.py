# -*- coding: utf-8 -*

import numpy as np
import collections

# bucket = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,100),(100,999)]
# data = [0] * len(bucket)
# arr = np.load(open('temp', 'r'))
#
# for e in arr:
#     for index, (low, high) in enumerate(bucket):
#         if low <= e < high:
#             data[index] += 1
#             break
#
# print data
# print sum(data)
# print len(arr)
# unknown = np.random.normal(size=(300))
# print unknown
embeddings_size = 300
unknown = np.random.normal(size=(embeddings_size))
padding = np.random.normal(size=(embeddings_size))
np.save(open('unknown_embedding', 'w'), unknown)
np.save(open('padding_embedding', 'w'), padding)