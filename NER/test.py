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

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass

a = BatchedInput(
      initializer=1,
      source=2,
      target_input=3,
      target_output=4,
      source_sequence_length=5,
      target_sequence_length=6)

print a.initializer