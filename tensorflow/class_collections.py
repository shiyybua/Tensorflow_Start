# -*- coding: utf-8 -*

import collections

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