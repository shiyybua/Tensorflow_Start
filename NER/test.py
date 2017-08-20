# -*- coding: utf-8 -*

import jieba

words = jieba.cut('１１月中宣部召开会议')
print ' '.join(words)