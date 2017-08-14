import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


IMG_PATH = '../../resource/faces/test.csv'

def read_img():
    with open(IMG_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            image = map(float,row['Image'].strip().split(' '))
            image = np.array(image).reshape([96, 96])

            plt.imshow(image)
            plt.show()
            raw_input()

read_img()