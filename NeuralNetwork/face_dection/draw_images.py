import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import csv
import numpy as np

IMG_PATH = '../../resource/faces/valid_training.csv'
IMAGE_SAVE_PATH = '/home/cai/Documents/images/'


def load_header():
    with open(IMG_PATH, 'r') as csvfile:
        header = csvfile.readline().strip().split(',')
        print header
        return header


def read_img(header):
    with open(IMG_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            image = map(float,row['Image'].strip().split(' '))
            image = np.array(image).reshape([96, 96])
            X = []
            Y = []
            for colunm in header[:-1]:
                if colunm.endswith('_x'):
                    X.append(float(row[colunm]))
                elif colunm.endswith('_y'):
                    Y.append(float(row[colunm]))

            plt.clf()
            plt.imshow(image)
            plt.scatter(X, Y, c='red')
            savefig(IMAGE_SAVE_PATH+'%d.jpg' % i)
            # plt.imsave('/home/cai/Documents/images/%d.jgp' % i, image)
            # plt.show()
            # raw_input()


header = load_header()
read_img(header)
