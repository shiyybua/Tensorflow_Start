import csv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = '../../resource/faces/training.csv'

def read_img():
    '''
    image : 96x96
    :return:
    '''
    with open(IMG_PATH, 'r') as csvfile:
        header = csvfile.readline().strip().split(',')
        print header

    with open(IMG_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image = map(float,row['Image'].strip().split(' '))
            image = np.array(image).reshape([96, 96]) / 255
            row['Image'] = image
            yield row


for x in read_img():
    plt.imshow(x['Image'])
    plt.show()
    break