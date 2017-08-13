import csv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = '../../resource/faces/training.csv'


def load_header():
    with open(IMG_PATH, 'r') as csvfile:
        header = csvfile.readline().strip().split(',')
        return header


def read_img():
    '''
    image : 96x96
    :return:
    '''

    with open(IMG_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # image = map(float,row['Image'].strip().split(' '))
            # image = np.array(image).reshape([96, 96]) / 255
            # row['Image'] = image
            yield row


def produce_valiad_data():
    with open('../../resource/faces/valid_training.csv', 'w') as file:
        header = load_header()
        file.write(','.join(header) + '\n')
        for element in read_img():
            traning_data = []
            for colunm in header[:-1]:
                if element[colunm].strip() == '':
                    break
                traning_data.append(element[colunm])

            traning_data.append(element[header[-1]])
            if len(header) == len(traning_data):
                file.write(','.join(traning_data) + '\n')

            # plt.imshow(element['Image'])
            # plt.show()
            # raw_input()
produce_valiad_data()
