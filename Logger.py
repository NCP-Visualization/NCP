import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from utils.config import config


class Logger(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.try_dir(self.save_dir)

    def save_as_npy(self, filename, data):
        np.save(self.save_dir + filename, data)

    def save_as_pkl(self, filename, data):
        with open(self.save_dir + filename, 'wb') as f:
            pickle.dump(data, f)

    def try_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def save_as_cv2_fig(self, filename, img):
        cv.imwrite(self.save_dir + filename, img)

    def load_as_cv2_fig(self, filename):
        return cv.imread(self.save_dir + filename)

    def save_time(self, name, time):
        with open(self.save_dir + 'time.txt', 'a') as f:
            f.write(str(name) + ' ' + str(time) + '\n')

    def paint_adapted_neighborhood_preservation(self, method):
        file_name = self.save_dir + 'adapted_neighborhood_preservation_ratio.npy'
        data = np.load(file_name).mean(axis=2)
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        candidate_k = list(range(1, data.shape[1] + 1))
        ratios = np.zeros((3, len(candidate_k)))

        for i in range(1, data.shape[0]):
            data[i] += data[i - 1]
        for index in range(data.shape[0]):
            j = index
            ax[j].plot(candidate_k, data[index, :], label=method, marker='o')
            ax[j].set_title('<= ' + str(index + 1) + ' hop')
            ax[j].set_xlabel('k')
            ax[j].set_ylabel('preservation ratio')
            ax[j].legend()
        plt.savefig(self.save_dir + 'preservation_ratio.png')
        plt.close()

    def draw_layout(self, pos, radii, labels, name='layout.png'):
        img_size = 4096
        img = 255 * np.ones((img_size, img_size, 3), np.uint8)
        colors = config.opencv_colors
        N = len(pos)
        for i in range(N):
            color = colors[labels[i]]
            x = int(pos[i][0] * img_size)
            y = int(pos[i][1] * img_size)
            r = int(radii[i] * img_size)
            cv.circle(img, (x, y), r, color, -1)
            cv.putText(img, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        self.save_as_cv2_fig(name, img)

    def load_layout_fig(self):
        img = self.load_as_cv2_fig('layout.png')
        return img
