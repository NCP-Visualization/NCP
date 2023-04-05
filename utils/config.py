import sys
import os
import numpy as np


ROOT = os.path.dirname(sys.modules[__name__].__file__)
ROOT = os.path.join(ROOT, '..')


def hex_to_dec(colors):
    res = []
    for color in colors:
        r = int(color[1 : 3], 16)
        g = int(color[3 : 5], 16)
        b = int(color[5 : 7], 16)
        res.append((b, g, r))
    return res


class Config(object):
    def __init__(self):
        self.server_root = ROOT
        self.data_root = os.path.join(ROOT, "data")

        self.metrics = [
            'neighborhood_preservation_degree_1_ring', 
            'neighborhood_preservation_degree_2_ring',
            'similarity_preservation_1_hop_5NN',
            'similarity_preservation_2_hop_15NN',
            'compactness',
            'overlap_ratio',
            'convexity_whole',
            'convexity_by_cluster'
        ]

        self.hex_colors = [
            "#4fa7ff", # 0
            "#ffa953", # 1
            "#55ff99", # 2
            "#ba9b96", # 3
            "#c982ce", # 4
            "#bcbd22", # 5
            "#e377c2", # 6
            "#990099", # 7
            "#17becf", # 8
            "#8c564b", # 9
            "#dc143c", # 10
            "#008000", # 11
            "#0000cd", # 12
            "#ff4500", # 13
            "#ffff00"  # 14
        ]

        self.opencv_colors = hex_to_dec(self.hex_colors)
        self.matplotlib_colors = [(it[2], it[1], it[0]) for it in self.opencv_colors]
        self.matplotlib_colors = np.array(self.matplotlib_colors) / 255

config = Config()
