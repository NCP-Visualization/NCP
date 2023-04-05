import utils.D3_simi.D3SimiFrontChain as D3SimiFrontChainPacking
import numpy as np


class D3SimiFrontChain(object):
    def __init__(self, data):
        self.data = data
        self.solver = D3SimiFrontChainPacking.D3SimiFrontChain()
        self.config = None
        self.pre_layout_positions = None
        self.logger = None

    def __str__(self):
        return "SimiFC"

    def run(self):
        importance = self.data['importance']
        similarity = self.data['similarity']
        index = np.argsort(importance)[:: - 1]

        self.solver.set_importance(importance)
        self.solver.set_similarity(similarity)
        self.solver.set_index(index)
        self.solver.layout()

        positions = self.solver.get_pos()
        radii = self.solver.get_r()
        final_positions = positions
        final_radii = radii

        return final_positions, final_radii

    def get_simi_sum(self, positions, radii):
        simi_sum = 0
        link = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                center_dist = ((positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2) ** 0.5
                if abs(center_dist - (radii[i] + radii[j])) < 1e-6 :
                    simi_sum += float(self.data['similarity'][i][j])
                    return simi_sum

    def set_logger(self, logger):
        self.logger = logger