import utils.AE.AEPacking as AEP
from utils.b2d.b2d import Box2DSimulator
from scipy.spatial.distance import cdist
import numpy as np
import os
from utils.config import config
import pickle


class AEPacking(object):
    def __init__(self, data):
        self.data = data
        self.solver = AEP.AESolver()
        self.cluster = self.data['label']
        self.config = {}
        self.extra_params = {}
        # Default params
        force = 85
        self.extra_params['attraction'] = force
        self.extra_params['gravity'] = force
        self.extra_params['iterations'] = 1000
        self.extra_params['alpha_min'] = 0.0009

    def __str__(self):
        return "AE"

    def run(self):
        self.solver.setRadii(self.data['importance'])
        try:
            dist = self.data['distance'].copy()
        except:
            dist = 1 - self.data['similarity']
        dataset = self.data['dataset_name']
        cluster = None
        data_path = os.path.join(config.data_root, 'high_dimensional_clustering/' + dataset + '-kmeans.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            cluster = data['cluster_labels']
        else:
            raise ValueError('Could not find hd clustering')
        for cluster_num in np.unique(cluster):
            selected_indices = np.where(cluster == cluster_num)[0]
            for i in selected_indices:
                for j in selected_indices:
                    if i != j:
                        dist[i, j] = dist[i, j] - 1e-3
        dist -= np.min(dist)
        np.fill_diagonal(dist, 0)
        self.solver.setDist(dist)
        self.solver.layout()
        positions = self.solver.getPositions()
        for pos in positions:
            pos[0] += 0.5
            pos[1] += 0.5
        radii = self.solver.getRadii()
        positions, radii = self.post_process(positions, radii)
        return positions, radii

    def set_logger(self, logger):
        self.logger = logger
    
    def post_process(self, positions, radii):
        N = len(radii)
        preservation = []
        dis_mat = cdist(positions, positions, metric='euclidean')
        np.fill_diagonal(dis_mat, np.inf)
        r_mat = np.tile(radii, N).reshape((N, N))
        r_mat += r_mat.T
        dis_mat -= r_mat
        for i in range(N):
            for j in range(i + 1, N):
                if np.abs(dis_mat[i, j]) < 1e-6:
                    preservation.append((i, j))
        preservation = set(preservation)

        importance = self.data['importance']
        index = importance.argmax()
        for i in range(N):
            radii[i] = radii[i] * (importance[i] / importance[index])


        config_b2d = {
            "size_mag": 1.0,
            "iterations": self.extra_params['iterations'],
            "attraction_mag": self.extra_params['attraction'],
            "density": 0,
            "gravity": self.extra_params['gravity'],
            "bullet": False,
            "time_step": 0.005,
        }

        alpha = 1
        alpha_min = self.extra_params['alpha_min']
        alpha_decay = (1 - (alpha_min ** (1 / config_b2d["iterations"])))
        alpha_target = 0

        attractions = [(i, j, self.data['similarity'][i, j]) for (i, j) in preservation]

        pre_pos = np.array(positions.copy())
        rad = np.array(radii.copy())

        config_b2d["size_mag"] = 1 / np.min(rad)
        config_b2d["gravity"] *= config_b2d["size_mag"]
        config_b2d["attraction_mag"] *= config_b2d["size_mag"]
        optimizer = Box2DSimulator(positions=pre_pos * config_b2d["size_mag"],
                                    radii=rad * config_b2d["size_mag"], size_mag=config_b2d["size_mag"], attractions=attractions,
                                    attraction_magnitude=config_b2d["attraction_mag"], density=config_b2d["density"],
                                    gravity=config_b2d["gravity"], bullet=config_b2d["bullet"],
                                    time_step=config_b2d["time_step"])
        
        for i in range(config_b2d["iterations"]):
            optimizer.clear_velocities()
            optimizer.apply_attractions_and_gravity()
            optimizer.step()
            alpha += alpha_decay * (alpha_target-alpha)
            optimizer.attraction_magnitude = alpha * config_b2d["attraction_mag"]
            optimizer.gravity = alpha * config_b2d["gravity"]
            if alpha <= alpha_min:
                break

        pos = np.array(optimizer.get_positions())
        rad = np.array(optimizer.get_radii())

        resolution = max(np.max(pos[:, 0] + rad) - np.min(pos[:, 0] - rad),
                            np.max(pos[:, 1] + rad) - np.min(pos[:, 1] - rad))
        pos = (pos - (pos - rad.reshape(-1, 1)).min(axis=0)) / resolution
        rad /= resolution

        return pos.tolist(), rad.tolist()