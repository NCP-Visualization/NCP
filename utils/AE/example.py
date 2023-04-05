import Packing
import math
import random
import numpy as np

WIDTH = HEIGHT = 1000
PADDING_CLUSTER = 55
RADIUS = 5
PADDING_POINT = 2
REP_WIDTH, REP_HEIGHT = 60, 30

if __name__ == "__main__":
    # Generate random dist matrices
    np.random.seed(0)
    CLUSTER_COUNT = 10
    SAMPLE_NUM = 2000
    labels = np.random.randint(CLUSTER_COUNT, size=2000)
    cluster_set = [[] for _ in range(CLUSTER_COUNT)]
    for i, lb in enumerate(labels):
        cluster_set[lb].append(i)
    dist_each_cluster = []
    for i in range(CLUSTER_COUNT):
        X = np.random.random_integers(0, 50, size=(len(cluster_set[i]), len(cluster_set[i])))
        d = (X + X.T) / 2
        row, col = np.diag_indices_from(d)
        d[row, col] = 0 
        dist_each_cluster.append(d)

    X = np.random.random_integers(0, 200, size=(CLUSTER_COUNT, CLUSTER_COUNT))
    center_dist = (X + X.T) / 2
    row, col = np.diag_indices_from(center_dist)
    center_dist[row, col] = 0 

    # Initialize the solver
    solver = Packing.Solver()
    # Configurate the screen
    solver.setLayoutSize(WIDTH, HEIGHT)
    solver.setLayoutConfig(RADIUS, PADDING_POINT, PADDING_CLUSTER)
    solver.setRepresentativeConfig(REP_WIDTH, REP_HEIGHT)
    
    # Init cluster data and set the layout parameters
    solver.init(len(cluster_set), labels)
    solver.setClustersDist(dist_each_cluster)
    solver.setLink(center_dist, 0.6, 1.0)
    solver.setCollision(0.5)

    # Set the previous positions for the current instances, [-1, -1] for no constraints
    anchors = np.ones((len(labels), 2)) * (-1)
    solver.setStability(anchors, 1.0)

    # Select representatives
    foc = []
    np.random.seed(100)
    for cid in range(len(cluster_set)):
        clus = cluster_set[cid]
        clus_len = len(clus)
        rep_num = 5
        res = random.sample(range(clus_len), rep_num)
        foc.append(res)
    solver.setFocuses(foc)
    solver.setRepresentatives(foc)

    # Compute layout positions
    offsets = solver.layoutClusters()
    cluster_centers = solver.layoutCenters(300)
    # Get layout positions
    rep_pos = solver.layoutRepresentatives()
    positions = solver.getLayoutPositions()

    print('circle packing', len(positions))
    print('structure glyphs', len(rep_pos))