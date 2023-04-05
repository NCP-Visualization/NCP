from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import numpy as np
import math
import pickle
import os
import cv2
from utils.Floyd import Floyd
from SCPPacking import computePowerDiagramBruteForce, cellArea
from utils.config import config
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import queue
import shapely.geometry


def cross_product(dir1, dir2):
    return dir1[0] * dir2[1] - dir2[0] * dir1[1]

def dot_product(dir1, dir2):
    return dir1[0] * dir2[0] + dir2[1] * dir1[1]

def get_k_hop(topology_dist):
    # get an (N, N) array k_hop
    # k_hop[i][j] are the nodes whose distance to node i is j
    N = topology_dist.shape[0]
    k_hop = [[[] for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(i):
            dis = topology_dist[i, j]
            if np.isinf(dis):
                continue
            k_hop[i][int(dis)].append(j)
            k_hop[j][int(dis)].append(i)
    # print(k_hop[0][1])
    return k_hop

def get_topo_dist(positions, weights, cluster):
    # get the matrix of topology distance
    N = positions.shape[0]
    topology_dist = np.ones((N,N)) * np.inf
    np.fill_diagonal(topology_dist, 0)
    _, _, pairs, _ = computePowerDiagramBruteForce(positions, weights, clipHull=True)
    # print(len(pairs))
    for (i , j) in pairs:
        if cluster[i] != cluster[j]:
            continue
        topology_dist[i, j] = min(topology_dist[i, j], 1)
        topology_dist[j, i] = min(topology_dist[j, i], 1)
    floyd = Floyd.Floyd()
    floyd.set_n(N)
    floyd.set_d(topology_dist)
    floyd.run()
    topology_dist = floyd.get_d()
    topology_dist = np.array(topology_dist)
    return topology_dist

def get_preservation_ratio(k, kk_hop, N, arg_sorted_similarities, k_hop):
    # return preservation of k & kk_hop
    preservation_count = np.zeros(N)
    
    for i in range(N):
        preservation_count[i] = len(set(arg_sorted_similarities[i, 1:k+1]) & set(k_hop[i][kk_hop]))
    preservation_ratio = preservation_count / k
    return preservation_ratio


class Evaluator(object):
    def __init__(self, positions, radii, pre_layout_positions, algorithm, dataset, data, logger, save_log=True):
        self.positions = positions
        self.radii = radii
        self.pre_layout_positions = pre_layout_positions
        self.data = data
        self.algorithm = algorithm
        self.dataset = dataset
        self.N = len(self.radii)
        self.logger = logger
        self.preservation_ratio = None
        self.save_log = save_log
        self.cluster = None

        self.all_connected_graph = None
        self.compute_all_connected_graph()

        self.overlap_graph = None
        self.compute_overlap_graph()

        self.convex_hull = None
        self.center_hull = None
        self.graph_hull = None

        self.high_dimensional_clustering()
        if self.save_log:
            self.logger.draw_layout(positions, radii, self.cluster, name='eval.png')

        # self.save_adapted_neighborhood_preservation()

    def compute_all_connected_graph(self):
        positions = np.array(self.positions)
        radii = np.array(self.radii)
        weights = radii ** 2
        _, _, pairs, _ = computePowerDiagramBruteForce(positions, weights, clipHull=True)
        links = []
        for it in pairs:
            dis = np.linalg.norm(positions[it[0]] - positions[it[1]])
            if dis / (radii[it[0]] + radii[it[1]]) < 1.2:
                links.append(it)
        self.all_connected_graph = {
            'node': self.positions,
            'link': links,
        }

    def high_dimensional_clustering(self):
        # get the high dimensional clustering result and store it in self.cluster
        data_path = os.path.join(config.data_root, 'high_dimensional_clustering/' + self.dataset + '-kmeans.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.cluster = data['cluster_labels']
            return

        features = self.data['feature']
        best_score = 0
        best_clustering_labels = None
        inertia = []
        for nc in range(2, 10):
            clustering_model = KMeans(n_clusters=nc, random_state=0)
            clustering_model.fit(features)
            clustering_labels = clustering_model.predict(features)
            # score = silhouette_score(features, clustering_labels)
            score = adjusted_rand_score(self.data['label'], clustering_labels)
            inertia.append(np.sqrt(clustering_model.inertia_))
            if score > best_score:
                best_score = score
                best_clustering_labels = clustering_labels

        self.cluster = best_clustering_labels

    def overlapping_space(self):
        # get the overlapping space in the layout
        points = np.array(self.positions)
        edges = self.overlap_graph['link']
        overlapping_pixels = []

        for edge in edges:
            i, j, w = edge
            d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if d < self.radii[i] + self.radii[j]:
                xi = (self.radii[i] ** 2 - self.radii[j] ** 2 + d ** 2) / (2 * d)
                xj = d - xi
                sin_i = np.sqrt(max(0, 1 - (xi / self.radii[i]) ** 2))
                s_fan_i = self.radii[i] ** 2 * math.asin(sin_i)
                s_tri_i = self.radii[i] * xi * sin_i
                s_bow_i = s_fan_i - s_tri_i
                sin_j = np.sqrt(max(0, 1 - (xj / self.radii[j]) ** 2))               
                s_fan_j = self.radii[j] ** 2 * math.asin(sin_j)
                s_tri_j = self.radii[j] * xj * sin_j
                s_bow_j = s_fan_j - s_tri_j
                overlapping_pixel = s_bow_i + s_bow_j
                overlapping_pixels.append(overlapping_pixel)

        overlap_area = np.array(overlapping_pixels).sum()
        return overlap_area

    def compactness(self):
        # get the compactness of the layout
        radii = np.array(self.radii)

        envelope_area = self.envelope_area()
        if envelope_area is None:
            return None

        overlap_area = self.overlapping_space()

        circle_area = np.sum(np.pi * radii * radii)
        # print(envelope_area, overlap_area, circle_area)

        return (circle_area - overlap_area) / envelope_area
    
    def opencv_compactness(self):
        # get the approximate compactness by opencv
        radii = np.array(self.radii)
        positions = np.array(self.positions)

        img_size = 10000
        img = 255 * np.ones((img_size, img_size, 3), np.uint8)
        N = len(positions)
        for i in range(N):
            color = (0, 0, 0)
            x = int(positions[i][0] * img_size)
            y = int(positions[i][1] * img_size)
            r = int(radii[i] * img_size) + 1
            cv2.circle(img, (x, y), r, color, -1)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 255 - 1, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 2)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)  

        if self.save_log:
            self.logger.save_as_cv2_fig('black.png', img)

        envelope_area = cv2.contourArea(contours[0]) / img_size / img_size

        overlap_area = self.overlapping_space()
        
        circle_area = np.sum(np.pi * radii * radii)
        # print(envelope_area, overlap_area, circle_area)

        return (circle_area - overlap_area) / envelope_area

    def overlap_ratio(self):
        # get the overlap ratio of the layout
        envelope_area = self.envelope_area()
        if envelope_area is None:
            return None

        overlap_area = self.overlapping_space()

        return overlap_area / envelope_area

    def convexity_whole(self):
        envelope_area = self.envelope_area()
        if envelope_area is None:
            return None
        convexhull_area = self.convexhull_area()
        return envelope_area / convexhull_area

    def convexity_by_cluster(self):
        convexity_by_cluster = []
        for cluster in range(len(np.unique(self.cluster))):
            selected_indices = np.where(self.cluster == cluster)[0]
            if self.save_log:
                positions = np.array(self.positions)[selected_indices]
                radii = np.array(self.radii)[selected_indices]
                labels = self.data['label'][selected_indices]
                self.logger.draw_layout(positions, radii, labels, name='cluster' + str(cluster) + '.png')
            groups = self.get_connected_components(selected_indices, cluster)
            groups_envelope_area = 0
            groups_convexhull_area = 0
            for group in groups:
                if self.envelope_area(group) is None:
                    return None
                groups_envelope_area += self.envelope_area(group)
                groups_convexhull_area += self.convexhull_area(group)
            convexity_by_cluster.append(groups_envelope_area / groups_convexhull_area)
        return np.mean(convexity_by_cluster)

    def neighborhood_preservation_degree_1_ring(self):
        N = len(self.radii)
        similarities = self.data['similarity']
        arg_sorted_similarities = np.argsort(-similarities, axis=1)
        positions = np.array(self.positions)
        radii = np.array(self.radii)
        weights = radii**2
        k_hop = get_k_hop(get_topo_dist(positions, weights, self.cluster))

        N_cluster = len(np.unique(self.cluster))
        NP1_by_cluster = [[] for _ in range(N_cluster)]
        NP_1 = 0
        for i in range(N):
            # symbols are the same as the paper
            # Force-directed graph layouts revisited: a new forcebased on the t-Distribution
            NG_i_1 = k_hop[i][1]
            ki = len(NG_i_1)
            NL_xi_ki = list(arg_sorted_similarities[i][1 : ki + 1])

            intersection_set = [it for it in NG_i_1 if it in NL_xi_ki]
            union_set = list(set(NG_i_1 + NL_xi_ki))

            if len(union_set) == 0:
                continue
            NP_1 += len(intersection_set) / len(union_set)
            NP1_by_cluster[self.cluster[i]].append(len(intersection_set) / len(union_set))

        NP_1 /= N
        for i in range(N_cluster):
            # print(np.array(NP1_by_cluster[i]).mean())
            pass
        return NP_1

    def neighborhood_preservation_degree_2_ring(self):
        N = len(self.radii)
        similarities = self.data['similarity']
        arg_sorted_similarities = np.argsort(-similarities, axis=1)
        positions = np.array(self.positions)
        radii = np.array(self.radii)
        weights = radii**2
        k_hop = get_k_hop(get_topo_dist(positions, weights, self.cluster))

        NP_2 = 0
        for i in range(N):
            # symbols are the same as the paper
            # Force-directed graph layouts revisited: a new forcebased on the t-Distribution
            NG_i_2 = k_hop[i][1] + k_hop[i][2]
            ki = len(NG_i_2)
            NL_xi_ki = list(arg_sorted_similarities[i][1 : ki + 1])

            intersection_set = [it for it in NG_i_2 if it in NL_xi_ki]
            union_set = list(set(NG_i_2 + NL_xi_ki))

            if len(union_set) == 0:
                continue
            NP_2 += len(intersection_set) / len(union_set)

        NP_2 /= N
        return NP_2

    def save_adapted_neighborhood_preservation(self):
        # save the array of preservation
        N = len(self.radii)
        positions = np.array(self.positions)
        radii = np.array(self.radii)
        # print(positions)
        # print(radii)
        weights = radii ** 2

        similarities = self.data['similarity']
        arg_sorted_similarities = np.argsort(-similarities, axis=1)

        MAX_K_HOP = 3
        MAX_K_NN = 25
        candidate_k = [i for i in range(1, MAX_K_NN + 1)]
        self.preservation_ratio = np.zeros((MAX_K_HOP, MAX_K_NN, N))

        k_hop = get_k_hop(get_topo_dist(positions, weights, self.cluster))
        for kk_hop in range(MAX_K_HOP):
            for k in candidate_k:
                self.preservation_ratio[kk_hop, k-1] = get_preservation_ratio(k, kk_hop+1, N, arg_sorted_similarities, k_hop)

        if self.save_log:
            self.logger.save_as_npy('adapted_neighborhood_preservation_ratio.npy', self.preservation_ratio)
            self.logger.paint_adapted_neighborhood_preservation(self.algorithm)

    def similarity_preservation_1_hop_5NN(self):
        data = self.preservation_ratio.mean(axis=2)
        return data[0][4]

    def similarity_preservation_2_hop_15NN(self):
        data = self.preservation_ratio.mean(axis=2)
        return data[0][14] + data[1][14]

    def compute_overlap_graph(self):
        # compute overlap graph for the overlapping computation
        node = self.positions
        link = []
        for (i, j) in self.all_connected_graph['link']:
                center_dist = cdist([self.positions[i]], [self.positions[j]], metric='euclidean')
                target_dist = self.radii[i] + self.radii[j]
                if center_dist - target_dist <= 1e-6:
                    link.append(
                        [i, j, float(self.data['similarity'][i][j])])
        self.overlap_graph = {
            'node': node,
            'link': link
        }

    def compute_connected_graph(self, selected_indices):
        # compute overlap graph for the overlapping computation
        node = self.positions
        link = []
        dist = cdist(node, node, metric='euclidean')
        error = 1e-2 * np.array(self.radii).mean()
        for it in self.all_connected_graph['link']:
            if it[0] in selected_indices and it[1] in selected_indices:
                i = it[0]
                j = it[1]
                center_dist = dist[i, j]
                target_dist = self.radii[i] + self.radii[j]
                if center_dist - target_dist <= error:
                    link.append(it)

        connected_graph = {
            'node': node,
            'link': link
        }
        return connected_graph

    def evaluate(self, metric_names):
        ret = {}
        for name in metric_names:
            # print(name)
            ret[name] = getattr(self, name)()
            # print(name, ret[name])
        return ret

    def get_connected_components(self, selected_indices, cluster_num):
        positions = np.array(self.positions)
        radii = np.array(self.radii)
        connected_graph = self.compute_connected_graph(selected_indices)

        positions = positions[selected_indices]
        radii = radii[selected_indices]

        selected_indices = list(selected_indices)

        N = len(radii)

        if N == 0:
            return []

        q = queue.Queue()
        res = []
        visited = np.zeros(N)

        while visited.sum() < N:
            start = -1
            current_group = []
            
            for i in range(N):
                if visited[i] == 0:
                    start = selected_indices[i]
                    visited[i] = 1
                    break

            if start != -1:
                q.put(start)
            
            while not q.empty():
                current = q.get()
                current_group.append(current)
                current_neighbor = []
                for link in connected_graph['link']:
                    if link[0] == current and link[1] in selected_indices and visited[selected_indices.index(link[1])] == 0:
                        current_neighbor.append(link[1])
                    elif link[1] == current and link[0] in selected_indices and visited[selected_indices.index(link[0])] == 0:
                        current_neighbor.append(link[0])

                for neighbor in current_neighbor:
                    visited[selected_indices.index(neighbor)] = 1
                    q.put(neighbor)

            res.append(current_group)
        
        if self.save_log:
            for i, group in enumerate(res):            
                self.logger.draw_layout(np.array(self.positions)[group], np.array(self.radii)[group], np.array(self.data['label'])[group], name='cluster' + str(cluster_num) + '_component' + str(i) + '.png')

        return res

    def compute_neighbor_graph(self, positions, radii):
        node = positions
        link = []
        dist = cdist(positions, positions, metric='euclidean')
        error = 1e-2 * np.array(self.radii).mean()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                center_dist = dist[i, j]
                target_dist = radii[i] + radii[j]
                if center_dist - target_dist <= error:
                    link.append([i, j])
        graph = {
            'node': node,
            'link': link
        }
        return graph

    def compute_boundary_circle(self, selected_indices=None):
        # compute the boundary circle for the envelope area
        if selected_indices is None:
            positions = np.array(self.positions)
            radii = np.array(self.radii)
            labels = self.data['label']
        else:
            positions = np.array(self.positions)[selected_indices]
            radii = np.array(self.radii)[selected_indices]
            labels = self.data['label'][selected_indices]

        N = len(radii)
        bound = []

        connected_graph = self.compute_neighbor_graph(positions, radii)

        count_neighbors = [[] for _ in range(N)]
        for it in connected_graph['link']:
            count_neighbors[it[0]].append(it[1])
            count_neighbors[it[1]].append(it[0])

        single_points = []
        while True:
            count_copy = count_neighbors.copy()
            old_single_points_count = len(single_points)
            for i in range(N):
                if len(count_copy[i]) == 1:
                    single_points.append(i)
                    count_copy[i] = []
                    for j in range(N):
                        if i in count_copy[j]:
                            count_copy[j].remove(i)
                elif len(count_copy[i]) == 0:
                    if i not in single_points:
                        single_points.append(i)

            count_neighbors = count_copy
            
            if len(single_points) == old_single_points_count:
                break

        bound_start = -1
        left_up = []
        pos_left = 1e8
        for i in range(len(positions)):
            if i in single_points:
                continue
            it = positions[i]
            if it[0] < pos_left:
                left_up.clear()
                left_up.append(i)
                pos_left = it[0]
            elif it[0] == pos_left:
                left_up.append(i)
    
        if len(left_up) == 1:
            bound_start = left_up[0]
        elif len(left_up) > 1:
            pos_up = 0
            for it in left_up:
                if positions[it][1] > pos_up:
                    bound_start = it
                    pos_up = positions[it][1]
        else:
            return [], single_points

        previous = bound_start
        current = bound_start
        bound.append(current)
        possible = np.ones(N)
        for it in single_points:
            possible[it] = 0
        center_of_gravity = []
        for i in range(N):
            if possible[i]:
                center_of_gravity.append(positions[i])
        center_of_gravity = np.array(center_of_gravity).mean(axis=0)

        times = 0

        while True:
            current_pos = positions[current]
            neighbors = []
            for link in connected_graph['link']:
                if link[0] == current and link[1] not in single_points:
                    neighbors.append(link[1])
                elif link[1] == current and link[0] not in single_points:
                    neighbors.append(link[0])
            neighbors = list(set(neighbors))
            if current != bound_start:
                previous_pos = positions[previous]
                max_angle = np.pi * 2
                candidate = -1

                for neighbor in neighbors:
                    if neighbor == previous or possible[neighbor] == 0:
                        continue
                    neighbor_pos = positions[neighbor]
                    x = previous_pos - current_pos
                    y = neighbor_pos - current_pos
                    cross = cross_product(x, y)
                    # sin = np.abs(cross) / np.linalg.norm(x) / np.linalg.norm(y)
                    dot = dot_product(x, y)
                    cos = dot / np.linalg.norm(x) / np.linalg.norm(y)
                    # angle = 0
                    if cos < -1:
                        cos = -1
                    elif cos > 1:
                        cos = 1
                    if cross < 0:
                        angle = np.pi * 2 - np.arccos(cos)
                    else:
                        angle = np.arccos(cos)
                    if angle < max_angle:
                        candidate = neighbor
                        max_angle = angle
                # print(previous, current)

                if candidate == -1:
                    # find no neighbor and go back
                    possible[current] = 0
                    single_points.append(current)
                    current = previous
                    
                    if previous != bound_start:
                        previous = bound[bound.index(current) - 1]
                    bound = bound[: -1]
                    continue

                if candidate == bound_start:
                    break
                
                bound.append(candidate)
                previous = current
                current = candidate

            else:
                previous_pos = current_pos.copy()
                previous_pos[1] = previous_pos[1] + 1
                max_angle = np.pi * 2
                candidate = -1

                for neighbor in neighbors:
                    if neighbor == previous or possible[neighbor] == 0:
                        continue
                    neighbor_pos = positions[neighbor]
                    x = previous_pos - current_pos
                    y = neighbor_pos - current_pos
                    cross = cross_product(x, y)
                    # sin = np.abs(cross) / np.linalg.norm(x) / np.linalg.norm(y)
                    dot = dot_product(x, y)
                    cos = dot / np.linalg.norm(x) / np.linalg.norm(y)
                    # angle = 0
                    if cos < -1:
                        cos = -1
                    elif cos > 1:
                        cos = 1
                    if cross < 0:
                        angle = np.pi * 2 - np.arccos(cos)
                    else:
                        angle = np.arccos(cos)
                    if angle < max_angle:
                        candidate = neighbor
                        max_angle = angle
                bound.append(candidate)
                current = candidate

            times += 1
            if times > 2000:
                return None, None

        single_points_copy = single_points.copy()
        polygon = shapely.geometry.Polygon(positions[bound])
        for i in range(len(single_points_copy)):
            point = shapely.geometry.Point(positions[single_points_copy[i]][0], positions[single_points_copy[i]][1])
            if polygon.intersects(point):
                single_points.remove(single_points_copy[i])

        if self.save_log:
            img_size = 4096
            img = 255 * np.ones((img_size, img_size, 3), np.uint8)
            colors = config.opencv_colors
            for i in bound:
                color = colors[labels[i]]
                x = int(positions[i][0] * img_size)
                y = int(positions[i][1] * img_size)
                r = int(radii[i] * img_size)
                cv2.circle(img, (x, y), r, color, -1)
            # self.logger.save_as_cv2_fig('bound.png', img)
            # cv2.imwrite('bound.png', img)
        # print(bound)

        return bound, single_points

    def envelope_area(self, selected_indices=None):
        # the area enclosed in the envelope

        if selected_indices is None:
            positions = np.array(self.positions)
            radii = np.array(self.radii)
        else:
            positions = np.array(self.positions)[selected_indices]
            radii = np.array(self.radii)[selected_indices]

        if len(radii) < 1:
            bound = []
            single_points = []
        else:
            bound, single_points = self.compute_boundary_circle(selected_indices)
        
        if bound is None:
            return None
        
        bound_pos = [positions[i] for i in bound]
        envelope_area = cellArea(bound_pos)
        if selected_indices is not None:
            if len(bound_pos) > 3:
                polygon = shapely.geometry.Polygon(bound_pos)
                label = self.cluster[selected_indices][0]
                for i in range(len(self.radii)):
                    if self.cluster[i] != label:
                        point = shapely.geometry.Point(self.positions[i][0], self.positions[i][1])
                        if polygon.intersects(point):
                            envelope_area -= np.pi * self.radii[i] * self.radii[i]
        for point in single_points:
            envelope_area += np.pi * radii[point] * radii[point]
        # area of the polygon
        
        for id, index in enumerate(bound):
            previous = 0
            current = index
            next = 0
            if id == 0:
                previous = bound[-1]
            else:
                previous = bound[id - 1]
            if id == len(bound) - 1:
                next = bound[0]
            else:
                next = bound[id + 1]
            current_pos = positions[current]
            previous_pos = positions[previous]
            next_pos = positions[next]

            x = previous_pos - current_pos
            y = next_pos - current_pos
            cross = cross_product(x, y)
            dot = dot_product(x, y)
            cos = dot / np.linalg.norm(x) / np.linalg.norm(y)
            angle = 0
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
            if cross < 0:
                angle = np.pi * 2 - np.arccos(cos)
            else:
                angle = np.arccos(cos)

            # print(angle * radii[current] / 2)
            
            envelope_area += angle * radii[current] * radii[current] / 2
            # 1/2 * theta * r * r
            # area of the fan
            # print(previous, current, next, angle/np.pi*180)

        return envelope_area 

    def occupying_space(self):
        return self.envelope_area() - self.overlapping_space()

    def convexhull_area(self, selected_indices=None):
        if selected_indices is None:
            positions = np.array(self.positions)
            radii = np.array(self.radii)
        else:
            positions = np.array(self.positions)[selected_indices]
            radii = np.array(self.radii)[selected_indices]
        N = len(radii)
        APPROX_NUM = 50
        approx_points = []
        for i in range(N):
            pos = positions[i]
            r = radii[i]
            for i in range(APPROX_NUM):
                approx_points.append(
                    [pos[0] + r * np.cos(np.pi * 2 * i / APPROX_NUM), pos[1] + r * np.sin(np.pi * 2 * i / APPROX_NUM)])
        convex_hull = ConvexHull(approx_points)
        # get the area of the convex_hull
        return convex_hull.volume
