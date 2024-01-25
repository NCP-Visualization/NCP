from interface import NCP
import os
import numpy as np


def load_dataset(dataset_name):
    data_keys = ['importance', 'similarity', 'distance', 'feature', 'label']
    data_path = os.path.join('data/{}.npz'.format(dataset_name))
    data = np.load(data_path, allow_pickle=True)
    ret = {}
    for key in data_keys:
        if key in data:
            ret[key] = data[key]
            if key in ['distance', 'similarity']:
                ret[key] /= np.max(data[key])
        else:
            ret[key] = None
    ret['dataset_name'] = dataset_name
    return ret


if __name__ == "__main__":
    dataset = load_dataset('boston_0.1_1')

    result = NCP('boston_0.1_1', dataset['feature'], dataset['label'],
                dataset['similarity'], dataset['importance'])

    print(len(result['positions']))
    print(len(result['radii']))
