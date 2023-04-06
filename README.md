# SCP: Similarity-Aware Nonuniform Circle Packing

The source code and the python interfaces for generating similarity-aware nonuniform circle packing layouts.

https://user-images.githubusercontent.com/129270678/230307020-150cc01d-cbd0-4cc2-8aa9-d343b506ef1a.mp4

## Contents

+ **data**: the copy of all datasets and cache in the quantitative experiments. The cache files include the clustering results and projection results.

+ **utils**: some necessary libraries written in C++, need to be compiled into python packages for usage.

+ **video**: a video that introduces the basic idea of **SCP**.

## Quick Start

Please download our source code and install/build the necessary python packages.

```bash
pip install -r requirements.txt
./compile.bat
```

 A simple example code to call our circle packing algorithm with the sample data is in `interface_test.py`:

```python
from interface import SCP
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

    result = SCP('boston_0.1_1', dataset['feature'], dataset['label'],
                dataset['similarity'], dataset['importance'])

    print(len(result['positions']))
    print(len(result['radii']))

```


A simple interface to call our SCP method to generate similarity-aware circle packing layout is in `interface.py`:
```python
def SCP(dataset_name,
        feature,
        label,
        similarity,
        importance,
        attraction=85,
        gravity=85,
        chebyshev=False,
        utopian=True
        ):
    my_solver = Solver()
    my_solver.dataset = dataset_name
    my_solver.raw_data = {
        'dataset_name': dataset_name,
        'feature': feature,
        'label': label,
        'importance': importance,
        'similarity': similarity,
    }

    algorithm_config = {
        'optimization': 'DivideAndConquer',
        'compaction': 'Box2D',
        'attraction': attraction,
        'gravity': gravity,
        'chebyshev': chebyshev,
        'utopian': utopian
    }

    my_solver.set_algorithm('SCP')
    my_solver.set_algorithm_config(algorithm_config)

    my_solver.run()

    result = my_solver.layout
    return result
```
