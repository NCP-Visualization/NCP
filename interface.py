from Solver import Solver


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
