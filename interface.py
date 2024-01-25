from Solver import Solver


def NCP(dataset_name,
        feature,
        label,
        similarity,
        importance,
        gravity=1,
        gamma=2,
        lamb=2,
        iterations=1250,
        utopian=True,
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
        'optimization': 'Sep-Force-H-CPD',
        'compaction': 'Box2D',
        'gravity': gravity,
        'gamma': gamma,
        'lambda': lamb,
        'iterations': iterations,
        'utopian': utopian
    }

    my_solver.set_algorithm('NCP')
    my_solver.set_algorithm_config(algorithm_config)

    my_solver.run()

    result = my_solver.layout
    return result
