import os
import numpy as np
from utils.config import config
from NCPPacking import NCPPacking
from D3SimiFrontChain import D3SimiFrontChain
from AEPacking import AEPacking
from Evaluator import Evaluator
import time
import Logger as Logger
from NCPPacking import computePowerDiagramBruteForce


class Solver(object):
    def __init__(self):
        self.dataset = None
        self.raw_data = None
        # Results
        self.packing_links = None
        self.positions = None
        self.cluster_positions = None
        self.radii = None
        self.pre_layout_positions = None
        self.layout = {
            'positions': [],
            'radii': []
        }
        self.algorithm = ""
        self.algorithm_config = {}
        self.packer = None
        self.packer_config = {}
        self.cost = -1
        self.time = -1
        self.debug = True
        self.logger = None
        self.save_dir = None
        self.run_flag = True

        self.save_log = False

    def update_save_dir(self):
        self.save_dir = './log/' + self.dataset + '/' + str(self.packer) + '/'
        self.save_dir += time.asctime().replace(' ', '_').replace(':', '_') + '/'

    def get_config(self):
        return {
            'metrics': ['time'] + config.metrics
        }

    def set_mode(self, debug):
        self.debug = debug

    def set_dataset(self, dataset_name):
        self.run_flag = True
        self.dataset = dataset_name
        self.raw_data = load_dataset(self.dataset)
        self.packing_links = None
        self.positions = None
        self.radii = None
        self.set_algorithm(self.algorithm, None)
        return {
            'dataset_name': dataset_name,
            'count': len(self.raw_data['importance'])
            # 'count': 2
        }

    def set_algorithm(self, algorithm_name):
        self.run_flag = True
        self.algorithm = algorithm_name
        ppl = None
        if self.algorithm == "NCP":
            ppl = NCPPacking
        elif self.algorithm == 'SimiFrontChain':
            ppl = D3SimiFrontChain
        elif self.algorithm == 'AE':
            ppl = AEPacking
        else:
            pass
        if ppl is not None and self.raw_data is not None:
            self.packer = ppl(self.raw_data)

    def set_algorithm_config(self, algorithm_config):
        self.run_flag = True
        for name in algorithm_config:
            self.packer_config[name] = algorithm_config[name]

    def run(self, evaluate=False):
        if self.run_flag:
            if self.packer:
                if self.packer_config is not None:
                    for name in self.packer_config:
                        if self.packer.config is not None and name in self.packer.config:
                            self.packer.config[name] = self.packer_config[name]
                        else:
                            try:
                                self.packer.extra_params[name] = self.packer_config[name]
                            except:
                                None
                self.update_save_dir()
                self.logger = Logger.Logger(self.save_dir)
                self.packer.set_logger(self.logger)
                if self.debug:
                    self.packer.debug = True
                start_time = time.time()
                positions, radii = self.packer.run()
                end_time = time.time()
                # self.logger.save_time('total time', end_time - start_time)
                self.positions = positions
                self.radii = radii
                if evaluate:
                    metrics = self.evaluate()
                else:
                    metrics = {}
                ret = {
                    'id': str(self.packer),
                    'status': True,
                    'time': end_time - start_time,
                }
                for name in metrics:
                    ret[name] = metrics[name]
                if hasattr(self.packer, 'intermediate_result'):
                    for name in self.packer.intermediate_result:
                        ret[name] = self.packer.intermediate_result[name]
                if self.save_log:
                    self.logger.draw_layout(self.positions, self.radii, self.raw_data['label'])
            self.run_flag = False
        else:
            ret = {
                'id': str(self.packer),
                'status': True,
                'time': 0
            }
        self.layout = {
            'positions': self.positions,
            'radii': self.radii
        }
        return ret

    def evaluate(self):
        evaluator = Evaluator(self.positions, self.radii,
                              self.pre_layout_positions, str(
                                  self.packer), self.dataset, self.raw_data,
                              self.logger, save_log=self.save_log)
        metrics = evaluator.evaluate(config.metrics)
        return metrics

    def save_metrics(self, ret):
        self.logger.save_as_pkl('metrics.pkl', ret)
