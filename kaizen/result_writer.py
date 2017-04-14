import logging
import datetime
import os
import pandas as pd


class CSVPerPipelineResultWriter:

    def __init__(self, result_iterator, results_dir, prefix='results', cache_limit=10):
        self.logger = logging.getLogger(__name__ + ':CSVPerPipelineResultWriter')
        self.results = result_iterator
        self.cache = {}
        self.results_dir = os.path.join(results_dir, "{}_{}".format(prefix, datetime.datetime.utcnow()).replace(' ', '_'))
        self.prefix = prefix
        self.cache_limit = cache_limit
        self.create_result_dir()

    def create_result_dir(self):

        if not os.path.exists(self.results_dir):
            self.logger.info('Creating results dir')
            os.mkdir(self.results_dir, mode=0o755)
        else:
            self.logger.error('Results path already existed')
            raise RuntimeError('Results dir already exists')

    def flush(self, force=False):

        for pipe, res in self.cache.items():

            if len(res) >= self.cache_limit or force:
                header = True
                file_path = os.path.join(self.results_dir, "{}_{}.csv".format(self.prefix, pipe))

                self.logger.info('Flushing results to file {}'.format(file_path))

                if os.path.exists(file_path):
                    header = False

                with(open(file_path, "a+")) as f:
                    df = pd.DataFrame(res)
                    df.to_csv(f, header=header, index=False)

                self.cache[pipe] = []

    def write(self):

        for result in self.results:
            pipe = result['pipeline']
            self.cache.setdefault(pipe, []).append(result)
            self.flush()

        self.flush(force=True)
