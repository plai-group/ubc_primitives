import os
import sys
import time
import json
import signal
import datetime
import subprocess
import pandas as pd
from os.path import split
# D3M
from d3m.metadata.problem import PerformanceMetric
from d3m.utils import silence
# GRPC client
from d3m_ta2s_eval.client_eval import D3MClient

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Docker images
TA2_DOCKER_IMAGES = {'NYU': 'registry.gitlab.com/vida-nyu/d3m/ta2:latest',
                     'CMU': 'registry.datadrivendiscovery.org/sheath/cmu-ta2:latest',
                     'SRI': 'registry.gitlab.com/daraghhartnett/autoflow:latest',
                     'TAMU': 'dmartinez05/tamuta2:latest'}

IGNORE_SUMMARY_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                             'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                             'd3m.primitives.data_transformation.denormalize.Common',
                             'd3m.primitives.data_transformation.column_parser.Common'}


class AutoML:
    def __init__(self, output_folder, local_dir, base_dir, dataset, dataset_dir, ta2_id='NYU'):
        if ta2_id not in TA2_DOCKER_IMAGES:
            raise ValueError('Unknown "%s" TA2, you should choose among: [%s]' % (ta2_id, ', '.join(TA2_DOCKER_IMAGES)))

        self.ta2           = None
        self.ta3           = None
        self.ta2_id        = ta2_id
        self.base_dir      = base_dir
        self.local_dir     = local_dir
        self.pipelines     = {}
        self.dataset       = dataset
        self.dataset_dir   = dataset_dir
        self.output_folder = output_folder


    def search_pipelines(self, save_pipeline, save_file, dataset, dataset_doc_path, problem_doc_path, time_bound, target=None, metric=None, task_keywords=None):
        logger.info('Search Solutions...')

        # self.start_ta2()
        search_id = None
        leaderboard  = None
        dataset_in_container = os.path.join(self.base_dir, 'datasets', dataset,\
                                            'TRAIN/dataset_TRAIN/datasetDoc.json')

        signal.signal(signal.SIGALRM, lambda signum, frame: self.ta3.do_stop_search(search_id))
        signal.alarm(time_bound * 60)

        start_time = datetime.datetime.utcnow()

        # Search
        # generator object D3MClient.do_search
        pipelines = self.ta3.do_search(dataset_in_container, problem_doc_path, ta2_id=self.ta2_id, time_bound=time_bound)

        for pipeline in pipelines:
            print('pipeline: ', pipeline)
            end_time = datetime.datetime.utcnow()
            try:
                pipeline_json = self.ta3.do_describe(pipeline['id'])
            except Exception as e:
                logger.error('Decoding pipeline id=%s', pipeline['id'])
                logger.error('Error due to: '+ str(e))
                continue
            # Summary
            summary_pipeline = self.get_summary_pipeline(pipeline_json)
            pipeline['json_representation'] = pipeline_json
            pipeline['summary'] = summary_pipeline
            pipeline['found_time'] = end_time.isoformat() + 'Z'
            duration = str(end_time - start_time)
            try:
                score_data = self.ta3.do_score(pipeline['id'], dataset_in_container, problem_doc_path, ta2_id=self.ta2_id)
            except Exception as e:
                logger.error('Scoring pipeline id=%s', pipeline['id'])
                logger.error('Error due to: '+ str(e))
                continue
            pipeline['score'] = score_data['score']
            pipeline['normalized_score'] = score_data['normalized_score']
            pipeline['metric'] = score_data['metric']
            logger.info('Found pipeline, id=%s, %s=%s, time=%s' % (pipeline['id'], pipeline['metric'], pipeline['score'], duration))
            self.pipelines[pipeline['id']] = pipeline
            search_id = pipeline['search_id']

        # Return a dataframe of scored pipelines
        if len(self.pipelines) > 0:
            leaderboard = []
            sorted_pipelines = sorted(self.pipelines.values(), key=lambda x: x['normalized_score'], reverse=True)
            metric = sorted_pipelines[0]['metric']
            for position, pipeline_data in enumerate(sorted_pipelines, 1):
                leaderboard.append([position, pipeline_data['id'], pipeline_data['summary'],  pipeline_data['score']])
                with open(os.path.join(save_pipeline, '%s.json' % pipeline_data['id']), 'w') as fout:
                    json.dump(self.pipelines[pipeline_data['id']]['json_representation'], fout, indent=2)
                fout.close()
            # Convert to a DataFrame
            leaderboard = pd.DataFrame(leaderboard, columns=['ranking', 'id', 'summary', metric])
        else:
            leaderboard = 'Not Found!'

        # saving the dataframe
        if isinstance(leaderboard, pd.DataFrame):
            leaderboard.to_csv(save_file, index=False)
        else:
            logger.info('No solutions found!')


    def train(self, dataset, solution_id):
        logger.info('Training model...')

        dataset_in_container = os.path.join(self.base_dir, 'datasets', dataset,\
                                            'TRAIN/dataset_TRAIN/datasetDoc.json')

        if solution_id not in self.pipelines:
            logger.error('Pipeline id=%s does not exist' % solution_id)
            return None

        fitted_solution_id = self.ta3.do_train(solution_id, dataset_in_container)

        return fitted_solution_id


    def test(self, dataset, fitted_solution_id):
        logger.info('Testing model...')

        dataset_in_container = os.path.join(self.base_dir, 'datasets', dataset,\
                                            'TRAIN/dataset_TRAIN/datasetDoc.json')

        predictions_path_in_container = self.ta3.do_test(fitted_solution_id, dataset_in_container)
        if not predictions_path_in_container.startswith('file://'):
            raise ValueError('Exposed output "%s" from TA2 cannot be read', predictions_path_in_container)

        logger.info('Testing finished!')

        predictions_path_in_container = predictions_path_in_container.replace('file:///output/', '')
        predictions = pd.read_csv(join(self.output_folder, predictions_path_in_container))

        return predictions


    def score(self, output_dir, ta2_id, dataset, pipeline_id, pipeline_path):
        logger.info('Score solution...')

        dataset_train_path = os.path.join('file://' + os.path.abspath(self.base_dir),\
                                          'datasets', dataset,\
                                          'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path  = os.path.join('file://' + os.path.abspath(self.base_dir),\
                                          'datasets', dataset,\
                                          'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = os.path.join('file://' + os.path.abspath(self.base_dir),\
                                          'datasets', dataset,\
                                          'SCORE/dataset_SCORE/datasetDoc.json')
        problem_doc_path   = os.path.join('file://' + os.path.abspath(self.base_dir),\
                                          'datasets', dataset,\
                                          'TRAIN/problem_TRAIN/problemDoc.json')

        score_pipeline_path = os.path.join(self.base_dir, ta2_id, dataset, pipeline_id, 'fit_score_%s.csv' % pipeline_id)
        metric = None
        score  = None

        try:
            process = subprocess.Popen(['docker', 'exec', 'ta2_container',
                                        'python3', '-m', 'd3m', 'runtime',
                                        '--context', 'TESTING',
                                        '--random-seed', '0',
                                        'fit-score',
                                        '--pipeline', pipeline_path,
                                        '--problem', problem_doc_path,
                                        '--input', dataset_train_path,
                                        '--test-input', dataset_test_path,
                                        '--score-input', dataset_score_path,
                                        '--scores', score_pipeline_path]
            )
            process.wait()
            df     = pd.read_csv(os.path.join(output_dir, 'fit_score_%s.csv' % pipeline_id))
            score  = round(df['value'][0], 5)
            metric = df['metric'][0].lower()
        except:
            logger.exception('Scoring pipeline in test dataset')

            return None, None

        return metric, score


    def repeat_until_success(self, function, max_try=720, sleep_time=30, desc=None):
        # repeatedly run the function until an exception isn't thrown anymore
        for i in range(max_try):
            if desc is not None:
                logger.info(desc)
            try:
                function()
                return
            except:
                time.sleep(sleep_time)
        logger.error(f'The supplied command failed after {max_try} tries. Aborting.')
        raise Exception()


    def start_ta2(self, port=45042):
        logger.info('Initializing %s TA2...', self.ta2_id)
        # Stop any running containers
        process = subprocess.Popen(['docker', 'stop', 'ta2_container'])
        process.wait()

        container_name = 'ta2_container'
        self.ta2 = subprocess.Popen(['docker', 'run', '--rm', '--name', container_name,
                                                      '-p', '{port}:{port}'.format(port=port),
                                                      '-e', 'D3MRUN=ta2ta3',
                                                      '-e', 'D3MINPUTDIR=/ta2-eval/datasets',
                                                      '-e', 'D3MOUTPUTDIR=/ta2-eval/{ta2_id}/{dataset}/runs'.format(ta2_id=self.ta2_id, dataset=self.dataset),
                                                      '-e', 'D3MSTATICDIR=/ta2-eval/{ta2_id}/{dataset}/static'.format(ta2_id=self.ta2_id, dataset=self.dataset),
                                                      '-v', '{base_dir}:/ta2-eval'.format(base_dir=self.local_dir),
                                                      '-v', '{dataset_dir}:/ta2-eval/datasets'.format(dataset_dir=self.dataset_dir),
                                                       TA2_DOCKER_IMAGES[self.ta2_id]])
        # Wait for TA2 to start
        # Since server.start() will not block, a sleep-loop is added to keep alive
        if self.ta2_id == 'CMU':
            time.sleep(120) # Both takes time for dockers to intialize
        elif self.ta2_id == 'TAMU':
            time.sleep(65)
        else:
            time.sleep(25)
        try:
            while True:
                subprocess.run(["docker", "ps"])
                self.ta3 = D3MClient(port=port, ta2_id=self.ta2_id)
                self.ta3.do_hello()
                logger.info('%s TA2 initialized!', self.ta2_id)
                break
        except KeyboardInterrupt:
            if self.ta3.channel is not None:
                self.ta3.channel.close()
                self.ta3 = None
                time.sleep(4)

        logger.info("TA2 server stopped")


    def end_session(self):
        logger.info('Ending session...')
        if self.ta2 is not None:
            process = subprocess.Popen(['docker', 'stop', 'ta2_container'])
            process.wait() # Wait till process stopped
            subprocess.run(['docker', 'rm', 'ta2_container'])

        subprocess.run(["docker", "ps"])
        logger.info('Session ended!')


    def get_summary_pipeline(self, pipeline_json):
        primitives_summary = []
        for primitive in pipeline_json['steps']:
            primitive_name = primitive['primitive']['python_path']
            if primitive_name not in IGNORE_SUMMARY_PRIMITIVES:
                primitive_name_short = '.'.join(primitive_name.split('.')[-2:]).lower()
                if primitive_name_short not in primitives_summary:
                    primitives_summary.append(primitive_name_short)

        return ', '.join(primitives_summary)


    @staticmethod
    def add_new_ta2(name, docker_image):
        TA2_DOCKER_IMAGES[name] = docker_image
        logger.info('%s TA2 added!', name)
