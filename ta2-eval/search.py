import os
import pandas
import unittest
import subprocess
# Test Automl
from d3m_ta2s_eval.automl_eval import AutoML


def main(dataset_dir, dataset_name, ta2_id, timeout):
        base_dir      = os.getcwd()
        container_dir = '/ta2-eval'
        dataset       = dataset_name
        save_dir      = os.path.join(base_dir, 'CMU', dataset)
        save_file     = os.path.join(base_dir, 'CMU', dataset, 'cmu_result.csv')
        save_pipeline = os.path.join(base_dir, 'CMU', dataset, 'pipelines')
        output_dir    = os.path.join(base_dir, 'CMU', dataset)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(save_pipeline):
            os.makedirs(save_pipeline)

        dataset_doc_path = os.path.join('file://' + os.path.abspath(dataset_dir),\
                                         dataset,\
                                        'TRAIN/dataset_TRAIN/datasetDoc.json')
        problem_doc_path = os.path.join('file://' + os.path.abspath(dataset_dir),\
                                         dataset,\
                                        'TRAIN/problem_TRAIN/problemDoc.json')

        # AutoML
        automl = AutoML(output_folder='CMU', local_dir=base_dir, base_dir=container_dir,\
                        dataset=dataset, dataset_dir=dataset_dir, ta2_id='CMU')

        # Start docker
        automl.start_ta2()

        # Run pipeline search
        automl.search_pipelines(save_pipeline, save_file, dataset, dataset_doc_path, problem_doc_path, time_bound=30)

        # End docker
        automl.end_session()


#---------------------------------MAIN-----------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D3M AutoML run')
    parser.add_argument('-d', '--dataset_dir',   type=str)
    parser.add_argument('-n', '--dataset_name',  type=str)
    parser.add_argument('-a', '--ta2_id',        type=str)
    parser.add_argument('-c', '--local_dir',     type=str)
    parser.add_argument('-t', '--timeout',       type=int)
    args = parser.parse_args()


     main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name,\
          ta2_id=args.ta2_id, timeout=args.timeout)

     print('Done')
