import os
import pandas as pd
# Test Automl
from d3m_ta2s_eval.automl_eval import AutoML


def main(dataset_dir, dataset_name, ta2_id, timeout):
        base_dir      = os.getcwd()
        container_dir = '/ta2-eval'
        dataset       = dataset_name
        save_dir      = os.path.join(base_dir, ta2_id, dataset, str(timeout))
        save_file     = os.path.join(base_dir, ta2_id, dataset, str(timeout), '{ta2_id}_result.csv'.format(ta2_id=ta2_id))
        save_pipeline = os.path.join(base_dir, ta2_id, dataset, str(timeout), 'pipelines')
        output_dir    = os.path.join(base_dir, ta2_id, dataset, str(timeout))

        top_k = 5

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

        results_csv = pd.read_csv(save_file)
        print(results_csv.shape)

        # AutoML
        automl = AutoML(output_folder=ta2_id, local_dir=base_dir, base_dir=container_dir,\
                        dataset=dataset, dataset_dir=dataset_dir, ta2_id=ta2_id)

        # Start docker
        automl.start_ta2(timeout=timeout)

        max_use = 0
        all_outputs = []
        for idx in range(results_csv.shape[0]):
            ranking, pipeline_id, summary, metric = results_csv.iloc[idx, :]

            # Run pipeline search
            output_dir_idx = os.path.join(output_dir, pipeline_id)
            pipeline_path  = os.path.join(container_dir, ta2_id, dataset, str(timeout), 'pipelines', '%s.json' % pipeline_id)

            if not os.path.exists(output_dir_idx):
                os.makedirs(output_dir_idx)

            # Score
            metric, score = automl.score(output_dir_idx, ta2_id, dataset, pipeline_id, pipeline_path)

            # Collect scores
            all_outputs.append([pipeline_id, metric, score])

            # Increment
            if metric != None:
                max_use += 1

            if max_use == top_k:
                break

        df = pd.DataFrame(all_outputs, columns=['Pipeline_ID', 'metric', 'score'])
        final_df   = df.sort_values(by=['score'], ascending=False)
        final_file = os.path.join(base_dir, ta2_id, dataset, str(timeout), 'ranked_scored_result.csv')
        final_df.to_csv(final_file, index=False)

        print('---------------------------------------------------------------')
        print(final_df)
        print('---------------------------------------------------------------')

        # End docker
        automl.end_session()


#---------------------------------MAIN-----------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D3M AutoML produce run')
    parser.add_argument('-d', '--dataset_dir',   type=str)
    parser.add_argument('-n', '--dataset_name',  type=str)
    parser.add_argument('-a', '--ta2_id',        type=str)
    parser.add_argument('-t', '--timeout',       type=int)
    args = parser.parse_args()


    main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name,\
          ta2_id=args.ta2_id, timeout=args.timeout)

    print('Done')
