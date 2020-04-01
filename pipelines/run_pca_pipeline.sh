#!/bin/bash

# Run Pipeline
echo "Running in local docker image for running pipeline..."

sudo docker run --rm \
                -v /home/tony/Documents/work_plai/d3m/codes/ubc_primitives:/ubc_primitives\
                -i -t registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 /bin/bash \
                -c "cd /ubc_primitives;\
                    pip3 install -e .;\
                    cd pipelines;\
                    python3 pca_pipeline.py -s 2;\
                    mkdir /ubc_primitives/static;\
                    python3 -m d3m index download -p d3m.primitives.feature_extraction.vggnet.UBC -o /ubc_primitives/static; \
                    python3 -m d3m runtime --volumes /ubc_primitives/static fit-score \
                            -p pca_pipeline.json \
                            -r /ubc_primitives/datasets/seed_datasets_current/22_handgeometry/TRAIN/problem_TRAIN/problemDoc.json \
                            -i /ubc_primitives/datasets/seed_datasets_current/22_handgeometry/TRAIN/dataset_TRAIN/datasetDoc.json \
                            -t /ubc_primitives/datasets/seed_datasets_current/22_handgeometry/TEST/dataset_TEST/datasetDoc.json \
                            -a /ubc_primitives/datasets/seed_datasets_current/22_handgeometry/SCORE/dataset_TEST/datasetDoc.json \
                            -o pca_results.csv \
                            -O pca_pipeline_run.yml;\
                    exit"

echo "Done!"
