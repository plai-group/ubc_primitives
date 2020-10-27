import os
import sys
import csv
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
import logging

import datamart_profiler
from d3m import container
from d3m.metadata import base as metadata_base
from profilers.randomimputer import RandomSamplingImputer
from common_primitives.simple_profiler import SimpleProfilerPrimitive
# from dummy_ta2.profilers.simpleprofiler import SemanticProfilerPrimitive

from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

LOGGER = logging.getLogger(__name__)


D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES = {
    'http://schema.org/Boolean':'boolean',
    'http://schema.org/Integer':'integer',
    'http://schema.org/Float': 'real',
    'http://schema.org/Text': 'string',
    'http://schema.org/DateTime': 'dateTime',
    'http://schema.org/Enumeration': 'categorical',
    'https://metadata.datadrivendiscovery.org/types/CategoricalData': 'categorical',
    'https://metadata.datadrivendiscovery.org/types/FloatVector': 'realVector',
    'https://metadata.datadrivendiscovery.org/types/JSON': 'json',
    'https://metadata.datadrivendiscovery.org/types/GeoJSON': 'geojson',
    'https://metadata.datadrivendiscovery.org/types/UnknownType': 'unknown'
}


def gen_digest(filename):
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K (in case of large files)
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()

def profiler(metadata, data):
    _map_semantic_type = {}
    _map_semantic_type["Text"]    = "string"
    _map_semantic_type["Float"]   = "real"
    _map_semantic_type["Boolean"] = "boolean"
    _map_semantic_type["Integer"] = "integer"
    _map_semantic_type["CategoricalData"] = "categorical"

    # Convert from pandas to d3m-pandas DataFrame
    data = container.DataFrame(data, generate_metadata=True)

    # Add column names
    for col in range(data.shape[1]):
        col_dict = dict(data.metadata.query((metadata_base.ALL_ELEMENTS, col)))
        try:
            if col_dict['name'] != 'd3mIndex':
                if 'structural_type' not in col_dict.keys():
                    col_dict['structural_type'] = type('all')
                    data.metadata = data.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
        except:
            print('name not found!')
        # col_dict['name'] = col_names[col]
        # print('Meta-data - {}'.format(col), col_dict)

    # Call imputer just to assign proper types when NaNs present
    RSI_hyperparams_class = RandomSamplingImputer.metadata.get_hyperparams()
    RSI_primitive = RandomSamplingImputer(hyperparams=RSI_hyperparams_class.defaults())
    RSI_primitive.set_training_data(inputs=data)
    RSI_primitive.fit()
    RSI_output = RSI_primitive.produce(inputs=data)
    RSI_output = RSI_output.value

    # Step 0: Profiler primitive
    SPP_hyperparams_class = SimpleProfilerPrimitive.metadata.get_hyperparams()
    SPP_primitive = SimpleProfilerPrimitive(hyperparams=SPP_hyperparams_class.defaults())
    SPP_primitive.set_training_data(inputs=data)
    SPP_primitive.fit()
    SPP_output = SPP_primitive.produce(inputs=data)
    SPP_output = SPP_output.value

    # static_files = {'distilbert-base-nli-stsb-mean-tokens.zip': '/dummy-ta2/dummy_ta2/profilers/distilbert-base-nli-stsb-mean-tokens.zip'}
    # SPP_hyperparams_class = SemanticProfilerPrimitive.metadata.get_hyperparams()
    # SPP_primitive = SemanticProfilerPrimitive(hyperparams=SPP_hyperparams_class.defaults(), volumes=static_files)
    # SPP_primitive.set_training_data(inputs=RSI_output)
    # SPP_primitive.fit()
    # SPP_output = SPP_primitive.produce(inputs=RSI_output)
    # SPP_output = SPP_output.value

    # print(SPP_output)

    # for col in range(SPP_output.shape[1]):
    #     col_dict = dict(SPP_output.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #     #col_dict['name'] = col_names[col]
    #     #predictions.metadata = data.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
    #     print('Meta-data - {}'.format(col), col_dict)

    # for cols in range(len(metadata['columns'])):
    #     print(cols, metadata['columns'][cols]['name'])
    #     print(cols, metadata['columns'][cols]['semantic_types'])
    #     print('-----------------------')

    # print('Meta-data after profiler.....')
    all_types = {}
    is_used = False
    for col in range(SPP_output.shape[1]):
        col_dict = dict(SPP_output.metadata.query((metadata_base.ALL_ELEMENTS, col)))
        # print('Meta-data - {}'.format(col), col_dict)
        semantic_types = col_dict["semantic_types"]
        for sem_type in semantic_types:
            Type = sem_type.split('/')[-1]
            if Type == 'Attribute':
                is_used = False
                continue
            else:
                try:
                    sem_type = _map_semantic_type[Type]
                    if sem_type == 'boolean':
                        sem_type = 'categorical'
                    is_used  = True
                except KeyError:
                    sem_type = "string"
                    is_used  = True
            all_types[col] = sem_type
            break
        # Default to string
        if is_used == False:
            all_types[col] = "string"


    return all_types


def dataset_json(pandas_dataset, dataset_name, mode, digest, targetColIdx, targetColType, col_types, col_names=None, MIN_META_DATA=False):
    dataset = {}

    #---------------------------------------------------------------------------
    dataset["about"] = {}
    dataset["about"]["datasetID"]            = dataset_name + "_dataset_" + mode
    dataset["about"]["datasetName"]          = dataset_name
    dataset["about"]["description"]          = "Sample Description."
    dataset["about"]["license"]              = "Private Data"
    dataset["about"]["citation"]             = "Private Data"
    dataset["about"]["source"]               = "Private Data"
    dataset["about"]["sourceURI"]            = "User upload"
    dataset["about"]["datasetSchemaVersion"] = "4.0.0"
    dataset["about"]["redacted"]             = False
    dataset["about"]["datasetVersion"]       = "4.0.0"
    dataset["about"]["digest"]               = digest

    #---------------------------------------------------------------------------
    dataset["dataResources"] = []
    data_resources_ = {}
    data_resources_["resID"]     = "learningData"
    data_resources_["resPath"]   = "tables/learningData.csv"
    data_resources_["resType"]   = "table"
    data_resources_["resFormat"] = {}
    data_resources_["resFormat"]["text/csv"] = ["csv"]
    #---------------------------------------------------------------------------

    data_resources_["isCollection"] = False
    data_resources_["columns"] = []

    if col_names ==  None:
        col_names = []
        for idx in range(pandas_dataset.shape[1]):
            col_names.append('column_'+str(idx))


    if not MIN_META_DATA:
        # For all columns is types available
        for idx in range(pandas_dataset.shape[1]):
            columns_ = {}

            if idx == 0:
                columns_["colIndex"] = idx
                columns_["colName"]  = "d3mIndex"
                columns_["colType"]  = "integer"
                columns_["role"]     = ["index"]
            else:
                columns_["colIndex"] = idx
                columns_["colName"]  = col_names[idx]
                if idx == targetColIdx:
                    # Most likely classification
                    if col_types[idx] == 'integer' or col_types[idx] == 'categorical' or col_types[idx] == 'boolean':
                        columns_["colType"] = 'categorical'
                    else:
                        columns_["colType"] = 'real'
                    columns_["role"] = ["suggestedTarget"]
                else:
                    columns_["colType"]  = col_types[idx]
                    columns_["role"]     = ["attribute"]

            data_resources_["columns"].append(columns_)
    else:
        # For d3mindex and target only columns, rest should be taken care of profiler.
        columns_ = {}

        columns_["colIndex"] = 0
        columns_["colName"]  = "d3mIndex"
        columns_["colType"]  = "integer"
        columns_["role"]     = ["index"]
        data_resources_["columns"].append(columns_)

        columns_ = {}
        columns_["colIndex"] = targetColIdx
        columns_["colName"]  = col_names[targetColIdx]
        columns_["colType"]  = targetColType
        columns_["role"]    = ["suggestedTarget"]

        data_resources_["columns"].append(columns_)

    #---------------------------------------------------------------------------
    dataset["dataResources"].append(data_resources_)

    dataset_json = json.dumps(dataset, sort_keys=False, indent=2)

    return dataset_json


def problem_json(dataset_name, taskKeywords, targetcolIndex, targetcolName, metric):
    problem = {}

    #---------------------------------------------------------------------------
    problem["about"] = {}

    problem["about"]["problemID"]            = dataset_name + "_problem"
    problem["about"]["problemName"]          = dataset_name
    problem["about"]["problemDescription"]   = "Sample Description."
    problem["about"]["problemSchemaVersion"] = "4.0.0"
    problem["about"]["problemVersion"]       = "4.0.0"
    problem["about"]["taskKeywords"]         = taskKeywords


    #---------------------------------------------------------------------------
    problem["inputs"] = {}

    problem["inputs"]["data"] = []
    data_ = {}
    data_["datasetID"] = dataset_name + '_dataset'
    data_["targets"]   = []
    targets_ = {}
    targets_["targetIndex"] = 0
    targets_["resID"]       = "learningData"
    targets_["colIndex"]    = targetcolIndex
    targets_["colName"]     = targetcolName
    data_["targets"].append(targets_)
    problem["inputs"]["data"].append(data_)

    #---------------------------------------------------------------------------
    problem["inputs"]["dataSplits"] = {}
    problem["inputs"]["dataSplits"]["splitsFile"]      = "dataSplits.csv"
    problem["inputs"]["dataSplits"]["datasetViewMaps"] = {}

    problem["inputs"]["dataSplits"]["datasetViewMaps"]["train"] = []
    train_ = {}
    train_["from"] = dataset_name + "_dataset"
    train_["to"]   = dataset_name + "_dataset_TRAIN"
    problem["inputs"]["dataSplits"]["datasetViewMaps"]["train"].append(train_)

    problem["inputs"]["dataSplits"]["datasetViewMaps"]["test"] = []
    test_ = {}
    test_["from"] = dataset_name + "_dataset"
    test_["to"]   = dataset_name + "_dataset_TEST"
    problem["inputs"]["dataSplits"]["datasetViewMaps"]["test"].append(test_)

    problem["inputs"]["dataSplits"]["datasetViewMaps"]["score"] = []
    score_ = {}
    score_["from"] = dataset_name + "_dataset"
    score_["to"]   = dataset_name + "_dataset_SCORE"
    problem["inputs"]["dataSplits"]["datasetViewMaps"]["score"].append(score_)

    #---------------------------------------------------------------------------
    problem["inputs"]["performanceMetrics"] = []
    performanceMetrics_ = {}
    performanceMetrics_["metric"] = metric
    problem["inputs"]["performanceMetrics"].append(performanceMetrics_)

    #---------------------------------------------------------------------------
    problem["expectedOutputs"] = {}
    problem["expectedOutputs"]["predictionsFile"] = "predictions.csv"

    problem_json = json.dumps(problem, sort_keys=False, indent=2)

    return problem_json

def problem_json_test(dataset_name, taskKeywords, targetcolIndex, targetcolName, metric):
    problem = {}

    #---------------------------------------------------------------------------
    problem["about"] = {}

    problem["about"]["problemID"]            = dataset_name + "_problem"
    problem["about"]["problemName"]          = dataset_name
    problem["about"]["problemDescription"]   = "Sample Description."
    problem["about"]["problemSchemaVersion"] = "4.0.0"
    problem["about"]["problemVersion"]       = "4.0.0"
    problem["about"]["taskKeywords"]         = taskKeywords


    #---------------------------------------------------------------------------
    problem["inputs"] = {}

    problem["inputs"]["data"] = []
    data_ = {}
    data_["datasetID"] = dataset_name + '_dataset'
    data_["targets"]   = []
    targets_ = {}
    targets_["targetIndex"] = 0
    targets_["resID"]       = "learningData"
    targets_["colIndex"]    = targetcolIndex
    targets_["colName"]     = targetcolName
    data_["targets"].append(targets_)
    problem["inputs"]["data"].append(data_)

    #---------------------------------------------------------------------------
    problem["inputs"]["dataSplits"] = {}
    problem["inputs"]["dataSplits"]["splitsFile"]      = "dataSplits.csv"
    problem["inputs"]["dataSplits"]["datasetViewMaps"] = {}

    problem["inputs"]["dataSplits"]["datasetViewMaps"]["test"] = []
    test_ = {}
    test_["from"] = dataset_name + "_dataset"
    test_["to"]   = dataset_name + "_dataset_TEST"
    problem["inputs"]["dataSplits"]["datasetViewMaps"]["test"].append(test_)

    problem["inputs"]["dataSplits"]["datasetViewMaps"]["score"] = []
    score_ = {}
    score_["from"] = dataset_name + "_dataset"
    score_["to"]   = dataset_name + "_dataset_SCORE"
    problem["inputs"]["dataSplits"]["datasetViewMaps"]["score"].append(score_)

    #---------------------------------------------------------------------------
    problem["inputs"]["performanceMetrics"] = []
    performanceMetrics_ = {}
    performanceMetrics_["metric"] = metric
    problem["inputs"]["performanceMetrics"].append(performanceMetrics_)

    #---------------------------------------------------------------------------
    problem["expectedOutputs"] = {}
    problem["expectedOutputs"]["predictionsFile"] = "predictions.csv"

    problem_json = json.dumps(problem, sort_keys=False, indent=2)

    return problem_json


#-------------------------------------------------------------------------------
def problem_type(data, targetColIdx, targetColName):
    """
    TODO: Better approach needed in future
    """
    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def RepresentsFloat(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    enc = OneHotEncoder(handle_unknown='ignore')
    # Find the taskKeywords based on target column type
    target_data = data.iloc[:, targetColIdx]

    # Check if all int:
    for data_item in target_data:
        is_int = RepresentsInt(s=data_item)

    if is_int:
        enc.fit_transform(data[[targetColName]])
        total_categories = enc.categories_
        if len(total_categories) < len(target_data):
            taskKeywords  = ["classification", "multiClass"]
            targetColType = "integer"
            metric = "accuracy"
        else:
            taskKeywords  = ["regression", "tabular"]
            targetColType = "real"
            metric = "meanSquaredError"
    else:
        for data_item in target_data:
            is_float = RepresentsFloat(s=data_item)

        if is_float:
            taskKeywords = ["regression", "tabular"]
            metric = "meanSquaredError"
            targetColType = "real"
        else:
            # Most likely string
            enc.fit_transform(data[[targetColName]])
            total_categories = enc.categories_
            if len(total_categories) < len(target_data):
                taskKeywords  = ["classification", "multiClass"]
                targetColType = "integer"
                metric = "accuracy"
            else:
                taskKeywords  = ["regression", "tabular"]
                targetColType = "real"
                metric = "meanSquaredError"

    return taskKeywords, targetColType, metric


#-------------------------------------------------------------------------------
def make_d3m_dataset(data, data_path, digest, save_dir, dataset_name, targetColIdx, use_random_split, test_split=20):
    """
    Accepts a single dataset table and splits into D3M dataset format
    Train/Test/Score
    """
    col_names = list(data.iloc[0, :])

    # Check if d3mIndex present
    if "d3mIndex" not in col_names:
        # Assuming the first row is column names
        col_names = ["d3mIndex"] + list(data.iloc[0, :])

        # Reset is data and reset index to 0
        data = data.iloc[1:, :]
        data.reset_index(drop=True, inplace=True)

        # Adding d3m indexs into the dataset
        d3m_idxs   = list(range(data.shape[0]))
        final_data = pd.DataFrame(data=d3m_idxs)

        final_data = pd.concat([final_data, data], axis=1, ignore_index=True)
        final_data.columns = col_names
    else:
        # Reset is data and reset index to 0
        data = data.iloc[1:, :]
        data.reset_index(drop=True, inplace=True)

        # Most likely a d3m dataset
        final_data = pd.DataFrame(data=data)
        final_data.columns = col_names

    # Random split the data into
    if use_random_split:
        final_data = final_data.sample(frac=1).reset_index(drop=True)
    # print(final_data)

    # TODO: Need to incorporate this with simple profiler
    # metadata = datamart_profiler.process_dataset(data_path)
    metadata = None
    # print(metadata['columns'])

    # Fill Missing values
    imp1 = SimpleImputer(missing_values='', strategy="most_frequent")
    imp2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    for col in final_data.columns:
        try:
            final_data[[col]] = imp1.fit_transform(final_data[[col]])
            final_data[[col]] = imp2.fit_transform(final_data[[col]])
        except ValueError:
            # Assuimg empty column and replace nan with 0
            final_data[col].fillna('0', inplace=True)

    all_col_types = profiler(metadata=metadata, data=final_data)
    # print(all_col_types)

    taskKeywords, targetColType, metric = problem_type(final_data, targetColIdx, targetColName=final_data.columns[targetColIdx])

    ## Split set -- Order matters in case of shuffle is True
    all_dataset_types  = ["SCORE", "TEST", "TRAIN"]
    all_dataset_types1 = ["TEST",  "TEST", "TRAIN"]
    total_data = final_data.shape[0]
    test_split = int(total_data * (test_split/100.0))

    all_dataset_splits = {}
    all_dataset_splits["SCORE"] = {}
    all_dataset_splits["SCORE"]["START"] = 0
    all_dataset_splits["SCORE"]["END"]   = test_split

    all_dataset_splits["TEST"] = {}
    all_dataset_splits["TEST"]["START"] = 0
    all_dataset_splits["TEST"]["END"]   = test_split

    all_dataset_splits["TRAIN"] = {}
    all_dataset_splits["TRAIN"]["START"] = test_split
    all_dataset_splits["TRAIN"]["END"]   = total_data

    problem_description = problem_json(dataset_name=dataset_name,\
                                       taskKeywords=taskKeywords,\
                                       targetcolIndex=targetColIdx,\
                                       targetcolName=col_names[targetColIdx],\
                                       metric=metric)

    main_dataset_dir = os.path.join(save_dir, dataset_name)
    try:
        os.makedirs(main_dataset_dir)
    except FileExistsError:
        LOGGER.info("Directory already exist %s", main_dataset_dir)

    for _type in range(len(all_dataset_types)):
        dataset_type = all_dataset_types[_type]
        # print(dataset_type)
        # Setup Problem
        problem_dir = os.path.join(main_dataset_dir, dataset_type, 'problem_'+all_dataset_types1[_type])
        try:
            os.makedirs(problem_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", problem_dir)

        csv_file = os.path.join(problem_dir, 'dataSplits.csv')
        with open(csv_file, 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(['d3mIndex', 'type', 'repeat', 'fold'])
            for idx in range(all_dataset_splits[dataset_type]["START"], all_dataset_splits[dataset_type]["END"]):
                interm_data = final_data.iloc[idx].tolist()
                wr.writerow([interm_data[0], all_dataset_types1[_type], 0, 0])
        f.close()

        json_file = os.path.join(problem_dir, 'problemDoc.json')
        with open(json_file, 'w') as write_file:
            write_file.write(problem_description)
        write_file.close()

        # Setup dataset
        dataset_dir = os.path.join(main_dataset_dir, dataset_type, 'dataset_'+dataset_type)
        try:
            os.makedirs(dataset_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", dataset_dir)

        dataset_description = dataset_json(pandas_dataset=final_data,\
                                           dataset_name=dataset_name,\
                                           mode=dataset_type,\
                                           digest=digest,\
                                           targetColIdx=targetColIdx,\
                                           targetColType=targetColType,\
                                           col_types=all_col_types,\
                                           col_names=col_names)
        # print(dataset_description)

        json_file = os.path.join(dataset_dir, 'datasetDoc.json')
        with open(json_file, 'w') as write_file:
            write_file.write(dataset_description)
        write_file.close()

        tables_dataset_dir = os.path.join(dataset_dir, 'tables')
        try:
            os.makedirs(tables_dataset_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", tables_dataset_dir)
        csv_file = os.path.join(tables_dataset_dir, 'learningData.csv')
        with open(csv_file, 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            wr.writerow(col_names)
            for idx in range(all_dataset_splits[dataset_type]["START"], all_dataset_splits[dataset_type]["END"]):
                interm_data = final_data.iloc[idx].tolist()
                if dataset_type == "TEST":
                    row_ = []
                    for ix_data in range(len(interm_data)):
                        # Remove the score
                        if ix_data == targetColIdx:
                            row_.append('')
                        else:
                            row_.append(interm_data[ix_data])
                    wr.writerow(row_)
                else:
                    wr.writerow([ix_data for ix_data in interm_data])
        f.close()


def make_d3m_test_dataset(data, digest, save_dir, dataset_name, train_dataset_uuid):
    """
    Accepts a single dataset table and makes it into D3M testing dataset format
    """
    col_names = list(data.iloc[0, :])

    # Check if d3mIndex present
    if "d3mIndex" not in col_names:
        # Assuming the first row is column names
        col_names = ["d3mIndex"] + list(data.iloc[0, :])

        # Reset is data and reset index to 0
        data = data.iloc[1:, :]
        data.reset_index(drop=True, inplace=True)

        # Adding d3m indexs into the dataset
        d3m_idxs   = list(range(data.shape[0]))
        final_data = pd.DataFrame(data=d3m_idxs)

        final_data = pd.concat([final_data, data], axis=1, ignore_index=True)
        final_data.columns = col_names
    else:
        # Reset is data and reset index to 0
        data = data.iloc[1:, :]
        data.reset_index(drop=True, inplace=True)

        # Most likely a d3m dataset
        final_data = pd.DataFrame(data=data)
        final_data.columns = col_names

    # Get task keywords from training problem json
    _dataset_train_doc_file = os.path.join(save_dir, train_dataset_uuid, 'TRAIN', 'problem_TRAIN', 'problemDoc.json')
    with open(_dataset_train_doc_file) as json_file:
        train_dataset_doc = json.load(json_file)

    taskKeywords  = train_dataset_doc['about']['taskKeywords']
    metric        = train_dataset_doc['inputs']['performanceMetrics'][0]['metric']
    targetColIdx  = train_dataset_doc['inputs']['data'][0]['targets'][0]['colIndex']
    targetColName = train_dataset_doc['inputs']['data'][0]['targets'][0]['colName']

    # Add target Empty Coloumn if not present
    if targetColName not in col_names:
        final_data[targetColName] = ""

    # Type
    if "classification" in taskKeywords:
        targetColType ="integer"
    else:
        targetColType ="real"

    all_dataset_types  = ["SCORE", "TEST"]
    all_dataset_types1 = ["TEST",  "TEST"]
    total_data = final_data.shape[0]

    all_dataset_splits = {}
    all_dataset_splits["SCORE"] = {}
    all_dataset_splits["SCORE"]["START"] = 0
    all_dataset_splits["SCORE"]["END"]   = total_data

    all_dataset_splits["TEST"] = {}
    all_dataset_splits["TEST"]["START"] = 0
    all_dataset_splits["TEST"]["END"]   = total_data

    problem_description = problem_json_test(dataset_name=train_dataset_uuid,\
                                            taskKeywords=taskKeywords,\
                                            targetcolIndex=targetColIdx,\
                                            targetcolName=targetColName,\
                                            metric=metric)

    main_dataset_dir = os.path.join(save_dir, dataset_name)
    try:
        os.makedirs(main_dataset_dir)
    except FileExistsError:
        LOGGER.info("Directory already exist %s", main_dataset_dir)

    metadata = None

    all_col_types = profiler(metadata=metadata, data=final_data)

    for _type in range(len(all_dataset_types)):
        dataset_type = all_dataset_types[_type]
        # print(dataset_type)
        # Setup Problem
        problem_dir = os.path.join(main_dataset_dir, dataset_type, 'problem_'+all_dataset_types1[_type])
        try:
            os.makedirs(problem_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", problem_dir)

        csv_file = os.path.join(problem_dir, 'dataSplits.csv')
        with open(csv_file, 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(['d3mIndex', 'type', 'repeat', 'fold'])
            for idx in range(all_dataset_splits[dataset_type]["START"], all_dataset_splits[dataset_type]["END"]):
                interm_data = final_data.iloc[idx].tolist()
                wr.writerow([interm_data[0], all_dataset_types1[_type], 0, 0])
        f.close()

        json_file = os.path.join(problem_dir, 'problemDoc.json')
        with open(json_file, 'w') as write_file:
            write_file.write(problem_description)
        write_file.close()

        # Setup dataset
        dataset_dir = os.path.join(main_dataset_dir, dataset_type, 'dataset_'+dataset_type)
        try:
            os.makedirs(dataset_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", dataset_dir)

        dataset_description = dataset_json(pandas_dataset=final_data,\
                                           dataset_name=train_dataset_uuid,\
                                           mode=dataset_type,\
                                           digest=digest,\
                                           targetColIdx=targetColIdx,\
                                           targetColType=targetColType,\
                                           col_types=all_col_types,\
                                           col_names=col_names)
        # print(dataset_description)

        json_file = os.path.join(dataset_dir, 'datasetDoc.json')
        with open(json_file, 'w') as write_file:
            write_file.write(dataset_description)
        write_file.close()

        tables_dataset_dir = os.path.join(dataset_dir, 'tables')
        try:
            os.makedirs(tables_dataset_dir)
        except FileExistsError:
            LOGGER.info("Directory already exist %s", tables_dataset_dir)
        csv_file = os.path.join(tables_dataset_dir, 'learningData.csv')
        with open(csv_file, 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            wr.writerow(col_names)
            for idx in range(all_dataset_splits[dataset_type]["START"], all_dataset_splits[dataset_type]["END"]):
                interm_data = final_data.iloc[idx].tolist()
                if dataset_type == "TEST":
                    row_ = []
                    for ix_data in range(len(interm_data)):
                        # Remove the score if present
                        if ix_data == targetColIdx:
                            row_.append('')
                        else:
                            row_.append(interm_data[ix_data])
                    wr.writerow(row_)
                else:
                    wr.writerow([ix_data for ix_data in interm_data])
        f.close()
