import grpc
# D3M TA-2 API
import ta3ta2_api.core_pb2 as pb_core
import ta3ta2_api.core_pb2_grpc as pb_core_grpc
import ta3ta2_api.value_pb2 as pb_value
from ta3ta2_api.value_pb2 import Value
from ta3ta2_api.utils import encode_problem_description
from ta3ta2_api.utils import encode_performance_metric, decode_performance_metric
from ta3ta2_api.utils import decode_value, decode_pipeline_description
# D3M
from d3m.metadata.problem import Problem, PerformanceMetric
from d3m.utils import fix_uri, silence
from d3m.metadata import pipeline as pipeline_module
# Logging
import logging
logger  = logging.getLogger(__name__)
version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]


class D3MClient:
    def __init__(self, port, ta2_id, user='D3M_TA2_Evaluation'):
        self.ta2_id = ta2_id
        self.user = user
        # open a gRPC channel
        try:
            self.channel = grpc.insecure_channel('localhost:{port}'.format(port=port))
        except:
            print('Could not open the GRPC channel!')
        # create a stub (client)
        self.core = pb_core_grpc.CoreStub(self.channel)


    def do_hello(self):
        self.core.Hello(pb_core.HelloRequest())


    def do_listprimitives(self):
        self.core.ListPrimitives(pb_core.ListPrimitivesRequest())


    def _build_problem(self, problem_path):
        problem = Problem.load(problem_uri=problem_path)
        return encode_problem_description(problem)

    def do_search(self, dataset_doc_path, problem_doc_path, ta2_id, time_bound=30.0, pipelines_limit=0, pipeline_template=None):
        # Search
        search = self.core.SearchSolutions(pb_core.SearchSolutionsRequest(
                                            user_agent='D3M_TA2_Evaluation',
                                            version=version,
                                            time_bound_search=time_bound,
                                            priority=10,
                                            rank_solutions_limit=pipelines_limit,
                                            allowed_value_types=['RAW', 'DATASET_URI', 'CSV_URI'],
                                            template=pipeline_template,
                                            problem=self._build_problem(problem_doc_path),
                                            inputs=[Value(dataset_uri='file://%s' % dataset_doc_path)],)
        )

        # Get request
        getsearch_request = pb_core.GetSearchSolutionsResultsRequest()
        getsearch_request.search_id = search.search_id

        # Make the call (loop cause streaming)-- It makes client docker remain open untill complete
        logger.warning('Solution Stream: ')
        for getsearch_response in self.core.GetSearchSolutionsResults(getsearch_request):
            logger.warning(getsearch_response)
            if getsearch_response.solution_id:
                pipeline_id = getsearch_response.solution_id
                yield {'id': pipeline_id, 'search_id': str(search.search_id)}
            logger.warning('------------------------------------')


    def do_score(self, solution_id, dataset_path, problem_path, ta2_id):
        try:
            problem = Problem.load(problem_uri=problem_path)
        except:
            logger.exception('Error parsing problem')

        # Encode metric
        metrics = []
        for metric in problem['problem']['performance_metrics']:
            metrics.append(encode_performance_metric(metric))

        # Showing only the first metric
        target_metric = problem['problem']['performance_metrics'][0]['metric']
        logger.info('target_metric %s !', target_metric)

        response = self.core.ScoreSolution(pb_core.ScoreSolutionRequest(
                                            solution_id=solution_id,
                                            inputs=[pb_value.Value(dataset_uri='file://%s' % dataset_path,)],
                                            performance_metrics=metrics,
                                            users=[],
                                            configuration=pb_core.ScoringConfiguration(method='HOLDOUT',
                                                                                       train_test_ratio=0.75,
                                                                                       shuffle=True,
                                                                                       random_seed=0),)
        )
        logger.info('ScoreSolution response %s !', response)

        # Get Results
        results = self.core.GetScoreSolutionResults(pb_core.GetScoreSolutionResultsRequest(request_id=response.request_id,))
        for result in results:
            logger.info('result %s !', result)
            if result.progress.state == pb_core.COMPLETED:
                scores = []
                for metric_score in result.scores:
                    metric = decode_performance_metric(metric_score.metric)['metric']
                    if metric == target_metric:
                        score = decode_value(metric_score.value)['value']
                        scores.append(score)
                if len(scores) > 0:
                    avg_score = round(sum(scores)/len(scores), 5)
                    normalized_score = PerformanceMetric[target_metric.name].normalize(avg_score)

                    return {'score': avg_score, 'normalized_score': normalized_score, 'metric': target_metric.name.lower()}


    def do_train(self, solution_id, dataset_path):
        fitted_solution = None
        try:
            response = self.core.FitSolution(pb_core.FitSolutionRequest(
                                              solution_id=solution_id,
                                              inputs=[pb_value.Value(dataset_uri=dataset_path,)],
                                              expose_outputs=[],
                                              expose_value_types=['CSV_URI'],
                                              users=[self.user],)
            )
            # Results
            results = self.core.GetFitSolutionResults(pb_core.GetFitSolutionResultsRequest(request_id=response.request_id,))
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    fitted_solution = result.fitted_solution_id
        except:
            logger.exception("Exception training %r", solution_id)

        return fitted_solution


    def do_test(self, fitted_solution_id, dataset_path):
        tested = None
        try:
            response = self.core.ProduceSolution(pb_core.ProduceSolutionRequest(
                                                  fitted_solution_id=fitted_solution_id,
                                                  inputs=[pb_value.Value(dataset_uri='file://%s' % dataset_path,)],
                                                  expose_outputs=['outputs.0'],
                                                  expose_value_types=['CSV_URI'],
                                                  users=[],)
            )
            # Results
            results = self.core.GetProduceSolutionResults(pb_core.GetProduceSolutionResultsRequest(request_id=response.request_id,))
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    tested = result.exposed_outputs['outputs.0'].csv_uri
        except:
            logger.exception("Exception testing %r", fitted_solution_id)

        return tested


    def do_export(self, fitted):
        for i, fitted_solution in enumerate(fitted.values()):
            try:
                self.core.SolutionExport(pb_core.SolutionExportRequest(solution_id=fitted_solution, rank=(i + 1.0)/(len(fitted) + 1.0),))
            except:
                logger.exception("Exception exporting %r", fitted_solution)


    def do_describe(self, solution_id):
        pipeline_description = None
        try:
            pipeline_description = self.core.DescribeSolution(pb_core.DescribeSolutionRequest(solution_id=solution_id,)).pipeline
        except:
            logger.exception("Exception during describe %r", solution_id)

        with silence():
            pipeline = decode_pipeline_description(pipeline_description, pipeline_module.NoResolver())

        return pipeline.to_json_structure()


    def do_stop_search(self, search_id):
        self.core.StopSearchSolutions(pb_core.StopSearchSolutionsRequest(search_id=search_id))
