import copy
import unittest

from lib.contracts import validate_qubo, validate_qubo_result
from lib.contracts.validation import _validate_qubo


class QuboContractValidationTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'qubo.v1',
            'problem_id': 'validation-example',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x0', 'x1'],
            'offset': 4.0,
            'terms': [
                [0, 0, -6.0],
                [0, 1, 8.0],
                [1, 1, -5.0],
            ],
            'metadata': {'owner': 'tests'},
        }
        self.result = {
            'schema': 'qubo-result.v1',
            'problem_id': 'validation-example',
            'solver': {
                'name': 'contract-test',
                'version': '1.0.0',
                'backend': 'local',
            },
            'status': 'optimal',
            'best_sample': [1, 0],
            'best_energy': -2.0,
            'runtime_seconds': 0.01,
            'termination_reason': 'search_exhausted',
            'metrics': {'iterations': 4},
            'trace': [
                {
                    'step': 0,
                    'time_seconds': 0.0,
                    'energy': 4.0,
                    'sample': [0, 0],
                    'metadata': {'phase': 'initial'},
                },
            ],
            'metadata': {'algorithm': 'enumeration'},
        }

    def test_accepts_schema_and_cross_semantically_valid_documents(self):
        self.assertIsNone(validate_qubo(self.problem))
        self.assertIsNone(validate_qubo_result(self.problem, self.result))
        self.assertIs(_validate_qubo, validate_qubo)

    def test_qubo_rejects_unknown_fields_and_non_object_metadata(self):
        unknown = copy.deepcopy(self.problem)
        unknown['surprise'] = True
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_qubo(unknown)

        invalid_metadata = copy.deepcopy(self.problem)
        invalid_metadata['metadata'] = 'not-an-object'
        with self.assertRaisesRegex(TypeError, 'must be an object'):
            validate_qubo(invalid_metadata)

    def test_qubo_requires_json_arrays_and_python_integer_indices(self):
        tuple_term = copy.deepcopy(self.problem)
        tuple_term['terms'][0] = (0, 0, -6.0)
        with self.assertRaisesRegex(ValueError, 'triples'):
            validate_qubo(tuple_term)

        boolean_index = copy.deepcopy(self.problem)
        boolean_index['terms'][0][0] = False
        with self.assertRaisesRegex(ValueError, 'outside the variable range'):
            validate_qubo(boolean_index)

    def test_qubo_rejects_non_finite_and_non_json_metadata_values(self):
        non_finite = copy.deepcopy(self.problem)
        non_finite['offset'] = float('nan')
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            validate_qubo(non_finite)

        non_json = copy.deepcopy(self.problem)
        non_json['metadata'] = {'tags': {'not', 'json'}}
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            validate_qubo(non_json)

        non_string_key = copy.deepcopy(self.problem)
        non_string_key['metadata'] = {1: 'not-a-json-object-key'}
        with self.assertRaisesRegex(TypeError, 'keys must be strings'):
            validate_qubo(non_string_key)

    def test_result_must_belong_to_problem_and_recompute_energy(self):
        wrong_problem = copy.deepcopy(self.result)
        wrong_problem['problem_id'] = 'another-problem'
        with self.assertRaisesRegex(ValueError, 'must match'):
            validate_qubo_result(self.problem, wrong_problem)

        wrong_energy = copy.deepcopy(self.result)
        wrong_energy['best_energy'] = -1.5
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            validate_qubo_result(self.problem, wrong_energy)

    def test_large_integer_energy_requires_the_exact_json_number(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'large-integer-contract',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': 10_000_000_000_000_000,
            'terms': [[0, 0, -1]],
            'metadata': {},
        }
        result = copy.deepcopy(self.result)
        result.update({
            'problem_id': problem['problem_id'],
            'best_sample': [1],
            'best_energy': 9_999_999_999_999_999,
            'trace': [],
        })
        self.assertIsNone(validate_qubo_result(problem, result))

        # Binary64 rounds this value up to 10**16. A relative tolerance would
        # accept it and erase the one-unit distinction that decides optimality.
        rounded = copy.deepcopy(result)
        rounded['best_energy'] = 1e16
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            validate_qubo_result(problem, rounded)

    def test_result_sample_is_exact_length_python_binary_list(self):
        invalid_samples = (
            ([1], 'length'),
            ([True, 0], 'Python integer binary'),
            ((1, 0), 'must be a list'),
        )
        for sample, message in invalid_samples:
            with self.subTest(sample=sample):
                result = copy.deepcopy(self.result)
                result['best_sample'] = sample
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate_qubo_result(self.problem, result)

    def test_result_status_controls_candidate_presence(self):
        missing_candidate = copy.deepcopy(self.result)
        missing_candidate['best_sample'] = None
        missing_candidate['best_energy'] = None
        with self.assertRaisesRegex(ValueError, 'requires'):
            validate_qubo_result(self.problem, missing_candidate)

        infeasible_candidate = copy.deepcopy(self.result)
        infeasible_candidate['status'] = 'infeasible'
        with self.assertRaisesRegex(ValueError, 'cannot include'):
            validate_qubo_result(self.problem, infeasible_candidate)

        half_candidate = copy.deepcopy(self.result)
        half_candidate['status'] = 'timeout'
        half_candidate['best_energy'] = None
        with self.assertRaisesRegex(ValueError, 'both be null'):
            validate_qubo_result(self.problem, half_candidate)

        no_incumbent = copy.deepcopy(self.result)
        no_incumbent['status'] = 'timeout'
        no_incumbent['best_sample'] = None
        no_incumbent['best_energy'] = None
        self.assertIsNone(validate_qubo_result(self.problem, no_incumbent))

    def test_result_rejects_invalid_runtime_and_closed_object_extras(self):
        for runtime in (-0.1, float('inf')):
            with self.subTest(runtime=runtime):
                result = copy.deepcopy(self.result)
                result['runtime_seconds'] = runtime
                with self.assertRaises((TypeError, ValueError)):
                    validate_qubo_result(self.problem, result)

        extra_top_level = copy.deepcopy(self.result)
        extra_top_level['artifact_id'] = 'not-in-v1'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_qubo_result(self.problem, extra_top_level)

        extra_solver_field = copy.deepcopy(self.result)
        extra_solver_field['solver']['nickname'] = 'not-in-v1'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_qubo_result(self.problem, extra_solver_field)

    def test_result_metrics_and_metadata_must_be_json_objects(self):
        invalid_metrics = copy.deepcopy(self.result)
        invalid_metrics['metrics'] = []
        with self.assertRaisesRegex(TypeError, 'must be an object'):
            validate_qubo_result(self.problem, invalid_metrics)

        invalid_metadata = copy.deepcopy(self.result)
        invalid_metadata['metadata'] = {'callback': lambda: None}
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            validate_qubo_result(self.problem, invalid_metadata)

        nested_non_finite = copy.deepcopy(self.result)
        nested_non_finite['metrics'] = {'loss': float('nan')}
        with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
            validate_qubo_result(self.problem, nested_non_finite)

    def test_trace_is_closed_json_and_cross_validated(self):
        missing_field = copy.deepcopy(self.result)
        del missing_field['trace'][0]['step']
        with self.assertRaisesRegex(ValueError, 'missing required'):
            validate_qubo_result(self.problem, missing_field)

        unknown_field = copy.deepcopy(self.result)
        unknown_field['trace'][0]['message'] = 'not-in-v1'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_qubo_result(self.problem, unknown_field)

        wrong_trace_energy = copy.deepcopy(self.result)
        wrong_trace_energy['trace'][0]['energy'] = 123.0
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            validate_qubo_result(self.problem, wrong_trace_energy)

        wrong_trace_sample = copy.deepcopy(self.result)
        wrong_trace_sample['trace'][0]['sample'] = [False, 0]
        with self.assertRaisesRegex(ValueError, 'Python integer binary'):
            validate_qubo_result(self.problem, wrong_trace_sample)

        negative_trace_time = copy.deepcopy(self.result)
        negative_trace_time['trace'][0]['time_seconds'] = -0.1
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            validate_qubo_result(self.problem, negative_trace_time)


if __name__ == '__main__':
    unittest.main()
