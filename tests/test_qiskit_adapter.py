import unittest

from lib.adapters.qiskit_quadratic_program import (
    QiskitQuadraticProgramAdapter,
)


class _FakeEnum:
    def __init__(self, name):
        self.name = name


class _FakeExpression:
    def __init__(self, coefficients=None):
        self._coefficients = coefficients or {}

    def to_dict(self, use_name=False):
        return dict(self._coefficients)


class _FakeVariable:
    def __init__(self, name):
        self.name = name
        self.vartype = _FakeEnum('BINARY')


class _FakeObjective:
    def __init__(self, sense, constant, linear, quadratic):
        self.sense = _FakeEnum(sense.upper())
        self.constant = constant
        self.linear = _FakeExpression(linear)
        self.quadratic = _FakeExpression(quadratic)


class _FakeConstraint:
    def __init__(self, name, linear, sense, rhs):
        self.name = name
        self.linear = _FakeExpression(linear)
        self.sense = _FakeEnum({'==': 'EQ', '>=': 'GE', '<=': 'LE'}[sense])
        self.rhs = rhs


class _FakeQuadraticProgram:
    def __init__(self, name=''):
        self.name = name
        self.variables = []
        self.objective = None
        self.linear_constraints = []
        self.quadratic_constraints = []

    def binary_var(self, name):
        self.variables.append(_FakeVariable(name))

    def minimize(self, constant, linear, quadratic):
        self.objective = _FakeObjective(
            'minimize',
            constant,
            linear,
            quadratic,
        )

    def maximize(self, constant, linear, quadratic):
        self.objective = _FakeObjective(
            'maximize',
            constant,
            linear,
            quadratic,
        )

    def linear_constraint(self, linear, sense, rhs, name):
        self.linear_constraints.append(
            _FakeConstraint(name, linear, sense, rhs)
        )


class QiskitQuadraticProgramAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = QiskitQuadraticProgramAdapter()
        self.cbqm = {
            'schema': 'cbqm.v1',
            'problem_id': 'adapter-example',
            'variables': [
                {
                    'index': 0,
                    'name': 'x0',
                    'vartype': 'BINARY',
                    'kind': 'selection',
                    'metadata': {'asset': 'A'},
                },
                {
                    'index': 1,
                    'name': 'x1',
                    'vartype': 'BINARY',
                    'kind': 'selection',
                },
            ],
            'objective': {
                'sense': 'minimize',
                'offset': 1.5,
                'linear': [[0, -2.0]],
                'quadratic': [[0, 1, 3.0]],
            },
            'constraints': [
                {
                    'name': 'budget',
                    'family': 'budget',
                    'linear': [[0, 1.0], [1, 1.0]],
                    'lower_bound': 1.0,
                    'upper_bound': 1.0,
                },
                {
                    'name': 'range',
                    'family': 'allocation',
                    'linear': [[0, 2.0], [1, 1.0]],
                    'lower_bound': 0.5,
                    'upper_bound': 1.5,
                    'metadata': {'source': 'policy'},
                },
            ],
            'fixed_values': [{'index': 1, 'value': 0}],
            'metadata': {'owner': 'test'},
        }

    def test_exports_variables_objective_constraints_and_fixed_values(self):
        program, context = self.adapter.to_qiskit(
            self.cbqm,
            quadratic_program_class=_FakeQuadraticProgram,
        )

        self.assertEqual([variable.name for variable in program.variables], ['x0', 'x1'])
        self.assertEqual(program.objective.constant, 1.5)
        self.assertEqual(program.objective.linear.to_dict(), {'x0': -2.0})
        self.assertEqual(
            program.objective.quadratic.to_dict(),
            {('x0', 'x1'): 3.0},
        )
        self.assertEqual(len(program.linear_constraints), 4)
        self.assertEqual(context['schema'], 'qiskit-qp-context.v1')

    def test_round_trip_restores_neutral_model_metadata_and_ranges(self):
        program, context = self.adapter.to_qiskit(
            self.cbqm,
            quadratic_program_class=_FakeQuadraticProgram,
        )

        restored = self.adapter.from_qiskit(program, context)

        self.assertEqual(restored['problem_id'], self.cbqm['problem_id'])
        self.assertEqual(restored['variables'], self.cbqm['variables'])
        self.assertEqual(restored['objective'], self.cbqm['objective'])
        self.assertEqual(restored['constraints'], self.cbqm['constraints'])
        self.assertEqual(restored['fixed_values'], self.cbqm['fixed_values'])
        self.assertEqual(restored['metadata']['owner'], 'test')
        self.assertEqual(restored['metadata']['source'], 'qiskit_optimization')

    def test_imports_external_qiskit_program_without_context(self):
        program = _FakeQuadraticProgram('external')
        program.binary_var('x')
        program.maximize(constant=0.0, linear={'x': 2.0}, quadratic={})
        program.linear_constraint(
            linear={'x': 1.0},
            sense='<=',
            rhs=1.0,
            name='cap',
        )

        converted = self.adapter.from_qiskit(program)

        self.assertEqual(converted['objective']['sense'], 'maximize')
        self.assertEqual(converted['objective']['linear'], [[0, 2.0]])
        self.assertEqual(converted['constraints'][0]['upper_bound'], 1.0)
        self.assertEqual(converted['constraints'][0]['family'], 'qiskit_linear')

    def test_rejects_constraint_name_collision_created_by_range_split(self):
        self.cbqm['constraints'].append(
            {
                'name': 'range__cbqm_lower',
                'family': 'collision',
                'linear': [[0, 1.0]],
                'upper_bound': 1.0,
            }
        )

        with self.assertRaisesRegex(ValueError, 'name collision'):
            self.adapter.to_qiskit(
                self.cbqm,
                quadratic_program_class=_FakeQuadraticProgram,
            )


if __name__ == '__main__':
    unittest.main()
