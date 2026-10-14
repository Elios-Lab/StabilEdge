import math
import unittest
from verification.core import Status, VerificationResult, aggregate_queries, solver_status
from verification.radius import RadiusBound, search_radius, schedule_tasks
from verification.transformer_request import position_tasks


class CertificationTests(unittest.TestCase):
    def test_no_successful_query_cannot_produce_radius(self):
        bound = search_radius(lambda radius: False)
        self.assertFalse(bound.certified)
        self.assertIsNone(bound.radius)

    def test_only_the_base_point_can_be_certified(self):
        bound = search_radius(lambda radius: radius == 0)
        self.assertTrue(bound.certified)
        self.assertEqual(bound.radius, 0)

    def test_radius_is_a_certified_lower_endpoint(self):
        seen = []
        def query(radius):
            seen.append(radius)
            return radius <= 0.135
        bound = search_radius(query, initial=0.01, maximum=0.3, iterations=8)
        self.assertLessEqual(bound.radius, 0.135)
        self.assertGreater(bound.upper, 0.135)
        self.assertIn(bound.radius, seen)

    def test_search_never_exceeds_ceiling(self):
        queried = []
        def query(radius):
            queried.append(radius)
            return True
        bound = search_radius(query, initial=0.03, maximum=0.1)
        self.assertEqual(bound.radius, 0.1)
        self.assertTrue(bound.at_cap)
        self.assertLessEqual(max(queried), 0.1)

    def test_tolerance_controls_refinement(self):
        bound = search_radius(lambda radius: radius <= 0.017, initial=.01, maximum=.1, tolerance=1e-5)
        self.assertLessEqual(bound.upper - bound.radius, 1e-5)

    def test_nonfinite_search_configuration_rejected(self):
        for initial in [0, -1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                search_radius(lambda radius: True, initial=initial)

    def test_scheduling_validates_skipped_tasks_and_searches_failures(self):
        searched, checked = [], []
        radii = {(1,): .2, (2,): .1, (3,): .3}
        def search(task):
            searched.append(task)
            return RadiusBound(radii[task], True, .4, 4)
        def validate(task, radius):
            checked.append((task, radius))
            return radii[task] >= radius
        result = schedule_tasks(list(radii), list(radii), 1, search, validate)
        self.assertEqual(result['radius'], .1)
        self.assertTrue(result['certified'])
        self.assertEqual(searched, [(1,), (2,)])
        self.assertEqual(checked, [((2,), .2), ((3,), .2)])
        self.assertEqual(result['fallback'], [(2,)])

    def test_unknown_fallback_does_not_certify(self):
        def search(task):
            return RadiusBound(.2, True, .3, 4) if task == (1,) else RadiusBound(None, False, .3, 0)
        result = schedule_tasks([(1,), (2,)], [(1,), (2,)], 1, search, lambda task, radius: False)
        self.assertFalse(result['certified'])
        self.assertIsNone(result['radius'])

    def test_empty_task_set_rejected(self):
        with self.assertRaises(ValueError):
            schedule_tasks([], [], 1, lambda _: None, lambda *_: True)

    def test_single_and_pair_properties_have_explicit_scope(self):
        self.assertEqual(position_tasks([1, 2, 3], [1, 3], 1), [(1,), (3,)])
        self.assertEqual(position_tasks([1, 2, 3], [], 2), [(1, 2), (1, 3), (2, 3)])
        with self.assertRaises(ValueError):
            position_tasks([1, 2], [0], 1)

    def test_cancelled_incomplete_and_unknown_queries_are_not_unsat(self):
        self.assertEqual(aggregate_queries([], 2), 'unknown')
        self.assertEqual(aggregate_queries(['unsat'], 2), 'unknown')
        self.assertEqual(aggregate_queries(['unsat', 'unknown'], 2), 'unknown')
        self.assertEqual(aggregate_queries(['unsat'], 2, cancelled=True), 'cancelled')
        self.assertEqual(aggregate_queries(['unsat', 'unsat'], 2), 'unsat')
        self.assertEqual(aggregate_queries(['sat'], 2, cancelled=True), 'sat')

    def test_unknown_does_not_become_counterexample(self):
        self.assertEqual(solver_status('unknown'), Status.UNKNOWN)
        self.assertEqual(solver_status('TIMEOUT'), Status.TIMEOUT)
        with self.assertRaises(ValueError):
            VerificationResult('transformer', '0', Status.UNKNOWN, radius=.1).to_dict()


if __name__ == '__main__':
    unittest.main()
