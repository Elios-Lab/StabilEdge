import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(os.environ.get('STABILEDGE_TRANSFORMER_REPO', ROOT.parent / 'Transformer-Position-Scheduling-Certification'))
PYTHON = Path(os.environ.get('STABILEDGE_TRANSFORMER_PYTHON', ROOT / '.venv-transformer/bin/python'))


@unittest.skipUnless(PYTHON.is_file() and REPOSITORY.is_dir(), 'Optional Transformer environment and source repository are required')
class TransformerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='stabiledge-test-')
        cls.directory = Path(cls.temporary.name)
        cls.environment = os.environ.copy()
        for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            cls.environment.pop(key, None)
        cls.environment['PYTHONDONTWRITEBYTECODE'] = '1'
        for name, extra in [('constant', []), ('tied', ['--tied'])]:
            subprocess.run([str(PYTHON), str(ROOT / 'tests/fixture_checkpoint.py'), '--repository', str(REPOSITORY),
                            '--output', str(cls.directory / name)] + extra, env=cls.environment, check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_engine(self, **changes):
        request = {'repository': str(REPOSITORY), 'checkpoint': str(self.directory / 'constant'),
                   'examples': [{'id': 'sentence', 'text': 'a good movie', 'label': 0}],
                   'methods': ['forward'], 'action': 'check', 'radius': .01, 'maximum_radius': .04,
                   'iterations': 2, 'device': 'cpu'}
        timeout = changes.pop('timeout', 0)
        request.update(changes)
        request_file = self.directory / 'request.json'
        result_file = self.directory / 'result.json'
        request_file.write_text(json.dumps(request))
        process = subprocess.run([str(PYTHON), str(ROOT / 'verification/transformer_runner.py'), '--request', str(request_file),
                                  '--output', str(result_file), '--timeout', str(timeout)], env=self.environment,
                                 capture_output=True, text=True, timeout=30)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(events[-1]['event'], 'finished', process.stderr)
        return json.loads(result_file.read_text()), events

    def test_real_forward_and_backward_certification(self):
        for method in ['forward', 'backward']:
            with self.subTest(method=method):
                result, events = self.run_engine(methods=[method])
                self.assertEqual(result['status'], 'CERTIFIED', result)
                self.assertEqual(result['results'][0]['radius'], .01)
                self.assertEqual(result['results'][0]['details']['task_count'], 3)

    def test_radius_search_and_pair_scheduling(self):
        for action, tokens in [('search', 1), ('schedule', 2)]:
            with self.subTest(action=action):
                result, _ = self.run_engine(action=action, perturbed_tokens=tokens)
                self.assertEqual(result['status'], 'CERTIFIED', result)
                self.assertEqual(result['results'][0]['radius'], .04)
                self.assertEqual(result['results'][0]['details']['task_count'], 3)

    def test_cascade_reuses_the_same_nominal_input(self):
        result, _ = self.run_engine(methods=['ibp', 'forward', 'backward'], action='search')
        self.assertEqual(result['status'], 'CERTIFIED', result)
        self.assertEqual(result['results'][0]['radius'], .04)

    def test_real_failure_to_certify_is_unknown(self):
        result, _ = self.run_engine(checkpoint=str(self.directory / 'tied'), action='search')
        self.assertEqual(result['status'], 'UNKNOWN', result)
        self.assertIsNone(result['results'][0]['radius'])

    def test_clean_misclassification_is_a_counterexample(self):
        result, _ = self.run_engine(examples=[{'text': 'a good movie', 'label': 1}])
        self.assertEqual(result['status'], 'COUNTEREXAMPLE')
        self.assertIsNone(result['results'][0]['radius'])

    def test_no_silent_text_truncation(self):
        result, _ = self.run_engine(max_length=3)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('no truncation', result['results'][0]['reason'])

    def test_invalid_token_selection_is_error(self):
        result, _ = self.run_engine(positions=[0])
        self.assertEqual(result['status'], 'ERROR')

    def test_deadline_does_not_become_certificate(self):
        result, _ = self.run_engine(timeout=.0001)
        self.assertEqual(result['status'], 'TIMEOUT')

    def test_preview_is_explicitly_unverified(self):
        result, _ = self.run_engine(action='preview')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['results'][0]['details']['valid_positions'], [1, 2, 3])
        self.assertIsNone(result['results'][0]['radius'])


if __name__ == '__main__':
    unittest.main()
