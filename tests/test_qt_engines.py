import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / 'Transformer-Position-Scheduling-Certification'
PYTHON = ROOT / '.venv-transformer/bin/python'


@unittest.skipUnless(importlib.util.find_spec('PyQt5') and PYTHON.is_file() and REPO.is_dir(), 'GUI and Transformer environments required')
class QtEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import StabilEdge_GUI  # QtWebEngine must be imported before QApplication.
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        cls.temporary = tempfile.TemporaryDirectory(prefix='stabiledge-qt-')
        cls.directory = Path(cls.temporary.name)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('LD_LIBRARY_PATH', None)
        subprocess.run([str(PYTHON), str(ROOT / 'tests/fixture_checkpoint.py'), '--repository', str(REPO),
                        '--output', str(cls.directory / 'checkpoint')], env=env, check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def request(self):
        from verification.transformer_request import TransformerRequest
        return TransformerRequest(str(REPO), str(self.directory / 'checkpoint'),
                                  [{'text': 'a good movie', 'label': 0}], methods=['forward'])

    def execute(self, executable=PYTHON, cancel=False, timeout=0):
        from PyQt5.QtCore import QEventLoop, QTimer
        from verification.process import TransformerProcess
        controller = TransformerProcess(runs_root=self.directory / 'runs')
        loop = QEventLoop()
        results, events = [], []
        controller.event_received.connect(events.append)
        controller.finished.connect(lambda result: (results.append(result), loop.quit()))
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        controller.start(self.request(), str(executable), timeout)
        if cancel:
            QTimer.singleShot(100, controller.cancel)
        guard.start(15000)
        loop.exec_()
        guard.stop()
        if not results:
            controller.shutdown()
        self.assertTrue(results, 'Process did not finish within the test deadline')
        self.assertFalse(controller.running)
        self.assertTrue((Path(results[0]['run_directory']) / 'results.json').exists())
        return results[0], events

    def test_gui_engine_selection(self):
        import StabilEdge_GUI as gui
        window = gui.MainWindow()
        self.assertEqual(window.engineSelector.count(), 2)
        window.engineSelector.setCurrentIndex(1)
        self.assertIs(window.enginePages.currentWidget(), window.transformerPanel)
        window.engineSelector.setCurrentIndex(0)
        self.assertEqual(window.engineSelector.currentData(), 'marabou')
        window.close()

    def test_configuration_roundtrip_preserves_embedded_batch(self):
        from verification.transformer_panel import TransformerPanel
        request = self.request()
        request.examples = [{'id': 'first', 'text': 'a good movie', 'label': 0},
                            {'id': 'second', 'text': 'a bad movie', 'label': 1}]
        request.methods = ['backward', 'forward']
        request.action = 'schedule'
        request.positions = [1, 3]
        path = self.directory / 'config.json'
        path.write_text(json.dumps({'schema_version': 1, 'engine': 'transformer', 'request': request.to_dict(),
                                    'python_executable': str(PYTHON), 'timeout': 12}))
        panel = TransformerPanel()
        with patch('verification.transformer_panel.QFileDialog.getOpenFileName', return_value=(str(path), '')):
            panel._load_config()
        self.assertEqual(panel.request().to_dict(), request.to_dict())
        self.assertEqual(panel.timeout.value(), 12)
        panel.close()

    def test_extracted_marabou_worker_with_grayscale_image(self):
        import numpy as np
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        from PIL import Image
        from PyQt5.QtCore import QEventLoop, QTimer
        from verification.registry import create_marabou_worker
        model_file = self.directory / 'affine.onnx'
        image_file = self.directory / 'grayscale.png'
        graph = helper.make_graph([helper.make_node('Gemm', ['input', 'weight', 'bias'], ['output'])],
                                  'affine', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
                                  [numpy_helper.from_array(np.array([[1, 0]] * 4, dtype=np.float32), 'weight'),
                                   numpy_helper.from_array(np.array([0, 1], dtype=np.float32), 'bias')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        onnx.save(model, str(model_file))
        Image.fromarray(np.full((2, 2), 128, dtype=np.uint8)).save(str(image_file))
        worker = create_marabou_worker(pipeline='ITD', vt_mode='Local Robustness', combos=[(1., None, 0)],
                                       perturb_all=True, model_file=str(model_file), input_path=str(image_file),
                                       json_path='', selected_idxs=[], plot=False, verbosity=0,
                                       feature_cfg={'features': [{'properties': {'input_domain': {'min': 0, 'max': 255}}}]},
                                       selected_classes=None)
        results = []
        loop = QEventLoop()
        worker.result.connect(results.append)
        worker.finished.connect(loop.quit)
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        guard.start(15000)
        worker.start()
        loop.exec_()
        guard.stop()
        self.assertTrue(worker.wait(1000))
        self.assertEqual([row['status'] for row in results], ['CERTIFIED'], results)
        worker.deleteLater()

    def test_real_qprocess_transport(self):
        result, events = self.execute()
        self.assertEqual(result['status'], 'CERTIFIED', result)
        self.assertTrue(any(event['event'] == 'result' for event in events))
        self.assertTrue((Path(result['run_directory']) / 'events.jsonl').exists())

    def test_stop_preserves_cancellation(self):
        result, _ = self.execute(cancel=True)
        self.assertEqual(result['status'], 'CANCELLED')

    def test_deadline_preserves_timeout(self):
        result, _ = self.execute(timeout=.0001)
        self.assertEqual(result['status'], 'TIMEOUT')

    def test_nonzero_exit_is_error(self):
        result, _ = self.execute(executable=Path('/bin/false'))
        self.assertEqual(result['status'], 'ERROR')

    def test_partial_result_file_is_not_a_completed_certificate(self):
        executable = self.directory / 'fake-python'
        executable.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                              'p=Path(sys.argv[sys.argv.index("--output")+1])\n'
                              'p.write_text(json.dumps({"schema_version":1,"engine":"transformer","status":"CERTIFIED","results":[]}))\n')
        executable.chmod(0o755)
        result, _ = self.execute(executable=executable)
        self.assertEqual(result['status'], 'ERROR')


if __name__ == '__main__':
    unittest.main()
