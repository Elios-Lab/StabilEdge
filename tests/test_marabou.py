"""Real solver regressions, run in verify_gui rather than the text environment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('maraboupy') and importlib.util.find_spec('onnx'), 'Marabou environment required')
class MarabouTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        cls.temporary = tempfile.TemporaryDirectory(prefix='stabiledge-marabou-')
        cls.model = str(Path(cls.temporary.name) / 'model.onnx')
        graph = helper.make_graph(
            [helper.make_node('Gemm', ['input', 'weight', 'bias'], ['output'])], 'affine_classifier',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.array([[1, 0]] * 4, dtype=np.float32), 'weight'),
             numpy_helper.from_array(np.array([0, 1], dtype=np.float32), 'bias')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        onnx.save(model, cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_local_and_targeted_universal_properties(self):
        import numpy as np
        from scripts.ITD import local_rob_sens, targeted_rob_sens
        for verifier in [local_rob_sens, targeted_rob_sens]:
            with self.subTest(verifier=verifier.__name__):
                report, status, stats = verifier(np.full((2, 2), .5), 'sample', .05, self.model,
                                                 tmin=0, tmax=1, perturb_all_pixels=True,
                                                 marabou_opts={'verbosity': 0})
                self.assertEqual(status, 'unsat')
                self.assertEqual(report['completed_queries'], report['expected_queries'])

    def test_normalized_delta_and_selected_features(self):
        import numpy as np
        from scripts import ITD
        seen = []
        original = ITD.set_pixel_bounds_adaptive
        def record(*args, **kwargs):
            seen.append((args[3], kwargs['perturb_all'], kwargs['selected_idxs']))
            return original(*args, **kwargs)
        with patch.object(ITD, 'set_pixel_bounds_adaptive', side_effect=record):
            ITD.local_rob_sens(np.full((2, 2), .5), 'sample', 5, self.model, tmin=0, tmax=100,
                              selected_idxs=[0], marabou_opts={'verbosity': 0})
        self.assertEqual(seen, [(.05, False, [0])])

    def test_cancellation_cannot_return_unsat(self):
        import numpy as np
        from scripts.ITD import local_rob_sens, targeted_rob_sens
        for verifier in [local_rob_sens, targeted_rob_sens]:
            report, status, stats = verifier(np.full((2, 2), .5), 'sample', .05, self.model,
                                             tmin=0, tmax=1, stop_requested=lambda: True)
            self.assertEqual(status, 'cancelled')
            self.assertEqual(report['completed_queries'], 0)

    def test_input_size_mismatch_is_rejected(self):
        import numpy as np
        from scripts.ITD import local_rob_sens
        with self.assertRaisesRegex(ValueError, 'model expects'):
            local_rob_sens(np.ones((3, 3)), 'sample', .1, self.model, tmin=0, tmax=1)

    def test_unknown_solver_status_remains_unknown(self):
        import numpy as np
        from scripts.ITD import local_rob_sens
        from maraboupy import Marabou
        original = Marabou.read_onnx
        def load(*args, **kwargs):
            network = original(*args, **kwargs)
            network.solve = lambda **kwargs: ('unknown', {}, None)
            return network
        with patch.object(Marabou, 'read_onnx', side_effect=load):
            _, status, _ = local_rob_sens(np.full((2, 2), .5), 'sample', .05, self.model,
                                         tmin=0, tmax=1, perturb_all_pixels=True)
        self.assertEqual(status, 'unknown')


if __name__ == '__main__':
    unittest.main()
