import unittest
import numpy as np
from lorenz import EDOs, RK4, sigma, rho, beta

class TestLorenzSystem(unittest.TestCase):
    
    def test_EDOs_values(self):
        t = 0
        r = [1.0, 2.0, 3.0]
        expected = np.array([
            sigma * (2.0 - 1.0),
            rho * 1.0 - 2.0 - 1.0 * 3.0,
            1.0 * 2.0 - beta * 3.0
        ])
        result = EDOs(t, r)
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_RK4_shape(self):
        t = 0
        r = [1.0, 2.0, 3.0]
        dt = 0.01
        result = RK4(t, r, EDOs, dt)
        self.assertEqual(len(result), 3)

    def test_RK4_small_step(self):
        t = 0
        r = np.array([1.0, 2.0, 3.0])
        dt = 1e-10
        result = RK4(t, r, EDOs, dt)
        np.testing.assert_array_almost_equal(result, r, decimal=8)

if __name__ == '__main__':
    unittest.main()
