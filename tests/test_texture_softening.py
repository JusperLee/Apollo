import unittest
import numpy as np
from texture_softening import soften_texture


class TextureTests(unittest.TestCase):
    def test_exact_bypass(self):
        x = np.random.default_rng(4).normal(0, .1, (2, 16000)).astype('float32')
        y, _ = soften_texture(x, max_reduction_db=0)
        np.testing.assert_array_equal(x, y)

    def test_steady_tone_stereo_and_length(self):
        a = .2*np.sin(2*np.pi*3000*np.arange(44101)/44100)
        x = np.array([a, -.3*a], dtype='float32')
        y, _ = soften_texture(x)
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(y[1], -.3*y[0], atol=1e-7)
        self.assertLess(np.sqrt(np.mean((x[:,2048:-2048]-y[:,2048:-2048])**2)), 1e-5)

    def test_silence(self):
        y, _ = soften_texture(np.zeros((2, 4410), dtype='float32'))
        self.assertEqual(float(abs(y).max()), 0)

    def test_actual_modulation_is_bounded(self):
        t = np.arange(44100)/44100
        envelope = .02*(.1 + (np.mod(t, .08) < .008))
        x = np.array([.4*np.sin(2*np.pi*220*t) + envelope*np.sin(2*np.pi*4000*t)], dtype='float32')
        y, stats = soften_texture(x)
        self.assertGreater(stats['max_reduction_db'], .01)
        self.assertLessEqual(stats['max_reduction_db'], 2)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(np.sqrt(np.mean((x-y)**2)), 1e-5)

    def test_invalid(self):
        for x in (np.zeros(4410), np.zeros((0,4410)), np.zeros((2,100)),
                  np.full((2,4410), np.nan), np.zeros((2,4410), dtype='complex64')):
            with self.assertRaises(ValueError):
                soften_texture(x)
        for cap in (-1, 4, np.nan, np.inf):
            with self.assertRaises(ValueError):
                soften_texture(np.zeros((1,4410)), max_reduction_db=cap)


if __name__ == '__main__':
    unittest.main()
