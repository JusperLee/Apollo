# Adapted from Neuracoust Sound Build-up experimental post-processing.
# Changes: explicit zero-strength bypass and input validation.
# Distributed under the repository's CC BY-SA 4.0 license.
"""Offline candidate: bounded temporal spectral excursion reduction, no dry input.
Not a defect classifier. Musical fluctuations may also trigger the detector.
"""
import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import median_filter, gaussian_filter1d, maximum_filter1d


def soften_texture(audio, sample_rate=44100, max_reduction_db=2.0):
    x = np.asarray(audio)
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[-1] < 2048
            or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()):
        raise ValueError('Expected finite channels-by-frames audio, at least 2048 frames')
    if not np.isfinite(sample_rate) or sample_rate < 16000 or not 0 <= max_reduction_db <= 3:
        raise ValueError('Unsupported rate or attenuation')
    if max_reduction_db == 0:
        return x.copy(), {'max_reduction_db': 0.0,
                          'power_weighted_reduction_db': 0.0,
                          'attack_frame_fraction': 0.0}
    kw = dict(fs=sample_rate, nperseg=1024, noverlap=768)
    f, _, z = stft(np.pad(x, ((0,0),(1024,1024)), mode='reflect'), **kw)
    power = np.mean(abs(z)**2, axis=0)
    level = 10*np.log10(np.maximum(power, 1e-16))
    # About 40 ms neighborhood: only positive, short-lived excursions.
    frames = max(3, int(round(.04*sample_rate/256)) | 1)
    baseline = median_filter(level, size=(1,frames), mode='nearest')
    excess = np.maximum(level-baseline-3, 0)
    # Broad attack gate protects simultaneous rising energy (not guaranteed perfect).
    energy = power.sum(axis=0)
    attack = np.r_[False, energy[1:] > 1.4*np.maximum(energy[:-1],1e-16)]
    gate = 1-maximum_filter1d(attack.astype(float), size=5)
    band = np.clip((f-1200)/800,0,1)*np.clip((14000-f)/4000,0,1)
    active = power > max(float(power.max())*1e-7,1e-15)
    reduction = np.minimum(excess*.35,max_reduction_db)*band[:,None]*gate*active
    reduction = gaussian_filter1d(gaussian_filter1d(reduction,.7,axis=0),.8,axis=1)
    _, y = istft(z*10**(-reduction[None,:,:]/20), **kw)
    y = y[:,1024:1024+x.shape[-1]].astype(np.float32)
    return y, {'max_reduction_db':float(reduction.max()),
               'power_weighted_reduction_db':float((reduction*power).sum()/max(power.sum(),1e-30)),
               'attack_frame_fraction':float(np.mean(gate==0))}
