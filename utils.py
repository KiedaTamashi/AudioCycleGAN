import librosa
import numpy as np
EPSILON = 1e-2

def load_wav(path,sr):
    return librosa.core.load(path, sr=sr)[0]


def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()


def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1


def q_zero(q_levels):
    return q_levels // 2