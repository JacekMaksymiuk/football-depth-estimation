import numpy as np

def compute_scale_and_shift_np(from_arr, to_arr, mask):
    a_00 = np.sum(mask * from_arr * from_arr, axis=(0, 1))
    a_01 = np.sum(mask * from_arr, axis=(0, 1))
    a_11 = np.sum(mask, axis=(0, 1))
    b_0 = np.sum(mask * from_arr * to_arr, axis=(0, 1))
    b_1 = np.sum(mask * to_arr, axis=(0, 1))
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1
