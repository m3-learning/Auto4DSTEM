import pytest
import numpy as np
from auto4dstem.Viz.viz import cal_diff

def test_cal_diff():
    # Mock input data
    exx_correlation = np.array([[1, 2], [3, 4]])
    eyy_correlation = np.array([[2, 3], [4, 5]])
    exy_correlation = np.array([[3, 4], [5, 6]])
    theta_correlation = np.array([[10, 20], [30, 40]])

    exx_ae = np.array([[1.5, 2.5], [3.5, 4.5]])
    eyy_ae = np.array([[2.5, 3.5], [4.5, 5.5]])
    exy_ae = np.array([[3.5, 4.5], [5.5, 6.5]])
    theta_ae = np.array([[15, 25], [35, 45]])

    label_xx = np.array([[1, 1], [2, 2]])
    label_yy = np.array([[2, 2], [3, 3]])
    label_xy = np.array([[3, 3], [4, 4]])
    label_rotation = np.array([[5, 5], [5, 5]])

    # Expected differences
    expected_dif_correlation_xx = np.array([[0, 1], [1, 2]])
    expected_dif_correlation_yy = np.array([[0, 1], [1, 2]])
    expected_dif_correlation_xy = np.array([[0, 1], [1, 2]])
    expected_dif_correlation_rotation = np.array([[5, 15], [25, 35]])

    expected_dif_ae_xx = np.array([[0.5, 1.5], [1.5, 2.5]])
    expected_dif_ae_yy = np.array([[0.5, 1.5], [1.5, 2.5]])
    expected_dif_ae_xy = np.array([[0.5, 1.5], [1.5, 2.5]])
    expected_dif_ae_rotation = np.array([[10, 20], [30, 40]])

    # Call the function
    diffs = cal_diff(exx_correlation, eyy_correlation, exy_correlation, theta_correlation,
                     exx_ae, eyy_ae, exy_ae, theta_ae,
                     label_xx, label_yy, label_xy, label_rotation)

    # Compare each difference with the expected result
    np.testing.assert_array_equal(diffs[0], expected_dif_correlation_xx)
    np.testing.assert_array_equal(diffs[1], expected_dif_correlation_yy)
    np.testing.assert_array_equal(diffs[2], expected_dif_correlation_xy)
    np.testing.assert_array_equal(diffs[3], expected_dif_correlation_rotation)
    np.testing.assert_array_equal(diffs[4], expected_dif_ae_xx)
    np.testing.assert_array_equal(diffs[5], expected_dif_ae_yy)
    np.testing.assert_array_equal(diffs[6], expected_dif_ae_xy)
    np.testing.assert_array_equal(diffs[7], expected_dif_ae_rotation)

