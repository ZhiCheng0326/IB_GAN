# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])

    unique_x, unique_x_indices, unique_inverse_x, unique_x_counts = np.unique(
        x, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )

    pxs = unique_x_counts / np.sum(unique_x_counts)

    unique_array_y, unique_y_indices, unique_inverse_y, unique_y_counts = np.unique(
        label, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )
    return pys, None, unique_x, unique_inverse_x, unique_inverse_y, pxs

def bin_calc_information(inputdata, layerdata, labelixs, num_of_bins):
    def get_h(d, num_of_bins):
        bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
#         digitized = np.floor(d / binsize).astype('int')
        digitized = bins[np.digitize(np.squeeze(d.reshape(1, -1)), bins) - 1].reshape(len(d), -1)
        p_ts, _ = get_unique_probs( digitized )
        return -np.sum(p_ts * np.log(p_ts))


    p_xs, unique_inverse_x = get_unique_probs(inputdata)
#     print(unique_inverse_x.shape) #(4096,)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs( digitized )

    H_LAYER = -np.sum(p_ts * np.log(p_ts))

    ##### calculate H(M|X) #####
    H_LAYER_GIVEN_INPUT = 0.
    for xval in np.arange(len(p_xs)):
#         print(digitized.shape) #(4096,10)
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))

    ##### calculate H(M|Y) #####
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs,:], num_of_bins)

    return H_LAYER - H_LAYER_GIVEN_INPUT, H_LAYER - H_LAYER_GIVEN_OUTPUT

def bin_calc_information2(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs( digitized )
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs,:])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT
