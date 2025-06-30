import numpy as np
import scipy as sp
from numba import njit
from scipy.special import i0
from numpy.random import vonmises
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import h5py





# Read through function 25 March 2025 by Niko Schonsheck.
# Passed validation testing: validation_testing/circular_distributions.py at git hash 06ab7c319f5139c33b99132c3efb907606b4d277
@njit
def vonmises_kde_on_circle_given_i_naught_kappa(data, kappa, i_naught_kappa, n_bins = 100):
    """
    Compute KDE on circle = [-pi, pi]/(pi ~ -pi) given I_0(kappa) using von Mises distribution.

    Arguments:
        - data (numpy array of shape (num_data_points, )) --- Represents values of a random variable taking values in the circle understood as [-np.pi, np.pi] where -np.pi ~ np.pi
        - kappa (float) --- Parameter for von Mises distribution
        - i_naught_kappa (float) --- Value of the modified Bessel function of the first kind of order 0 at kappa
        - n_bins (int) --- Number of discrete bins into which to partition the circle
    Returns:
        - kde_result (numpy array of shape (num_bins, 2)) --- Representing the kernel density estimation. That is, plotting x = kde_result[:, 0], y = kde_result[:, 1] will draw the estimated probability density function of X.
    """

    # Set x-points at which we'll evaluate the kde
    xs_for_kde = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # Remove last entry because of gluing -np.pi to np.pi
    xs_for_kde = xs_for_kde[:-1]
    
    # Initialize zero array for values of kde
    ys_for_kde = np.zeros(xs_for_kde.shape)
    
    # For each x at which to evaluate kde
    for index, x in enumerate(xs_for_kde):
        # Calculate kde at x by summing kernel function over all data points
        y = 0
        for x_i in data:
            y = y + np.exp(kappa*np.cos(x - x_i))/(2*np.pi*i_naught_kappa)
        ys_for_kde[index] = y/len(xs_for_kde)
    
    # Normalize so that integral over circle is 1
    total_area = (2*np.pi/n_bins)*np.sum(ys_for_kde)
    ys_for_kde = ys_for_kde/total_area

    # Stack x and y values for final return
    kde_result = np.column_stack((xs_for_kde, ys_for_kde))

    # Final return
    return kde_result

    

# Read through function 25 March 2025 by Niko Schonsheck.
# Passed validation testing: validation_testing/circular_distributions.py at git hash 06ab7c319f5139c33b99132c3efb907606b4d277
def vonmises_kde_on_circle(data, kappa, n_bins = 100):
    """
    Compute KDE on circle = [-pi, pi]/(pi ~ -pi) using von Mises distribution.

    Arguments:
        - data (numpy array of shape (num_data_points, )) --- Represents values of a random variable taking values in the circle understood as [-np.pi, np.pi] where -np.pi ~ np.pi
        - kappa (float) --- Parameter for von Mises distribution
        - n_bins (int) --- Number of discrete bins into which to partition the circle
    Returns:
        - kde_result (numpy array of shape (num_bins, 2)) --- Represents the kernel density estimation. That is, plotting x = kde_result[:, 0], y = kde_result[:, 1] will draw the estimated probability density function of X.
    """
    # Get value of modified Bessel function of the first kind of order 0
    i_naught_kappa = i0(kappa)

    # Calculate result with njit speedup
    result = vonmises_kde_on_circle_given_i_naught_kappa(data, kappa, i_naught_kappa, n_bins)

    # Final return
    return result



# Passed validation_testing/circular_distance_neg_pi_to_pi at git hash 6052f4679b6803645d7014a171cca43ed08dfe5b
def circular_distance_neg_pi_to_pi(a, b):
    """
    Calculate distance on a circle = [-pi, pi]/(-pi ~ pi)

    Arguments:
        - a (float) --- a number in [-pi, pi]
        - b (float) --- a number in [-pi, pi]

    Returns:
        - distance (float) --- distance on the circle [-pi, pi]/(-pi ~ pi) between `a` and `b`
    """

    # Check valid inputs
    if a < -np.pi or a > np.pi or b < - np.pi or b > np.pi:
        raise ValueError('Variables a and b must be in [-pi, pi]')

    # If numbers are equal, return 0.0
    if a == b:
        return 0.0

    # Otherwise, find smaller and larger
    elif a < b:
        smaller = a
        larger = b

    else:
        smaller = b
        larger = a
    
    # Compute distance without wrapping
    direct_distance = larger - smaller

    # Compute distance wrapping
    wrap_distance = np.abs(np.pi - larger) + np.abs(smaller - -1*np.pi)

    # Return min of direct and wrap distances 
    distance = min(direct_distance, wrap_distance)

    return distance






# Checked through 25 March 2025 by Niko Schonsheck.
# Modified 26 March 2025 by Niko Schonsheck to include flag value for empty empirical distribution.
# Read through again 26 March 2025 by Niko Schonsheck as check.
def get_kde_and_locality_score(output_neuron_index, nonzero_input_indices, weight_matrix, input_neurons_tuning_curve_tensor, kappa = 20, n_bins = 100):
    """
    Summary:
        Given a FF network from N_in excitatory neurons in a circular stimulus space to N_out neurons and fixed `output_neuron_index`, get
            - KDE approximation of the PDF of the distribution on S^1 determined by synaptic weights
            - locality score for `output_neuron_index` defined as the maximum value of the KDE over S^1
    Arguments:
        - output_neuron_index (int) --- the output neuron of which to compute the KDE and locality score
        - nonzero_input_indices (list of int) --- input neuron indices over which to construct the KDE. (E.g., those that are nonzero at the outset of learning.)
        - weight_matrix (numpy array of shape (N_out, N_in)) --- weight matrix of the FF neetwork
        - input_neurons_tuning_curve_tensor --- (numpy array of shape (?, 2, N_in)) tuning curves of input neurons such that input_neurons_tuning_curve_tensor[:, :, index] is the tuning curve of input neuron `index` (col 1 is sorted x_vales, col_2 is associated firing rate)
            - WARNING WARNING: we assume each tuning curve has x-values in [0, 1) (so, S^1 = R/Z) and convert these values to S^1 = [-pi, pi]/(-pi ~ pi)
        - kappa (float) --- parameter for von Mises KDE
        - n_bins (int) --- number of bins to use for the kernel density estimation
    
    Returns:
        - kde (numpy array of shape (n_bins, 2)) --- kernel density estimation where each row is a pair [point_on_circle, value_of_kde]
        - locality_score (float) --- maximum value of kde over the interval [-pi, pi)

    Further details:
        - For sake of illustration, assume input neurons (determined by nonzero_input_indices)are arrayed uniformly over S^1 = R/Z. (Note that in the function, we use [-pi, pi] with -pi ~ pi.)
        - Define a measure (i.e., distribution) on S^1 by defining measure of each interval of width 1/N_in around each input neuron to be the synaptic weight from `output_neuron_index` to that input neuron.
        - Take samples from this distribution on S^1 to obtain an empirical estimation of this distribution.
        - Approximate the PDF of this distribution using VonMises kernel density estimation on S^1.
        - Return this KDE and its maximum value on S^1.
        - So, we assumpe synaptic weights come from an unknown distribution on S^1 and are approximating this distribution.
        - One could also understand this as defining a measure /nu on S^1 as above that is absolutely continuous with respsect to Lebesgue measue and then approximating the Radon-Nikodym derivative.

    Notes:
        - flag return value for the case where empirical distribution is empty: (None, np.nan)
    """

    # Initialize array of samples
    circular_data = []

    # For each input neuron to consider...
    for presyn_index in nonzero_input_indices:

        # Get tuning curve
        tuning_curve = input_neurons_tuning_curve_tensor[:, :, presyn_index]

        # Find position of input neuron in circular stimulus space = R/Z by taking value of its maximum
        row_index_of_max_firing_rate = np.argmax(tuning_curve[:, 1])
        position_of_max_firing_rate = tuning_curve[row_index_of_max_firing_rate, 0]

        # Convert position to interval [-pi, pi)
        position_of_max_firing_rate = 2*np.pi*position_of_max_firing_rate - np.pi

        # Get synaptic weight
        weight = weight_matrix[output_neuron_index, presyn_index]

        # Generate samples
        for _ in range(int(np.floor(weight*1000))):
            circular_data.append(position_of_max_firing_rate)

    
    # Flag value for empty empirical distribution
    if len(circular_data) == 0:
        kde, locality_score = (None, np.nan)
    
    else:
        # Do KDE
        kde = vonmises_kde_on_circle(circular_data, kappa, n_bins = n_bins)

        # Get locality score
        locality_score = np.max(kde[:, 1])

    # Final return
    return kde, locality_score






# Checke by read-through on 25 March 2025 by Niko Schonsheck.
# Modified to remove flag value check (this is moved to function get_kde_and_locality_score) and re-checked with read-through
# on 26 March 2025 by Niko Schonsheck.
def make_locality_dictionary(initial_weight_matrix, weight_matrix_to_analyze, input_neurons_tuning_curve_tensor, kappa = 20, n_bins = 100):
    """
    Summary:
        - Get KDE and locality score for each output neuron in `weight_matrix_to_analyze` using `initial_weight_matrix` to find, for each output neuron, nonzero initial connections.
        - Suppose network is FF architecture N_in -> N_out neurons
    
    Arguments:
        - initial_weight_matrix (numpy array of shape (N_out, N_in)) --- initial weight matrix of the network (e.g., pre-learning)
        - weight_matrix_to_analyze (numpy array of shape (N_out, N_in)) --- weight matrix from which to compute KDE's and locality score
        - input_neurons_tuning_curve_tensor --- (numpy array of shape (?, 2, N_in)) tuning curves of input neurons such that input_neurons_tuning_curve_tensor[:, :, index] is the tuning curve of input neuron `index` (col 1 is sorted x_vales, col_2 is associated firing rate)
            - WARNING WARNING: we assume each tuning curve has x-values in [0, 1) (so, S^1 = R/Z) and convert these values to S^1 = [-pi, pi]/(-pi ~ pi)
        - kappa (float) --- parameter for von Mises KDE
        - n_bins (int) --- number of bins to use for the kernel density estimation
    Returns:
        - locality_dict (dict) --- key => val where 
            - key (int) ranges over output neuron indices (i.e., row indices of the weight matrices)
            - val (tuple) is a tuple (kde, locality_score) that is the return of function get_kde_and_locality_score
    """

    # Check that matrices are the same shape
    if initial_weight_matrix.shape != weight_matrix_to_analyze.shape:
        raise ValueError('Variables `initial_weight_matrix` and `weight_matrix_to_analyze` must be the same shape.')

    # Get number of output neurons
    num_output_neurons, num_input_neurons = initial_weight_matrix.shape

    # Initialize locality_dict
    locality_dict = {}

    # For each output neuron index...
    for output_neuron_index in range(num_output_neurons):

        ## Get input neuron indices to which the output neuron starts with nonzero connection
        nontrivial_input_indices = []
        for input_neuron_index in range(num_input_neurons):
            if initial_weight_matrix[output_neuron_index, input_neuron_index] != 0:
                nontrivial_input_indices.append(input_neuron_index)

        ## Compute KDE and locality score
        (kde, locality_score) = get_kde_and_locality_score(output_neuron_index, nontrivial_input_indices, weight_matrix_to_analyze, input_neurons_tuning_curve_tensor, kappa, n_bins)

        ## Add to dictionary
        locality_dict[output_neuron_index] = (kde, locality_score)

    # Final return
    return locality_dict






# Passed test_find_indices_of_peaks_on_circle() at git hash 214c0bd9c9b4617b056f5977e4e91069f85746a9
def find_indices_of_peaks_on_circle(circular_signal, width_in_samples, min_height, pad):
    """
    Arguments:
        - circular_signal (numpy array of shape (length, )) --- a 1-d array of values on a circle = `[a, b]/(a ~ b)` such that `circular_signal[0]` is the signal value at `a` and `circular_signal[length - 1]` is the signal value at `b - (b-a)/length`.
        - width_in_samples (int) --- minimum width used to find peaks (see documentation for scipy.signal.find_peaks())
        - min_height (int) --- minimum height used to find peaks (see documentation for scipy.signal.find_peaks())
        - pad (int) --- number of samples with which to pad `circular_signal` to account for periodic boundary condition

    Returns:
        - actual_indices_of_peak_indices (numpy array of shape (num_peaks, )) --- list of indices at which peak values occur, taking into account periodic boundary condition.
    
    Notes:
        - From scipy.signal.find_peaks() documentation and https://stackoverflow.com/questions/67157264/how-can-i-interpret-the-parameter-width-in-scipy-peak-detection, parameter `width` is the number of consecutive samples above set `height` required to be considered a peak.
        """
    
    # Get length of the signal
    length = len(circular_signal)

    # Pad signal to account for periodic boundary conditions
    padded_raster = np.concatenate((circular_signal[-pad:], circular_signal, circular_signal[:pad]))

    # Get indices of the peaks of the padded signal
    indices_of_peaks_of_padded, _ = find_peaks(padded_raster, height = min_height, width = width_in_samples)

    # Convert to indices of the original signal
    actual_indices_of_peaks = []
    for peak_index in indices_of_peaks_of_padded:
        actual_index = int((peak_index - pad) % length)
        actual_indices_of_peaks.append(actual_index)
    
    # Convert to array and find unique peaks
    actual_indices_of_peaks = np.array(actual_indices_of_peaks)
    actual_indices_of_peaks = np.unique(actual_indices_of_peaks)

    # Final return
    return actual_indices_of_peaks




def test_find_indices_of_peaks_on_circle():
    # Unimodal at boundary
    sample_1 = np.zeros((100))
    sample_1[0] = 50
    sample_1[1] = 40
    sample_1[2] = 12
    sample_1[99] = 49
    sample_1[98] = 30
    sample_1[97] = 35

    # Bimodal, but second peak below default threshold height
    sample_2 = np.zeros((100))
    sample_2[9] = 10
    sample_2[10] = 20
    sample_2[11] = 30
    sample_2[12] = 50
    sample_2[13] = 40
    sample_2[14] = 20
    sample_2[15] = 5
    sample_2[80] = 5
    sample_2[81] = 15
    sample_2[82] = 20
    sample_2[83] = 20
    sample_2[84] = 5
    sample_2[85] = 5



    # Test sample_1
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_1, 3, min_height=25, pad=5), np.array([0]))
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_1, 4, min_height=25, pad=5), np.array([0]))
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_1, 5, min_height=25, pad=5), np.array([]))

    # Test sample_2
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_2, 5, min_height=25, pad=5), np.array([]))
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_2, 3, min_height=25, pad=5), np.array([12]))
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_2, 3, min_height=20, pad=5), np.array([12, 82]))
    assert np.array_equal(find_indices_of_peaks_on_circle(sample_2, 4, min_height=20, pad=5), np.array([]))



    print('Function find_indices_of_peaks_on_circle: passed.')



# Passed test_classify_tuning_curve() at git hash f7eb0f0d0675e17a1df838a28555edf544fffebb for multiple values of initialization distribution and Skaggs info of L1 neurons.
# Please note that classifying tuning curves as strictly unimodal, bimodal, or neither will always be open to interpretation and is not strictly defined. Therefore, this classification is
# coarse and imperfect and should only be used as a rough measure of the types of tuning curves produced in L3.
def classify_tuning_curve(tuning_curve, width_in_samples, sigma = 5, min_height = 25, pad = 10):
    """
    Arguments:
        - tuning_curve (numpy array of shape (num_samples, )) --- tuning curve of a neuron where tuning_curve[index] is the firing rate of the neuron at some x-position corresponding to index (implicit)
        - width_in_samples (int) --- minimum width used to find peaks (see documentation for find_indices_of_peaks_on_circle)
        - sigma (int) --- sigma used for Gaussian smoothing.
        - min_height (int) --- minimum height used to find peaks (see documentation for find_indices_of_peaks_on_circle)
        - pad (int) --- number of samples with which to pad `circular_signal` to account for periodic boundary condition
    
    Returns:
        - tuple (classification_type, smoothed_tuning_curve) 
            - classification_type (int) --- 1 for unimodal, 2 for bimodal, 3 for other. Based on number of peaks detected after Gaussian smoothing. 
            - smoothed_tuning_curve (numpy array of shape (num_samples, )) --- tuning curve after Gaussian smoothing and normalization to same max value.
    """

    # Get max of original tuning curve
    max_rate = np.max(tuning_curve)

    # Smooth signal
    smoothed_tuning_curve = gaussian_filter1d(tuning_curve, sigma = sigma, mode='wrap')

    # Normalize for matching max vals
    if max_rate > 0:
        smoothed_tuning_curve = smoothed_tuning_curve*(max_rate/np.max(smoothed_tuning_curve))

    # Get peaks
    peaks_on_circle = find_indices_of_peaks_on_circle(smoothed_tuning_curve, width_in_samples, min_height = min_height, pad = pad)

    # Get classification
    if len(peaks_on_circle) == 1:
        classification_type = 1
    elif len(peaks_on_circle) == 2:
        classification_type = 2
    else:
        classification_type = 3
    
    # Final return
    return classification_type, smoothed_tuning_curve



def test_classify_tuning_curve():
    # Path to data is hard-coded. Recommend checking a few values of initialization distribution and Skaggs of L1 neurons.

    fig, ax = plt.subplots(nrows = 6, ncols = 10, figsize = (20,15))

    with h5py.File('local_data/vault_read_only/delta_tuning_curves/tuning_curves_2_2_0_0_3_0_0.h5', 'r') as file:
        L3_response_matrix_epoch_10000 = file['L3_response_matrix_epoch_10000'][:]
    
    num_L3_neurons = L3_response_matrix_epoch_10000.shape[0]
    
    num_found_unimodal = 0
    num_found_bimodal = 0
    num_found_other = 0

    # Find and plot unimodal
    search_counter = 0
    while search_counter < num_L3_neurons and num_found_unimodal < 10:
        tuning_curve = L3_response_matrix_epoch_10000[search_counter, :]
        result, signal = classify_tuning_curve(tuning_curve, width_in_samples = 5)
        if result == 1:
            ax[0, num_found_unimodal].plot(range(100), tuning_curve)
            ax[0, num_found_unimodal].set_title('Unimodal')
            ax[1, num_found_unimodal].plot(range(100), signal)
            ax[1, num_found_unimodal].set_title('Smoothed')
            num_found_unimodal = num_found_unimodal + 1

        search_counter = search_counter + 1

    # Find and plot bimodal
    search_counter = 0
    while search_counter < num_L3_neurons and num_found_bimodal < 10:
        tuning_curve = L3_response_matrix_epoch_10000[search_counter, :]
        result, signal = classify_tuning_curve(tuning_curve, width_in_samples = 5)
        if result == 2:
            ax[2, num_found_bimodal].plot(range(100), tuning_curve)
            ax[2, num_found_bimodal].set_title('Bimodal')
            ax[3, num_found_bimodal].plot(range(100), signal)
            ax[3, num_found_bimodal].set_title('Smoothed')
            num_found_bimodal = num_found_bimodal + 1

        search_counter = search_counter + 1

    
    # Find and plot other
    search_counter = 0
    while search_counter < num_L3_neurons and num_found_other < 10:
        tuning_curve = L3_response_matrix_epoch_10000[search_counter, :]
        result, signal = classify_tuning_curve(tuning_curve, width_in_samples = 5)
        if result == 3:
            ax[4, num_found_other].plot(range(100), tuning_curve)
            ax[4, num_found_other].set_title('Other')
            ax[5, num_found_other].plot(range(100), signal)
            ax[5, num_found_other].set_title('Smoothed')
            num_found_other = num_found_other + 1

        search_counter = search_counter + 1


    for axis in ax.flatten():
        axis.set_xlabel('Circular position (a.u.)')
        axis.set_ylabel('Firing rate (Hz)')
        
        
    fig.suptitle('Test function classify_tuning_curve()')


    plt.tight_layout()
    plt.savefig('unsafe_local_data/test_classify_tuning_curves.png')

    print('Function test_classify_tuning_curve: manually check figure produced.')







def main():
    
    test_classify_tuning_curve()




if __name__ == "__main__":
    main()