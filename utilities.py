import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from numba import njit
from numba.typed import Dict
from numba.core import types

# Function to sample froma  numpy distribution
def sample_from_distribution(distribution, *args):
    return distribution(*args)




# Generates a (biased) walk on the circle, understood as R/Z. That is, we identify the circle with
# the unit interval [0,1] subject to the identification 0 ~ 1.k
# Returns a np.array of shape (num_steps, ).
# Checked and last modified 10 Oct 2024 by Niko Schonsheck. 
def generate_biased_random_circular_walk(initial_position, bias, num_steps, max_step_size):
    probability_to_continue_in_same_direction = bias

    # Generate a biased random walk on (-infty, infty). To get the appropriate circular walk, we will
    # mod by 1.
    walk_on_real_line = np.zeros(num_steps)

    # Initialize the walk, and make the first step
    walk_on_real_line[0] = initial_position
    second_position  = initial_position + random.uniform(-max_step_size, max_step_size)
    walk_on_real_line[1] = second_position

    for step_index in range(1, num_steps - 1):
        current_position = walk_on_real_line[step_index]

        previous_position = walk_on_real_line[step_index - 1]

        which_direction_last_step = np.sign(current_position - previous_position)

        sample_to_determine_which_direction_next_step = random.uniform(0,1)

        next_step_size = random.uniform(0, max_step_size)

        if sample_to_determine_which_direction_next_step < probability_to_continue_in_same_direction: # keep going in the same direction
            next_position = current_position + which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
        else: # reverse direction
            next_position = current_position + (-1)*which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
    
    circular_walk = [(x % 1) for x in walk_on_real_line]

    return np.array(circular_walk)



# Arguments: 
# position: a number in [0,1) representing a position on the circle S^1 = R/Z
# tuning_curve_array: tuning curve on a circle, represented as R/Z. First column should be x positions (in [0,1)) and second column should be firing rates
# Function: computes firing rate by linearly interpolating between data points that describe the tuning curve
def calculate_firing_rate_from_tuning_curve(position, tuning_curve_array):
    if position < 0 or position >= 1:
        raise Exception('Invalid position.')
    num_positions = tuning_curve_array.shape[0]

    # Handle edge case where we run into 0~1 identification
    if position >= tuning_curve_array[num_positions - 1, 0] or position <= tuning_curve_array[0, 0]:
        x_1 = tuning_curve_array[num_positions - 1, 0]
        y_1 = tuning_curve_array[num_positions - 1, 1]
        x_2 = tuning_curve_array[0, 0] + 1
        y_2 = tuning_curve_array[0, 1]
        slope = (y_2 - y_1)/(x_2 - x_1)
        if position <= tuning_curve_array[0, 0]:
            rate = slope*((position + 1) - x_1) + y_1
            return rate
        else:
            rate = slope*((position) - x_1) + y_1
            return rate
    
    # Otherwise, just linearly interpolate
    else:
        index = np.searchsorted(tuning_curve_array[:, 0], position) - 1
        x_1 = tuning_curve_array[index, 0]
        y_1 = tuning_curve_array[index, 1]
        x_2 = tuning_curve_array[index + 1, 0]
        y_2 = tuning_curve_array[index + 1, 1]
        slope = (y_2 - y_1)/(x_2 - x_1)
        rate = slope*(position - x_1) + y_1
        return rate
    



# Arguments 
# path_on_circle: a path on a circle = R/Z, i.e., a  np.array of shape (num_steps,) consisting of values in [0,1)
# tuning_curves: a dictionary of num_neurons (neuron_index, tuning_curve)-pairs
# Output
# a (num_neurons x num_steps) matrix where row i is neuron i's response to the path_on_circle as a time series 
def calculate_response_matrix_given_walk_on_circle_and_tuning_curves(path_on_circle, tuning_curves):
    num_steps = len(path_on_circle)
    num_neurons = len(tuning_curves.keys())
    response_matrix = np.zeros((num_neurons, num_steps))
    for neuron_index in range(num_neurons):
        tuning_curve = tuning_curves[neuron_index]
        for step_index in range(num_steps):
            response_matrix[neuron_index, step_index] = calculate_firing_rate_from_tuning_curve(path_on_circle[step_index], tuning_curve)
    return response_matrix




# Adds normal random noise to a matrix with fixed integer seed for reproducability
def add_normal_random_noise(matrix, integer_seed, nonzero_entry_std_dev_factor, min_std_dev):
    np.random.seed(integer_seed)

    noisy_matrix = matrix.copy()

    noisy_matrix = np.maximum(0, np.random.normal(noisy_matrix, np.maximum(nonzero_entry_std_dev_factor*noisy_matrix, min_std_dev)))

    return noisy_matrix



# Arguments:
# input_matrix: a (num_neurons_input_layer x num_timesteps) matrix (i.e., numpy array) representing neuronal responses to some stimulus
# weight matrix: a (num_neurons_output_layer x num_neurons_input_layer) matrix (i.e., numpy array) representing synaptic weights
# bias_vector: a (num_neurons_output_layer, ) numpy array of biases
# Note: bias vector entries should be POSITIVE (or zero)
# output:
# responses in output layer of neurons based on one layer feedforward ReLU architecture
def calculate_one_layer_ff_relu_output_matrix(input_matrix, weight_matrix, bias_vector):
    
    # Check dimensions
    if input_matrix.shape[0] != weight_matrix.shape[1]:
        raise Exception("Number of input neurons implied by input matrix and weight matrix do not match.")
    elif weight_matrix.shape[0] != len(bias_vector):
        raise Exception("Number of output neurons implied by weight matrix and bias vector do not match.")
    
    # Check bias vector values are nonnegative
    if np.any(bias_vector < 0):
        raise ValueError("All bias vector entries must be positive.")
    
    # Multiply, apply bias, and ReLU
    output_response = np.maximum(0.0, weight_matrix@input_matrix - bias_vector.reshape((-1,1)))

    return output_response



# PLEASE ENSURE THAT NUMBER OF LAYER 2 NEURONS IS EVEN
# Initializes a two-layer feedforward network L1 -> L2 -> L3, where
# L1 neurons are all understood as excitatory Dale neurons
# Neurons in L2 with indices 0, 1, 2, ..., num_layer_2_neurons/2 - 1 are understood as inhibitory Dale neurons with outgoing connection strengths -1
# Neurons in L2 with indices num_layer_2_neurons, num_layer_2_neurons + 1, ..., num_layer_2_neurons - 1 are understood as excitatory Dale neurons with outgoing connection strengths 1
# L3 is a readout layer. Each neuron in L3 receives one unique excitatory connection from L2 and one unique inhibitory connection from L2
# Specifically, neuron with index i in L3 receives a synapse with weight -1 from neuron i in L2 and a synapse with weight +1 from neuron (i + num_layer_2_neurons) in L2
# Connections L1 -> L2 are formed as follows. For an L1 neuron, one draws from the specified distribution. If the result x is negative, then one forms a connection
# with weight |x| between that L1 neuron and a randomly chosen inhibitory L2 neuron. If the result x is positive, then one forms a connection with weight x between
# that L1 neuron and a randomly chosen excitatory L2 neuron. A maximum of 1 and minimum of -1 is enforced by hand.
# By choosing distributions with negative means, one can model inhibitory bias in interneuron populations.
# Sparsity for L1 -> L2 means, e.g., if sparsity_parameter = 0.1, then if there are 100 neurons in L2, each neuron in L1 will be connected to a total of 10 neurons in L2
# Bias vectors for L2 are warm-started by computing L1 responses for a uniform sampling of the input circle, recording the max firing rate of each L2 neuron, and
# then taking a factor of the max firing rate. 
# L3 neurons do not have biases.
# On 3 Dec 2024, we added a check that each output neuron has at least one nonzero synapse. Otherwise, there are problems with normalizations
# and to ensure a (functionaly) fixed output and input size, we want to ensure each output neuron receives at least one input.
# Function returns: (L1_to_L2_weight_matrix, initial_row_sums, bias_vector, L2_to_L3_weight_matrix)
# Where 
# "L1_to_L2_weight_matrix" is the initial L1 -> L2 weight matrix
# "initial_row_sums" is a vector of row sums of the initial L1 -> L2 weight matrix. This is useful for normalization, if we desire to keep the total weight into an L2 neuron constant, e.g.
# "bias_vector" is the initial bias_vector for L2 neurons
# "L2_to_L3_weight_matrix" is the initial L2 -> L3 weight matrix 
# Checked and last modified by Niko Schonsheck on 3 Dec 2024

def initialize_network(tuning_curve_dict, num_layer_1_neurons, num_layer_2_neurons, initial_layer_2_bias_vector_factor, sparsity_parameter_layer_1_to_layer_2, distribution, *args):

    num_layer_3_neurons = int(num_layer_2_neurons/2)

    num_layer_2_inhibitory_neurons = int(num_layer_2_neurons/2)

    found_valid_L1_to_L2_weight_matrix = False

    while found_valid_L1_to_L2_weight_matrix == False:

        # Make L1 -> L2 connection matrix, enforcing maximum of 1 and minimum of -1 and save
        L1_to_L2_weight_matrix = np.zeros((num_layer_2_neurons, num_layer_1_neurons))

        num_outgoing_connections_of_each_L1_neuron = int(np.floor(num_layer_2_neurons * sparsity_parameter_layer_1_to_layer_2))

        # For each L1 neuron...
        for col_index in range(num_layer_1_neurons):

            for index in range(num_outgoing_connections_of_each_L1_neuron):
                weight = sample_from_distribution(distribution, *args)
                weight = np.minimum(1, weight)
                weight = np.maximum(-1, weight)

                if weight < 0:
                    layer_2_neuron_to_connect = np.random.randint(0, num_layer_2_inhibitory_neurons)
                    L1_to_L2_weight_matrix[layer_2_neuron_to_connect, col_index] = np.abs(weight)

                else:
                    layer_2_neuron_to_connect = np.random.randint(num_layer_2_inhibitory_neurons, num_layer_2_neurons)
                    L1_to_L2_weight_matrix[layer_2_neuron_to_connect, col_index] = np.abs(weight)
        
        # Sum rows of first connection matrix for normalization downstream (e.g., want to keep total weight into each L2 neuron constant through learning)
        initial_row_sums = np.sum(L1_to_L2_weight_matrix, axis = 1)

        # Warm start bias vector for L2 neurons
        walk_for_bias_vector_warm_start = np.linspace(0,1,1000)%1
        L1_response_for_bias_vector_warm_start = calculate_response_matrix_given_walk_on_circle_and_tuning_curves(walk_for_bias_vector_warm_start, tuning_curve_dict)
        L2_respone_for_bias_vector_warm_start = calculate_one_layer_ff_relu_output_matrix(L1_response_for_bias_vector_warm_start, L1_to_L2_weight_matrix, np.zeros(num_layer_2_neurons))
        bias_vector = initial_layer_2_bias_vector_factor*np.max(L2_respone_for_bias_vector_warm_start, axis = 1)
                                                                                        
        L2_to_L3_weight_matrix = np.zeros((num_layer_3_neurons, num_layer_2_neurons))

        # For each L3 neuron...
        for row_index in range(num_layer_3_neurons):
            L2_to_L3_weight_matrix[row_index, row_index] = -1
            L2_to_L3_weight_matrix[row_index, row_index + num_layer_2_inhibitory_neurons] = 1


        if not np.any(initial_row_sums == 0.0):
            found_valid_L1_to_L2_weight_matrix = True



    return (L1_to_L2_weight_matrix, initial_row_sums, bias_vector, L2_to_L3_weight_matrix)









# Calculates cross-corellogram of two populations of neurons, using the formula 
# in supplementary Figure 3 of Giusti et al. Clique topology reveals...
# See document "dissimilarity_calculations_and_functions_check.pdf" on 10/7/24 for full details.
# Function last modified on 7 Oct 2024 by Niko Schonsheck.
def calculate_ccg_fixed_tau(array_1, array_2, tau):
    T = array_1.shape[1]
    padded_array_1 = np.pad(array_1, ((0,0), (tau, 0)))
    padded_array_2 = np.pad(array_2, ((0,0), (0, tau)))
    return (1/T)*(padded_array_1@(padded_array_2.T))



# Calculates cross-correlogram of two populations of neurons, using the forumla
# in supplementary Figure 3 of Giusti et al. Clique topology reveals...
# for a range of tau's. Note that the range of tau's computed is 0, 1, 2, ..., tau_max - 1
# See document "dissimilarity_calculations_and_functions_check.pdf" on 10/7/24 for full details.
# Function last modified on 7 Oct 2024 by Niko Schonsheck. 
def calculate_full_ccg(array_1, array_2, tau_max):
    num_neurons_pop_1 = array_1.shape[0]
    num_neurons_pop_2 = array_2.shape[0]
    full_ccg = np.zeros((num_neurons_pop_1, num_neurons_pop_2, tau_max))
    for tau in range(tau_max):
        full_ccg[:, :, tau] = calculate_ccg_fixed_tau(array_1, array_2, tau)
    return full_ccg


# Computes neuron dissimilarities within a single population using the formula in 
# supplementary Figure 3 of Giusti et al. Clique topology reveals...
# We first compute C_ij, then normalize to have a maximum of 1. We compute
# D_ij = 1 - C_ij, add a regularization term (since D_ij will likely have a 0 on an off-diagonal)
# then renormalize if necessary and manually enforce 0's on the diagonal.
# See document "dissimilarity_calculations_and_functions_check.pdf" on 10/7/24 for full details.
# Last modified 9 Oct 2024 by Niko Schonsheck
def calculate_intrapopulation_neuron_dissimilarity(array, tau_max, regularization_factor):

    num_neurons = array.shape[0]

    similarity_matrix = np.zeros((num_neurons, num_neurons))

    full_ccg = calculate_full_ccg(array, array, tau_max)

    summed_ccg = np.zeros((num_neurons, num_neurons))

    for row_index in range(num_neurons):
        for col_index in range(num_neurons):
            summed_ccg[row_index, col_index] = sum(full_ccg[row_index, col_index, :])


    
    for row_index in range(num_neurons):
        for col_index in range(row_index, num_neurons):
            mean_rate_row_index = np.mean(array[row_index, :])
            mean_rate_col_index = np.mean(array[col_index, :])
            unnormalized_entry = np.maximum(summed_ccg[row_index, col_index], summed_ccg[col_index, row_index])
            norm_factor = tau_max*mean_rate_row_index*mean_rate_col_index
            similarity_matrix[row_index, col_index] = (1/norm_factor)*unnormalized_entry
            similarity_matrix[col_index, row_index] = (1/norm_factor)*unnormalized_entry


    # Normalize similarity_matrix to have max of 1
    max_val_similarity_matrix = np.max(similarity_matrix)
    similarity_matrix = (1/max_val_similarity_matrix)*similarity_matrix



    # Create dissimilarity_matrix
    dissimilarity_matrix = 1 - similarity_matrix



    # Regularize dissimilarity matrix
    # Find largest nonzero entry of dissimilarity_matrix, if it exists.
    unique_and_flattened_dissimilarity_matrix = np.unique(dissimilarity_matrix)
    if len(unique_and_flattened_dissimilarity_matrix) > 1:
        smallest_nonzero_entry_dissimilarity_matrix = unique_and_flattened_dissimilarity_matrix[1]
    else:
        smallest_nonzero_entry_dissimilarity_matrix = 0

    # Add regularization_factor*(smallest nonzero entry) to each element of dissimilarity_matrix
    dissimilarity_matrix = dissimilarity_matrix + (regularization_factor*smallest_nonzero_entry_dissimilarity_matrix)


    # If maximum of dissimilarity_matrix is larger than 1, normalize so max is 1
    max_val_dissimilarity_matrix = np.max(dissimilarity_matrix)
    if max_val_dissimilarity_matrix > 1:
        dissimilarity_matrix = dissimilarity_matrix/max_val_dissimilarity_matrix

    
    # Set diagonal entries of dissimilarity_matrix equal to 0
    for index in range(num_neurons):
        dissimilarity_matrix[index, index] = 0

    return dissimilarity_matrix

# Computes the anlaogue of "calculate_intrapopulation_neuron_dissimilarity" given two populations of neurons
# where one is "upstream" (or, presynaptic) of another. Thus, we only wish to consider postsynaptic responses
# that occur after presynaptic ones. 
# See document "dissimilarity_calculations_and_functions_check.pdf" on 10/7/24 for full details.
# Note that the returned matrix is of dimension (num_presynaptic_neurons x num_postsynaptic_neurons)
# so that the (i,j) entry is the dissimilarity if presynaptic neuron i with postsynaptic neuron j. 
# In terms of the Yoon et al. Analogous bars pipeline, this means that if the presynaptic population is P
# and postsynaptic population is Q, then assuming columns are landmarks and rows are witnesses, this function
# returns the matrix W_QP, not W_PQ as it will have num_postsynaptic_neurons columns. 
# Function last modified on 9 Oct 2024 by Niko Schonsheck.
def calculate_interpopulation_neuron_dissimilarity_nonsymmetric(presynaptic_response_matrix, postsynaptic_response_matrix, tau_max, regularization_factor):

    num_presynaptic_neurons = presynaptic_response_matrix.shape[0]
    
    num_postsynaptic_neurons = postsynaptic_response_matrix.shape[0]

    similarity_matrix = np.zeros((num_presynaptic_neurons, num_postsynaptic_neurons))

    full_ccg = calculate_full_ccg(presynaptic_response_matrix, postsynaptic_response_matrix, tau_max)

    summed_ccg = np.zeros((num_presynaptic_neurons, num_postsynaptic_neurons))

    for row_index in range(num_presynaptic_neurons):
        for col_index in range(num_postsynaptic_neurons):
            summed_ccg[row_index, col_index] = sum(full_ccg[row_index, col_index, :])
    
    for row_index in range(num_presynaptic_neurons):
        for col_index in range(num_postsynaptic_neurons):
            mean_rate_row_index = np.mean(presynaptic_response_matrix[row_index, :])
            mean_rate_col_index = np.mean(postsynaptic_response_matrix[col_index, :])
            unnormalized_entry = summed_ccg[row_index, col_index]
            norm_factor = tau_max*mean_rate_row_index*mean_rate_col_index
            similarity_matrix[row_index, col_index] = (1/norm_factor)*unnormalized_entry


    # Normalize similarity_matrix to have max of 1
    max_val_similarity_matrix = np.max(similarity_matrix)
    similarity_matrix = (1/max_val_similarity_matrix)*similarity_matrix


    # Create dissimilarity_matrix
    dissimilarity_matrix = 1 - similarity_matrix


    # Regularize dissimilarity matrix
    # Find largest nonzero entry of dissimilarity_matrix, if it exists.
    unique_and_flattened_dissimilarity_matrix = np.unique(dissimilarity_matrix)
    if len(unique_and_flattened_dissimilarity_matrix) > 1:
        smallest_nonzero_entry_dissimilarity_matrix = unique_and_flattened_dissimilarity_matrix[1]
    else:  
        smallest_nonzero_entry_dissimilarity_matrix = 0

    # Add regularization_factor*(smallest nonzero entry) to each element of dissimilarity_matrix
    dissimilarity_matrix = dissimilarity_matrix + regularization_factor*smallest_nonzero_entry_dissimilarity_matrix


    # If maximum of dissimilarity_matrix is larger than 1, normalize so max is 1
    max_val_dissimilarity_matrix = np.max(dissimilarity_matrix)
    if max_val_dissimilarity_matrix > 1:
        dissimilarity_matrix = dissimilarity_matrix/max_val_dissimilarity_matrix



    return dissimilarity_matrix





# Calculate the circular variance of a list of positions on the circle, understood as R/Z, i.e., [0,1]/(1~0)
# See Kanti V. Mardia, Peter E. Jupp "Directional Statistics" for a useful reference.
# Function last modified on 7 Oct 2024 by Niko Schonsheck.
def circular_variance(positions):
    
    # Convert positions to radians
    angles = [position * 2 * np.pi for position in positions]

    # Calculate components of the resultant vector
    C = np.sum([np.cos(angle) for angle in angles]) / len(angles)

    S = np.sum([np.sin(angle) for angle in angles]) / len(angles)

    # Calculate the resultant vector length

    R = np.sqrt(C**2 + S**2)

    # Circular variance
    V = 1 - R
    return V

# To test:
# print(np.isclose(1, circular_variance(np.array([0, .25, .5, .75]))))

# print(np.isclose(0, circular_variance(np.array([0, 0, .000001, .99999]))))



# Given tuning_curve a numpy array of shape (N,2), column 1 is position, column 2 is firing rate,
# Return position (*not* index) of maximum firing rate.
# By default, if more than one position has the same firing rate, we return the position
# with the smallest index in the input array.
# Last modified on 7 Oct 2024 by Niko Schonsheck.
def find_position_of_max_firing_rate(tuning_curve):
    max_firing_rate = -1.0
    position_of_max_firing_rate = -1.0

    num_positions = tuning_curve.shape[0]

    for index in range(num_positions):
        firing_rate = tuning_curve[index, 1]
        if firing_rate > max_firing_rate:
            max_firing_rate = firing_rate
            position_of_max_firing_rate = tuning_curve[index, 0]

    if position_of_max_firing_rate != -1.0:
        return position_of_max_firing_rate
    else:  
        raise Exception("Did not find a maximum rate.")
    

# Convert a spiketrain to a time series of firing rates using bins
# Input should be a spiketrain, i.e., a numpy array of 0's and 1's where each index is 1ms,
# a 1 corresponds to a spike, and a 0 corresponds to no spike
# Input should be a numpy array of shape (number of milliseconds in spike trains, )
# And returns a time series of firing rates of shape (number of milliseconds in spike trians/size_of_firing_rate_bins_in_ms, )
# Note that the number of milliseconds in the spiketrain (i.e., its length as a numpy array) must be divisible by the size_of_firing_rate_bin_in_ms
# Last modified 8 Oct 2024 by Niko Schonsheck
def spiketrain_to_rates_in_Hz(spiketrain, size_of_firing_rate_bin_in_ms):
    if (len(spiketrain)/size_of_firing_rate_bin_in_ms) % 1 != 0:
        raise ValueError("Length of spike train must be divisible by size of firing rate bin in ms.")
    
    num_firing_rate_bins = np.int64(len(spiketrain)/size_of_firing_rate_bin_in_ms)

    firing_rate_as_time_series = []

    for bin_index in range(num_firing_rate_bins):
        relevant_portion_of_spiketrain = spiketrain[bin_index*size_of_firing_rate_bin_in_ms:bin_index*size_of_firing_rate_bin_in_ms + size_of_firing_rate_bin_in_ms]
        num_spikes = np.sum(relevant_portion_of_spiketrain)
        firing_rate_in_hz = (num_spikes*1000)/size_of_firing_rate_bin_in_ms
        firing_rate_as_time_series.append(firing_rate_in_hz)
    
    firing_rate_as_time_series = np.array(firing_rate_as_time_series)

    if len(firing_rate_as_time_series) != num_firing_rate_bins:
        raise Exception("Length of returned time series of firing rates is not equal to specified number of firing rate bins.")
    
    return firing_rate_as_time_series



# This function calls spikes_to_rates and then smooths the resultant time series with a Gaussian kernel
# Last modified 8 Oct 2024 by Niko Schonsheck
def spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bin_in_ms, sigma_for_smoothing):
    unsmoothed_rates = spiketrain_to_rates_in_Hz(spiketrain, size_of_firing_rate_bin_in_ms)
    smoothed_rates = gaussian_filter1d(unsmoothed_rates, sigma_for_smoothing)
    return smoothed_rates


# Generate a spike train from a given set of firing rates specified in Hz using the method described
# by David Heegar in "Poisson Model of Spike Generation"
# available at https://www.cns.nyu.edu/~david/handouts/poisson.pdf

# # Description
# This function simulates a spike train based on input firing rates (in Hz) for specified durations
# of firing rate bins (in milliseconds). It produces a spike train with a temporal resolution of 1ms.
# Each bin in the firing rate array represents the firing rate for a duration defined by the size of the
# firing rate bins. The output is an array where each element indicates the presence (1) or absence (0)
# of a spike in each millisecond.

# # Arguments
# - `firing_rates`: Numpy array of firing rates in Hz, where each rate corresponds to a bin in the time series. Should be of size (timesteps, )
# - `size_of_firing_rate_bins_in_ms`: Size of each firing rate bin in milliseconds.

# # Returns
# - `spike_train`: Array of integers (0 or 1), where each element represents whether a spike occurred
#   at each millisecond of the simulation period.

# # Warnings
# This function assumes that the firing rates are passed as Hz and outputs a time series with 1ms temporal resolution.
# Also notice the @njit call. This can result in orders of magnitude speed up and should be used when possible.
# Function last modified on 8 Oct 2024 by Niko Schonsheck
@njit
def simulate_spiketrain_with_firing_rates_in_Hz(firing_rates, size_of_firing_rate_bins_in_ms):
    num_input_bins = len(firing_rates)

    spike_train = np.zeros(num_input_bins * size_of_firing_rate_bins_in_ms)

    for input_bin_index in range(num_input_bins):
        rate = firing_rates[input_bin_index] # Obtain firing rate from current bin

        probability = rate/1000 # Convert firing rate to probability of spike, assuming 1ms time bins

        # Simulate spikes for each 1ms sub-bin within the current bin
        for sub_bin in range(size_of_firing_rate_bins_in_ms):
            random_number = np.random.rand()
            if random_number < probability:
                spike_train[size_of_firing_rate_bins_in_ms*(input_bin_index - 1) + sub_bin] = 1
    
    return spike_train




# Given a presynaptic and postsynaptic spike train -- i.e., numpy arrays of shape (num_bins, ) -- this function calculates
# the change in synaptic weight between the two neurons acccording to the formulas in
# "Synaptic Modification by Correlated Activity: Hebb's Postulate Revisited" by Bi and Poo, Figure 1.
# Note: this function considers presynaptic spikes at spike indices:
# search_window_limit_in_ms, search_window_limit_in_ms + 1, ..., spike_train_length - search_window_limit_in_ms - 2, spike_train_length - search_window_limit_in_ms - 1
# So, it considers a total of (spike_train_length - search_window_limit_in_ms) - search_window_limit_len_in_ms = spike_train_length - 2*search_window_limit_in_ms indices.
# Note: to avoid excessive changes in weight, one should use relatively short spike trains. For instance, if the presynaptic neuron has a maximum firing rate of 50 Hz,
# then over a spike train of 1200ms with a search_window_limit_in_ms of 100, we consider a total of 1000ms to look for presynaptic spikes and therefore
# one can expect 50 presynaptic spikes to consider. Using the formula in Bi and Poo, this can result in a maximal increase of 45.23% in synaptic weight if there are postsynaptic
# spikes 1ms after each presynaptic spike, and a maximal decrease (i.e., most negative) of -25.73% in synaptic weight if there are postsynaptic spikes 1ms before each
# presynaptic spike. 
# Please note the use of @njit. This can have serious (e.g., 3 orders of magnitude) speed implications and should be used whenever possible.
# Passed testing at git hash cfd4ea1c46f5b93fe78e8b0da021e82ef7bfc2cc
@njit
def calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presynaptic_spiketrain, postsynaptic_spiketrain, search_window_limit_in_ms):

    # Check spike trains are the same length
    if len(presynaptic_spiketrain) != len(postsynaptic_spiketrain):
        raise ValueError("Spike trains must be of the same length.")
    
    # Since the lengths are the same, define spike_train_length
    spike_train_length = len(presynaptic_spiketrain)

    # Check spike trains are long enough   
    if spike_train_length < 2*search_window_limit_in_ms + 1:
        raise ValueError("Spike trains not long enough relative to search window.")
    
    # In general, pre_before_post_delta_W will be positive if there are updates via this mechanism 
    pre_before_post_delta_W = 0.0

    # In general, post_before_pre_delta_W will be negative if there are updates via this mechanism 
    post_before_pre_delta_W = 0.0



    
    # For a given spike at index spike_index in the presynaptic spike train, we must be able to look for postsynaptic spikes at indices:
    # spike_index - search_window_limit_in_ms, spike_index - search_window_limit_in_ms + 1, ..., spike_index - 1 and
    # spike_index + 1, spike_index + 2, ..., spike_index + search_window_limit_in_ms - 1, spike_index + seach_window_limit_in_ms
    # Thus, we must have spike_index - search_window_limit_in_ms >= 0 and spike_index + search_window_limit_in_ms <= spike_train_length - 1 (because indexing begins at 0)
    # Therefore, we must have spike_index >= search_window_limit_in_ms and spike_index <= spike_train_length - search_window_limit_in_ms - 1
    for presynaptic_spike_index in range(search_window_limit_in_ms, spike_train_length - search_window_limit_in_ms):
        
        # If there is a presynaptic spike...
        if presynaptic_spiketrain[presynaptic_spike_index] == 1:

            # Look for postsynaptic spikes
            counter = 0
            while counter < search_window_limit_in_ms - 1:
                counter = counter + 1

                # If there are postsynaptic spikes equidistant from the presynaptic spike, do nothing
                if postsynaptic_spiketrain[presynaptic_spike_index + counter] == 1 and postsynaptic_spiketrain[presynaptic_spike_index - counter] == 1:
                    break

                # Otherwise, suppose we have a postsynaptic spike occuring at presynaptic_spike_index + counter but no spike at presynaptic_spike_index - counter.
                # Then we have a pre- before post-, update pre_before_post_delta_W and stop the search (winner take all)
                elif postsynaptic_spiketrain[presynaptic_spike_index + counter] == 1 and postsynaptic_spiketrain[presynaptic_spike_index - counter] == 0:
                    delta_t = counter
                    pre_before_post_delta_W = pre_before_post_delta_W + A_plus*np.exp(
                        (-1*delta_t)/tau_plus
                    )
                    break

                # Otherwise, suppose we have a postsynaptic spike occuring at presynaptic_spike_index - counter but no spiek at presynaptic spike_index + counter.
                # Then we have a post- before -pre, update post_before_pre_delta_W and stop the search (winner take all)
                elif postsynaptic_spiketrain[presynaptic_spike_index + counter] == 0 and postsynaptic_spiketrain[presynaptic_spike_index - counter] == 1:
                    delta_t = -1*counter
                    post_before_pre_delta_W = post_before_pre_delta_W + A_minus*np.exp(
                        (-1*delta_t)/tau_minus
                    )
                    break

                # Otherwise, we have not yet found a postsynaptic spike, so continue the search
                else:
                    continue

    
    return pre_before_post_delta_W + post_before_pre_delta_W



def test_calculate_STDP_delta_W():

    A_plus = 0.0096
    A_minus = -0.0053
    tau_plus = 16.8
    tau_minus = -33.7

    # No spikes in valid range
    presyn_spiketrain_1 = np.zeros(100)
    postsyn_spiketrain_1 = np.zeros(100)
    presyn_spiketrain_1[[0, 1, 2, 99]] = 1
    postsyn_spiketrain_1[[10, 11, 20, 40, 50, 23]] = 1
    actual_result_1 = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presyn_spiketrain_1, postsyn_spiketrain_1, 10)
    expected_result_1 = 0.0
    assert(np.isclose(actual_result_1, expected_result_1))


    # Two spikes pre before post
    presyn_spiketrain_2 = np.zeros(100)
    postsyn_spiketrain_2 = np.zeros(100)
    presyn_spiketrain_2[[11, 50]] = 1
    postsyn_spiketrain_2[[20, 51]] = 1
    actual_result_2 = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presyn_spiketrain_2, postsyn_spiketrain_2, 10)
    expected_result_2 = 0.01466365636
    assert(np.isclose(actual_result_2, expected_result_2))


    # Two spikes post before pre
    presyn_spiketrain_3 = np.zeros(100)
    postsyn_spiketrain_3 = np.zeros(100)
    presyn_spiketrain_3[[11, 50]] = 1
    postsyn_spiketrain_3[[5, 47]] = 1
    actual_result_3 = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presyn_spiketrain_3, postsyn_spiketrain_3, 10)
    expected_result_3 = -0.009284191399
    assert(np.isclose(actual_result_3, expected_result_3))


    # One presynaptic spike not in valid range, a pair of offsetting, one post before pre that beats a pre before post, and one pre before post that beats a pre before post
    presyn_spiketrain_4 = np.zeros(100)
    postsyn_spiketrain_4 = np.zeros(100)
    presyn_spiketrain_4[[5, 20, 30, 40]] = 1
    postsyn_spiketrain_4[[1, 19, 21, 25, 33, 35, 60]] = 1
    actual_result_4 = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presyn_spiketrain_4, postsyn_spiketrain_4, 10)
    expected_result_4 = 0.003460854
    assert(np.isclose(actual_result_4, expected_result_4))


    print('test_calculate_STDP_delta_W(): passed.')
    



# Updates a bias vector by:
# max(current entry, factor*(max firing rate over previous response))
# Note that bias vectors should be numpy arrays of shape (num_L2_neurons, )
# Checked and last modified 8 Oct 2024 by Niko Schonsheck
def update_L2_bias_vector(current_L2_bias_vector, L2_response_matrix, factor):
    
    # Sanity check
    if len(current_L2_bias_vector) != L2_response_matrix.shape[0]:
        raise ValueError("current_L2_bias_vector and L2_response_matrix imply different number of L2 neurons.")
    
    num_L2_neurons = len(current_L2_bias_vector)
    
    vector_of_max_firing_rates = np.max(L2_response_matrix, axis = 1)
    new_bias_vector = current_L2_bias_vector.copy()

    for neuron_index in range(num_L2_neurons):
        potential_new_bias = factor*vector_of_max_firing_rates[neuron_index]
        if potential_new_bias > current_L2_bias_vector[neuron_index]:
            new_bias_vector[neuron_index] = potential_new_bias
    
    return new_bias_vector



# This function is not correct. It generates a new spiketrain for each pair.
# # Given L1 and L2 responses of shapes (neurons x time steps) = (neurons x T), each with the same number of columns, and a time lag for propagation between L1 and L2, this function updates a given weight matrix 
# # by iterating over applictaions of calculate_STDP_delta_W.
# # Note that weight matrix should be (num_L2_neurons x num_L1_neurons)
# # We require that time_lag_in_ms < size_of_firing_rate_bins_in_ms and time_lag_in_ms < search_window_limit_in_ms and that size_of_firing_rate_bins_in_ms divides search_window_limit_in_ms
# # In this function, learning will occurr on the following column indices of the rate response matrices, where we let a = size_of_firing_rate_bins_in_msk b = search_window_limit_in_ms, and c = time_lag_in_ms
# # 2(b/a), 2(b/a) + 1, ..., T - 2(b/a) - 1
# # i.e., over the range [2(b/a), ..., T - 2(b/a))
# # See STDP_update_weight_matrix_docs.pdf for further details.
# # Last modified on 8 Oct 2024 by Niko Schonsheck
# @njit
# def STDP_update_weight_matrix(L1_rate_response_matrix, L2_rate_response_matrix, current_weight_matrix, time_lag_in_ms, size_of_firing_rate_bins_in_ms, A_plus, A_minus, tau_plus, tau_minus, search_window_limit_in_ms):
    
#     num_L1_neurons = L1_rate_response_matrix.shape[0]
#     num_L2_neurons = L2_rate_response_matrix.shape[0]
    
#     if L1_rate_response_matrix.shape[1] != L2_rate_response_matrix.shape[1]:
#         raise ValueError("L1_rate_response_matrix and L2_rate_response_matrix have different number of columns, i.e., time steps.")

#     if num_L1_neurons != current_weight_matrix.shape[1]:
#         raise ValueError("L1_rate_response_matrix and current_weight_matrix imply different number of L1 neurons.")

#     if num_L2_neurons != current_weight_matrix.shape[0]:
#         raise ValueError("L2_rate_response_matrix and current_weight_matrix imply different number of L2 neurons.")
    
#     if time_lag_in_ms >= size_of_firing_rate_bins_in_ms:
#         raise ValueError("Size of firing rate bins in ms must be greater than time lag in ms.")
    
#     if time_lag_in_ms >= search_window_limit_in_ms:
#         raise ValueError("Size of search window limit in ms must be greater than time lag in ms.")
    
#     if (search_window_limit_in_ms/size_of_firing_rate_bins_in_ms) % 1 != 0:
#         raise ValueError("Size of firing rate bins  in ms must divide search window limit in ms.")
    

#     delta_w_matrix = np.zeros((num_L2_neurons, num_L1_neurons))

#     for L1_neuron_index in range(num_L1_neurons):
#         for L2_neuron_index in range(num_L2_neurons):

#             if current_weight_matrix[L2_neuron_index, L1_neuron_index] != 0:
#                 presynaptic_spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L1_rate_response_matrix[L1_neuron_index, :], size_of_firing_rate_bins_in_ms)
#                 postsynaptic_spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L2_rate_response_matrix[L2_neuron_index, :], size_of_firing_rate_bins_in_ms)

#                 presynaptic_spiketrain_to_learn = presynaptic_spiketrain[search_window_limit_in_ms:-search_window_limit_in_ms]
#                 postsynaptic_spiketrain_to_learn = postsynaptic_spiketrain[(search_window_limit_in_ms - time_lag_in_ms):-(search_window_limit_in_ms + time_lag_in_ms)]


#                 delta_w_matrix[L2_neuron_index, L1_neuron_index] = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presynaptic_spiketrain_to_learn, postsynaptic_spiketrain_to_learn, search_window_limit_in_ms)
        
#     new_weight_matrix = np.zeros((num_L2_neurons, num_L1_neurons))

#     for L2_neuron_index in range(num_L2_neurons):
#         for L1_neuron_index in range(num_L1_neurons):
#             new_weight_matrix[L2_neuron_index, L1_neuron_index] = (1 + delta_w_matrix[L2_neuron_index, L1_neuron_index])*current_weight_matrix[L2_neuron_index, L1_neuron_index] 
    

#     return new_weight_matrix



# Given a NeuronDictionary constructed as we have, a desired number of neurons per quintile of Skaggs information (across the whole dictionary)
# and a requirement of minimum circular variances for each quintile, this function returns neuron indices
# of the NeuronDictionary that, collectively, meet the desired requirements.
# Note that the quintiles are increasing. So, if one desires 20 neurons all of which are in the top quintile of Skaggs indices, then
# samples_per_quantile = [0, 0, 0, 0, 20]
# Checked and last modified on 8 Oct 2024 by Niko Schonsheck
def sample_neurons_with_variance_condition(neuron_dict, samples_per_quintile, min_circular_variances, max_attempts):
    if len(samples_per_quintile) != 5 or len(min_circular_variances) != 5:
        raise ValueError("samples_per_quintile and min_circular_variances must each have exactly 5 elements.")
    
    # Extract Skaggs indices from the dictionary
    skaggs_indices = [v["Skaggs index"] for k, v in neuron_dict.items()]
    
    # Compute quintile thresholds based on Skaggs index
    thresholds = np.quantile(skaggs_indices, np.arange(0, 1.2, 0.2))
    
    neuron_keys = list(neuron_dict.keys())
    
    # Create lists of keys for each quintile
    quintile_keys = [
        [neuron_keys[i] for i in range(len(skaggs_indices)) 
         if thresholds[q] < skaggs_indices[i] <= thresholds[q+1]] 
        for q in range(len(thresholds) - 1)
    ]
    
    sampled_keys = []
    
    # Iterate over each quintile
    for q, num_samples in enumerate(samples_per_quintile):
        attempt = 0
        satisfied = False
        
        while not satisfied and attempt < max_attempts:
            if num_samples > 0 and quintile_keys[q]:
                # Randomly sample without replacement
                candidate_sample = random.sample(quintile_keys[q], num_samples)
                
                # Compute max firing positions and circular variance
                max_positions = [find_position_of_max_firing_rate(neuron_dict[key]["Tuning curve"]) for key in candidate_sample]
                cv = circular_variance(max_positions)
                
                # Check if the circular variance condition is satisfied
                satisfied = cv >= min_circular_variances[q]
                if satisfied:
                    sampled_keys.extend(candidate_sample)
            else:
                satisfied = True  # Skip sampling if no samples requested or quintile is empty
            
            attempt += 1
        
        if attempt == max_attempts:
            raise ValueError(f"Warning: Reached maximum attempts for quintile {q}. Circular variance condition may not be satisfied.")

    return sampled_keys






# Given a neuron dictionary as we have created, a desired number of samples per quintile, and a desired list of minimum circular variances
# This function will return a list of sampled_neuron_indices (i.e., indices in the input neuron_dict) and a dictionary of tuning_curves
# re-indexed to give neurons 0, 1, ...
# Note that the quintiles are increasing. So, if one desires 20 neurons all of which are in the top quintile of Skaggs indices, then
# samples_per_quantile = [0, 0, 0, 0, 20]
# Checked and last modified on 8 Oct 2024 by Niko Schonsheck
def make_standard_tuning_curve_dictionary(neuron_dict, samples_per_quintile, min_circular_variances):
    sampled_neuron_indices = sample_neurons_with_variance_condition(neuron_dict, samples_per_quintile, min_circular_variances, 1000000)

    sample_dict = {}

    for (list_index, neuron_index) in enumerate(sampled_neuron_indices):
        sample_dict[list_index] = neuron_dict[neuron_index]['Tuning curve']
    
    return sampled_neuron_indices, sample_dict







# This function represents one of those rare times when the code really is the documentation. Please read through the code to understand the function.
# For additional information on the shifting of spikes trains used to model delay between layers, please see shifting_in_run_propagation_trials.pdf
# Please ensure that time_lag_for_L1_to_L3_in_ms is less than firing_rate_bin_size_in_ms
# Note that, since L3 is essentially a "readout" layer, we do not take it to have any bias vector (only threshold-linearity)
# Checked and last modified 9 Oct 2024 by Niko Schonsheck
def run_propagation_trial(
        walk_on_circle, # A random walk on a circle representing a circular stimulus. Note that if this walk has steps 0, 1, 2, ..., N, (which will correspond to firing rate bins) then the walk used for dissimilarity calculations is the same walk but indexed on 1, 2, ..., N. 
        time_lag_for_L1_to_L3_in_ms, # time lag for signal to propagate from L1 to L3 in milliseconds. Essentially, we will shift spike trains by this amount.
        size_of_firing_rate_bins_in_ms, # how many milliseconds each firing rate bin is treated as. Used for generating spike trains from firing rates, and vice versa. We generally use a value of 10 here.
        sigma_for_Gaussian_smoothing, # sigma used when converting spike trains to firing rates. If the firing rate bin size in ms is relatively small (~10 ms), and the time series is relatively long (~100k firing rate bins), we recommend a *large* value here, around 100.
        random_seed_for_L1_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L1 response matrix
        random_seed_for_L2_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L2 response matrix
        random_seed_for_L3_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L3 response matrix
        tuning_curves_for_L1_neurons, # should be a dictionary int64 (neuron index) => np.array of shape (20,2) (tuning curve)
        L1_to_L2_weight_matrix, # (num_L2_neurons x num_L1_neurons) numpy array representing synaptic weights between neurons in L1 and neurons in L2
        L2_bias_vector, # a (num_L2_neurons, ) array representing biases for L2 neurons
        L2_to_L3_weight_matrix, # (num_L3_neurons x num_L2_neurons) numpy array representing synaptic weights between neurons in L2 and neurons in L3
        tau_max_in_firing_rate_bins, # tau_max value used in dissimilarity calculations, measured in number of firing rate bins. Generally we take this to be 100, so that if each firing rate bin is 10ms, we compute dissimilarities over a window of size 1000ms = 1s
        Gaussian_noise_factor, # A value betwen 0 and 1 used to add noise to response matrices. In general, given a firing rate r, we sample from a normal distribution centered at r with standard deviation = max(r*Gaussian_noise_factor, min_std_dev_Gaussian_noise)
        min_std_dev_Gaussian_noise, # See "Gaussian_noise_factor." In general, we suggest Gaussian_noise_factor ~ 0.05 and min_std_dev_Gaussian_noise = 2.5, if firing rates are in Hz
        regularization_factor_for_dissimilarity_calculations # See documentation for calculate_intrapopulation_neuron_dissimilarity and calculate_interpopulation_neuron_dissimilarity_nonsymmetric. We generally recommend a value of 0.1.
    ):

    # Check variables
    if time_lag_for_L1_to_L3_in_ms >= size_of_firing_rate_bins_in_ms:
        raise ValueError("time_lag_for_L1_to_L3_in_ms must be less than firing_rate_bin_size_in_ms")

    # Simulate L1 neuron responses
    L1_response_matrix = calculate_response_matrix_given_walk_on_circle_and_tuning_curves(walk_on_circle, tuning_curves_for_L1_neurons)

    # Add noise to L1 response matrix
    L1_response_matrix = add_normal_random_noise(L1_response_matrix, random_seed_for_L1_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L2 neuron responses
    L2_response_matrix = calculate_one_layer_ff_relu_output_matrix(L1_response_matrix, L1_to_L2_weight_matrix, L2_bias_vector)

    # Add noise to L2 response matrix 
    L2_response_matrix = add_normal_random_noise(L2_response_matrix, random_seed_for_L2_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L3 neuron responses. Note that, by convention, we do not have a bias vector for L3. Thus, we
    # construct a bias vector comprising all 0's of the appropriate size to compute L3 responses
    num_L3_neurons = L2_to_L3_weight_matrix.shape[0]
    L3_bias_vector = np.zeros(num_L3_neurons)
    L3_response_matrix = calculate_one_layer_ff_relu_output_matrix(L2_response_matrix, L2_to_L3_weight_matrix, L3_bias_vector)

    # Add noise to L3 response matrix
    L3_response_matrix = add_normal_random_noise(L3_response_matrix, random_seed_for_L3_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Convert L1 responses to spike trains to and back to rates.
    # We do this to ensure robustness and to account for the time lag between L1 and L3
    # That is, we will be converting L3 rates to spikes in order to account for a delay on the order of milliseconds
    # and then back to rates in order to do a similarity calculation. So as to perform the same transformation for all inputs
    # to the cycle matching pipeline, we perform the same transformations on L1 responses
    # Note that we omit the first firing_rate_bin_size_in_ms milliseconds from this spiketrain to be able to account for the time lag
    # See shifting_in_run_propagation_trials.pdf for further details on the shifting. 
    L1_response_matrix_from_spikes = np.zeros((L1_response_matrix.shape[0], len(walk_on_circle) - 1))
    for L1_neuron_index in range(L1_response_matrix.shape[0]):
        spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L1_response_matrix[L1_neuron_index, :], size_of_firing_rate_bins_in_ms)
        spiketrain = spiketrain[size_of_firing_rate_bins_in_ms:]
        rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
        if len(rates) != len(walk_on_circle) - 1:
            raise ValueError("Incorrect number of rate bins after converting from spikes to rates.")
        L1_response_matrix_from_spikes[L1_neuron_index, :] = rates
    

    # Convert L3 responses to spike trains, shift by time lag, and then back to rates
    # See shifting_in_run_propagation_trials.pdf for further details on the shifting.
    L3_response_matrix_from_spikes_with_shift = np.zeros((L3_response_matrix.shape[0], len(walk_on_circle) - 1))
    for L3_neuron_index in range(L3_response_matrix.shape[0]):
        spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L3_response_matrix[L3_neuron_index, :], size_of_firing_rate_bins_in_ms)
        spiketrain = spiketrain[size_of_firing_rate_bins_in_ms - time_lag_for_L1_to_L3_in_ms: -time_lag_for_L1_to_L3_in_ms]
        rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
        if len(rates) != len(walk_on_circle) - 1:
            raise ValueError("Incorrect number of rate bins after converting from spikes to rates.")
        L3_response_matrix_from_spikes_with_shift[L3_neuron_index, :] = rates

    # Calculate L1 intrapopulation dissimilarity matrix
    L1_intrapopulation_dissimilarity_matrix = calculate_intrapopulation_neuron_dissimilarity(L1_response_matrix_from_spikes, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)

    # Calculate L3 intrapopulation dissimilarity matrix
    L3_intrapopulation_dissimilarity_matrix = calculate_intrapopulation_neuron_dissimilarity(L3_response_matrix_from_spikes_with_shift, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)

    # Calculate L1 -> L3 interpopulation dissimilarity matrix
    # Note that the return matrix is (num_L1_neurons x num_L3_neurons)
    L1_to_L3_interpopulation_dissimilarity_matrix = calculate_interpopulation_neuron_dissimilarity_nonsymmetric(L1_response_matrix_from_spikes, L3_response_matrix_from_spikes_with_shift, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)


    return L1_intrapopulation_dissimilarity_matrix, L3_intrapopulation_dissimilarity_matrix, L1_to_L3_interpopulation_dissimilarity_matrix, L1_response_matrix_from_spikes, L2_response_matrix, L3_response_matrix_from_spikes_with_shift

# Uses a simple for loop and numba to find (min, max) of an array
# Checked and last modified by Niko Schonsheck on 29 Oct 2024.
@njit
def numba_min_max(arr):
    if len(arr) == 0:
        raise ValueError("Array input to numba_min_max is empty")
    min_val = arr[0]
    max_val = arr[0]
    for val in arr:
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    return min_val, max_val



# This function implements local variance scaling and synaptic scaling as detailed in STDP_scalings.pdf
# Please see that document for details.
# Note that delta_w_matrix is (num_postsynaptic_neurons x num_presynaptic_neurons)
# Please note the use of @njit. This can have serious (e.g., 3 orders of magnitude) speed implications and should be used whenever possible.
# Additional modification to avoid division by 0 added by Niko Schonsheck on 13-Nov-2024 (see below). 
@njit
def do_local_variance_and_synaptic_scaling_of_delta_w_matrix(delta_w_matrix, short_term_presynaptic_response_matrix, long_term_presynaptic_response_matrix, short_term_presynaptic_mean_firing_rates, long_term_presynaptic_mean_firing_rates, numba_dict_L1_to_L2_connections):
    
    # Collect variables and initialize the normalized (or, scaled) delta w matrix 
    num_time_bins_short_term_presynaptic_response_matrix = short_term_presynaptic_response_matrix.shape[1]
    normalized_delta_w_matrix = np.zeros(delta_w_matrix.shape)
    num_postsynaptic_neurons = delta_w_matrix.shape[0]
    num_presynaptic_neurons = delta_w_matrix.shape[1]


    # DO LOCAL VARIANCE SCALING
    # Find postsynaptic neuron indices that have at least one delta_w entry (i.e., that have a synapse whose strength has changed)
    nontrivial_postsynaptic_neuron_indices = []
    for postsynaptic_neuron_index in range(num_postsynaptic_neurons):
        found_delta_synapse = False
        presynaptic_neuron_index = -1
        while found_delta_synapse == False and presynaptic_neuron_index < num_presynaptic_neurons - 1:
            presynaptic_neuron_index = presynaptic_neuron_index + 1
            if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
                nontrivial_postsynaptic_neuron_indices.append(postsynaptic_neuron_index)
                found_delta_synapse = True

    
    # For each postsynaptic neuron which has had a change in synapse strength to at least one presynaptic neuron...
    for postsynaptic_neuron_index in nontrivial_postsynaptic_neuron_indices:
        short_term_deviations_from_long_term_means = []

        # Loop through presynaptic neurons to which that postsynaptic neuron is connected
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:

            # The following condition is guaranteed to be true at least once
            # If the synpase strength between a given pre- and post-synaptic neuron has changed...
            if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
                deviation = np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index])
                short_term_deviations_from_long_term_means.append(deviation)

        # Find max and min of deviations between short term (over learning window) and long term mean firing rates
        minimum_short_term_deviation_from_long_term_means, maximum_short_term_deviation_from_long_term_means = numba_min_max(short_term_deviations_from_long_term_means)

        # Do the scaling step
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
            current_delta_w = delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index]
            if current_delta_w != 0:

                # If given postsynaptic neuron has only one synapse that changed strength, do not scale
                if minimum_short_term_deviation_from_long_term_means == maximum_short_term_deviation_from_long_term_means:
                    normalization_factor = 1
                    normalized_delta_w = current_delta_w*normalization_factor
                    normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w
                
                # Otherwise, do scaling
                else:
                    normalization_factor = (np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]) - minimum_short_term_deviation_from_long_term_means)/(maximum_short_term_deviation_from_long_term_means - minimum_short_term_deviation_from_long_term_means)
                    normalized_delta_w = current_delta_w*normalization_factor
                    normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w

    

    # DO SYNAPTIC SCALING
    for postsynaptic_neuron_index in range(num_postsynaptic_neurons):

        synaptic_scaling_factor_numerator = 0
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
            synaptic_scaling_factor_numerator = synaptic_scaling_factor_numerator + short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]

        synaptic_scaling_factor_denominator = 0
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
            synaptic_scaling_factor_denominator = synaptic_scaling_factor_denominator + long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]

        # Added the following check for division by 0 on 13-Nov-2024 by Niko Schonsheck.
        if synaptic_scaling_factor_denominator != 0:
            synaptic_scaling_factor = synaptic_scaling_factor_numerator/synaptic_scaling_factor_denominator
        else:
            synaptic_scaling_factor = 1



        normalized_delta_w_matrix[postsynaptic_neuron_index, :] = (1/synaptic_scaling_factor)*normalized_delta_w_matrix[postsynaptic_neuron_index, :]

    # Update long_term_presynaptic_response_matrix to now include responses over the learning window (throwing out the first num_time_bins_short_term_presynaptic_response_matrix columns)
    long_term_presynaptic_response_matrix = long_term_presynaptic_response_matrix[:, num_time_bins_short_term_presynaptic_response_matrix:]
    long_term_presynaptic_response_matrix = np.concatenate((long_term_presynaptic_response_matrix, short_term_presynaptic_response_matrix), axis = 1)

    return normalized_delta_w_matrix, long_term_presynaptic_response_matrix
    




# This is a copy of do_local_variance_and_synaptic_scaling_of_delta_w_matrix without synaptic scaling, created 29 May 2025 by Niko Schonsheck.
@njit
def do_local_variance_scaling_of_delta_w_matrix(delta_w_matrix, short_term_presynaptic_response_matrix, long_term_presynaptic_response_matrix, short_term_presynaptic_mean_firing_rates, long_term_presynaptic_mean_firing_rates, numba_dict_L1_to_L2_connections):
    
    # Collect variables and initialize the normalized (or, scaled) delta w matrix 
    num_time_bins_short_term_presynaptic_response_matrix = short_term_presynaptic_response_matrix.shape[1]
    normalized_delta_w_matrix = np.zeros(delta_w_matrix.shape)
    num_postsynaptic_neurons = delta_w_matrix.shape[0]
    num_presynaptic_neurons = delta_w_matrix.shape[1]


    # DO LOCAL VARIANCE SCALING
    # Find postsynaptic neuron indices that have at least one delta_w entry (i.e., that have a synapse whose strength has changed)
    nontrivial_postsynaptic_neuron_indices = []
    for postsynaptic_neuron_index in range(num_postsynaptic_neurons):
        found_delta_synapse = False
        presynaptic_neuron_index = -1
        while found_delta_synapse == False and presynaptic_neuron_index < num_presynaptic_neurons - 1:
            presynaptic_neuron_index = presynaptic_neuron_index + 1
            if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
                nontrivial_postsynaptic_neuron_indices.append(postsynaptic_neuron_index)
                found_delta_synapse = True

    
    # For each postsynaptic neuron which has had a change in synapse strength to at least one presynaptic neuron...
    for postsynaptic_neuron_index in nontrivial_postsynaptic_neuron_indices:
        short_term_deviations_from_long_term_means = []

        # Loop through presynaptic neurons to which that postsynaptic neuron is connected
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:

            # The following condition is guaranteed to be true at least once
            # If the synpase strength between a given pre- and post-synaptic neuron has changed...
            if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
                deviation = np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index])
                short_term_deviations_from_long_term_means.append(deviation)

        # Find max and min of deviations between short term (over learning window) and long term mean firing rates
        minimum_short_term_deviation_from_long_term_means, maximum_short_term_deviation_from_long_term_means = numba_min_max(short_term_deviations_from_long_term_means)

        # Do the scaling step
        for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
            current_delta_w = delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index]
            if current_delta_w != 0:

                # If given postsynaptic neuron has only one synapse that changed strength, do not scale
                if minimum_short_term_deviation_from_long_term_means == maximum_short_term_deviation_from_long_term_means:
                    normalization_factor = 1
                    normalized_delta_w = current_delta_w*normalization_factor
                    normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w
                
                # Otherwise, do scaling
                else:
                    normalization_factor = (np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]) - minimum_short_term_deviation_from_long_term_means)/(maximum_short_term_deviation_from_long_term_means - minimum_short_term_deviation_from_long_term_means)
                    normalized_delta_w = current_delta_w*normalization_factor
                    normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w

    

    # Update long_term_presynaptic_response_matrix to now include responses over the learning window (throwing out the first num_time_bins_short_term_presynaptic_response_matrix columns)
    long_term_presynaptic_response_matrix = long_term_presynaptic_response_matrix[:, num_time_bins_short_term_presynaptic_response_matrix:]
    long_term_presynaptic_response_matrix = np.concatenate((long_term_presynaptic_response_matrix, short_term_presynaptic_response_matrix), axis = 1)

    return normalized_delta_w_matrix, long_term_presynaptic_response_matrix
    




# Given L1 and L2 responses of shapes (neurons x time steps) = (neurons x T), each with the same number of columns, and a time lag for propagation between L1 and L2, this function updates a given weight matrix 
# by iterating over applictaions of calculate_STDP_delta_W.
# Note that weight matrix should be (num_L2_neurons x num_L1_neurons)
# We require that time_lag_in_ms < size_of_firing_rate_bins_in_ms and time_lag_in_ms < search_window_limit_in_ms and that size_of_firing_rate_bins_in_ms divides search_window_limit_in_ms
# In this function, learning will occurr on the following column indices of the rate response matrices, where we let a = size_of_firing_rate_bins_in_msk b = search_window_limit_in_ms, and c = time_lag_in_ms
# 2(b/a), 2(b/a) + 1, ..., T - 2(b/a) - 1
# i.e., over the range [2(b/a), ..., T - 2(b/a))
# See STDP_update_weight_matrix_docs.pdf for further details.
# Note that this also employs the function do_local_variance_scaling_of_delta_w_matrix. See the documentation of that function
# and the document STDP_scalings.pdf for more details. 
@njit
def STDP_update_weight_matrix_with_local_variance_scaling(numba_dict_L1_to_L2_connections, short_term_L1_rate_response_matrix, short_term_L2_rate_response_matrix, long_term_L1_rate_response_matrix, current_weight_matrix, time_lag_in_ms, size_of_firing_rate_bins_in_ms, A_plus, A_minus, tau_plus, tau_minus, search_window_limit_in_ms):

    num_L1_neurons = short_term_L1_rate_response_matrix.shape[0]
    num_L2_neurons = short_term_L2_rate_response_matrix.shape[0]
    
    if short_term_L1_rate_response_matrix.shape[1] != short_term_L2_rate_response_matrix.shape[1]:
        raise ValueError("short_term_L1_rate_response_matrix and short_term_L2_rate_response_matrix have different number of columns, i.e., time steps.")

    if num_L1_neurons != current_weight_matrix.shape[1]:
        raise ValueError("short_term_L1_rate_response_matrix and current_weight_matrix imply different number of L1 neurons.")

    if num_L2_neurons != current_weight_matrix.shape[0]:
        raise ValueError("short_term_L2_rate_response_matrix and current_weight_matrix imply different number of L2 neurons.")
    
    if time_lag_in_ms >= size_of_firing_rate_bins_in_ms:
        raise ValueError("Size of firing rate bins in ms must be greater than time lag in ms.")
    
    if time_lag_in_ms >= search_window_limit_in_ms:
        raise ValueError("Size of search window limit in ms must be greater than time lag in ms.")
    
    if (search_window_limit_in_ms/size_of_firing_rate_bins_in_ms) % 1 != 0:
        raise ValueError("Size of firing rate bins  in ms must divide search window limit in ms.")
    

    delta_w_matrix = np.zeros((num_L2_neurons, num_L1_neurons))

    # We expect the length of spiketrains for learning to be (number of columns in rate matrices)*(size of firing rate bins in ms) - 2*(search window limit in ms)
    expected_length_of_spiketrains_for_learning = short_term_L1_rate_response_matrix.shape[1]*size_of_firing_rate_bins_in_ms - 2*(search_window_limit_in_ms)

    # Construct a matrices to hold spiketrains for learning for L1 and L2 neurons
    matrix_of_presynaptic_spiketrains_for_learning = np.zeros((num_L1_neurons, expected_length_of_spiketrains_for_learning))
    for L1_neuron_index in range(num_L1_neurons):
        presynaptic_spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(short_term_L1_rate_response_matrix[L1_neuron_index, :], size_of_firing_rate_bins_in_ms)
        presynaptic_spiketrain_to_learn = presynaptic_spiketrain[search_window_limit_in_ms:-search_window_limit_in_ms]
        if len(presynaptic_spiketrain_to_learn) != expected_length_of_spiketrains_for_learning:
            raise ValueError('Variable `presynaptic_spiketrain_to_learn` is not of expected length.')
        else:
            matrix_of_presynaptic_spiketrains_for_learning[L1_neuron_index, :] = presynaptic_spiketrain_to_learn


    matrix_of_postsynaptic_spiketrains_for_learning = np.zeros((num_L2_neurons, expected_length_of_spiketrains_for_learning))
    for L2_neuron_index in range(num_L2_neurons):
        postsynaptic_spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(short_term_L2_rate_response_matrix[L2_neuron_index, :], size_of_firing_rate_bins_in_ms)
        postsynaptic_spiketrain_to_learn = postsynaptic_spiketrain[(search_window_limit_in_ms - time_lag_in_ms):-(search_window_limit_in_ms + time_lag_in_ms)]
        if len(postsynaptic_spiketrain_to_learn) != expected_length_of_spiketrains_for_learning:
            raise ValueError('Variable `postsynaptic_spiketrain_to_learn` is not of expected length.')
        else:
            matrix_of_postsynaptic_spiketrains_for_learning[L2_neuron_index, :] = postsynaptic_spiketrain_to_learn





    for L1_neuron_index in range(num_L1_neurons):
        for L2_neuron_index in range(num_L2_neurons):

            if current_weight_matrix[L2_neuron_index, L1_neuron_index] != 0:

                presynaptic_spiketrain_to_learn = matrix_of_presynaptic_spiketrains_for_learning[L1_neuron_index, :]
                postsynaptic_spiketrain_to_learn = matrix_of_postsynaptic_spiketrains_for_learning[L2_neuron_index, :]


                delta_w_matrix[L2_neuron_index, L1_neuron_index] = calculate_STDP_delta_W(A_plus, A_minus, tau_plus, tau_minus, presynaptic_spiketrain_to_learn, postsynaptic_spiketrain_to_learn, search_window_limit_in_ms)
    

    short_term_presynaptic_mean_firing_rates = np.array([short_term_L1_rate_response_matrix[i, :].mean() for i in range(short_term_L1_rate_response_matrix.shape[0])])
    long_term_presynaptic_mean_firing_rates = np.array([long_term_L1_rate_response_matrix[i, :].mean() for i in range(long_term_L1_rate_response_matrix.shape[0])])
    
    normalized_delta_w_matrix, new_long_term_presynaptic_response_matrix = do_local_variance_scaling_of_delta_w_matrix(
        delta_w_matrix, 
        short_term_L1_rate_response_matrix, 
        long_term_L1_rate_response_matrix, 
        short_term_presynaptic_mean_firing_rates, 
        long_term_presynaptic_mean_firing_rates, 
        numba_dict_L1_to_L2_connections)
    
    new_weight_matrix = np.zeros((num_L2_neurons, num_L1_neurons))

    for L2_neuron_index in range(num_L2_neurons):
        for L1_neuron_index in range(num_L1_neurons):
            new_weight_matrix[L2_neuron_index, L1_neuron_index] = (1 + normalized_delta_w_matrix[L2_neuron_index, L1_neuron_index])*current_weight_matrix[L2_neuron_index, L1_neuron_index] 

    return new_weight_matrix, new_long_term_presynaptic_response_matrix




# Passed testing at git hash 74cd5e9fd3544cf87ecdf9ad88f68a6793283fb1
def determine_direction_on_circle(init_pos, final_pos, threshold):
    """
    Let S^1 = [a, b]/(a ~ b). This function returns +1 if the direction of movement on S^1 is positive and -1 if the direction is negative.
    If the difference between `init_pos` and `final_pos` is greater than `threshold` it assumes that the trajectory has wrapped around the circle.
    It panics if `init_pos == final_pos`

    Parameters
    ----------
    init_pos : float
        Initial position on S^1
    
    final_pos : float
        Final position on S^1

    threshold : float
        Threshold beyond which to assume a wrap has occurred.
    
    Returns
    ----------
    orientation : float
        +1 if positive direction, -1 if negative direction, 0 if init_pos = final_pos
    
    """

    # If no movement, return 0
    if init_pos == final_pos:
        orientation = 0
        return orientation
    
    # Compute magnitude of difference
    abs_val_diff = np.abs(final_pos - init_pos)

    # If below threshold, no wrapping
    if abs_val_diff < threshold:
        orientation = np.sign(final_pos - init_pos)
        return orientation
    
    # Otherwise, wrapping
    else:
        orientation = -1*np.sign(final_pos - init_pos)
        return orientation


def test_determine_direction_on_circle():
    
    # No wrapping, positive direction
    actual_result_1 = determine_direction_on_circle(.1, .2, .5)
    expected_result_1 = 1
    assert(np.isclose(actual_result_1, expected_result_1))

    # No wrapping, negative direction
    actual_result_2 = determine_direction_on_circle(10, 9.9, .5)
    expected_result_2 = -1
    assert(np.isclose(actual_result_2, expected_result_2))

    # Wrapping, negative direction
    actual_result_3 = determine_direction_on_circle(.1, 10, .5)
    expected_result_3 = -1
    assert(np.isclose(actual_result_3, expected_result_3))

    # Wrapping, positive direction
    actual_result_4 = determine_direction_on_circle(2*np.pi - 1e-6, 1e-6, .2)
    expected_result_4 = 1
    assert(np.isclose(actual_result_4, expected_result_4))

    # No movement
    actual_result_5 = determine_direction_on_circle(.1, .1, .5)
    expected_result_5 = 0
    assert(np.isclose(actual_result_5, expected_result_5))

    print('test_determine_direction_on_circle(): passed.')




def count_num_direction_changes_on_circle(positions, threshold):
    """
    Given a sequence of posititions, `positions` on S^1 = [a, b]/(a ~ b), count the number of times the trajectory changes directions.
    Note that we count stopping as a change in direction or starting from stop as a change in direction.
    We assume a wrap on the circle has occurred if the difference between consecutive positions is greater, in absolute value, than `threshold`.

    Parameters
    ----------
    positions : 1-d numpy array or list
        List of floats corresponding to positions on S^1

    threshold : float
        Parameter beyond which to assume a wrap has occurred

    
    Returns
    --------
    num_direction_changes : float
        Number of times the trajectory determined by `positions` has changed direction.
    """

    # Initialize counter
    num_direction_changes = 0

    # Compare current direction to next direction, add to counter if there is a change in direction
    for index in range(1, len(positions) - 1):
        current_direction = determine_direction_on_circle(positions[index - 1], positions[index], threshold)
        next_direction = determine_direction_on_circle(positions[index], positions[index + 1], threshold)

        if np.sign(current_direction) != np.sign(next_direction):
            num_direction_changes = num_direction_changes + 1

    return num_direction_changes


# Passed testing at git hash 74cd5e9fd3544cf87ecdf9ad88f68a6793283fb1
def test_counter_num_direction_changes_on_circle():
    
    # No changes, no wraps
    positions_1 = [.1, .2, .3, .4]
    actual_result_1 = count_num_direction_changes_on_circle(positions_1, .5)
    expected_result_1 = 0
    assert(actual_result_1 == expected_result_1)

    # No changes, two wraps on [0, 1]/(0 ~ 1)
    positions_2 = [.2, .4, .6, .8, .95, .05, .25, .45, .65, .85, .05]
    actual_result_2 = count_num_direction_changes_on_circle(positions_2, .3)
    expected_result_2 = 0
    assert(actual_result_2 == expected_result_2)

    # One change, no wraps
    positions_3 = [.2, .3, .4, .5, .4, .3, .2]
    actual_result_3 = count_num_direction_changes_on_circle(positions_3, .5)
    expected_result_3 = 1
    assert(actual_result_3 == expected_result_3)

    # Two changes, two wraps
    positions_4 = np.array(
        [
            .1, .2, .3, .4, .35, .31, .2, .1, 6, 5.9, 6, .1, .2, .3
        ]
    )
    actual_result_4 = count_num_direction_changes_on_circle(positions_4, .5)
    expected_result_4 = 2
    assert(actual_result_4 == expected_result_4)

    # One short pause, no wraps
    positions_5 = [.2, .3, .3, .4, .5, .6]
    actual_result_5 = count_num_direction_changes_on_circle(positions_5, .5)
    expected_result_5 = 2
    assert(actual_result_5 == expected_result_5)

    # One long pause, no wraps
    positions_6 = [.2, .3, .3, .3, .3, .3, .4, .5]
    actual_result_6 = count_num_direction_changes_on_circle(positions_6, .5)
    expected_result_6 = 2
    assert(actual_result_6 == expected_result_6)



    print('test_counter_num_direction_changes_on_circle(): passed.')







# Passed testing at git hash 74cd5e9fd3544cf87ecdf9ad88f68a6793283fb1
def estimate_bias_from_walk(circular_positions, length_of_walk_in_ms, threshold):
    """
    Given a walk on S^1 = [a, b]/(a ~ b), return the number of direction changes divided by length of walk in milliseconds to give
    average number of direction changes per millisecond.

    Parameters
    ----------
    circular_positions : 1-d numpy array or list of floats
        List of positions on S^1

    length_of_walk_in_ms : float
        Temporal length of the walk in milliseconds

    threshold : float   
        Threshold used in function count_num_direction_changes_on_circle()

    
    Returns
    ----------
    num_direction_changes_per_ms : float
        Total number of direction changes divided by `length_of_walk_in_ms`
    """
    
    # Count total number of direction changes
    num_direction_changes = count_num_direction_changes_on_circle(circular_positions, threshold)

    num_direction_changes_per_ms = num_direction_changes/length_of_walk_in_ms
    # Final return
    return num_direction_changes_per_ms


def test_estimate_bias_from_walk():
    
    # No changes, no wraps, 10ms
    positions_1 = [.1, .2, .3, .4]
    actual_result_1 = estimate_bias_from_walk(positions_1, 10, .5)
    expected_result_1 = 0
    assert(np.isclose(actual_result_1, expected_result_1))


    # No changes, two wraps on [0, 1]/(0 ~ 1), 21.5 ms
    positions_2 = [.2, .4, .6, .8, .95, .05, .25, .45, .65, .85, .05]
    actual_result_2 = estimate_bias_from_walk(positions_2, 21.5, .5)
    expected_result_2 = 0
    assert(np.isclose(actual_result_2, expected_result_2))

    # One change, no wraps, 109.1ms
    positions_3 = [.2, .3, .4, .5, .4, .3, .2]
    actual_result_3 = estimate_bias_from_walk(positions_3, 109.1, 0.5)
    expected_result_3 = 1/109.1
    assert(np.isclose(actual_result_3, expected_result_3))

    # Two changes, two wraps, 1.2ms
    positions_4 = np.array(
        [
            .1, .2, .3, .4, .35, .31, .2, .1, 6, 5.9, 6, .1, .2, .3
        ]
    )
    actual_result_4 = estimate_bias_from_walk(positions_4, 1.2, 0.5)
    expected_result_4 = 2/1.2
    assert(np.isclose(actual_result_4, expected_result_4))

    # One short pause, no wraps, 10ms
    positions_5 = [.2, .3, .3, .4, .5, .6]
    actual_result_5 = estimate_bias_from_walk(positions_5, 10, .5)
    expected_result_5 = 2/10
    assert(np.isclose(actual_result_5, expected_result_5))

    # One long pause, no wraps, 15ms
    positions_6 = [.2, .3, .3, .3, .3, .3, .4, .5]
    actual_result_6 = estimate_bias_from_walk(positions_6, 15, 0.5)
    expected_result_6 = 2/15
    assert(np.isclose(actual_result_6, expected_result_6))

    print('test_estimate_bias_from_walk(): passed.')




def generate_biased_normal_random_circular_walk(initial_position, bias, num_steps, step_size_mean, step_size_std_dev):
    probability_to_continue_in_same_direction = bias

    # Generate a biased random walk on (-infty, infty). To get the appropriate circular walk, we will
    # mod by 1.
    walk_on_real_line = np.zeros(num_steps)

    # Initialize the walk, and make the first step
    walk_on_real_line[0] = initial_position
    second_position  = initial_position + max(1e-6, np.random.normal(step_size_mean, step_size_std_dev))
    walk_on_real_line[1] = second_position

    for step_index in range(1, num_steps - 1):
        current_position = walk_on_real_line[step_index]

        previous_position = walk_on_real_line[step_index - 1]

        which_direction_last_step = np.sign(current_position - previous_position)

        sample_to_determine_which_direction_next_step = random.uniform(0,1)

        next_step_size = max(1e-6, np.random.normal(step_size_mean, step_size_std_dev))

        if sample_to_determine_which_direction_next_step < probability_to_continue_in_same_direction: # keep going in the same direction
            next_position = current_position + which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
        else: # reverse direction
            next_position = current_position + (-1)*which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
    
    circular_walk = [(x % 1) for x in walk_on_real_line]

    return np.array(circular_walk)






def generate_biased_lognormal_random_circular_walk(initial_position, bias, num_steps, step_size_mean_of_underlying_normal, step_size_std_dev_of_underlying_normal):
    probability_to_continue_in_same_direction = bias

    # Generate a biased random walk on (-infty, infty). To get the appropriate circular walk, we will
    # mod by 1.
    walk_on_real_line = np.zeros(num_steps)

    # Initialize the walk, and make the first step
    walk_on_real_line[0] = initial_position
    second_position  = initial_position + np.random.lognormal(step_size_mean_of_underlying_normal, step_size_std_dev_of_underlying_normal)
    walk_on_real_line[1] = second_position

    for step_index in range(1, num_steps - 1):
        current_position = walk_on_real_line[step_index]

        previous_position = walk_on_real_line[step_index - 1]

        which_direction_last_step = np.sign(current_position - previous_position)

        sample_to_determine_which_direction_next_step = random.uniform(0,1)

        next_step_size = np.random.lognormal(step_size_mean_of_underlying_normal, step_size_std_dev_of_underlying_normal)

        if sample_to_determine_which_direction_next_step < probability_to_continue_in_same_direction: # keep going in the same direction
            next_position = current_position + which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
        else: # reverse direction
            next_position = current_position + (-1)*which_direction_last_step*next_step_size
            walk_on_real_line[step_index + 1] = next_position
    
    circular_walk = [(x % 1) for x in walk_on_real_line]

    return np.array(circular_walk)



# DEPRECATED FUNCTIONS


# # Initializes a two-layer feedforward network L1 -> L2 -> L3, where
# # Connections L1 -> L2 are drawn from specified distribution (make the distribution non-negative if L1 neurons are to be understood as excitatory) with specified sparsity enforced; max of 1, min of -1 enforced.
# # and e.g. if sparsity_parameter = 0.1, then if there are 100 neurons in L2, each neuron in L1 will be connected to 10 randomly chosen neurons in L2
# # L2 is made up of Dale neurons, with outgoing connection strengths equal to 1 or -1
# # By convention, if L2 comprises N_2_total neurons, of which N_2_inh are inhibitory, then neurons in L2 indexed 0 to N_2_inh-1 will be inhibitory, and neurons indexed N_2_inh to N_2_total - 1 will be excitatory
# # Neurons in L2 have biases specified by initial_bias_vector
# # Neurons in L3 do not have biases
# # What we return (in order)
# # "L1_to_L2_connection_matrix" is the initial L1 -> L2 connection matrix
# # "initial_row_sums" is a vector of row sums of the initial L1 -> L2 connection matrix. This is useful for normalization, if we desire to keep the total weight into an L2 neuron constant, e.g.
# # "bias_vector" is the initial bias_vector for L2 neurons
# # "L2_to_L3_connection_matrix" is the initial L2 -> L3 connection matrix 
# # Last modified 9 Oct 2024 by Niko Schonsheck.
# def initialize_network(num_layer_1_neurons, num_layer_2_neurons, num_layer_2_inhibitory_neurons, num_layer_3_neurons, initial_bias_vector_value, sparsity_parameter_layer_1_to_layer_2, sparsity_parameter_layer_2_to_layer_3, distribution, *args):


#     # Make L1 -> L2 connection matrix, enforcing maximum of 1 and minimum of -1 and save
#     L1_to_L2_weight_matrix = np.zeros((num_layer_2_neurons, num_layer_1_neurons))

#     num_outgoing_connections_of_each_L1_neuron = np.floor(num_layer_2_neurons * sparsity_parameter_layer_1_to_layer_2).astype(int)

#     # For each L1 neuron...
#     for col_index in range(num_layer_1_neurons):

#         # Select which L2 neurons to connect to by shuffling the list [0, 1, ..., num_layer_2_neurons] and selecting the first num_outgoing_connections_of_each_L1_neuron of this list
#         shuffled_layer_neuron_indices = list(range(num_layer_2_neurons))
#         np.random.shuffle(shuffled_layer_neuron_indices)

#         for index in range(num_outgoing_connections_of_each_L1_neuron):
#             weight = sample_from_distribution(distribution, *args)
#             weight = np.minimum(1, weight)
#             weight = np.maximum(-1, weight)

#             layer_2_neuron_to_connect = shuffled_layer_neuron_indices[index]

#             L1_to_L2_weight_matrix[layer_2_neuron_to_connect, col_index] = weight

    
#     # Sum rows of first connection matrix for normalization downstream (e.g., want to keep total weight into each L2 neuron constant through learning)
#     initial_row_sums = np.sum(L1_to_L2_weight_matrix, axis = 1)

    
#     # Make initial bias vector for L2 neurons
#     bias_vector = np.zeros(num_layer_2_neurons)
#     for index in range(num_layer_2_neurons):
#         bias_vector[index] = initial_bias_vector_value


#     # Make L2 -> L3 connection matrix (this will not update through learning)
#     L2_to_L3_weight_matrix = np.zeros((num_layer_3_neurons, num_layer_2_neurons))

#     num_incoming_connections_to_each_L3_neuron = np.floor(num_layer_2_neurons * sparsity_parameter_layer_2_to_layer_3).astype(int)


#     # For each L3 neuron...
#     for row_index in range(num_layer_3_neurons):
#         # Select which L2 neurons to connect to by shuffling the list [0, 1, ..., num_layer_2_neurons] and selecting the first num_incoming_connections_to_each_L3_neuron of this list
#         shuffled_layer_neuron_indices = list(range(num_layer_2_neurons))
#         np.random.shuffle(shuffled_layer_neuron_indices)

#         for index in range(num_incoming_connections_to_each_L3_neuron):
#             layer_2_neuron_to_connect = shuffled_layer_neuron_indices[index]

#             if layer_2_neuron_to_connect < num_layer_2_inhibitory_neurons:
#                 # Then the L2 neuron is inhibitory
#                 L2_to_L3_weight_matrix[row_index, layer_2_neuron_to_connect] = -1
#             elif layer_2_neuron_to_connect >= num_layer_2_inhibitory_neurons:
#                 # Then the L2 neuron is excitatory
#                 L2_to_L3_weight_matrix[row_index, layer_2_neuron_to_connect] = 1
#             else:
#                 raise Exception("Incomparable types layer_2_neuron_to_connect.")


#     return (L1_to_L2_weight_matrix, initial_row_sums, bias_vector, L2_to_L3_weight_matrix)




# # Note that delta_w_matrix is (num_postsynaptic_neurons x num_presynaptic_neurons)
# @njit
# def old_do_long_term_presynaptic_normalization_of_delta_w_matrix(delta_w_matrix, short_term_presynaptic_response_matrix, long_term_presynaptic_response_matrix, short_term_presynaptic_mean_firing_rates, long_term_presynaptic_mean_firing_rates, numba_dict_L1_to_L2_connections):
#     num_time_bins_short_term_presynaptic_response_matrix = short_term_presynaptic_response_matrix.shape[1]
#     normalized_delta_w_matrix = np.zeros(delta_w_matrix.shape)
#     num_postsynaptic_neurons = delta_w_matrix.shape[0]
#     num_presynaptic_neurons = delta_w_matrix.shape[1]


#     # Do local variance scaling
#     # Find postsynaptic neuron indices that have at least one delta_w entry (i.e., that have a synapse whose strength has changed)
#     nontrivial_postsynaptic_neuron_indices = []
#     for postsynaptic_neuron_index in range(num_postsynaptic_neurons):
#         found_delta_synapse = False
#         presynaptic_neuron_index = -1
#         while found_delta_synapse == False and presynaptic_neuron_index < num_presynaptic_neurons - 1:
#             presynaptic_neuron_index = presynaptic_neuron_index + 1
#             if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
#                 nontrivial_postsynaptic_neuron_indices.append(postsynaptic_neuron_index)
#                 found_delta_synapse = True

    

#     for postsynaptic_neuron_index in nontrivial_postsynaptic_neuron_indices:
#         short_term_deviations_from_long_term_means = []
#         for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
            

#             # The following condition is guaranteed to be true at least once
#             if delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] != 0:
#                 deviation = np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index])
#                 short_term_deviations_from_long_term_means.append(deviation)

#         minimum_short_term_deviation_from_long_term_means, maximum_short_term_deviation_from_long_term_means = numba_min_max(short_term_deviations_from_long_term_means)

#         for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
#             current_delta_w = delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index]
#             if current_delta_w != 0:
#                 if minimum_short_term_deviation_from_long_term_means == maximum_short_term_deviation_from_long_term_means:
#                     normalization_factor = 1
#                     normalized_delta_w = current_delta_w*normalization_factor
#                     normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w
#                 else:
#                     normalization_factor = (np.abs(long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index] - short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]) - minimum_short_term_deviation_from_long_term_means)/(maximum_short_term_deviation_from_long_term_means - minimum_short_term_deviation_from_long_term_means)
#                     normalized_delta_w = current_delta_w*normalization_factor
#                     normalized_delta_w_matrix[postsynaptic_neuron_index, presynaptic_neuron_index] = normalized_delta_w

    

#     # Do synaptic scaling
#     for postsynaptic_neuron_index in range(num_postsynaptic_neurons):

#         synaptic_scaling_factor_numerator = 0
#         for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
#             synaptic_scaling_factor_numerator = synaptic_scaling_factor_numerator + short_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]

#         synaptic_scaling_factor_denominator = 0
#         for presynaptic_neuron_index in numba_dict_L1_to_L2_connections[postsynaptic_neuron_index]:
#             synaptic_scaling_factor_denominator = synaptic_scaling_factor_denominator + long_term_presynaptic_mean_firing_rates[presynaptic_neuron_index]

#         synaptic_scaling_factor = synaptic_scaling_factor_numerator/synaptic_scaling_factor_denominator



#         normalized_delta_w_matrix[postsynaptic_neuron_index, :] = (1/synaptic_scaling_factor)*normalized_delta_w_matrix[postsynaptic_neuron_index, :]

    
#     long_term_presynaptic_response_matrix = long_term_presynaptic_response_matrix[:, num_time_bins_short_term_presynaptic_response_matrix:]
#     long_term_presynaptic_response_matrix = np.concatenate((long_term_presynaptic_response_matrix, short_term_presynaptic_response_matrix), axis = 1)

#     return normalized_delta_w_matrix, long_term_presynaptic_response_matrix
    

def main():
    test_determine_direction_on_circle()
    test_counter_num_direction_changes_on_circle()
    test_estimate_bias_from_walk()
    test_calculate_STDP_delta_W()


if __name__ == "__main__":
    main()