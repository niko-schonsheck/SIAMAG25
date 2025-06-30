from utils_new import *
from utilities import *
import seaborn as sns
from ripser import ripser
from persim import plot_diagrams
from scipy.ndimage import rotate
from scipy.ndimage import shift
import plotly.graph_objects as go
from skimage.feature import peak_local_max
from numba import njit














# Tested and validated by Niko Schonsheck on 8 Jan 2025.
def make_standard_walk_in_square_enclosure(num_steps, x_min, x_max, y_min, y_max):
    """
    Simulate a walk in a square enclosure bounded, in Cartesian coordinates, by the lines x = x_min, x = x_max, y = y_min, and y = y_max
    Arguments:
        num_steps: total number of steps in the walk
        x_min: float representing left-hand side of enclosure
        x_max: float representing right-hand side of enclsoure
        y_min: float representing bottom of enclosure
        y_max: float representing top of enclosure
    Note:
        Each step is assumed to be 10ms in time.
    Returns:
        num_steps x 2 numpy array where each row is a Cartesian coordinate of a point in the walk
    """

    expected_delta_position_per_10_ms = 0.0014480015756437703

    x_positions = [0.0, 0.0 +0.0014480015756437703]
    y_positions = [0.0, 0.0]

    for step_index in range(1, num_steps - 1):
        current_x_position = x_positions[step_index]
        current_y_position = y_positions[step_index]

        previous_x_position = x_positions[step_index - 1]
        previous_y_position = y_positions[step_index - 1]

        delta_x = current_x_position - previous_x_position
        delta_y = current_y_position - previous_y_position

        current_heading = (np.arctan2(delta_y, delta_x))

        next_heading = (np.random.normal(current_heading, 0.1))
        next_step_size = np.random.uniform(expected_delta_position_per_10_ms/2, 2*expected_delta_position_per_10_ms)
        next_x_position = current_x_position + next_step_size*np.cos(next_heading)
        next_y_position = current_y_position + next_step_size*np.sin(next_heading)

        if next_x_position > x_max or next_x_position < x_min or next_y_position > y_max or next_y_position < y_min:
            next_heading = np.random.uniform(0, 2*np.pi)
            next_step_size = np.random.uniform(expected_delta_position_per_10_ms/2, 2*expected_delta_position_per_10_ms)
            next_x_position = current_x_position + next_step_size*np.cos(next_heading)
            next_y_position = current_y_position + next_step_size*np.sin(next_heading)


        next_x_position = np.clip(next_x_position, x_min, x_max).item()
        next_y_position = np.clip(next_y_position, y_min, y_max).item()
        
        

        x_positions.append(next_x_position)
        y_positions.append(next_y_position)


    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    complete_walk = np.zeros((num_steps, 2))

    complete_walk[:, 0] = x_positions
    complete_walk[:, 1] = y_positions

    return complete_walk




# This function can be checked by running script documentation/grid_cell_generate_stimulus.py
# Passed testing (using this script) at git hash 34d0590d4c6d520e9133b2057cd3238eaffb3bc8
def make_standard_walk_in_square_enclosure_siulated_to_data(num_steps, x_min, x_max, y_min, y_max, mean_delta_position_per_bin, std_dev_delta_position_per_bin, mean_log_delta_heading_per_bin, std_dev_log_delta_heading_per_bin, max_delta_heading_per_bin, bias):
    """
    Simulate a walk in a square enclosure bounded, in Cartesian coordinates, by the lines x = x_min, x = x_max, y = y_min, and y = y_max. Note that each step is assumed to be 10ms in time. 
    Note further that we assume changes in position to be normally distributed (with a minimum of 1e-4, for simplicity of computing heading) and changes in heading to be lognormally distributed.

    Parameters
    ----------
    num_steps : float
        Number of steps in the walk

    x_min : float
        Represents left-hand side of enclosure
    
    x_max : float
        Represents right-hand side of enclosure

    y_min : float
        Represents bottom of enclosure

    y_max : float
        Represents top of enclosure

    mean_delta_position_per_bin : float
        Average change in position per bin

    std_dev_delta_position_per_bin : float
        Standard deviation of changes in position per bin

    mean_log_delta_heading_per_bin : float
        Mean of the underlying normal distribution of changes in heading per bin

    std_dev_log_delta_heading_per_bin : float
        Standard deviation of the underlying normal distribution of changes in heading per bin

    max_delta_heading_per_bin : float
        Enforced maximum change in heading per bin


    Returns
    ----------
    complete_walk : numpy array of shape (num_steps, 2)
        Each row is a Cartesian coordinate of a point in the walk. 
    """

    # Randomly initialize the walk.
    # By default, this function returns values in [0, 1), so we multiply by 2pi.
    first_two_headings = 2*np.pi*generate_biased_lognormal_random_circular_walk(np.random.rand(), bias, 3, mean_log_delta_heading_per_bin, std_dev_log_delta_heading_per_bin)

    x_positions = [0.0]

    y_positions = [0.0]

    for index in range(2):

        current_x_position = x_positions[index]
        current_y_position = y_positions[index]

        # To ensure we stay within the enclosure, for these first steps, delta_position will be set to the average
        delta_position = mean_delta_position_per_bin
        heading = first_two_headings[index]

        next_x_position = current_x_position + delta_position*np.cos(heading)
        next_y_position = current_y_position + delta_position*np.sin(heading)

        x_positions.append(next_x_position)
        y_positions.append(next_y_position)



    for step_index in range(2, num_steps - 1):
        current_x_position = x_positions[step_index]
        current_y_position = y_positions[step_index]



        # Step t-2 to t-1
        dx1 = x_positions[step_index - 1] - x_positions[step_index - 2]
        dy1 = y_positions[step_index - 1] - y_positions[step_index - 2]
        # Get heading in [0, 2pi)
        previous_heading = np.arctan2(dy1, dx1) % (2*np.pi)

        # Step t-1 to t
        dx2 = x_positions[step_index] - x_positions[step_index - 1]
        dy2 = y_positions[step_index] - y_positions[step_index - 1]
        # Get heading in [0, 2pi)
        current_heading = np.arctan2(dy2, dx2) % (2*np.pi)


        # This gives +1 for counter-clockwise, -1 for clockwise
        last_direction = determine_direction_on_circle(current_heading, previous_heading, threshold=0.25)



        next_delta_heading = min(max_delta_heading_per_bin, np.random.lognormal(mean_log_delta_heading_per_bin, std_dev_log_delta_heading_per_bin))

        test_for_continue_in_same_direction = np.random.rand()
        if test_for_continue_in_same_direction < bias:
            direction_change = 1
        else:
            direction_change = -1

        
        
        next_heading = current_heading + direction_change*last_direction*next_delta_heading
        next_step_size = max(1e-4, np.random.normal(mean_delta_position_per_bin, std_dev_delta_position_per_bin))
        next_x_position = current_x_position + next_step_size*np.cos(next_heading)
        next_y_position = current_y_position + next_step_size*np.sin(next_heading)

        # If the walk hits a bounday, randomly choose a new direction
        while next_x_position >= x_max or next_x_position <= x_min or next_y_position >= y_max or next_y_position <= y_min:
            next_heading = np.random.uniform(0, 2*np.pi)
            next_step_size = max(1e-4, np.random.normal(mean_delta_position_per_bin, std_dev_delta_position_per_bin))
            next_x_position = current_x_position + next_step_size*np.cos(next_heading)
            next_y_position = current_y_position + next_step_size*np.sin(next_heading)
       
        

        x_positions.append(next_x_position)
        y_positions.append(next_y_position)


    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    complete_walk = np.zeros((num_steps, 2))

    # Check correct number of steps
    if len(x_positions) != num_steps or len(y_positions) != num_steps:
        raise ValueError('Incorrect number of steps created.')

    complete_walk[:, 0] = x_positions
    complete_walk[:, 1] = y_positions

    return complete_walk



        
# Tested and validated by Niko Schonsheck on 8 Jan 2025.
@njit
def convert_cartesian_coords_to_matrix_indices(cartesian_x, cartesian_y, x_min, x_max, y_min, y_max, num_rows, num_cols):
    """
    Given a Cartesian coordinate (cartesian_x, cartesian_y) in a square enclsoure bounded by the lines x = x_min, x = x_max, y = y_min, and y = y_max
    return the matrix index (of a num_rows x num_cols matrix) of the coordinate where we identify:
    - matrix index [0,0] with Cartesian point (x_min, y_max)
    - matrix index [0, num_cols-1] with Cartesian point (x_max, y_max)
    - matrix index [num_rows-1, 0] with Cartesian point (x_min, y_min)
    - matrix index [num_rows-1, num_cols-1] with Cartesian point (x_max, y_min)
    """

    # Compute the width and height of each cell in the matrix
    x_step = (x_max - x_min) / num_cols
    y_step = (y_max - y_min) / num_rows

    # Compute the row and column indices
    row_index = int((y_max - cartesian_y) / y_step)  # Rows decrease as y decreases
    col_index = int((cartesian_x - x_min) / x_step)  # Columns increase as x increases

    # Ensure the indices are within bounds
    row_index = min(max(row_index, 0), num_rows - 1)
    col_index = min(max(col_index, 0), num_cols - 1)

    return row_index, col_index


# Tested and validated by Niko Schonsheck on 8 Jan 2025.
@njit
def convert_walk_in_square_enclosure_to_list_of_matrix_indices(num_steps, walk_in_square_enclsoure, x_min, x_max, y_min, y_max, num_rows, num_cols):
    """
    Given a num_steps x 2 numpy array, walk_in_square_enclosure, representing a walk in an enclosure bounded by the lines x = x_min, x = x_max, y = y_min, and y = y_max
    return a num_steps x 2 numpy array of dtype int where each row is the pair of matrix indices (of a num_rows x num_cols matrix) corresponding to the associated Cartesian point
    """
    
    list_of_matrix_indices = np.zeros((num_steps, 2), dtype=np.int64)
    for step_index in range(num_steps):
        cartesian_x = walk_in_square_enclsoure[step_index, 0]
        cartesian_y = walk_in_square_enclsoure[step_index, 1]
        row_index, col_index = convert_cartesian_coords_to_matrix_indices(cartesian_x, cartesian_y, x_min, x_max, y_min, y_max, num_rows, num_cols)
        list_of_matrix_indices[step_index, 0] = row_index
        list_of_matrix_indices[step_index, 1] = col_index
    
    return list_of_matrix_indices


# Tested and validated by Niko Schonsheck on 8 Jan 2025.
@njit
def compute_grid_cell_response_given_rate_map_tensor_and_walk_as_matrix_indices(rate_map_tensor, walk_as_matrix_indices):
    """
    Arguments:
        rate_map_tensor: a numpy array where rate_map_tensor[:, :, neuron_index] is the rate map, as a matrix, of a given neuron. 
        walk_as_matrix_indices: a num_steps x 2 numpy array of dtype int where each row is a pair of matrix indices corresponding to a point on a walk in a square enclsoure
    Returns:
        Assume rate_map_tensor.shape = (N, N, num_neurons) and walk_as_matrix_indices.shape = (num_steps, 2). Function returns a
        num_neurons x num_steps array, call it response_matrix, where response_matrix[fixed_neuron, :] is a time series of a fixed neuron's repsonse to the walk in the square enclosure
        and response_matrix[:, time_step] is a vector of each neuron's firing rate at a given time.
    """

    num_neurons = rate_map_tensor.shape[2]

    num_steps = walk_as_matrix_indices.shape[0]

    response_matrix = np.zeros((num_neurons, num_steps))

    for fixed_neuron_index in range(num_neurons):
        relevant_rate_map = rate_map_tensor[:, :, fixed_neuron_index]
        for step_index in range(num_steps):
            row_index_of_current_step = walk_as_matrix_indices[step_index, 0]
            col_index_of_current_step = walk_as_matrix_indices[step_index, 1]
            firing_rate = relevant_rate_map[row_index_of_current_step, col_index_of_current_step]
            response_matrix[fixed_neuron_index, step_index] = firing_rate

    return response_matrix


# Tested and validated by Niko Schonsheck on 8 Jan 2025.
def sample_grid_cell_dictionary(dictionary, samples_per_quintile):
    """
    Arguments:
        dictionary: a dictionary made using notebook "make_grid_cell_tuning_curve_dict_v4.ipynb"
        samples_per_quintile: numpy array [N_5, N_4, N_3, N_2, N_1]. PLEASE OBSERVE THE NONSTANDARD ORDERING. This is to be consistent
        with a similar function defined for head-direction cells and because of how np.percentile works.
    Returns:
        A list of indices where N_1 of the indices corresponsd to cells with grid scores in the first quintile (i.e., highest) of the whole population's scores, 
        N_2 of the indices correspond to cells with grid scores in the second quintile (i.e., second highest) of the whole population's scores, etc.
    """

    # Initialize return list sampled_indices
    sampled_indices = []

    # Populate a list of indices [0, 1, 2, ..., num_neurons -1]
    list_of_neuron_indices = list(range(len(dictionary.keys())))

    # Ensure that the sampling is possible
    for num_samples in samples_per_quintile:
        if num_samples >= 0.2*len(list_of_neuron_indices):
            raise ValueError("Not enough neurons to sample as desired.")
    
    # Shuffle the list
    np.random.shuffle(list_of_neuron_indices)

    # Populate a list comprising all grid scores
    all_grid_scores = []
    for key in dictionary.keys():
        grid_score = dictionary[key]['grid_score'][0]
        all_grid_scores.append(grid_score)

    # Compute quintile cutoffs of grid scores
    # Note that quintiles is *increasing*, hence the strange ordering of samples_per_quintile
    quintiles = np.percentile(all_grid_scores, [0, 20, 40, 60, 80, 100])
    
    # For each quintile
    for aux_index in range(len(quintiles) - 1):

        # Set cutoff range
        lower_bound = quintiles[aux_index]
        upper_bound = quintiles[aux_index + 1]

        # Set number of neurons to sample from this quintile
        num_samples_from_this_quintile_to_draw = samples_per_quintile[aux_index]

        samples_drawn = 0
        
        # Proceed through shuffled list greedily adding neurons in selected quintile to return list sampled_indices until we have enough
        counter = 0
        while samples_drawn < num_samples_from_this_quintile_to_draw:
            potential_neuron_index = list_of_neuron_indices[counter]

            # Note that since we are using closed intervals, it is theoretically possible one neuron will appear two times (as part of two different quintiles). We do this to ensure we sample the full range.
            score = dictionary[potential_neuron_index]['grid_score'][0]
            if score <= upper_bound and score >= lower_bound:
                sampled_indices.append(potential_neuron_index)
                samples_drawn = samples_drawn + 1
            counter = counter + 1

    return sampled_indices




    

# Tested and validated by Niko Schonsheck on 8 Jan 2025.
def build_rate_map_tensor_given_cell_indices(dictionary, list_of_indices):
    """
    Arguments:
        dictionary: a dictionary made using notebook "make_grid_cell_tuning_curve_dict_v4.ipynb"
        list_of_indices: a list of indices corresponding to keys of dictionary
    Returns:
        Suppose len(list_of_indices) = num_neurons and each rate map in dictionary is of shape num_rows x num_cols.
        This function returns a num_rows x num_cols x num_neurons numpy array, call it rate_map_tensor, where
        rate_map_tensor[:, :, index] is the fully_processed_rate_map of cell index in dictionary.
    Notes:
        We assume each rate map in dictionary is of the same size
    """

    num_neurons = len(list_of_indices)

    num_rows = dictionary[0]['fully_processed_rate_map'].shape[0]

    num_cols = dictionary[0]['fully_processed_rate_map'].shape[1]

    rate_map_tensor = np.zeros((num_rows, num_cols, num_neurons))

    for list_index, neuron_index in enumerate(list_of_indices):
        rate_map = dictionary[neuron_index]['fully_processed_rate_map']
        rate_map_tensor[:, :, list_index] = rate_map
    
    return rate_map_tensor



# The function below is adapted from function "initialize_network". Its documentation (of the original) is below.

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

# The grid cell version is completely analogous. All changes from original are documented below. Note that instead of passing "tuning_curve_dict", we pass "rate_map_tensor"
# which is a num_rows x num_cols x num_neurons np. array of where rate_map_tensor[:, :, neuron_index] is the rate map (in matrix indices) of a given neuron
# We also pass min_x, max_x, min_y, max_y as arguments; these are the bounds of the square enclosure in Cartesian coordinates
# We assume all rate maps are of the same num_rows and num_cols
# Checked and last modified by Niko Schonsheck on 9 January 2025.


def grid_cell_initialize_network(rate_map_tensor, min_x, max_x, min_y, max_y, num_layer_1_neurons, num_layer_2_neurons, initial_layer_2_bias_vector_factor, sparsity_parameter_layer_1_to_layer_2, distribution, *args):

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

        # This is the only change for grid cells. Original is commented out. 
        # Warm start bias vector for L2 neurons
        # walk_for_bias_vector_warm_start = np.linspace(0,1,1000)%1
        # L1_response_for_bias_vector_warm_start = calculate_response_matrix_given_walk_on_circle_and_tuning_curves(walk_for_bias_vector_warm_start, tuning_curve_dict)
        # L2_respone_for_bias_vector_warm_start = calculate_one_layer_ff_relu_output_matrix(L1_response_for_bias_vector_warm_start, L1_to_L2_weight_matrix, np.zeros(num_layer_2_neurons))
        # bias_vector = initial_layer_2_bias_vector_factor*np.max(L2_respone_for_bias_vector_warm_start, axis = 1)

        # Warm start bias vector for L2 neurons
        num_rows = rate_map_tensor[:, :, 0].shape[0]
        num_cols = rate_map_tensor[:, :, 0].shape[1]
        walk_for_bias_vector_warm_start_in_cartesian_coordinates = make_standard_walk_in_square_enclosure(100000, min_x, max_x, min_y, max_y)
        walk_for_bias_vector_warm_start_in_matrix_indices = convert_walk_in_square_enclosure_to_list_of_matrix_indices(100000, walk_for_bias_vector_warm_start_in_cartesian_coordinates, min_x, max_x, min_y, max_y, num_rows, num_cols)
        L1_response_for_bias_vector_warm_start = compute_grid_cell_response_given_rate_map_tensor_and_walk_as_matrix_indices(rate_map_tensor, walk_for_bias_vector_warm_start_in_matrix_indices)
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




# This function is adapted from "run_propagation_trial"
# The original documentation is below:
# This function represents one of those rare times when the code really is the documentation. Please read through the code to understand the function.
# For additional information on the shifting of spikes trains used to model delay between layers, please see shifting_in_run_propagation_trials.pdf
# Please ensure that time_lag_for_L1_to_L3_in_ms is less than firing_rate_bin_size_in_ms
# Note that, since L3 is essentially a "readout" layer, we do not take it to have any bias vector (only threshold-linearity)
# Checked and last modified 9 Oct 2024 by Niko Schonsheck

# All changes are documented in code or listed here:
# walk_on_circle is now walk_in_square_enclosure_as_matrix_indices: a num_steps x 2 numpy array of dtype Int where each row is a matrix index corresponding to a position on the rate map
# tuning_curves_for_L1_neurons is now rate_map_tensor, which is a num_rows x num_cols x num_neurons np. array of where rate_map_tensor[:, :, neuron_index] is the rate map (in matrix indices) of a given neuron
# len(walk_on_circle) replaced with variable "num_steps = walk_in_square_enclosure_as_matrix_indices.shape[0]"
def grid_cell_run_propagation_trial(
        walk_in_square_enclosure_as_matrix_indices, # a num_steps x 2 numpy array of dtype Int where each row is a matrix index corresponding to a position on the rate map.  Note that if this walk has steps 0, 1, 2, ..., N, (which will correspond to firing rate bins) then the walk used for dissimilarity calculations is the same walk but indexed on 1, 2, ..., N. 
        time_lag_for_L1_to_L3_in_ms, # time lag for signal to propagate from L1 to L3 in milliseconds. Essentially, we will shift spike trains by this amount.
        size_of_firing_rate_bins_in_ms, # how many milliseconds each firing rate bin is treated as. Used for generating spike trains from firing rates, and vice versa. We generally use a value of 10 here.
        sigma_for_Gaussian_smoothing, # sigma used when converting spike trains to firing rates. If the firing rate bin size in ms is relatively small (~10 ms), and the time series is relatively long (~100k firing rate bins), we recommend a *large* value here, around 100.
        random_seed_for_L1_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L1 response matrix
        random_seed_for_L2_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L2 response matrix
        random_seed_for_L3_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L3 response matrix
        rate_map_tensor, # a num_rows x num_cols x num_neurons np. array of where rate_map_tensor[:, :, neuron_index] is the rate map (in matrix indices) of a given neuron
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

    # # Simulate L1 neuron responses
    # L1_response_matrix = calculate_response_matrix_given_walk_on_circle_and_tuning_curves(walk_on_circle, tuning_curves_for_L1_neurons)
    L1_response_matrix = compute_grid_cell_response_given_rate_map_tensor_and_walk_as_matrix_indices(rate_map_tensor, walk_in_square_enclosure_as_matrix_indices)

    # Add noise to L1 response matrix
    L1_response_matrix = add_normal_random_noise(L1_response_matrix, random_seed_for_L1_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L2 neuron responses
    L2_response_matrix = calculate_one_layer_ff_relu_output_matrix(L1_response_matrix, L1_to_L2_weight_matrix, L2_bias_vector)

    # Add noise to L2 response matrix 
    L2_response_matrix = add_normal_random_noise(L2_response_matrix, random_seed_for_L2_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L3 neuron responses. Note that, by convention, we do not add noise to L3 repsonse matrix and do not have a bias vector for L3. Thus, we
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

    num_steps = walk_in_square_enclosure_as_matrix_indices.shape[0]

    L1_response_matrix_from_spikes = np.zeros((L1_response_matrix.shape[0], num_steps - 1))
    for L1_neuron_index in range(L1_response_matrix.shape[0]):
        spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L1_response_matrix[L1_neuron_index, :], size_of_firing_rate_bins_in_ms)
        spiketrain = spiketrain[size_of_firing_rate_bins_in_ms:]
        rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
        if len(rates) != num_steps - 1:
            raise ValueError("Incorrect number of rate bins after converting from spikes to rates.")
        L1_response_matrix_from_spikes[L1_neuron_index, :] = rates
    

    # Convert L3 responses to spike trains, shift by time lag, and then back to rates
    # See shifting_in_run_propagation_trials.pdf for further details on the shifting.
    L3_response_matrix_from_spikes_with_shift = np.zeros((L3_response_matrix.shape[0], num_steps - 1))
    for L3_neuron_index in range(L3_response_matrix.shape[0]):
        spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L3_response_matrix[L3_neuron_index, :], size_of_firing_rate_bins_in_ms)
        spiketrain = spiketrain[size_of_firing_rate_bins_in_ms - time_lag_for_L1_to_L3_in_ms: -time_lag_for_L1_to_L3_in_ms]
        rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
        if len(rates) != num_steps - 1:
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





def grid_cell_run_propagation_trial_no_spiketrain_conversion(
        walk_in_square_enclosure_as_matrix_indices, # a num_steps x 2 numpy array of dtype Int where each row is a matrix index corresponding to a position on the rate map.  Note that if this walk has steps 0, 1, 2, ..., N, (which will correspond to firing rate bins) then the walk used for dissimilarity calculations is the same walk but indexed on 1, 2, ..., N. 
        time_lag_for_L1_to_L3_in_ms, # time lag for signal to propagate from L1 to L3 in milliseconds. Essentially, we will shift spike trains by this amount.
        size_of_firing_rate_bins_in_ms, # how many milliseconds each firing rate bin is treated as. Used for generating spike trains from firing rates, and vice versa. We generally use a value of 10 here.
        sigma_for_Gaussian_smoothing, # NOT USED IN THIS FUNCTION, BUT KEPT TO MAINTAIN SYMMETRY WITH ORIGINAL FUNCTION. sigma used when converting spike trains to firing rates. If the firing rate bin size in ms is relatively small (~10 ms), and the time series is relatively long (~100k firing rate bins), we recommend a *large* value here, around 100.
        random_seed_for_L1_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L1 response matrix
        random_seed_for_L2_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L2 response matrix
        random_seed_for_L3_response_matrix_noise, # integer used as a random seed to add Gaussian noise to L3 response matrix
        rate_map_tensor, # a num_rows x num_cols x num_neurons np. array of where rate_map_tensor[:, :, neuron_index] is the rate map (in matrix indices) of a given neuron
        L1_to_L2_weight_matrix, # (num_L2_neurons x num_L1_neurons) numpy array representing synaptic weights between neurons in L1 and neurons in L2
        L2_bias_vector, # a (num_L2_neurons, ) array representing biases for L2 neurons
        L2_to_L3_weight_matrix, # (num_L3_neurons x num_L2_neurons) numpy array representing synaptic weights between neurons in L2 and neurons in L3
        tau_max_in_firing_rate_bins, # tau_max value used in dissimilarity calculations, measured in number of firing rate bins. Generally we take this to be 100, so that if each firing rate bin is 10ms, we compute dissimilarities over a window of size 1000ms = 1s
        Gaussian_noise_factor, # A value betwen 0 and 1 used to add noise to response matrices. In general, given a firing rate r, we sample from a normal distribution centered at r with standard deviation = max(r*Gaussian_noise_factor, min_std_dev_Gaussian_noise)
        min_std_dev_Gaussian_noise, # See "Gaussian_noise_factor." In general, we suggest Gaussian_noise_factor ~ 0.05 and min_std_dev_Gaussian_noise = 2.5, if firing rates are in Hz
        regularization_factor_for_dissimilarity_calculations # See documentation for calculate_intrapopulation_neuron_dissimilarity and calculate_interpopulation_neuron_dissimilarity_nonsymmetric. We generally recommend a value of 0.1.
    ):

    """
    This is an exact copy of function grid_cell_run_propagation_trial that just skips the rates -> spikes -> rates conversion steps. Note that, as a result,
    there is no simulated time lag between layers.
    """

    # Check variables
    if time_lag_for_L1_to_L3_in_ms >= size_of_firing_rate_bins_in_ms:
        raise ValueError("time_lag_for_L1_to_L3_in_ms must be less than firing_rate_bin_size_in_ms")

    # # Simulate L1 neuron responses
    # L1_response_matrix = calculate_response_matrix_given_walk_on_circle_and_tuning_curves(walk_on_circle, tuning_curves_for_L1_neurons)
    L1_response_matrix = compute_grid_cell_response_given_rate_map_tensor_and_walk_as_matrix_indices(rate_map_tensor, walk_in_square_enclosure_as_matrix_indices)

    # Add noise to L1 response matrix
    L1_response_matrix = add_normal_random_noise(L1_response_matrix, random_seed_for_L1_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L2 neuron responses
    L2_response_matrix = calculate_one_layer_ff_relu_output_matrix(L1_response_matrix, L1_to_L2_weight_matrix, L2_bias_vector)

    # Add noise to L2 response matrix 
    L2_response_matrix = add_normal_random_noise(L2_response_matrix, random_seed_for_L2_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)

    # Simulate L3 neuron responses. Note that, by convention, we do not add noise to L3 repsonse matrix and do not have a bias vector for L3. Thus, we
    # construct a bias vector comprising all 0's of the appropriate size to compute L3 responses
    num_L3_neurons = L2_to_L3_weight_matrix.shape[0]
    L3_bias_vector = np.zeros(num_L3_neurons)
    L3_response_matrix = calculate_one_layer_ff_relu_output_matrix(L2_response_matrix, L2_to_L3_weight_matrix, L3_bias_vector)

    # Add noise to L3 response matrix
    L3_response_matrix = add_normal_random_noise(L3_response_matrix, random_seed_for_L3_response_matrix_noise, Gaussian_noise_factor, min_std_dev_Gaussian_noise)



    # # Convert L1 responses to spike trains to and back to rates.
    # # We do this to ensure robustness and to account for the time lag between L1 and L3
    # # That is, we will be converting L3 rates to spikes in order to account for a delay on the order of milliseconds
    # # and then back to rates in order to do a similarity calculation. So as to perform the same transformation for all inputs
    # # to the cycle matching pipeline, we perform the same transformations on L1 responses
    # # Note that we omit the first firing_rate_bin_size_in_ms milliseconds from this spiketrain to be able to account for the time lag
    # # See shifting_in_run_propagation_trials.pdf for further details on the shifting. 

    # num_steps = walk_in_square_enclosure_as_matrix_indices.shape[0]

    # L1_response_matrix_from_spikes = np.zeros((L1_response_matrix.shape[0], num_steps - 1))
    # for L1_neuron_index in range(L1_response_matrix.shape[0]):
    #     spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L1_response_matrix[L1_neuron_index, :], size_of_firing_rate_bins_in_ms)
    #     spiketrain = spiketrain[size_of_firing_rate_bins_in_ms:]
    #     rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
    #     if len(rates) != num_steps - 1:
    #         raise ValueError("Incorrect number of rate bins after converting from spikes to rates.")
    #     L1_response_matrix_from_spikes[L1_neuron_index, :] = rates
    

    # # Convert L3 responses to spike trains, shift by time lag, and then back to rates
    # # See shifting_in_run_propagation_trials.pdf for further details on the shifting.
    # L3_response_matrix_from_spikes_with_shift = np.zeros((L3_response_matrix.shape[0], num_steps - 1))
    # for L3_neuron_index in range(L3_response_matrix.shape[0]):
    #     spiketrain = simulate_spiketrain_with_firing_rates_in_Hz(L3_response_matrix[L3_neuron_index, :], size_of_firing_rate_bins_in_ms)
    #     spiketrain = spiketrain[size_of_firing_rate_bins_in_ms - time_lag_for_L1_to_L3_in_ms: -time_lag_for_L1_to_L3_in_ms]
    #     rates = spiketrain_to_rates_in_Hz_with_smoothing(spiketrain, size_of_firing_rate_bins_in_ms, sigma_for_Gaussian_smoothing)
    #     if len(rates) != num_steps - 1:
    #         raise ValueError("Incorrect number of rate bins after converting from spikes to rates.")
    #     L3_response_matrix_from_spikes_with_shift[L3_neuron_index, :] = rates

    # Calculate L1 intrapopulation dissimilarity matrix
    L1_intrapopulation_dissimilarity_matrix = calculate_intrapopulation_neuron_dissimilarity(L1_response_matrix, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)

    # Calculate L3 intrapopulation dissimilarity matrix
    L3_intrapopulation_dissimilarity_matrix = calculate_intrapopulation_neuron_dissimilarity(L3_response_matrix, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)

    # Calculate L1 -> L3 interpopulation dissimilarity matrix
    # Note that the return matrix is (num_L1_neurons x num_L3_neurons)
    L1_to_L3_interpopulation_dissimilarity_matrix = calculate_interpopulation_neuron_dissimilarity_nonsymmetric(L1_response_matrix, L3_response_matrix, tau_max_in_firing_rate_bins, regularization_factor_for_dissimilarity_calculations)


    return L1_intrapopulation_dissimilarity_matrix, L3_intrapopulation_dissimilarity_matrix, L1_to_L3_interpopulation_dissimilarity_matrix, L1_response_matrix, L2_response_matrix, L3_response_matrix


