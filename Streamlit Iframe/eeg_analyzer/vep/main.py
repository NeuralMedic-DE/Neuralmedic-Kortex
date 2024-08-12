import pandas as pd
import numpy as np
from eeg_analyzer.vep.vep_tools import VEPAnalysisTools

def load_eeg_data(file_path):
    """
    Load EEG data from a text file.
    
    Parameters:
    file_path (str): Path to the text file containing EEG data.
    
    Returns:
    list: A list of NumPy arrays, each representing a channel.
    """
    eeg_data = pd.read_csv(file_path, delimiter="\t", skiprows=10)
    array_1 = np.array(eeg_data['12000002 samples'])
    array_2 = np.array(eeg_data['12000002 samples.1'])
    return [array_1, array_2]

def check_equal_length(arrays):
    """
    Check if all arrays in the list have the same length.
    
    Parameters:
    arrays (list): A list of NumPy arrays.
    
    Returns:
    bool: True if all arrays have the same length, False otherwise.
    """
    if not arrays:
        return False
    first_length = len(arrays[0])
    return all(len(arr) == first_length for arr in arrays)

def find_increasing_indices(arr):
    """
    Find indices where the previous value is less than the current value.
    
    Parameters:
    arr (numpy array): The array to analyze.
    
    Returns:
    numpy array: Indices where the previous value is less than the current value.
    """
    prev = np.roll(arr, 1)
    prev[0] = np.nan  # Replace the first element as there's no previous element
    indices = np.where(prev < arr)[0]
    return indices

def calculate_diff_list(indices):
    """
    Calculate the difference between consecutive indices.
    
    Parameters:
    indices (numpy array): Array of indices.
    
    Returns:
    list: List of differences between consecutive indices.
    """
    diff_list = []
    prev = indices[0]
    for j in range(1, len(indices)):
        diff = indices[j] - prev
        prev = indices[j]
        diff_list.append(diff)
    return diff_list

def main():
    # Load EEG data
    list_channels = load_eeg_data("test_data.txt")
    
    # Check if all arrays in list_channels have the same length
    if not check_equal_length(list_channels):
        return
    
    # Find indices where the previous value is less than the current value
    indices = find_increasing_indices(list_channels[1])
    
    # Calculate the difference between consecutive indices
    diff_list = calculate_diff_list(indices)
    
    size_row = min(diff_list)  # You can adjust this based on your needs
    size_col = len(diff_list)
    
    # Extract a subset of the original array using the calculated indices
    eeg_data = list_channels[0][indices[0]:indices[0] + size_row]
    
    # Calculate number of columns for reshaping
    n_col = len(eeg_data) // size_row
    
    # Reshape the array
    reshaped_eeg_array = eeg_data[:int(n_col) * size_row]
    reshaped_eeg_array = reshaped_eeg_array.reshape(-1, size_row).transpose()
    
    reshaped_eeg_row_mean = np.mean(reshaped_eeg_array, axis=1)
    
    Fs = 5000  # Sampling frequency
    
    # Initialize the VEPAnalysisTools class
    vep_tools = VEPAnalysisTools(reshaped_eeg_row_mean, Fs)
    
    # Reshape data
    reshaped_data = vep_tools.reshape_data(size_row)
    
    # Plot VEP data
    vep_tools.plot_vep(reshaped_data)
    
    # Define intervals for amplitude and latency calculation
    intervals = [
        {'name': 'P100', 't_min': 80, 't_max': 130, 'type': 'positive'},
        {'name': 'N75', 't_min': 55, 't_max': 85, 'type': 'negative'},
        {'name': 'N135', 't_min': 130, 't_max': 180, 'type': 'negative'}
    ]
    
    # Display amplitude and latency results
    vep_tools.display_amplitude_latency_results(reshaped_data, intervals)

if __name__ == "__main__":
    main()
