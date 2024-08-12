import pandas as pd
import numpy as np
import mne
from eeg_analyzer.ssvep.ssvep_tools import SSVEPAnalysisTools
import io

def load_eeg_data(file_path):
    """
    Load EEG data from a text file.
    
    Parameters:
    file_path (str): Path to the text file containing EEG data.
    
    Returns:
    list: A list of NumPy arrays, each representing a channel.
    """
    eeg_data = pd.read_csv(file_path, delimiter="\t", skiprows=12)
    array_1 = np.array(eeg_data['5309625 samples'])
    array_2 = np.array(eeg_data['5309625 samples.1'])
    array_3 = np.array(eeg_data['5309625 samples.2'])
    return [array_1, array_2, array_3]

def preprocess_ssvep_data(eeg_data, Fs):
    # Create MNE Info object
    info = mne.create_info(ch_names=['Fp1', 'Fp2', 'Fz'], sfreq=Fs, ch_types='eeg')

    # Convert the data to an MNE RawArray
    raw = mne.io.RawArray(eeg_data, info)

    # Set montage
    montage = mne.channels.make_standard_montage("easycap-M1")
    raw.set_montage(montage, verbose=False)

    # Set common average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Apply bandpass filter
    raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

    # Generate synthetic events: here, every 2 seconds we create an event
    event_id = 1  # This is the ID for your synthetic event
    event_duration = int(Fs * 2)  # Event every 2 seconds
    n_events = raw.n_times // event_duration

    events = np.array([[i * event_duration, 0, event_id] for i in range(n_events)])

    # Construct epochs based on synthetic events
    epochs = mne.Epochs(raw, events=events, event_id={'Stimulus': event_id},
                        tmin=0, tmax=2, baseline=None, verbose=False)
    return epochs.get_data()

def main(file_path):
    # Load EEG data
    list_channels = load_eeg_data(file_path)
    
    # Check if all arrays in list_channels have the same length
    if len(set(map(len, list_channels))) != 1:
        raise ValueError("Error: Not all arrays have the same length.")

    eeg_data = np.array(list_channels)
    Fs = 5000  # Sampling frequency
    
    # Preprocess the data
    preprocessed_data = preprocess_ssvep_data(eeg_data, Fs)
    
    reshaped_eeg_row_mean = np.mean(preprocessed_data, axis=1)
    
    # Initialize the SSVEPAnalysisTools class
    ssvep_tools = SSVEPAnalysisTools(reshaped_eeg_row_mean, Fs)
    
    # Reshape data
    size_row = preprocessed_data.shape[1]
    reshaped_data = ssvep_tools.reshape_data(size_row)
    
    return reshaped_data, ssvep_tools

if __name__ == "__main__":
    # Example usage
    reshaped_data, ssvep_tools = main("test_data.txt")

    # Define target frequencies for SSVEP analysis
    frequencies = [
        {'name': 'Frequency 1', 'frequency': 12},
        {'name': 'Frequency 2', 'frequency': 15}
    ]

    # Display PSD and SNR results
    results = ssvep_tools.display_psd_snr_results(reshaped_data, [f['frequency'] for f in frequencies])
    print(results)
