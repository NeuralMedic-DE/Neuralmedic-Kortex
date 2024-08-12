import matplotlib.pyplot as plt
import numpy as np

class VEPAnalysisTools:
    def __init__(self, reshaped_eeg_row_mean, Fs):
        self.reshaped_eeg_row_mean = reshaped_eeg_row_mean
        self.Fs = Fs

    def reshape_data(self, size_row):
        reshaped_data = self.reshaped_eeg_row_mean.reshape((size_row, -1), order='F')
        return reshaped_data

    def plot_vep(self, reshaped_data, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        t = np.arange(reshaped_data.shape[0]) / self.Fs * 1000  # time in ms
        ax.plot(t, reshaped_data)
        ax.set_title('VEP Plot')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        return ax

    def display_amplitude_latency_results(self, reshaped_data, intervals):
        results = []
        for interval in intervals:
            name = interval['name']
            t_min = interval['t_min']
            t_max = interval['t_max']
            peak_type = interval['type']

            t = np.arange(reshaped_data.shape[0]) / self.Fs * 1000  # time in ms
            start_idx = np.where(t >= t_min)[0][0]
            end_idx = np.where(t <= t_max)[0][-1]

            if peak_type == 'positive':
                peak_value = np.max(reshaped_data[start_idx:end_idx])
                peak_time = t[np.argmax(reshaped_data[start_idx:end_idx]) + start_idx]
            else:
                peak_value = np.min(reshaped_data[start_idx:end_idx])
                peak_time = t[np.argmin(reshaped_data[start_idx:end_idx]) + start_idx]

            results.append(f"The latency of the maximum {name} amplitude is: {peak_time} ms")
            results.append(f"The maximum {name} amplitude is: {peak_value} µV")

        return '\n'.join(results)
