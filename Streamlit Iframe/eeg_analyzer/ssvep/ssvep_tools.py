import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

class SSVEPAnalysisTools:
    def __init__(self, reshaped_eeg_row_mean, Fs):
        self.reshaped_eeg_row_mean = reshaped_eeg_row_mean
        self.Fs = Fs

    def reshape_data(self, size_row):
        reshaped_data = self.reshaped_eeg_row_mean.reshape((size_row, -1), order='F')
        return reshaped_data

    def plot_ssvep(self, reshaped_data, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        t = np.arange(reshaped_data.shape[0]) / self.Fs * 1000  # time in ms
        ax.plot(t, reshaped_data)
        ax.set_title('SSVEP Plot')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        return ax

    def calculate_psd(self, reshaped_data):
        f, Pxx = welch(reshaped_data, self.Fs, nperseg=1024)
        return f, Pxx

    def calculate_snr(self, psd_freqs, psd_values, target_freq):
        signal_power = psd_values[np.argmin(np.abs(psd_freqs - target_freq))]
        noise_power = np.mean(psd_values[(psd_freqs < target_freq - 2) | (psd_freqs > target_freq + 2)])
        snr = signal_power / noise_power
        return snr

    def plot_psd_snr(self, psd_freqs, psd_values, snr_values, target_frequencies):
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (V^2/Hz)', color=color)
        ax1.plot(psd_freqs, psd_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('SNR (dB)', color=color)
        ax2.plot(psd_freqs, 10 * np.log10(snr_values), color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        for freq in target_frequencies:
            ax1.axvline(freq, color='gray', linestyle='--')

        fig.tight_layout()
        plt.title('PSD and SNR Spectra')
        plt.show()

    def display_psd_snr_results(self, reshaped_data, target_frequencies):
        psd_freqs, psd_values = self.calculate_psd(reshaped_data)
        snr_values = np.zeros_like(psd_values)
        results = []

        for target_freq in target_frequencies:
            snr = self.calculate_snr(psd_freqs, psd_values, target_freq)
            snr_values[np.argmin(np.abs(psd_freqs - target_freq))] = snr
            results.append(f"SNR at {target_freq} Hz: {10 * np.log10(snr):.2f} dB")

        self.plot_psd_snr(psd_freqs, psd_values, snr_values, target_frequencies)
        return '\n'.join(results)
