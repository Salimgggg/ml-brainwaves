from pathlib import Path
import numpy as np
import pandas as pd
from utils import butter_bandpass_filter, reshape_signal_into_windows, reshape_targets_for_windows
import matplotlib.pyplot as plt

class ECGDataset:
    def __init__(
        self, 
        data_path, 
        lowcut=0.1, 
        highcut=18, 
        order=4,
        window_duration=2, 
        flatten_channels=True
    ):
        assert window_duration % 2 == 0 and window_duration >= 2, "Window duration must be an even number over 2"
        self.fs = 250
        self.window_length = 250
        self.window_duration = window_duration
        self.data_path = data_path
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.flatten_channels = flatten_channels

        self.data, self.targets = self._load_and_format_data()
        print(f"Data records: {len(self.data)}, Target records: {len(self.targets)}")
        assert len(self.data) == len(self.targets), "Number of data and target records do not match"

    def _load_and_format_data(self):
        data_files = sorted(Path(self.data_path).glob("data_*.npy"))
        target_files = sorted(Path(self.data_path).glob("target_*.npy"))

        data_dict = {}
        target_dict = {}

        has_targets = len(target_files) > 0

        for i, data_file in enumerate(data_files):
            record_number = i  # Use index as record number
            data = np.load(data_file)

            # Check if corresponding target file exists
            if has_targets:
                target_file = target_files[i]
                target = np.load(target_file)
            else:
                target = None  # No targets provided

            # Filter data
            filtered_data = butter_bandpass_filter(data, self.lowcut, self.highcut, self.fs, self.order)
            # Reshape into (num_channels, num_windows, window_length)
            reshaped_data = reshape_signal_into_windows(filtered_data, self.window_length, self.window_duration)

            # If no targets, create dummy targets
            if target is None:
                num_windows = reshaped_data.shape[1]
                num_channels = reshaped_data.shape[0]
                target = np.zeros((num_channels, num_windows), dtype=np.float32)
                print(f"Dummy targets created for file {data_file.name}, shape: {target.shape}")

            reshaped_targets = reshape_targets_for_windows(target, self.window_length, self.window_duration)

            # Store data and targets in dictionaries
            data_dict[record_number] = np.transpose(reshaped_data, (1, 0, 2))  # (num_windows, num_channels, window_length)
            target_dict[record_number] = np.transpose(reshaped_targets, (1, 0, 2))  # (num_windows, num_channels, target_length)

        return data_dict, target_dict

    def get_data_and_targets(self):
        return self.data, self.targets

    def plot_signal_with_label(self, record_number: int, window_index: int = 0, data_sampling_rate: int = 250):
        data = self.data[record_number][window_index]  # Single window
        target = self.targets[record_number][window_index]

        if self.flatten_channels:
            data = data.flatten()
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(data)) / data_sampling_rate, data)
            plt.title(f"Record {record_number}, Window {window_index}, Label: {target}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.show()
        else:
            # Plot each channel separately
            _, axs = plt.subplots(5, 1, figsize=(10, 12))
            for i in range(data.shape[0]):  # Iterate over channels
                axs[i].plot(np.arange(len(data[i])) / data_sampling_rate, data[i])
                axs[i].set_title(f"Record {record_number}, Channel {i}, Window {window_index}, Label: {target[i]}")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel("Amplitude")
            plt.tight_layout()
            plt.show()

    def __getitem__(self, idx):
        # Returns the data and targets for a specific record
        return self.data[idx], self.targets[idx]

    def __len__(self):
        # Returns the number of records
        return len(self.data)
