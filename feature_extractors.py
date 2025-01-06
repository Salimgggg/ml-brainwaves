import numpy as np
import pandas as pd
from scipy.fft import fft
import pywt
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import pywt


import torch
from torch.utils.data import Dataset

class FeatureExtractorDataset(Dataset):
    def __init__(self, signals, targets, feature_functions=None, index_start=0):
        """
        A PyTorch Dataset that extracts features and stores them in a format compatible 
        with PyTorch and sklearn models.

        Parameters:
        - signals: np.ndarray, shape (num_samples, window_length) for single-channel 
                   or (num_windows, num_channels, window_length) for multi-channel.
        - targets: np.ndarray, shape (num_samples,) or (num_windows,)
        - feature_functions: dict, maps feature names to functions.
        - index_start: int, starting index for data points (default 0).
        """
        self.signals = signals
        self.targets = targets
        self.index_start = index_start

        if not feature_functions:
            feature_functions = {
                "mean_value": self._mean_value,
                "std_dev": self._std_dev,
                "amplitude": self._amplitude,
                "fourier_power": self._fourier_power,
                "wavelet_energy": self._wavelet_energy,
            }
        
        self.feature_functions = feature_functions
        self.features, self.feature_names = self._extract_features()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
        - A tuple (features, target, index)
        """
        return self.features[idx], self.targets[idx], idx + self.index_start

    def _extract_features(self):
        """
        Extract features from the signals and return them as a list of tensors.
        Also tracks feature names.

        Returns:
        - features: torch.FloatTensor, shape (num_samples, num_features)
        - feature_names: list of str, names of the extracted features
        """
        feature_dict = {}
        indices = []

        # Extract features for each signal
        for fname, ffunc in self.feature_functions.items():
            # Apply feature function
            if self.signals.ndim == 2:  # Single-channel
                feats = np.array([ffunc(signal) for signal in self.signals])
            elif self.signals.ndim == 3:  # Multi-channel
                feats = np.array([ffunc(window) for window in self.signals])
                if feats.ndim == 2:  # (num_windows, num_channels)
                    # Expand channel features into separate columns
                    feats = feats.reshape(feats.shape[0], -1)
            else:
                raise ValueError("Signals must be 2D or 3D.")

            feature_dict[fname] = feats
            indices.append(fname)

        # Combine features into a single matrix
        combined_features = np.column_stack(list(feature_dict.values()))
        feature_names = []

        for fname, feat_array in feature_dict.items():
            if feat_array.ndim == 1:  # Single feature column
                feature_names.append(fname)
            else:  # Multi-channel features (expanded into separate columns)
                feature_names.extend([f"{fname}_ch{ch}" for ch in range(feat_array.shape[1])])

        return torch.FloatTensor(combined_features), feature_names

    def get_feature_names(self):
        """
        Returns the names of all extracted features.
        """
        return self.feature_names

    def get_indices(self):
        """
        Returns the indices of all data points.
        """
        return list(range(self.index_start, self.index_start + len(self)))

    def to_dataframe(self):
        """
        Converts the dataset into a Pandas DataFrame for inspection or export.
        """
        import pandas as pd

        df = pd.DataFrame(self.features.numpy(), columns=self.feature_names)
        df["target"] = self.targets
        df["index"] = self.get_indices()
        return df

    # Feature functions
    def _mean_value(self, signal):
        return np.mean(signal, axis=-1)

    def _std_dev(self, signal):
        return np.std(signal, axis=-1)

    def _amplitude(self, signal):
        return np.ptp(signal, axis=-1)

    def _fourier_power(self, signal):
        fft_vals = fft(signal, axis=-1)
        power_spectrum = np.abs(fft_vals)**2
        return np.mean(power_spectrum, axis=-1)

    def _wavelet_energy(self, signal, wavelet='db4', level=4):
        if signal.ndim == 1:
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            energy_details = [np.sum(np.array(detail)**2) for detail in coeffs[1:]]
            return np.sum(energy_details)
        else:
            energies = []
            for channel in signal:
                coeffs = pywt.wavedec(channel, wavelet, level=level)
                energy_details = [np.sum(np.array(detail)**2) for detail in coeffs[1:]]
                energies.append(np.sum(energy_details))
            return np.array(energies)

