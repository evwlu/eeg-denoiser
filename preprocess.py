import pandas as pd
import numpy as np
import pywt
from scipy import signal
from collections import Counter, OrderedDict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

'''
Preprocessing function for the MindWave dataset, which records EEG signals
when viewing images of digits 0-9. The dataset also includes images of
non-digits, which are labeled as -1.
'''
def load_fixedSize(signal_size=476, data_type='MU'):
    # Read the file
    print(f"Preprocessing {data_type} dataset...")
    file_path = f"data/{data_type}.txt"
    df = pd.read_csv(file_path, delimiter="\t")
    data = df.to_numpy()

    ## Preprocessing. We will select samples of denoted size (second last column)
    ## of input size (default 952 because most frequent). This column denotes the
    ## size of the EEG signal frequency list (last column) and consequently allows
    ## for more uniform data.

    isolated_events = data[:, -2]
    data = data[isolated_events == signal_size, :]

    # Remove the first four columns. Now our table is as follows:
    # 0: Label (MNIST Digit: 0-9). Includes -1 for non-digit images.
    # 1: EEG Signal Size
    # 2: EEG Signal Frequency (list of size EEG Signal Size)
    data = data[:, 4:]

    # Convert data[:, 2] from strings to lists of floats.
    for i in range(len(data)):
        split_string = data[i, 2].split(',')
        data[i, 2] = np.array([float(i) for i in split_string], dtype=np.float32)

    # Convert data[:, 2] from lists of floats to numpy arrays. We have now isolated
    # the EEG signal frequency list in labels (0-9) and inputs (EEG signal frequency).
    labels = data[:, 0]
    inputs = data[:, 2]

    if len(inputs) == 0:
        raise Exception("No inputs found. Check your preprocessing.")

    inputs = np.vstack(inputs)  # Vertically stack the arrays within inputs
    encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = encoder.fit_transform(labels.reshape(-1, 1))

    print()
    print("Preprocessing complete.")
    print(f"Inputs shape: {inputs.shape} || Labels shape: {labels_onehot.shape}")

    return inputs, labels_onehot, labels

'''
Preprocessing function for the MindWave dataset, which records EEG signals
when viewing images of digits 0-9. The dataset also includes images of
non-digits, which are labeled as -1.
'''
def returnEventDistribution(data_type='MU'):
    file_path = f"data/{data_type}.txt"
    df = pd.read_csv(file_path, delimiter="\t")
    data = df.to_numpy()
    isolated_events = data[:, -2]
    return OrderedDict(sorted(Counter(isolated_events).items(), key=lambda t: t[1], reverse=True))

'''
Preprocessing function. Allows for some preliminary preprocessing techniques
as described in the paper Mahapatra et al. (2023). These include:
    - Butterworth high-pass filter (0.5 Hz)
    - Applying the DWT coefficient decomposition with 3 layers
        - Normalizing the data using mean and standard deviation [this step has been removed. the results are MUCH better without it.]
    - Scaling inputs to range [0, 1]
'''
def preprocessInputs(data):
    data = butter_highpass_filter_multi(data, 0.5, 128, order=5)
    data = DWT_transform(data)
    # data = normalize_scale(data)
    return data

'''
Butterpass filter. Taken from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
'''
def butter_highpass_filter_multi(data, cutoff, fs, order=5):
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    b, a = butter_highpass(cutoff, fs, order=order)
    n_signals, _ = data.shape
    filtered_signals = np.zeros_like(data)

    for i in range(n_signals):
        filtered_signals[i, :] = signal.filtfilt(b, a, data[i, :])

    return filtered_signals

'''
DWT transform. Decomposes the signal into 3 layers of coefficients.
'''
def DWT_transform(data):
    coefficients = pywt.wavedec(data, 'db4', mode='sym', level=3)
    coefficients_thresholded = [pywt.threshold(c, 0.1, mode='soft') for c in coefficients]
    eeg_preprocessed = pywt.waverec(coefficients_thresholded, 'db4', mode='sym')
    return eeg_preprocessed

'''
Normalize data using mean and standard deviation.
'''
def normalize_scale(data):
    # data = (data - np.mean(data)) / np.std(data)
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)