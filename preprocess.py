import pandas as pd
import numpy as np
import pywt
from scipy import signal
from collections import Counter, OrderedDict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

"""
Reads in the dataset and converts it to a numpy array.
Removes the first four columns.
"""
def getData(data_type="MU"):
    # Read the file
    print(f"Reading {data_type} dataset...")
    file_path = f"data/{data_type}.txt"
    df = pd.read_csv(file_path, delimiter="\t")
    data = df.to_numpy()

    # Remove the first four columns. Now our table is as follows:
    # 0: Label (MNIST Digit: 0-9). Includes -1 for non-digit images.
    # 1: EEG Signal Size
    # 2: EEG Signal Frequency (list of size EEG Signal Size)
    data = data[:, 4:]

    return data


"""
Preprocesses the signal data from a string into an array of floats.
Split the data into inputs, labels, and one-hot encoded labels.
"""
def splitData(data):
    # Convert data[:, 2] from strings to lists of floats.
    for i in range(len(data)):
        split_string = data[i, 2].split(",")
        data[i, 2] = np.array([float(i) for i in split_string], dtype=np.float32)

    labels = data[:, 0]
    inputs = data[:, 2]

    if len(inputs) == 0:
        raise Exception("No inputs found. Check your preprocessing.")

    inputs = np.vstack(inputs)  # Vertically stack the arrays within inputs
    encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = encoder.fit_transform(labels.reshape(-1, 1))
    return inputs, labels_onehot, labels


"""
Preprocessing function for a given dataset that records EEG signals
when viewing images of digits 0-9. The dataset also includes images of
non-digits, which are labeled as -1. Only keeps signals of the most common signal size.
"""
def load_autoSize(data_type="MU"):
    data = getData(data_type)

    # get most common event size
    isolated_events = data[:, 1]
    most_common_event_size = Counter(isolated_events).most_common(1)[0][0]
    data = data[isolated_events == most_common_event_size, :]

    inputs, labels_onehot, labels = splitData(data)

    print("Preprocessing complete.")
    print(f"Inputs shape: {inputs.shape} || Labels shape: {labels_onehot.shape}")

    return inputs, labels_onehot, labels


"""
Preprocessing function for a given dataset that records EEG signals
when viewing images of digits 0-9. The dataset also includes images of
non-digits, which are labeled as -1. Only keeps signals of a fixed size.
"""
def load_fixedSize(signal_size=476, data_type="MU"):
    data = getData(data_type)

    isolated_events = data[:, 1]
    data = data[isolated_events == signal_size, :]

    inputs, labels_onehot, labels = splitData(data)

    print("Preprocessing complete.")
    print(f"Inputs shape: {inputs.shape} || Labels shape: {labels_onehot.shape}")

    return inputs, labels_onehot, labels


"""
Linearly scales inputs to range [0, 1].
"""
def scaleInputs(inputs):
    scaler = MinMaxScaler()
    return scaler.fit_transform(inputs)


"""
Preprocessing function for the MindWave dataset, which records EEG signals
when viewing images of digits 0-9. The dataset also includes images of
non-digits, which are labeled as -1.
"""
def returnEventDistribution(data_type="MU"):
    data = getData(data_type)
    isolated_events = data[:, 1]
    return OrderedDict(
        sorted(Counter(isolated_events).items(), key=lambda t: t[1], reverse=True)
    )


"""
Preprocessing function. Allows for some preliminary preprocessing techniques
as described in the paper Mahapatra et al. (2023). These include:
    - Butterworth high-pass filter (0.5 Hz)
    - Applying the DWT coefficient decomposition with 3 layers
        - Normalizing the data using mean and standard deviation [this step has been removed. the results are MUCH better without it.]
    - Scaling inputs to range [0, 1]
"""
def preprocessInputs(data):
    data = butter_highpass_filter_multi(data, 0.5, 128, order=5)
    data = DWT_transform(data)
    # data = normalize_scale(data)
    return data


"""
Butterpass filter. Taken from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
"""
def butter_highpass_filter_multi(data, cutoff, fs, order=5):
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    b, a = butter_highpass(cutoff, fs, order=order)
    n_signals, _ = data.shape
    filtered_signals = np.zeros_like(data)

    for i in range(n_signals):
        filtered_signals[i, :] = signal.filtfilt(b, a, data[i, :])

    return filtered_signals


"""
DWT transform. Decomposes the signal into 3 layers of coefficients.
"""
def DWT_transform(data):
    coefficients = pywt.wavedec(data, "db4", mode="sym", level=3)
    coefficients_thresholded = [
        pywt.threshold(c, 0.1, mode="soft") for c in coefficients
    ]
    eeg_preprocessed = pywt.waverec(coefficients_thresholded, "db4", mode="sym")
    return eeg_preprocessed


"""
Normalize data using mean and standard deviation.
"""
def normalize_scale(data):
    # data = (data - np.mean(data)) / np.std(data)
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)