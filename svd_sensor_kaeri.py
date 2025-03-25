import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def evaluate_sensor_stability(sensor_data, window_size, overlap=0):
    """
    Evaluates sensor stability using Truncated SVD.

    Args:
        sensor_data (np.array): Time series sensor data.
        window_size (int): Size of the time window for SVD.
        overlap (int): Number of overlapping samples between windows.

    Returns:
        list: List of singular values for each time window.
    """

    singular_values_list = []
    num_samples = len(sensor_data)
    start = 0

    while start + window_size <= num_samples:
        window_data = sensor_data[start:start + window_size]

        # Create a data matrix (if needed, e.g., for multi-channel data)
        data_matrix = window_data.reshape(1, -1) #Reshape to a row vector for simplicity in this example.
        #Apply Truncated SVD.
        U, S, V = svd(data_matrix)

        # Truncate SVD - Keep top K singular values (e.g., K = 5)
        k = min(5, len(S))  # Ensure k is not larger than the number of singular values
        truncated_S = S[:k]

        singular_values_list.append(truncated_S)
        start += (window_size - overlap)

    return singular_values_list

def plot_singular_values(singular_values_list, title):
    """Plots the singular values over time."""

    num_windows = len(singular_values_list)
    singular_values_array = np.array(singular_values_list)

    plt.figure(figsize=(12, 6))
    for i in range(singular_values_array.shape[1]): # Iterate through each singular value
        plt.plot(range(num_windows), singular_values_array[:, i], label=f'Singular Value {i+1}')

    plt.xlabel('Time Window')
    plt.ylabel('Singular Value')
    plt.title(f'Singular Values Over Time : {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage:
# Generate some synthetic sensor data (replace with your actual data)


data_dir = "./"
normal_matrix = np.load(data_dir + 'normal_data.npy') # load
leak_matrix = np.load( data_dir + 'leak_data.npy') # load

print
sensor_data = all_matrix = np.concatenate((normal_matrix, leak_matrix), axis=0)

#sensor_data = sensor_data[0:2000:4] 

sensor_data = normal_matrix[0:4000:2] 
print (sensor_data.shape)

window_size = 40
overlap = 2

singular_values_list = evaluate_sensor_stability(sensor_data, window_size, overlap)
title = "Normal Context (40,2)"
#title = "Normal Context"
plot_singular_values(singular_values_list, title)

# Example of comparing to a reference (you'd need to create a reference set)
if len(singular_values_list) > 0:
  reference_values = singular_values_list[0] #using first window as reference.
  for current_values in singular_values_list[1:]:
      difference = np.linalg.norm(np.array(current_values) - np.array(reference_values))
      print(f"Difference from reference: {difference}")