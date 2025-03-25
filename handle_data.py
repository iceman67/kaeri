import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_directories_recursive(root_dir, target_dir_name):
    """
    Finds all directories with a specific name recursively under a given root directory.

    Args:
        root_dir (str): The starting directory for the search.
        target_dir_name (str): The name of the directory to find.

    Returns:
        list: A list of absolute paths to the found directories.
    """
    found_directories = []
    for root, dirs, files in os.walk(root_dir):
        if target_dir_name in dirs:
            found_directories.append(os.path.join(root, target_dir_name))
    return found_directories


def read_dat_data(filename):

    n_packet = 320
    data = np.memmap(filename, dtype=np.float32)
    return np.reshape(data, [-1, n_packet])


def read_csv_data(filename):

    try:
        csv_data = pd.read_csv(filename, encoding="cp949")
        data_list = csv_data.loc[:, ["SENSOR_DATA"]].values
        data = np.empty((0, 320))

        for value in data_list:
            value_split = np.array([float(x) for x in value[0].split("|")])

            # 이상치 필터링
            if np.sum(value_split**2) < 1e10:
                data = np.concatenate([data, value_split[np.newaxis, :]], axis=0)

        return data
    except Exception as e:
        print(f"Error reading CSV file {filename}: {str(e)}")
        return np.empty((0, 320))


def load_data(data_dir, data_type="leak"):

    directories = find_directories_recursive(data_dir, data_type)
    print(directories)
    data = np.empty((0, 320))

    for data_path in directories:
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            _, ext = os.path.splitext(file_path)

            if ext == ".dat":
                data_tmp = read_dat_data(file_path)
            elif ext == ".csv":
                data_tmp = read_csv_data(file_path)
            else:
                continue

            data = np.append(data, data_tmp, axis=0)

    return data


def plot_average_signals(normal_data, leak_data):

    # 각 데이터의 평균 계산
    normal_mean = np.mean(normal_data, axis=0)
    leak_mean = np.mean(leak_data, axis=0)

    # 시각화
    plt.figure(figsize=(15, 6))
    time_steps = np.arange(len(normal_mean))

    # 정상 데이터 (파란색 실선)
    plt.plot(time_steps, normal_mean, "b-", label="Normal", linewidth=2)

    # 누수 데이터 (빨간색 점선)
    plt.plot(time_steps, leak_mean, "r--", label="Leak", linewidth=2)

    plt.title("Average Signal Comparison: Normal vs Leak")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    total_data_dir = "./data"

    leak_data = load_data(total_data_dir, data_type="leak")
    normal_data = load_data(total_data_dir, data_type="normal")

    print(f"Leak data shape: {leak_data.shape}")
    print(f"Normal data shape: {normal_data.shape}")

    np.save("./normal_data.npy", normal_data)  # save
    np.save("./leak_data.npy", leak_data)  # save

    plot_average_signals(normal_data, leak_data)
