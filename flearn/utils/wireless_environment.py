import copy

import numpy
import numpy as np
import math
import random


# device_num
def calculate_receive_powers(device_num, transmit_power_pi, path_losses, shadow_fading_standard_deviation):

    shadow_fading_arrays = []
    for _ in range(1000):
        shadow_fading = np.random.normal(0, shadow_fading_standard_deviation, device_num)
        shadow_fading_arrays.append(shadow_fading)

    # 计算每列的均值
    shadow_fadings = np.mean(shadow_fading_arrays, axis=0)
    print(f"Shadow_fadings(dB):",shadow_fadings)

    # 计算每台设备的接收功率
    receive_powers = transmit_power_pi - path_losses - shadow_fadings

    return receive_powers

def calculate_wireless_environment(device_num, radius, B, tx, rx):
    print("Radius(Km):", radius)
    print("Device_num:", device_num)
    device_distances = np.random.uniform(low=0.1, high=radius, size=device_num)
    print(f"Device_distances(km):", device_distances)
    path_losses = np.array([128.1 + 37.6 * np.log10(d) for d in device_distances])
    print(f"Path_losses(dB):", path_losses)
    shadow_fading_standard_deviation = 8
    # transmit_power_pi = 10
    upload_power_pi = 10 * np.log10(tx) - 30  # dB
    download_power_pi = 10 * np.log10(rx) - 30
    N0 = -174

    upload_power_pi_w = tx / 1000
    download_power_pi_w = rx / 1000
    print("Upload power(w):", upload_power_pi_w)
    print("Download power(w):", download_power_pi_w)

    print("Upload_power(dB):", upload_power_pi)
    print("Download_power(dB):", download_power_pi)

    receive_powers_upload = calculate_receive_powers(device_num, upload_power_pi, path_losses, shadow_fading_standard_deviation)
    print(f"Upload_powers:", receive_powers_upload)

    up_ri_list = np.array(
        [B / device_num * np.log2(1 + (10 ** ((receive_power + 30 - 30) / 10)) / ((B / device_num) * (10 ** ((N0 - 30) / 10)))) for
         receive_power in receive_powers_upload])
    print(f"Upload_rates:", up_ri_list)

    receive_powers_download = calculate_receive_powers(device_num, download_power_pi, path_losses, shadow_fading_standard_deviation)
    print(f"Download_powers:", receive_powers_download)

    down_ri_list = np.array(
        [B / device_num * np.log2(1 + (10 ** ((receive_power + 30 - 30) / 10)) / ((B / device_num) * (10 ** ((N0 - 30) / 10)))) for
         receive_power in receive_powers_download])
    print(f"Download_rates:", down_ri_list)

    return [upload_power_pi_w] * device_num, [download_power_pi_w] * device_num, up_ri_list, down_ri_list

def calculate_device_powers(device_num, total_flops):
    # 计算GPU频率
    min_frequency = 1e9  # 1 GHz
    max_frequency = 3e9  # 3 GHz

    cpu_frequencies = np.random.uniform(min_frequency, max_frequency, device_num)
    print(f"Cpu_frequencies", cpu_frequencies)
    voltages = np.random.uniform(0.5, 1.2, device_num)
    print(f"voltages", voltages)
    cpu_powers = voltages ** 2 * cpu_frequencies / 10e9
    print(f"cpu_powers", cpu_powers)

    C = 0.1
    print(f"Total_flops ", total_flops)
    energy_per_flops = C / cpu_frequencies
    print(f"Energy_per_flops", energy_per_flops)
    train_times = C * total_flops / cpu_frequencies
    print(f"Train_times:", train_times)
    print(max(train_times))

    return cpu_powers, energy_per_flops, train_times

