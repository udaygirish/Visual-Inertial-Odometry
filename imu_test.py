# Imu test
import numpy as np
from utils.imu_model import * 

def generate_imu_data_test():
    t_array = []
    x_array = []
    y_array = []
    z_array = []
    roll_array = []
    pitch_array = []
    yaw_array = []


    initial_dist = 0
    dt = 0
    initial_pos = 100 
    accel = 2
    imu_rate = 100 
    v_rate = 30 
    angle = 0
    dt = 0
    initial_angle = 359
    angular_vel = 2 
    imu_rate = 100
    v_rate = 30

    for i in range(102):
        t_array.append(dt)
        dist = initial_pos + accel * dt * dt / 2
        x_array.append(dist)
        y_array.append(dist)
        z_array.append(dist)

    for i in range(101):
        angle = initial_angle + angular_vel * dt
        roll_array.append(angle)
        pitch_array.append(angle)
        yaw_array.append(angle)
        dt += 1/imu_rate

    # Output - Real Data - Pure linear model N*3
    accel_data = cal_linear_acc(x_array, y_array, z_array, imu_rate)

    # Output - Real Data - Pure angular model N*3   
    gyro_data = cal_angular_vel(roll_array, pitch_array, yaw_array, imu_rate)


    # Synthetic data- Real - Adding Error and Vibration according to model 

    fs = 100  
    num_samples = 1000
    acc_err = accel_high_accuracy 
    gyro_err = gyro_high_accuracy

    accel_env = '[0.03 0.001 0.01]-random'
    gyro_env = '[6 5 4]d-0.5Hz-sinusoidal'

    vib_accel_def = vib_from_env(accel_env, fs)

    vib_gyro_def = vib_from_env(gyro_env, fs)

    # Output - Synthetic Data - Adding Error and Vibration according to model
    real_acc = acc_gen(fs, accel_data, acc_err, vib_accel_def)
    real_gyro = gyro_gen(fs, gyro_data, gyro_err, vib_gyro_def)

    print("Accel Data Shape: ", accel_data.shape)
    print("Gyro Data Shape: ", gyro_data.shape)
    print("Real Accel Data Shape: ", real_acc.shape)
    print("Real Gyro Data Shape: ", real_gyro.shape)


    # Find the overall error between the real and synthetic data
    acc_error = np.mean(np.abs(real_acc - accel_data))

    gyro_error = np.mean(np.abs(real_gyro - gyro_data))

    print("Accel Error: ", acc_error)
    print("Gyro Error: ", gyro_error)

if __name__ == "__main__":
    generate_imu_data_test()