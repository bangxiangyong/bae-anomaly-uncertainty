import numpy as np
from scipy.fft import fft
import pandas as pd
from scipy.signal import resample
from scipy.interpolate import interp1d


def apply_along_sensor(sensor_data, func1d, seq_axis=2, *args, **kwargs):
    """
    Apply a func1d on every independent sequence with data of shape (batch,sensor,seq)
    """
    if seq_axis == 2 or seq_axis == -1:
        sensor_axis = 1
    else:
        sensor_axis = 2

    transformed_x = []
    for sensor_i in range(sensor_data.shape[sensor_axis]):
        transform_batch = np.apply_along_axis(
            func1d=func1d,
            arr=np.take(sensor_data, sensor_i, sensor_axis),
            axis=1,
            *args,
            **kwargs
        )
        transformed_x.append(transform_batch)

    transformed_x = np.array(transformed_x)
    transformed_x = np.moveaxis(transformed_x, 0, sensor_axis)

    return transformed_x


class Resample_Sensor:
    """
    Resample by grouping into equal-sized bins and calculate mean of each bin
    """

    def downsample_data(self, data_series, n=10):
        temp_df = pd.DataFrame(data_series)
        resampled_data = (
            temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
        )
        return resampled_data

    def upsample_data(self, data_series, n=10):
        x_ori = np.linspace(0, 1, len(data_series))
        x_upsampled = np.linspace(0, 1, len(data_series) * n)
        inter_f = interp1d(x_ori, data_series)
        resampled_data = inter_f(x_upsampled)
        return resampled_data

    def transform(self, x, seq_axis=2, n=10, mode="down"):
        # set mode
        if mode == "down":
            func1d = self.downsample_data
        elif mode == "up":
            func1d = self.upsample_data
        else:
            raise NotImplemented("Resample mode can be either up or down only.")

        return apply_along_sensor(
            sensor_data=x,
            func1d=func1d,
            seq_axis=seq_axis,
            n=n,
        )


class Resample_Sensor_Fourier:
    """
    Resample by grouping into equal-sized bins and calculate mean of each bin
    """

    def resample_data(self, data_series, n=10):
        temp_df = pd.DataFrame(data_series)
        resampled_data = (
            temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
        )
        return resampled_data

    def transform(self, x, seq_axis=2, seq_len=600):
        # specify seq and sens axis
        if seq_axis == 2:
            sens_axis = 1
            x_resampled = np.zeros((x.shape[0], x.shape[sens_axis], seq_len))
        elif seq_axis == 1:
            sens_axis = 2
            x_resampled = np.zeros((x.shape[0], seq_len, x.shape[sens_axis]))
        else:
            raise ValueError("Seq axis can be either 1 or 2.")

        # ignore if already same size as seq_len ?
        if seq_len == x.shape[seq_axis]:
            return np.copy(x)

        # apply transformation over sensor axis
        for sensor_id in range(x.shape[sens_axis]):
            if sens_axis == 1:
                x_resampled[:, sensor_id] = resample(x[:, sensor_id], seq_len, axis=1)
            else:
                x_resampled[:, :, sensor_id] = resample(
                    x[:, :, sensor_id], seq_len, axis=1
                )
        return x_resampled


class FFT_Sensor:
    def apply_fft(self, seq_1d):
        N = len(seq_1d)
        trace_fft = 2.0 / N * np.abs(fft(seq_1d)[: N // 2])
        trace_fft = trace_fft[1:]
        return trace_fft

    def transform(self, x, seq_axis=2):
        return apply_along_sensor(
            sensor_data=x, func1d=self.apply_fft, seq_axis=seq_axis
        )


class MinMaxSensor:
    def __init__(self, num_sensors=11, axis=1, clip=True):
        self.num_sensors = num_sensors
        self.max_sensors = []
        self.min_sensors = []
        self.axis = axis
        self.clip = clip

    def fit(self, x):
        self.max_sensors = []
        self.min_sensors = []
        for sensor in range(self.num_sensors):
            slice_ = np.take(x, sensor, axis=self.axis)
            self.max_sensors.append(np.max(slice_))
            self.min_sensors.append(np.min(slice_))

    def transform(self, x):
        trans = np.copy(x)
        for sensor in range(self.num_sensors):
            slice_ = np.take(trans, sensor, axis=self.axis)
            slice_ = (slice_ - self.min_sensors[sensor]) / (
                self.max_sensors[sensor] - self.min_sensors[sensor]
            )
            if self.clip:
                slice_ = np.clip(slice_, 0, 1)
            if self.axis == 1:
                trans[:, sensor] = slice_
            elif self.axis == 2:
                trans[:, :, sensor] = slice_
            else:
                raise NotImplemented("Axis for sensor must be 1 or 2")
        return trans

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
