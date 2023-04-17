import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import numpy as np
import librosa


class MimiiDataset(Dataset):
    def __init__(self, audio_dir, n_mel=128):
        super(MimiiDataset, self).__init__()
        self.audio_dir = audio_dir
        self.n_mel = n_mel

    def get_data(self, device):
        self.train_files, self.train_labels = self._train_file_list(device)
        self.test_files, self.test_labels = self._test_file_list(device)

        self.train_data = self.derive_melspect(self.train_files)
        self.test_data = self.derive_melspect(self.test_files)

        self.valid_data = self.test_data[:int(len(self.test_data) / 2)]
        self.valid_labels = self.test_labels[:int(len(self.test_data) / 2)]

        self.test_data = self.test_data[int(len(self.test_data) / 2):]
        self.test_labels = self.test_labels[int(len(self.test_data) / 2):]

        return self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, self.test_labels

    def _train_file_list(self, device):
        query = os.path.abspath(
            f"{self.audio_dir}/{device}/train/*_normal_*.wav"
        )
        train_normal_files = sorted(glob.glob(query))
        train_normal_labels = np.zeros(len(train_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/train/*_anomaly_*.wav"
        )
        train_anomaly_files = sorted(glob.glob(query))
        train_anomaly_labels = np.zeros(len(train_anomaly_files))

        train_file_list = np.concatenate(
            (train_normal_files, train_anomaly_files), axis=0)
        train_labels = np.concatenate(
            (train_normal_labels, train_anomaly_labels), axis=0)

        return train_file_list, train_labels

    def _test_file_list(self, device):
        query = os.path.abspath(
            f"{self.audio_dir}/{device}/target_test/*_normal_*.wav"
        )
        test_trg_normal_files = sorted(glob.glob(query))
        test_trg_normal_labels = np.zeros(len(test_trg_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/target_test/*_anomaly_*.wav"
        )
        test_trg_anomaly_files = sorted(glob.glob(query))
        test_trg_anomaly_labels = np.zeros(len(test_trg_anomaly_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/source_test/*_normal_*.wav"
        )
        test_src_normal_files = sorted(glob.glob(query))
        test_src_normal_labels = np.zeros(len(test_src_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/source_test/*_anomaly_*.wav"
        )
        test_src_anomaly_files = sorted(glob.glob(query))
        test_src_anomaly_labels = np.zeros(len(test_src_anomaly_files))

        test_file_list = np.concatenate((test_trg_normal_files,
                                         test_trg_anomaly_files,
                                         test_src_normal_files,
                                         test_src_anomaly_files), axis=0)
        test_labels = np.concatenate((test_trg_normal_labels,
                                      test_trg_anomaly_labels,
                                      test_src_normal_labels,
                                      test_src_anomaly_labels), axis=0)

        return test_file_list, test_labels

    def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    def derive_melspect(self, file_list):
        data = []
        max_length = 440
        for file in file_list:
            amplitudes, sr = librosa.load(file)
            melspect = librosa.feature.melspectrogram(y=amplitudes, sr=sr,
                                                      n_mels=128, fmin=1,
                                                      fmax=8192)
            melspect = np.pad(melspect, [[0, 0], [0, max(0, max_length -
                                                         melspect.shape[1])]],
                              mode='constant')
            melspect_norm = librosa.util.normalize(melspect)
            data.append(melspect_norm)
        return data

class Dataloader:
    def __init__(self, spectrograms, targets):
        self.data = list(zip(spectrograms, targets))

    def next_batch(self, batch_size, device):
        indices = np.random.randint(len(self.data), size=batch_size)

        input = [self.data[i] for i in indices]

        source = [line[0] for line in input]
        target = [line[1] for line in input]

        return self.torch_batch(source, target, device)

    @staticmethod
    def torch_batch(source, target, device):
        return tuple(
            [
                torch.tensor(val, dtype=torch.float).to(device, non_blocking=True)
                for val in [source, target]
            ]
        )