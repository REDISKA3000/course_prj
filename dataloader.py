import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import numpy as np
import librosa


class MimiiDataset(Dataset):
    def __init__(self, audio_dir, n_mel=128):
        super(Dataset, self).__init__()
        self.audio_dir = audio_dir
        self.n_mel = n_mel

    def get_data(self, device):
        self.train_files, self.train_labels = self._train_file_list(device)
        self.test_files, self.test_labels = self._test_file_list(device)

        self.train_data = self._derive_data(self.train_files.copy())
        self.test_data = self._derive_data(self.test_files.copy())

        return self.train_data, self.test_data, self.train_labels, self.test_labels

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

    @staticmethod
    def get_melspect(self, file):
        y, sr = librosa.load(file, sr=16000, mono=True)

        features = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            n_mels=self.n_mel,
            win_length=1024,
            hop_length=512,
            power=2.0,
        )
        # features = features.reshape(-1, features.shape[0], features.shape[1])

        return features

    def _derive_data(self, file_list):
        data = []
        for i in range(len(file_list)):
            vectors = self.get_melspect(self, file_list[i])
            n_objs = vectors.shape[0]

            data.append(vectors)

        return data