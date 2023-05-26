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
        train_anomaly_labels = np.ones(len(train_anomaly_files))

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
        test_trg_anomaly_labels = np.ones(len(test_trg_anomaly_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/source_test/*_normal_*.wav"
        )
        test_src_normal_files = sorted(glob.glob(query))
        test_src_normal_labels = np.zeros(len(test_src_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/source_test/*_anomaly_*.wav"
        )
        test_src_anomaly_files = sorted(glob.glob(query))
        test_src_anomaly_labels = np.ones(len(test_src_anomaly_files))

        test_file_list = np.concatenate((test_trg_normal_files,
                                         test_trg_anomaly_files,
                                         test_src_normal_files,
                                         test_src_anomaly_files), axis=0)
        test_labels = np.concatenate((test_trg_normal_labels,
                                      test_trg_anomaly_labels,
                                      test_src_normal_labels,
                                      test_src_anomaly_labels), axis=0)

        return test_file_list, test_labels

    def normalize(self, tensor):
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean / np.absolute(tensor_minusmean).max()

    def make0min(self, tensornd):
        tensor = tensornd.numpy()
        res = np.where(tensor == 0, 1E-19, tensor)
        return torch.from_numpy(res)

    def spectrogrameToImage(self, waveform):
        specgram = torchaudio.transforms.MelSpectrogram(n_fft=1024,
                                                        win_length=1024,
                                                        hop_length=512,
                                                        power=2,
                                                        normalized=True,
                                                        n_mels=128)(waveform)
        specgram = self.make0min(specgram)
        specgram = specgram.log2()[0, :, :].numpy()

        tr2image = transforms.Compose([transforms.ToPILImage()])

        specgram = self.normalize(specgram)
        # specgram = img_as_ubyte(specgram)
        specgramImage = tr2image(specgram)
        return specgramImage

    def _derive_data(self, file_list):
        tr2tensor = transforms.Compose([transforms.PILToTensor()])
        data = []
        for i in range(len(file_list)):
            y, sr = torchaudio.load(file_list[i])
            spec = self.spectrogrameToImage(y)
            spec = spec.convert('RGB')
            vectors = tr2tensor(spec)

            data.append(vectors)

        return data