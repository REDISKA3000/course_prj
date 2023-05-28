import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import numpy as np
import librosa


class MimiiDataset(Dataset):
    def __init__(self, audio_dir, n_fft=1024, win_length=1024,
                 hop_length=512, power=2, n_mels=128, pad_mode='reflect',
                 sr=16000, center=True, norm=None):
        super(MimiiDataset, self).__init__()
        self.audio_dir = audio_dir
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.pad_mode = pad_mode
        self.sr = sr
        self.center = center
        self.norm = norm

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

    def spectrogrameToImage(self, specgram):
        # specgram = torchaudio.transforms.MelSpectrogram(n_fft=1024, win_length=1024,
        #                                                 hop_length=512, power=2,
        #                                                 normalized=True, n_mels=128)(waveform )
        specgram = self.make0min(specgram)
        specgram = specgram.log2()[0, :, :].numpy()

        tr2image = transforms.Compose([transforms.ToPILImage()])

        specgram = self.normalize(specgram)
        # specgram = img_as_ubyte(specgram)
        specgramImage = tr2image(specgram)
        return specgramImage

    def get_logmelspectrogram(self, waveform):
        melspec = librosa.feature.melspectrogram(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power, n_mels=self.n_mels, pad_mode=self.pad_mode,
            sr=self.sr,
            center=self.center, norm=self.norm, htk=True,
            y=waveform.numpy()
        )  # melspectrogram

        logmelspec = librosa.power_to_db(melspec)  # log-melspectrogram

        return logmelspec

    def get_melspectrogram(self, waveform):
        melspec = librosa.feature.melspectrogram(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power, n_mels=self.n_mels, pad_mode=self.pad_mode,
            sr=self.sr,
            center=self.center, norm=self.norm, htk=True,
            y=waveform.numpy()
        )  # melspectrogram

        return melspec

    def get_mfcc(self, waveform):
        mfcc = librosa.feature.mfcc(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, pad_mode=self.pad_mode, sr=self.sr,
            center=self.center, norm=self.norm, n_mfcc=40,
            y=waveform.numpy()
        )

        return mfcc

    def get_chroma_stft(self, waveform):
        stft = librosa.feature.chroma_stft(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, pad_mode=self.pad_mode, sr=self.sr,
            center=self.center, norm=self.norm, n_chroma=12,
            y=waveform.numpy()
        )

        return stft

    def get_spectral_contrast(self, waveform):
        spec_contrast = librosa.feature.spectral_contrast(
            n_fft=self.n_fft, win_length=self.win_length, center=self.center,
            hop_length=self.hop_length, pad_mode=self.pad_mode, sr=self.sr,
            y=waveform.numpy()
        )

        return spec_contrast

    def get_tonnetz(self, waveform):
        harmonic = librosa.effects.harmonic(waveform.numpy())
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sr)

        return tonnetz

    def _derive_data(self, file_list):
        tr2tensor = transforms.Compose([transforms.PILToTensor()])
        data = []
        for i in range(len(file_list)):
            y, sr = torchaudio.load(file_list[i])
            spec = self.get_melspectrogram(y)
            spec = self.spectrogrameToImage(spec)
            spec = spec.convert('RGB')
            vectors = tr2tensor(spec)

            data.append(vectors)

        return data