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

    def get_files(self):
        return self.train_files, self.test_files

    def get_data(self, device, id):

        self.train_files, self.train_labels = self._train_file_list(device, id)
        self.test_files, self.test_labels = self._test_file_list(device, id)

        self.train_data = self.get_audios(self.train_files)
        self.test_data = self.get_audios(self.test_files)

        return self.train_data, self.test_data, self.train_labels, self.test_labels

    def _train_file_list(self, device, id):
        query = os.path.abspath(
            f"{self.audio_dir}/{device}/train/normal_id_0{id}*.wav"
        )
        train_normal_files = sorted(glob.glob(query))
        train_normal_labels = np.zeros(len(train_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/train/anomaly_id_0{id}*.wav"
        )
        train_anomaly_files = sorted(glob.glob(query))
        train_anomaly_labels = np.ones(len(train_anomaly_files))

        train_file_list = np.concatenate(
            (train_normal_files, train_anomaly_files), axis=0)
        train_labels = np.concatenate(
            (train_normal_labels, train_anomaly_labels), axis=0)

        return train_file_list, train_labels

    def _test_file_list(self, device, id):
        query = os.path.abspath(
            f"{self.audio_dir}/{device}/test/normal_id_0{id}*.wav"
        )
        test_normal_files = sorted(glob.glob(query))
        test_normal_labels = np.zeros(len(test_normal_files))

        query = os.path.abspath(
            f"{self.audio_dir}/{device}/test/anomaly_id_0{id}*.wav"
        )
        test_anomaly_files = sorted(glob.glob(query))
        test_anomaly_labels = np.ones(len(test_anomaly_files))

        test_file_list = np.concatenate((test_normal_files,
                                         test_anomaly_files), axis=0)
        test_labels = np.concatenate((test_normal_labels,
                                      test_anomaly_labels), axis=0)

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
        )

        logmelspec = librosa.power_to_db(melspec)

        return logmelspec

    def get_melspectrogram(self, waveform):
        melspec = librosa.feature.melspectrogram(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power, n_mels=self.n_mels, pad_mode=self.pad_mode,
            sr=self.sr,
            center=self.center, norm=self.norm, htk=True,
            y=waveform.numpy()
        )

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

    def get_audios(self, file_list):
        data = []
        for i in range(len(file_list)):
            y, sr = torchaudio.load(file_list[i])
            data.append(y)

        return data

    def _derive_data(self, file_list):
        train_data = []
        test_data = []
        train_mode = True
        for file_list in [self.train_files, self.test_files]:
            tr2tensor = transforms.Compose([transforms.PILToTensor()])
            data = []
            for j in range(len(file_list)):
                y, sr = torchaudio.load(file_list[j])
                spec = self.get_melspectrogram(y)
                spec = self.spectrogrameToImage(spec)
                spec = spec.convert('RGB')
                vectors = tr2tensor(spec)
                if train_mode:
                    train_data.append(vectors)
                else:
                    test_data.append(vectors)

            train_mode = False

        return data