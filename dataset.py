from utils import *
import torch
from torch.utils.data import dataset,DataLoader as DataLoaderBase
import os
from natsort import natsorted
from hparams import hparams

class AudioDataset(dataset):
    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        '''
        :param path: natural folder path which contains the specific data
        :param overlap_len:
        :param q_levels: quantitize level 256
        :param ratio_min: decide get how many percent data in this folder.
        :param ratio_max:decide get how many percent data in this folder.
        '''
        super().init()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        file_names = natsorted(
            [os.path.join(path, file_name) for file_name in os.listdir(path)]
        )
        self.file_names = file_names[
                          int(ratio_min * len(file_names)): int(ratio_max * len(file_names))
                          ]

    def __getitem__(self, index):
        (seq, _) = librosa.core.load(self.file_names[index], sr=hparams.s, mono=True)
        return torch.cat([
            torch.LongTensor(self.overlap_len).fill_(q_zero(self.q_levels)),linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

    def __len__(self):
        return len(self.file_names)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len :].contiguous()

                yield (input_sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()