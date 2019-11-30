from utils import *
import torch
from torch.utils.data import Dataset
import os
from natsort import natsorted
from hparams import hparams
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, path, q_levels=256, ratio_min=0, ratio_max=1,max_len = hparams.audio_max_length,isPreprocess=True):
        '''
        :param path: natural folder path which contains the specific data
        :param q_levels: quantitize level 256
        :param ratio_min: decide get how many percent data in this folder.
        :param ratio_max:decide get how many percent data in this folder.
        '''
        self.q_levels = q_levels
        file_names_a = natsorted(
            [os.path.join(path+"/A", file_name) for file_name in os.listdir(path+"/A")]
        )
        file_names_b = natsorted(
            [os.path.join(path + "/B", file_name) for file_name in os.listdir(path+"/B")]
        )
        self.file_names = {}
        self.file_names["A"] = file_names_a[
                          int(ratio_min * len(file_names_a)): int(ratio_max * len(file_names_a))
                          ]
        self.file_names["B"] = file_names_b[
                               int(ratio_min * len(file_names_b)): int(ratio_max * len(file_names_b))
                               ]
        self.max_len = max_len
        self.isPreprocessed = isPreprocess

    def __getitem__(self, index):
        if self.isPreprocessed:
            input_data_a = np.load(self.file_names["A"][index]).astype(np.float)
            input_data_b = np.load(self.file_names["B"][index]).astype(np.float)
            ori_length = input_data_a.shape[0]
            if ori_length < self.max_len:
                npi = np.zeros(self.q_levels, dtype=np.float32)
                npi = np.tile(npi, (self.max_len - ori_length, 1))
                input_data_a = np.row_stack((input_data_a, npi))
                npo = np.zeros(self.q_levels, dtype=np.float32)
                npo = np.tile(npo, (self.max_len - ori_length, 1))
                input_data_b = np.row_stack((input_data_b, npo))
            #TODO if needed, change the net to LSTM with "pack_padded_sequence"
        else:

            (seq_a, _) = librosa.core.load(self.file_names["A"][index], sr=hparams.s, mono=True) #if we load audio directly, we have double
            (seq_b, _) = librosa.core.load(self.file_names["B"][index], sr=hparams.s, mono=True)
            input_data_a = linear_quantize(torch.from_numpy(seq_a), self.q_levels)
            input_data_b = linear_quantize(torch.from_numpy(seq_b), self.q_levels)

            ori_length = len(self.file_names["A"])

        return {"A":input_data_a,"B":input_data_b,"ori_length":ori_length}


    def __len__(self):
        return len(self.file_names["A"])


# class DataLoader(DataLoaderBase):
#
#     def __init__(self, dataset, batch_size, seq_len, overlap_len,
#                  *args, **kwargs):
#         super().__init__(dataset, batch_size, *args, **kwargs)
#         self.seq_len = seq_len
#         self.overlap_len = overlap_len
#
#     def __iter__(self):
#         for batch in super().__iter__():
#             (batch_size, n_samples) = batch.size()
#
#             reset = True
#
#             for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
#                 from_index = seq_begin - self.overlap_len
#                 to_index = seq_begin + self.seq_len
#                 sequences = batch[:, from_index : to_index]
#                 input_sequences = sequences[:, : -1]
#                 target_sequences = sequences[:, self.overlap_len :].contiguous()
#
#                 yield (input_sequences, reset, target_sequences)
#
#                 reset = False
#
#     def __len__(self):
#         raise NotImplementedError()