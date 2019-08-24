import numpy as np
import os
import pickle
import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.martemev_utils import normalize


event_step = 30720
ada_step = 12
collection_step = 960
readout_step = 800
time_len = 4492

def load_files(path_clear='/data/martemev/Data/output_data/',
               path_noise='/data/martemev/Data/input_noised_data/'):
    for index in range(0, 100):
        clear_data = np.load(os.path.join(path_clear, 'onlysignal_evt{}.npy'.format(index+1)))[:, 2:]
        noised_data = np.load(os.path.join(path_noise, 'event_{}_noised.npy'.format(index)))[:, 3:]
        yield torch.Tensor(clear_data), torch.Tensor(noised_data)
        
        
def load_file(index, path_clear='/data/martemev/Data/output_data/',
               path_noise='/data/martemev/Data/input_noised_data/'):
    clear_data = np.load(os.path.join(path_clear, 'onlysignal_evt{}.npy'.format(index+1)))[:, 2:]
    noised_data = np.load(os.path.join(path_noise, 'event_{}_noised.npy'.format(index)))[:, 3:]
    return torch.Tensor(clear_data), torch.Tensor(noised_data)


def get_planes(clear_file, noised_file):
    """
    Returning only first collection plane
    """
    signal_panes = [1, 2, 5, 6, 9, 10]
    for index in signal_panes:
        clear_plane = clear_file[index*(readout_step*2 + collection_step):
                                  (index+1)*(readout_step*2 + collection_step)]
        noised_plane = noised_file[index*(readout_step*2 + collection_step):
                                            (index+1)*(readout_step*2 + collection_step)]
        if clear_plane[2*readout_step:].max() == 0:
            continue
        
        yield normalize(clear_plane[2*readout_step:]), \
            normalize(noised_plane[2*readout_step:])

#         yield normalize(clear_plane[2*readout_step:][:collection_step - collection_step%33, 
#                                                      :time_len - time_len%33]), \
#             normalize(noised_plane[2*readout_step:][:collection_step - collection_step%33, 
#                                                      :time_len - time_len%33])

        
def get_crop(clear_plane, noised_plane, total_crops=1000, crop_shape=(33, 33), num_trials=5):
    probs = torch.clone(clear_plane)
    probs[probs == 0] += probs.mean()
    distr = torch.distributions.binomial.Binomial(total_count=num_trials, probs=probs)
    x, y = clear_plane.shape
    c_x, c_y = crop_shape[0]//2, crop_shape[1]//2

    for i in tqdm.tqdm_notebook(range(total_crops), desc='crops', leave=False):
        samples = torch.nonzero(distr.sample())
        while len(samples) < 1:
            samples = torch.nonzero(distr.sample())
        sample = np.random.choice(len(samples))
        sample = samples[sample]

        sample = (min(max(int(sample[0]), c_x), x-c_x), min(max(int(sample[1]), c_y), y-c_y))
        clear_crop = clear_plane[sample[0]-c_x:sample[0]+c_x, sample[1]-c_y:sample[1]+c_y]
        noised_crop = noised_plane[sample[0]-c_x:sample[0]+c_x, sample[1]-c_y:sample[1]+c_y]
        yield clear_crop.unsqueeze(0), noised_crop.unsqueeze(0)
