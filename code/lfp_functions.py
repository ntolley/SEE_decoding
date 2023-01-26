import numpy as np
import neo
import scipy.io as sio
import pandas as pd
#from externals.SpectralEvents import spectralevents_functions as tse

def load_ns6_analog(fpath, downsample_rate, from_ns6=True, save=False, channel_step=1):

    # Load LFP data directly from NS5
    if from_ns6:
        stream_index = 1
        ns6 = neo.BlackrockIO(fpath + '.ns6')

        num_channels = ns6.signal_channels_count(stream_index=stream_index)
        lfp_channels = list(range(num_channels))

        # Loop over each channel to avoid blowing up RAM, really slow...
        lfp_data_list = list()
        for idx_start in range(0, num_channels, channel_step):
            if idx_start + channel_step < num_channels:
                channel_indexes = tuple(lfp_channels[idx_start:idx_start+channel_step])
            else:
                channel_indexes = tuple(lfp_channels[idx_start:])   

            print(channel_indexes, end=' ')
            channel_data = ns6.get_analogsignal_chunk(
                stream_index=stream_index, channel_indexes=[channel_indexes]).squeeze().transpose()
                
            channel_data = channel_data[:, ::downsample_rate]
            lfp_data_list.append(channel_data)

            lfp_data = np.concatenate(lfp_data_list)
            tstart, tstop = ns6._seg_t_starts, ns6._seg_t_stops
            lfp_times = np.linspace(tstart, tstop, lfp_data.shape[1]).squeeze()
                
            if save:
                np.save(fpath + f'_lfp_channels_{downsample_rate}x_downsample.npy', lfp_data)
                np.save(fpath + f'_lfp_times_{downsample_rate}x_downsample.npy', lfp_times)

    else:
        lfp_data = np.load(fpath+ f'_lfp_channels_{downsample_rate}x_downsample.npy')
        lfp_times = np.load(fpath+ f'_lfp_times_{downsample_rate}x_downsample.npy')

    return lfp_data, lfp_times