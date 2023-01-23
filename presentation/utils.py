from brainflow import BoardShim
import mne
import logging
from mne.datasets import eegbci
import os
from pathlib import Path
import argparse
import json
from psychopy import visual

#Load config
path = os.getcwd() + '\presentation'
parser = argparse.ArgumentParser(description='Config file name')
parser.add_argument('-f', '--file', metavar='ConfigFile', type=str,
                    default='speller_config.json', help="Name of the config file for freq "
                                    "and amplitude. Default: %(default)s.")
args = parser.parse_args()
config_path = os.path.join(path, args.file)

with open(config_path, 'r') as config_file:
    params = json.load(config_file)


participant_id = params["participant_id"]
type_of_file = params["type_of_file"]
csv_dir = params["csv_dir"]

def create_session_folder(subj,dir):
    base_path = os.getcwd() + "\\"
    dir = base_path + dir
    folder_name = f'{subj}'
    print(folder_name)
    if os.path.isdir(os.path.join(dir, folder_name)):
        folder_path = os.path.join(dir, folder_name)
    else:
        folder_path = os.path.join(dir, folder_name)
        os.makedirs(folder_path)
    return folder_path

def getdata(data,board,clear_buffer=False,n_samples=None,dropEnable = False):
    """
        Get data that has been recorded to the board. (and clear the buffer by default)
        if n_samples is not passed all of the data is returned.
    """
    # Creating MNE objects from brainflow data arrays
    # the only relevant channels are eeg channels + marker channel
    # get row index which holds markers
    print("INSIDE GET DATA")
    print(data.shape)
    marker_channel = BoardShim.get_marker_channel(board)
    
    #row which hold eeg data
    eeg_channels = BoardShim.get_eeg_channels(board)
    #print(f'Before {data[eeg_channels]}')
    data[eeg_channels] = data[eeg_channels] / 1e6
    #print(f'After {data[eeg_channels]}')
    #eeg row + marker row (8 + 1)
    data = data[eeg_channels + [marker_channel]]
    
    #string of all channel name ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    ch_names = BoardShim.get_eeg_names(board)
    ch_types = (['eeg'] * len(eeg_channels)) + ['stim']
    ch_names = ch_names + ["Stim Markers"]
    print(ch_names)
    #sample rate
    sfreq = BoardShim.get_sampling_rate(board)
    
    #Create Raw data from MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    print(raw)
    logging.info(f"{str(raw)}")
    raw_data = raw.copy()
    eegbci.standardize(raw_data)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_data.set_montage(montage)
    raw_data=raw_data.notch_filter([50,75,100])
    # raw_data=raw_data.filter(8,14, method='fir', verbose=20)
    #2 electrode
    
    if dropEnable == True:
        raw_data.pick_channels(['C3','C4','STIM MARKERS']) 
        #raw_data = raw_data.drop_channels(['Fp1', 'Fp2', 'P7', 'P8', 'O1', 'O2'])
        #raw_data = raw_data.drop_channels(['Fz'])

    # print(raw_data.info['ch_names'])

    return raw_data


def save_raw(raw, name,dir):
    print("RAW is here")
    print(raw)
    folder_path = create_session_folder(participant_id,dir)
    raw.save(os.path.join(folder_path, f'{name}{type_of_file}'), overwrite = True)
    return os.path.basename(folder_path)

def save_raw_to_dataframe(raw,name):
    epoch_dataframe = raw.copy().to_data_frame()
    csv_folder = create_session_folder(participant_id,csv_dir)
    csv_name = f'{name}.csv'
    epoch_dataframe.to_csv(os.path.join(csv_folder,csv_name),encoding='utf-8')


def drawTextOnScreen(message,window) :
    message = visual.TextStim(window, text=message, color=(-1., -1., -1.))
    message.draw() # draw on screen
    window.flip()   # refresh to show what we have draw
