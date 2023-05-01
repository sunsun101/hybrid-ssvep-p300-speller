import os
import platform
import sys
from turtle import fillcolor, pos

from psychopy import core, event, visual  # import some libraries from PsychoPy

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
													   
import argparse
import json
import logging
			 
								   
import multiprocessing
import pickle
import random
import threading
				
																
import time
from multiprocessing import Process

import brainflow
import numpy as np
from beeply.notes import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams

from models.cca import ECCA
from utils.common import drawTextOnScreen, getdata, save_raw, save_csv
from utils.gui import CheckerBoard, get_screen_settings
from speller_config import *
from realtime_plot import Graph
from models.trca import TRCA
from scipy import signal

a = beeps(800)
# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

#create a window
window = visual.Window([width, height], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="pix", fullscr=True)

# window = visual.Window([1920, 1080], screen=1, color=[1,1,1],blendMode='avg', monitor="hybrid-speller-monitor", useFBO=True, units="deg", fullscr=True)
# mywin = visual.Window(SCREEN_SIZE, color="black",monitor="Experiment Monitor" , units='norm',screen=SCREEN_NUM,fullscr=True) 
refresh_rate = round(window.getActualFrameRate())
print("Refresh Rate ==>", refresh_rate)

# Time conversion to frames
epoch_frames = int(EPOCH_DURATION * refresh_rate)

print("Epoch frames ==>",epoch_frames)
cue_frames = int(CUE_DURATION * refresh_rate)   
print("Cue frames ==>",cue_frames)


#Presentation content

cue = visual.Rect(window, width=WIDTH, height=HEIGHT, pos=[0, 0], lineWidth=3, lineColor='red')

calib_text_start = "Starting callibration phase.Please avoid moving or blinking.\n\
You may blink when shifting your gaze.Focus your target on the characters presented with red cue."

calib_text_end = "Calibration phase completed"
cal_start = visual.TextStim(window, text=calib_text_start, color=(-1., -1., -1.))
cal_end = visual.TextStim(window, text=calib_text_end, color=(-1., -1., -1.))

																										   
															 

targets = {f"{target}": visual.TextStim(win=window, text=target, pos=pos, color=(-1., -1., -1.), height=HEIGHT_OF_TARGET)
            for pos, target in zip(POSITIONS, TARGET_CHARACTERS)}

wave_type = "sin"

flickers = {f"{target}": CheckerBoard(window=window, size=SIZE, frequency=f, phase=phase, amplitude=AMPLITUDE, 
                                    wave_type=wave_type, duration=EPOCH_DURATION, fps=refresh_rate,
                                    base_pos=pos, height=HEIGHT, width=WIDTH)
            for f, pos, phase, target in zip(FREQS, POSITIONS, PHASES, TARGET_CHARACTERS)}

# for _,flicker in flickers.items():
#     flicker.wave_func += -1 * 0.2 


block_break_text = "Block Break 1 Minutes"
block_break_start = visual.TextStim(window, text=block_break_text, color=(-1., -1., -1.))
display_text_start = visual.TextStim(window, text=">", color=(-1., -1., -1.), pos=DISPLAY_BOX_POS)
display_box = visual.Rect(window, size=DISPLAY_BOX_SIZE, pos=DISPLAY_BOX_POS, lineColor='black', lineWidth=2.5)

def get_keypress():
    keys = event.getKeys()
    if keys and keys[0] == 'escape':
        window.close()
        core.quit()
    else: 
        return None


def eegMarking(board,marker):
    print("Inserting marker", marker)
    board.insert_marker(marker)
    time.sleep(0.1)

def get_predicted_result(data):
    list_freqs = FREQS
    list_phases = PHASES
    fs = 250
    num_harms = 5
    num_fbs = 5
    loaded_model = pickle.load(open(r"C:\Users\bci\Documents\projects\hybrid-ssvep-p300-speller\three_flicker\TRCA_model.sav", 'rb'))
    result = loaded_model.predict(data)
    print("Here is the result", list(filter(lambda x: MARKERS[x] == result[0], MARKERS))[0])
    # result = fbcca_realtime(data, list_freqs, list_phases, fs, num_harms, num_fbs)
    # print("Target Character found", TARGET_CHARACTERS[result])
    # return TARGET_CHARACTERS[result]
    
    return list(filter(lambda x: MARKERS[x] == result[0], MARKERS))[0]

def get_prediction(data):
    marker_channel = BoardShim.get_marker_channel(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    data[eeg_channels] = data[eeg_channels] / 1e6
    data = data[eeg_channels + [marker_channel]]

    _CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    data = data[:8,:1740]
    # order = 1
    # l_freq = 4
    # sos = signal.butter(order, l_freq, 'highpass', analog=False, fs=250, output='sos')
    # notch_freq = 50
    # quality = 1
    # b,a = signal.iirnotch(notch_freq, quality, fs=250)
    # for i in range(8):
    #     data[i] = signal.lfilter(b, a, data[i])
    #     data[i] = signal.sosfilt(sos, data[i])
    
    b,a = signal.iirfilter(10, Wn=[7, 90],  btype='band', analog=False, fs=250,  ftype='butter')
    data = signal.filtfilt(b,a,data,axis=1)
    X = np.expand_dims(data[:],axis=0)
    print("Shape of data", X.shape)

    loaded_model = pickle.load(open(r"C:\Users\bci\Documents\projects\hybrid-ssvep-p300-speller\three_flicker\TRCA_model.sav", 'rb'))
    offset = 476
    pred = loaded_model.predict(X[:,:,offset:offset + 1000])

    return list(filter(lambda x: MARKERS[x] == pred, MARKERS))[0]

def flicker(trial):
    
    global frames
    global t0
    global correct_count
    global incorrect_count
    # For the flickering
   
    for target in sequence:

        get_keypress()
        target_flicker = flickers[str(target)]
        target_pos = (target_flicker.base_x, target_flicker.base_y)
        # print(target_flicker.base_x)

        t0 = trialClock.getTime()  # Retrieve time at start of cue presentation
        
        #Display the cue
        cue.pos = target_pos
        for frame in range(cue_frames):
            cue.draw()
            window.flip()

        board_shim.get_board_data()
        core.wait(1)
        frames = 0
        for frame, j in enumerate(range(epoch_frames)):
            get_keypress()
            for flicker in flickers.values():
                flicker.draw2(frame = frame)
            # target_flicker.draw2(frame = frame)
            frames += 1
            window.flip() 
        # predicting the output
        core.wait(1)
        data = board_shim.get_board_data()
        save_csv(data, str(trial)+target, RECORDING_DIR, PARTICIPANT_ID)
        data_copy = data.copy()
        # print("Shape of the data is ==>", data.shape)
        raw = getdata(data_copy,BOARD_ID,n_samples = 250,dropEnable = False)
        # raw.plot_psd()
        # output = get_predicted_result(raw.get_data()[:8,250:1500])
        save_raw(raw, str(trial)+target,RECORDING_DIR, PARTICIPANT_ID)
        output = get_prediction(data)
        if (output == target):
            correct_count += 1
        else:
            incorrect_count +=1
        display_text_start.text += output
        display_text_start.draw()
        window.flip()

        


def main():
    global sequence
    global trialClock
    global board_shim
    global correct_count
    global incorrect_count

    BoardShim.enable_dev_board_logger()

    #brainflow initialization 
    params = BrainFlowInputParams()
    # params.serial_port = serial_port
    board_shim = BoardShim(BOARD_ID, params)

    #prepare board
    try:
        board_shim.prepare_session()
    except brainflow.board_shim.BrainFlowError as e:
        print(f"Error: {e}")
        print("The end")
        time.sleep(1)
        sys.exit()
    
    logging.info('Begining the experiment')

    while True:
        correct_count = 0
        incorrect_count = 0
        # Starting the display
        trialClock = core.Clock()
        cal_start.draw()
        window.flip()
        core.wait(3)


        a.hear('A_')
        drawTextOnScreen("Please donot move now",window)
					
        sequence = random.sample(TARGET_CHARACTERS, len(TARGET_CHARACTERS))
        
        #board start streaming
        board_shim.start_stream()
        core.wait(10)
        # Graph(board_shim)

        for trial in range(NUM_TRIAL):
            get_keypress()
            # Drawing display box
            display_box.autoDraw = True
            display_text_start.autoDraw = True
            window.flip()

            # Drawing the grid
            # Display target characters
            for target in targets.values():
                target.autoDraw = True
                # get_keypress()
            flicker(trial)

        # At the end of the trial, calculate real duration and amount of frames
        t1 = trialClock.getTime()  # Time at end of trial
        elapsed = t1 - t0
        print(f"Time elapsed: {elapsed}")
        print(f"Total frames: {frames}")
        
        acc = correct_count/(correct_count + incorrect_count)
        print("correct count", correct_count)
        print("incorrect count", incorrect_count)
        print("Accuracy ==>", acc)

        display_box.autoDraw = False
        display_text_start.autoDraw = False
        window.flip()
        for target in targets.values():
            target.autoDraw = False
     
        drawTextOnScreen('End of experiment, Thank you',window)
        core.wait(10)
        break


    if board_shim.is_prepared():
        logging.info('Releasing session')
        # stop board to stream
        board_shim.stop_stream()
        board_shim.release_session()

    #cleanup
    window.close()
    core.quit()


if __name__ == "__main__":
    main()
