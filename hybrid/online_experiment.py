import os
import sys

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import logging
import platform
import random
import time
from multiprocessing import Process

import brainflow
import numpy as np
from beeply.notes import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from psychopy import core, event, visual  # import some libraries from PsychoPy
from speller_config import *

from utils.common import drawTextOnScreen, save_raw, getdata, save_csv
from utils.gui import CheckerBoard, get_screen_settings
from scipy import signal
import pickle

a = beeps(800)

# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

#create a window
window = visual.Window([width, height], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="pix", fullscr=True)

refresh_rate = round(window.getActualFrameRate())
print("Refresh Rate ==>", refresh_rate)

# Time conversion to frames
epoch_frames = int(EPOCH_DURATION * refresh_rate)
print("Epoch frames ==>",epoch_frames)
iti_frames = int(ITI_DURATION * refresh_rate)
iti_frames_cal = int(0.8 * refresh_rate)
cue_frames = int(CUE_DURATION * refresh_rate)   

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

hori_divider = visual.Line(window, start=HORI_DIVIDER_START, end=HORI_DIVIDER_END, lineColor='black')
ver_divider_1 = visual.Line(window, start=VER_DIVIDER_1_START, end=VER_DIVIDER_1_END, lineColor='black')
display_box = visual.Rect(window, size=DISPLAY_BOX_SIZE, pos=DISPLAY_BOX_POS, lineColor='black', lineWidth=2.5)
display_text_start = visual.TextStim(window, text=">", color=(-1., -1., -1.), pos=DISPLAY_BOX_POS)

block_break_text = "Block Break 1 Minute. Please donot move towards the end of break."
block_break_start = visual.TextStim(window, text=block_break_text, color=(-1., -1., -1.))
counter = visual.TextStim(window, text='', pos=(0, 50), color=(-1., -1., -1.))

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
    loaded_model = pickle.load(open(r"C:\Users\bci\Documents\projects\hybrid-ssvep-p300-speller\nine_flicker\TRCA_model.sav", 'rb'))
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

    loaded_model = pickle.load(open(r"C:\Users\bci\Documents\projects\hybrid-ssvep-p300-speller\nine_flicker\TRCA_model.sav", 'rb'))
    # offset = int(250 * 1.5)
    offset = 475
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
        marker = MARKERS[str(target)]


        t0 = trialClock.getTime()  # Retrieve time at start of cue presentation
        #Display the cue
        cue.pos = target_pos
        for frame in range(cue_frames):
                cue.draw()
                window.flip()

        frames = 0
        # IDEA
        # Generating an entire epoch of frames
        # The shape is (n, m, f) where 
        # n: number of sub-speller
        # m: is each character in the sub speller
        # f: is frame_idx
        timeline = gen_timeline(n=4, m=2, overlap=0.5, isShuffle=True)
        marked:bool = False
        for t_idx in range(timeline.shape[2]):
            get_keypress()
            for n_idx in range(timeline.shape[0]):
                frame = timeline[n_idx,:,t_idx]
                chars = SUBSPELLERS[str(n_idx+1)]
                for idx, char in enumerate(chars):
                    get_keypress()
                    if(frame[idx] == -1):
                        flickers[char].draw2(frame=frame[idx], amp_override=-1)
                    else:
                        if(marked == False and char == target):
                            marked = True
                        flickers[char].draw2(frame=frame[idx])
            window.flip()
        #predicting the output
        core.wait(1)
        data = board_shim.get_board_data()
        save_csv(data, str(trial)+target, RECORDING_DIR, PARTICIPANT_ID)
        data_copy = data.copy()
        raw = getdata(data_copy,BOARD_ID,n_samples = 250,dropEnable = False)
        save_raw(raw, str(trial)+target,RECORDING_DIR, PARTICIPANT_ID)
        output = get_prediction(data)
        if (output == target):
            correct_count += 1
        else:
            incorrect_count +=1
        display_text_start.text += output
        display_text_start.draw()
        window.flip()

def gen_timeline(n:int, m:int, overlap:float, isShuffle:bool=False):
    import numpy as np
    timeline = []
    for _ in range(n):
        timeline.append(gen_timeline_subspeller(m, overlap, isShuffle))
    timeline = np.vstack(timeline)
    return timeline

def gen_timeline_subspeller(m:int, overlap:float, isShuffle:bool=False):
    # overlap:float
    #   0: No 2 stimuli flicker at the same time
    # 0.5: 2 stimuli overlap by half
    #   1: 2 stimuli flicker at the same time 
    import numpy as np

    # import random
    # characters = list(range(m))
    # if(isShuffle):
    #     random.shuffle(characters)

    n = m
    d = epoch_frames
    t = int(d*(((n-1) * (1-overlap)) + 1))
    # print(f"{n=} {m=} {d=} {t=}")
    timeline = np.zeros((n, t), dtype=int)
    # print(f"{timeline.shape}")
    for i in range(n):
        start_offset = int(i * d * (1 - overlap))
        end_offset = start_offset + d
        # print(f"{i=} {start_offset=} {end_offset=}")
        # idx = characters.index(characters[i])
        timeline[i, start_offset:end_offset] = range(1,d+1)
    timeline += -1

    if(isShuffle):
        np.random.shuffle(timeline)

    timeline = np.expand_dims(timeline, axis=0)
    return timeline


def main():
    global sequence
    global trialClock
    global board_shim
    global correct_count
    global incorrect_count

    BoardShim.enable_dev_board_logger()

    #brainflow initialization 
    params = BrainFlowInputParams()
    board_shim = BoardShim(BOARD_ID, params)

    #prepare board
    try:
        board_shim.prepare_session()
    except brainflow.board_shim.BrainFlowError as e:
        print(f"Error: {e}")
        print("The end")
        time.sleep(1)
        sys.exit()
    #board start streaming
    board_shim.start_stream()

    logging.info('Begining the experiment')

    while True:

        # Starting the display
        trialClock = core.Clock()
        cal_start.draw()
        window.flip()
        core.wait(6)

        for block in range(NUM_BLOCK):
            a.hear('A_')
            drawTextOnScreen('Starting block ' + str(block + 1) ,window)
            core.wait(0.5)
            sequence = random.sample(TARGET_CHARACTERS, len(TARGET_CHARACTERS))
            for trial in range(NUM_TRIAL):
                get_keypress()
                # Drawing display box
                display_box.autoDraw = True
                display_text_start.autoDraw = True
                # Drawing the grid
                hori_divider.autoDraw = True
                ver_divider_1.autoDraw = True
                # Display target characters
                for target in targets.values():
                    target.autoDraw = True
                    # get_keypress()
                print("Sequence is", sequence)
                flicker(trial)
                # At the end of the trial, calculate real duration and amount of frames
                t1 = trialClock.getTime()  # Time at end of trial
                elapsed = t1 - t0
                print(f"Time elapsed: {elapsed}")
                print(f"Total frames: {frames}")

           
            # clearing the screen
            hori_divider.autoDraw = False
            ver_divider_1.autoDraw = False
            for target in targets.values():
                target.autoDraw = False

            countdown_timer = core.CountdownTimer(BLOCK_BREAK)
            if (block + 1) < NUM_BLOCK: 
                block_break_start.autoDraw = True
                while countdown_timer.getTime() > 0:
                    time_remaining = countdown_timer.getTime()
                    counter.text = f'Time remaining: {int(time_remaining)}'
                    counter.draw()
                    window.flip()

            block += 1
            block_break_start.autoDraw = False
            window.flip()

        drawTextOnScreen('End of experiment, Thank you',window)
        core.wait(3)
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