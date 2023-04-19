from turtle import fillcolor, pos
from psychopy import visual, core, event #import some libraries from PsychoPy
import platform
import os
import sys
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
from utils.gui import get_screen_settings, CheckerBoard
import argparse
import json
import numpy as np
import random
from multiprocessing import Process
import multiprocessing
import threading
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import time
import logging
from utils.common import getdata_offline, save_raw, drawTextOnScreen, save_csv
from beeply.notes import *
from speller_config import *

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


#Presentation content

cue = visual.Rect(window, width=WIDTH, height=HEIGHT, pos=[0, 0], lineWidth=3, lineColor='red')

calib_text_start = "Starting callibration phase.Please avoid moving or blinking.\n\
You may blink when shifting your gaze.Focus your target on the characters presented with red cue."

calib_text_end = "Calibration phase completed"
cal_start = visual.TextStim(window, text=calib_text_start, color=(-1., -1., -1.))
cal_end = visual.TextStim(window, text=calib_text_end, color=(-1., -1., -1.))

targets = {f"{target}": visual.TextStim(win=window, text=target, pos=pos, color=(-1., -1., -1.), height=35)
        for pos, target in zip(POSITIONS, TARGET_CHARACTERS)}


wave_type = "sin"

flickers = {f"{target}": CheckerBoard(window=window, size=SIZE, frequency=f, phase=phase, amplitude=AMPLITUDE, 
                                    wave_type=wave_type, duration=EPOCH_DURATION, fps=refresh_rate,
                                    base_pos=pos)
            for f, pos, phase, target in zip(FREQS, POSITIONS, PHASES, TARGET_CHARACTERS)}


block_break_text = "Block Break 1 Minutes"
block_break_start = visual.TextStim(window, text=block_break_text, color=(-1., -1., -1.))

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

def flicker(board):
    print("POSITIONS", POSITIONS)
    global frames
    global t0
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
        #flicker random sequence of each speller parallely
        # runInParallel(flicker_subspeller(randomized_subspeller[1]), flicker_subspeller(randomized_subspeller[2]), flicker_subspeller(randomized_subspeller[3]),flicker_subspeller(randomized_subspeller[4]))
        eegMarking(board,marker)
        for frame, j in enumerate(range(epoch_frames)):
            get_keypress()
            # for flicker in flickers.values():
            #     flicker.draw2(frame = frame)
            target_flicker.draw2(frame = frame)
            frames += 1
            window.flip()
        core.wait(0.5)

def main():
    global sequence
    global trialClock

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
    #board start streaming
    board_shim.start_stream()

    logging.info('Begining the experiment')

    while True:

        # Starting the display
        trialClock = core.Clock()
        cal_start.draw()
        window.flip()
        core.wait(3)


        drawTextOnScreen("Starting the experiment.Please donot move now",window)
        #Adding buffer of 10 sec at the beginning of experiment
        core.wait(10)
        sequence = random.sample(TARGET_CHARACTERS, len(TARGET_CHARACTERS))

        for trials in range(NUM_TRIAL):
            get_keypress()
            # Drawing display box

            # Drawing the grid
            # Display target characters
            for target in targets.values():
                target.autoDraw = True
                # get_keypress()
            flicker(board_shim)

            # At the end of the trial, calculate real duration and amount of frames
            t1 = trialClock.getTime()  # Time at end of trial
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print(f"Total frames: {frames}")
        
        #Adding buffer of 10 sec at the end
        core.wait(10)
        # saving the data from 1 block
        block_name = f'{PARTICIPANT_ID}'
        data = board_shim.get_board_data()
        # data_copy = data.copy()
        # raw = getdata_offline(data_copy,BOARD_ID,n_samples = 250,dropEnable = False)
        # save_raw(raw,block_name,RECORDING_DIR, PARTICIPANT_ID)
        save_csv(data, RECORDING_DIR, PARTICIPANT_ID)

        for target in targets.values():
            target.autoDraw = False

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
