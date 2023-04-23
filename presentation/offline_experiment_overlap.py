import os
import sys

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)

import argparse
import json
import logging
import multiprocessing
import platform
import random
import threading
import time
from multiprocessing import Process

import brainflow
import numpy as np
from beeply.notes import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from psychopy import core, event, visual  # import some libraries from PsychoPy

from utils.common import drawTextOnScreen, getdata_offline, save_raw
from utils.gui import CheckerBoard, get_screen_settings
from utils.speller_config import *

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
ver_divider_2 = visual.Line(window, start=VER_DIVIDER_2_START, end=VER_DIVIDER_2_END, lineColor='black')
display_box = visual.Rect(window, size=DISPLAY_BOX_SIZE, pos=DISPLAY_BOX_POS, lineColor='black', lineWidth=2.5)

block_break_text = "Block Break 1 Minutes"
block_break_start = visual.TextStim(window, text=block_break_text, color=(-1., -1., -1.))

def get_keypress():
    keys = event.getKeys()
    if keys and keys[0] == 'escape':
        window.close()
        core.quit()
    else: 
        return None

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    proc.append(p)
  for p in proc:
    p.start()
  for p in proc:
    p.join()

def flicker_subspeller(sub_characters):
   for frame, n in enumerate(range(epoch_frames)):
        # get_keypress()
        for char in sub_characters:
            flickers[char].draw2(frame=frame)
            window.flip()


def randomize_characters():
    randomized_subspeller = {}
    for n in range(1, NO_SUBSPELLER + 1):
        subspeller_char = SUBSPELLERS[str(n)]
        random_seq = random.sample(subspeller_char, len(subspeller_char))
        randomized_subspeller[n] = random_seq
    print(randomized_subspeller)
    return randomized_subspeller

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
        target_freq = target_flicker.freq
        target_phase = target_flicker.phase
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
        eegMarking(board, MARKERS['trial_start'])
        
        # IDEA
        # Generating an entire epoch of frames
        # The shape is (n, m, f) where 
        # n: number of sub-speller
        # m: is each character in the sub speller
        # f: is frame_idx
        timeline = gen_timeline(n=6, m=9, overlap=0.5, isShuffle=True)
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
                            eegMarking(board,marker)
                            marked = True
                        flickers[char].draw2(frame=frame[idx])
            window.flip()


        # for m in range(9):
        #     get_keypress()
        #     elementsArray = [] # stores each character from randomized subspeller to flicker at once
        #     for n in range(1,7):
        #         get_keypress()
        #         if len(randomized_subspeller[n])>m:
        #             elementsArray.append(flickers[str(randomized_subspeller[n][m])])
        #     if target_flicker in elementsArray :
        #         eegMarking(board,marker)
        #     for frame, j in enumerate(range(epoch_frames)):
        #         get_keypress()
        #         for flicker in elementsArray:
        #             flicker.draw2(frame = frame)
        #         frames += 1
        #         window.flip()

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
    global randomized_subspeller
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
        core.wait(6)

        for block in range(NUM_BLOCK):
            a.hear('A_')
            drawTextOnScreen('Starting block ' + str(block + 1) + ".Please donot move now",window)
            core.wait(7)
            sequence = random.sample(TARGET_CHARACTERS, len(TARGET_CHARACTERS))
            #randomize the characters of the sub speller. returns a dictionary of randomized characters in each sub speller. {1: ['S', 'L', 'U', 'J', 'T', 'K', 'C', 'B', 'A'], 2: ['F', 'M', 'V', 'D', 'W', 'N', 'O', 'E', 'X'], 3: ['Q', '0', 'I', 'Z', 'H', 'R', 'Y', 'P', 'G'], 4: ['2', '5', '1', '6', '4', '3'], 5: ['8', '9', '?', '7', ',', '.'], 6: ['Space', '<<', '-', '!', '(', ')']}
            randomized_subspeller = randomize_characters()
            
            get_keypress()
            # Drawing display box
            display_box.autoDraw = True
            # Drawing the grid
            hori_divider.autoDraw = True
            ver_divider_1.autoDraw = True
            ver_divider_2.autoDraw = True
            # Display target characters
            for target in targets.values():
                target.autoDraw = True
                # get_keypress()
            print("Sequence is", sequence)
            flicker(board_shim)
            core.wait(1)

            # At the end of the trial, calculate real duration and amount of frames
            t1 = trialClock.getTime()  # Time at end of trial
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print(f"Total frames: {frames}")

            # saving the data from 1 block
            block_name = f'{PARTICIPANT_ID}{block}'
            data = board_shim.get_board_data()
            print("WHAT IS IN DATA")
            print(data)
            data_copy = data.copy()
            raw = getdata_offline(data_copy,BOARD_ID,n_samples = 250,dropEnable = False)
            print("WHAT IS IN RAW")
            print(raw)
            save_raw(raw,block_name,RECORDING_DIR, PARTICIPANT_ID)

        
            #giving block break
            # clearing the screen
            display_box.autoDraw = False
            hori_divider.autoDraw = False
            ver_divider_1.autoDraw = False
            ver_divider_2.autoDraw = False
            for target in targets.values():
                target.autoDraw = False
            if (block + 1) < NUM_BLOCK: 
                drawTextOnScreen('Block Break 1 Minute',window)
                core.wait(BLOCK_BREAK)
            #throw data
            data = board_shim.get_board_data()

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