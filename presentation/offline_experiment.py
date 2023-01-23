from turtle import fillcolor, pos
from psychopy import visual, core, event #import some libraries from PsychoPy
import platform
import os
from utils_experiments import get_screen_settings, CheckerBoard
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
import sys
import logging
from utils import getdata, save_raw_to_dataframe, save_raw, drawTextOnScreen


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

# Experimental params
size = params['size']
num_block = params['num_block']
num_trial = params['num_trial']
epoch_duration = params['epoch_duration']
iti_duration = params['iti_duration']
cue_duration = params['cue_duration']
freq = params['freqs']
positions = [tuple(position) for position in params['positions']]
phases = params['phases']
target_characters = params['target_characters']
amp = params['amplitude']
no_subspeller = params['no_subspeller']


subspellers = params["subspellers"]
subspeller_1 = params['subspeller_1']
subspeller_2 = params['subspeller_2']
subspeller_3 = params['subspeller_3']
subspeller_4 = params['subspeller_4']
subspeller_5 = params['subspeller_5']
subspeller_6 = params['subspeller_6']

serial_port = params["serial_port"]
board_id = params["board_id"]
participant_id = params["participant_id"]
recording_dir = params["recording_dir"]
block_break = params["block_break"]

# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

#create a window
window = visual.Window([width, height], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="pix", fullscr=True)
refresh_rate = round(window.getActualFrameRate())
print("Refresh Rate ==>", refresh_rate)

# Time conversion to frames
epoch_frames = int(epoch_duration * refresh_rate)
print("Epoch frames ==>",epoch_frames)
iti_frames = int(iti_duration * refresh_rate)
iti_frames_cal = int(0.8 * refresh_rate)
cue_frames = int(cue_duration * refresh_rate)   
markers = params["markers"]

#Presentation content

cue = visual.Rect(window, width=100, height=80, pos=[0, 0], lineWidth=3, lineColor='red')

calib_text_start = "Starting callibration phase.Please avoid moving or blinking.\n\
You may blink when shifting your gaze.Focus your target on the characters presented with red cue."

calib_text_end = "Calibration phase completed"
cal_start = visual.TextStim(window, text=calib_text_start, color=(-1., -1., -1.))
cal_end = visual.TextStim(window, text=calib_text_end, color=(-1., -1., -1.))

targets = {f"{target}": visual.TextStim(win=window, text=target, pos=pos, color=(-1., -1., -1.), height=35)
        for pos, target in zip(positions, target_characters)}


wave_type = "sin"
flickers = {f"{target}": CheckerBoard(window=window, size=size, frequency=f, phase=phase, amplitude=amp, 
                                    wave_type=wave_type, duration=epoch_duration, fps=refresh_rate,
                                    base_pos=pos)
            for f, pos, phase, target in zip(freq, positions, phases, target_characters)}

hori_divider = visual.Line(window, start=[-850,-75], end=[850,-75], lineColor='black')
ver_divider_1 = visual.Line(window, start=[-300,350], end=[-300,-350], lineColor='black')
ver_divider_2 = visual.Line(window, start=[300, 350], end=[300,-350], lineColor='black')
display_box = visual.Rect(window, size=[1700,100], pos=(0,450), lineColor='black', lineWidth=2.5)

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
    for n in range(1, no_subspeller + 1):
        subspeller_char = subspellers[str(n)]
        random_seq = random.sample(subspeller_char, len(subspeller_char))
        randomized_subspeller[n] = random_seq
    print(randomized_subspeller)
    return randomized_subspeller

def eegMarking(board,marker):
    print("Inserting marker", marker)
    board.insert_marker(marker)

def flicker(board):
    global frames
    global t0
    # For the flickering
    for target in sequence:
        get_keypress()
        target_flicker = flickers[str(target)]
        target_pos = (target_flicker.base_x, target_flicker.base_y)
        target_freq = target_flicker.freq
        target_phase = target_flicker.phase
        marker = markers[str(target)]


        t0 = trialClock.getTime()  # Retrieve time at start of cue presentation
        #Display the cue
        cue.pos = target_pos
        for frame in range(cue_frames):
                cue.draw()
                window.flip()

        frames = 0
        #flicker random sequence of each speller parallely
        # runInParallel(flicker_subspeller(randomized_subspeller[1]), flicker_subspeller(randomized_subspeller[2]), flicker_subspeller(randomized_subspeller[3]),flicker_subspeller(randomized_subspeller[4]))
        
        for m in range(9):
            get_keypress()
            elementsArray = [] # stores each character from randomized subspeller to flicker at once
            for n in range(1,7):
                get_keypress()
                if len(randomized_subspeller[n])>m:
                    elementsArray.append(flickers[str(randomized_subspeller[n][m])])
            for frame, j in enumerate(range(epoch_frames)):
                get_keypress()
                for flicker in elementsArray:
                    flicker.draw2(frame = frame)
                frames += 1
                window.flip()
            eegMarking(board,marker)

def main():
    global sequence
    global randomized_subspeller
    global trialClock

    BoardShim.enable_dev_board_logger()

    #brainflow initialization 
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_shim = BoardShim(board_id, params)

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

        for block in range(num_block):

            drawTextOnScreen('Starting block ' + str(block + 1) ,window)
            core.wait(3)
            sequence = random.sample(target_characters, 45)
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

            # At the end of the trial, calculate real duration and amount of frames
            t1 = trialClock.getTime()  # Time at end of trial
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print(f"Total frames: {frames}")

            # saving the data from 1 block
            block_name = f'{participant_id}{block}'
            data = board_shim.get_board_data()
            print("WHAT IS IN DATA")
            print(data)
            data_copy = data.copy()
            raw = getdata(data_copy,board_id,n_samples = 250,dropEnable = False)
            print("WHAT IS IN RAW")
            print(raw)
            save_raw(raw,block_name,recording_dir)

        
            #giving block break
            # clearing the screen
            display_box.autoDraw = False
            hori_divider.autoDraw = False
            ver_divider_1.autoDraw = False
            ver_divider_2.autoDraw = False
            for target in targets.values():
                target.autoDraw = False
            drawTextOnScreen('Block Break 1 Minute',window)
            core.wait(block_break)
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
