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
from utils.common import getdata, save_raw, drawTextOnScreen
from beeply.notes import *
from utils.speller_config import *

a = beeps(800)
# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

#create a window
window = visual.Window([width, height], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="norm", fullscr=True)


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

def main():
    global sequence
    global trialClock

    BoardShim.enable_dev_board_logger()
    drawTextOnScreen('Begining the experiment',window)
    core.wait(7)
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

        sequence = random.sample(TARGET_CHARACTERS, len(TARGET_CHARACTERS))

        for trials in range(NUM_TRIAL):
            for target in TARGET_CHARACTERS:
                eegMarking(board_shim, 1)
                get_keypress()
                fixation = visual.ShapeStim(window,
                                    vertices=((0, -0.5), (0, 0.5), (0,0), (-0.5,0), (0.5, 0)),
                                    lineWidth=5,
                                    closeShape=False,
                                    lineColor="black"
                )
                fixation.draw()
                window.flip()
                core.wait(1.7)
    
        data = board_shim.get_board_data()
        data_copy = data.copy()
        raw = getdata(data_copy,BOARD_ID,n_samples = 250,dropEnable = False)
        raw.save(f'{PARTICIPANT_ID}{TYPE_OF_FILE}', overwrite = True)


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
