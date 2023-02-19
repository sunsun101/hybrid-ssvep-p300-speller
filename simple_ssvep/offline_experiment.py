import numpy as np
import pandas as pd
import time
from psychopy import visual, core, event
from glob import glob
from random import choice, random
from psychopy.visual import ShapeStim
import math
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from utils.speller_config import *
import sys
sys.path.append('E:\Thesis\HybridSpeller\presentation')
import logging
from utils import getdata, save_ssvep_raw, save_raw, drawTextOnScreen


markernames = [1, 2, 3]


def getFrames(freq):
    framerate = 60  # mywin.getActualFrameRate()
    frame = int(round(framerate / freq))
    frame_on = math.ceil(frame / 2)
    frame_off = math.floor(frame / 2)
    return frame_on, frame_off


def three_stimuli_blinking(frame_on1, frame_off1, frame_on2, frame_off2, frame_on3, frame_off3, shapes, flipCount, count):
    looptime = math.gcd(frame_on1, math.gcd(frame_on2, frame_on3))

    # reset clock for next trial
    trialclock.reset()
    while trialclock.getTime() < soa:
        # if count% freq_len ==0:
        if (flipCount == 0 or (flipCount % frame_on1 == 0 and flipCount % (frame_on1+frame_on1) != 0)):
            shapes[0].setAutoDraw(True)
            shapes[1].setAutoDraw(False)
        if (flipCount % (frame_off1+frame_off1) == 0):
            shapes[1].setAutoDraw(True)
            shapes[0].setAutoDraw(False)

        # if count% freq_len ==1:
        if (flipCount == 0 or (flipCount % frame_on2 == 0 and flipCount % (frame_on2+frame_on2) != 0)):
            shapes[2].setAutoDraw(True)
            shapes[3].setAutoDraw(False)
        if (flipCount % (frame_off2+frame_off2) == 0):
            shapes[3].setAutoDraw(True)
            shapes[2].setAutoDraw(False)
        # if count% freq_len ==2:
        if (flipCount == 0 or (flipCount % frame_on3 == 0 and flipCount % (frame_on3+frame_on3) != 0)):
            shapes[4].setAutoDraw(True)
            shapes[5].setAutoDraw(False)
        if (flipCount % (frame_off3+frame_off3) == 0):
            shapes[5].setAutoDraw(True)
            shapes[4].setAutoDraw(False)

        for frameN in range(looptime):
            mywin.flip()
            flipCount += 1
    shapes[0].setAutoDraw(False)
    shapes[1].setAutoDraw(False)
    shapes[2].setAutoDraw(False)
    shapes[2].setAutoDraw(False)
    shapes[3].setAutoDraw(False)
    shapes[4].setAutoDraw(False)
    shapes[5].setAutoDraw(False)


def eegMarking(board, marker):
    print("Inserting marker", marker)
    board.insert_marker(marker)
    time.sleep(0.1)


def main():
    global mywin 
    global trialclock
    global soa

    mywin = visual.Window([1536, 864], fullscr=False)

    soa = 6  # stimulus onset asynchrony
    iti = 2  # inter trial interval

    trials_no = 5
    test_freq = [6, 10, 16]  # , 15]
    stimuli_seq = [0, 1, 2] * trials_no  # five trials for each freq in test_freq
    freq_len = len(test_freq)

    frame_on1, frame_off1 = getFrames(test_freq[0])
    frame_on2, frame_off2 = getFrames(test_freq[1])
    frame_on3, frame_off3 = getFrames(test_freq[2])

    # print(getFrames(16))

    count = 0
    trialclock = core.Clock()

    patternup1Pos = [0, 0.65]
    patternright1Pos = [0.65, -0.5]
    patternleft1Pos = [-0.65, -0.5]

    # Arrow position is now y+0.2
    arrowUp1Pos = [0, 0.85]
    arrowRigh1Pos = [0.65, -0.3]
    arrowLeft1Pos = [-0.65, -0.3]

    # array to identify the sequence of the stimuli
    arrowSequence = [arrowUp1Pos, arrowRigh1Pos, arrowLeft1Pos]

    patternup1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern1', autoLog=False, color=[1, 1, 1], pos=patternup1Pos)
    patternup2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern2', autoLog=False, color=[-1, -1, -1], pos=patternup1Pos)

    patternright1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern1', autoLog=False, color=[1, 1, 1], pos=patternright1Pos)
    patternright2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern2', autoLog=False, color=[-1, -1, -1], pos=patternright1Pos)

    # patterndown1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.3,
    #    name='pattern1', autoLog=False, color=[1,1,1], pos=(0, -0.5))
    # patterndown2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.3,
    #    name='pattern2', autoLog=False, color=[-1,-1,-1], pos=(0, -0.5))

    patternleft1 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern1', autoLog=False, color=[1, 1, 1], pos=patternleft1Pos)
    patternleft2 = visual.GratingStim(mywin, tex=None, sf=0, size=0.6,
                                    name='pattern2', autoLog=False, color=[-1, -1, -1], pos=patternleft1Pos)

    # prepare the arrow shape
    arrowVert = [(0, 0), (-0.1, 0.15), (-0.05, 0.15), (-0.05, 0.3),
                (0.05, 0.3), (0.05, 0.15), (0.1, 0.15)]

    shapes = [patternup1, patternup2, patternright1,
            patternright2, patternleft1, patternleft2]

    # fixation cross
    fixation = visual.ShapeStim(mywin,
                                vertices=((0, -0.5), (0, 0.5), (0, 0),
                                        (-0.5, 0), (0.5, 0)),
                                lineWidth=5,
                                closeShape=False,
                                lineColor="white"
                                )
    BoardShim.enable_dev_board_logger()

    # brainflow initialization
    params = BrainFlowInputParams()
    print("Here is the board id", BOARD_ID)
    # params.serial_port = serial_port
    board_shim = BoardShim(BOARD_ID, params)

    while True:
        message = visual.TextStim(
            mywin, text='Start recording and press space to continue')
        message.draw()
        mywin.flip()
        keys = event.getKeys()

        if 'space' in keys:  # If space has been pushed

            # prepare board
            try:
                board_shim.prepare_session()
            except brainflow.board_shim.BrainFlowError as e:
                print(f"Error: {e}")
                print("The end")
                time.sleep(1)
                sys.exit()
            # board start streaming
            board_shim.start_stream()
            for block in range(5):
                count = 0
                message.setText = ''
                message.draw()
                mywin.flip()

                fixation.draw()
                mywin.flip()  # refresh
                core.wait(iti)
                mywin.flip()

                # create arrow shape for the first sequence
                arrow = ShapeStim(mywin, vertices=arrowVert, fillColor='darkred',
                                size=.5, lineColor='red', pos=arrowSequence[0])
                arrow.setAutoDraw(True)
                mywin.flip()
                core.wait(iti)
                mywin.flip()

                while count < len(stimuli_seq):
                    print("Count: ", count)

                    # draw the stimuli and update the window
                    print("freq: ", test_freq[count % freq_len])
                    # print("frameon-off: ", frame_on, frame_off)
                    print("markername: ", markernames[count % freq_len])
                    print("======")

                    eegMarking(board_shim, markernames[count % freq_len])
                    # outlet.push_sample([markernames[count%freq_len]])  #(x, timestamp)

                    flipCount = 0
                    # one_stimuli_blinking(frame_on, frame_off, shapes[count%freq_len*2], shapes[count%freq_len*2+1])
                    three_stimuli_blinking(frame_on1, frame_off1, frame_on2,
                                        frame_off2, frame_on3, frame_off3, shapes, flipCount, count)

                    # close the finish arrow
                    arrow.setAutoDraw(False)

                    # draw the next arrow
                    arrow = ShapeStim(mywin, vertices=arrowVert, fillColor='darkred',
                                    size=.5, lineColor='red', pos=arrowSequence[(count+1) % freq_len])
                    arrow.setAutoDraw(True)

                    # clean black screen off
                    mywin.flip()
                    # wait certain time for next trial
                    core.wait(iti)
                    # clear fixation
                    mywin.flip()
                    # count number of trials
                    count += 1

                    if 'escape' in event.getKeys():
                        core.quit()
                # saving the data
                block_name = f'{PARTICIPANT_ID}{block}'
                data = board_shim.get_board_data()
                data_copy = data.copy()
                raw = getdata(data_copy, BOARD_ID, n_samples=250, dropEnable=False)
                save_raw(raw, block_name, "simple_ssvep/record")
                arrow.setAutoDraw(False)
                if (block + 1) < 5: 
                    drawTextOnScreen('Block Break 1 Minute',mywin)
                    core.wait(30)
                #throw data
                data = board_shim.get_board_data()
            break
       

    if board_shim.is_prepared():
        logging.info('Releasing session')
        # stop board to stream
        board_shim.stop_stream()
        board_shim.release_session()
    mywin.close()


if __name__ == "__main__":
    main()
