import argparse
import time
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
import mne
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations,find_events
from mne.channels import make_standard_montage
from mne.filter import construct_iir_filter,create_filter
from config import *
from data_utils import save_raw,getdata,getepoch,save_raw_to_dataframe,send_raw,Erd_Plot,randomStimuli
import requests
from numpy import ndarray
import keyboard
from threading import Thread
import time
import logging
from database import Database 
from psychopy import visual, core, event,monitors
from experiment_gui import *
import multiprocessing
import concurrent.futures
import random
import sounddevice as sd
import soundfile as sf
a = beeps(1000)
b = beeps(800)
logging.getLogger().setLevel(logging.DEBUG)
def main():
    global EXE_COUNT
    global IMAGINE_COUNT
    global STIM_CHECK
    global PLAY_VIDEO
    #Window setup
    mywin = visual.Window(SCREEN_SIZE, color="black",monitor="Experiment Monitor" , units='norm',screen=4,fullscr=True)
    #erdWin = visual.Window(SCREEN_SIZE, color="black",monitor="ERD view" , units='norm',screen=0)
    stimuli = get_stimuli()
    
    #Board setup
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(1750000)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        while True:  
            #wait to avoid bombardment of gpu
            #ตรงนี้ต้องเป็น experiment ซ้ายขวา
            drawTextOnScreen('Experiment session : Press space bar to start',mywin)
            keys = event.getKeys()
            if 'space' in keys: #start
                throw_data = board.get_board_data()
                start = time.time()
                drawTextOnScreen('',mywin)
                for session in range(NUM_SESSION):
                    image_list,numIm_list,video_list,numVi_list = randomStimuli(NUM_TRIAL)
                    num = random.randint(0,1)
                    # 4 block
                    for block in range(NUM_BLOCK):
                        if IS_VIDEO:
                            if (block+1) % 2 != 0:
                                #Executed use image list
                                PLAY_VIDEO = False
                                stim = IMAGE_DICT
                            else:
                                if(block+1) / 2 == 1:
                                    #block 2      
                                    stim = VIDEO_DICT[num]
                                else:
                                    #block 4
                                    if num == 0:
                                        stim = VIDEO_DICT[1]
                                    else:
                                        stim = VIDEO_DICT[0]
                                #Imagine use video list
                                PLAY_VIDEO = True
                        #1:'execute_left',2:'executed_right',3:'imagine_left',4:'imagine_right'
                        #12 trials
                        STIM_CHECK = 0
                        a.hear('A_')
                        for trials in range(NUM_TRIAL):
                            drawFixation(FIXATION_TIME,board,mywin)
                            #สลับซ้ายขวา = ใช้ mod
                            #check is_video == true       
                            if PLAY_VIDEO == True:
                                if PLAY_SOUND == True:
                                    data, fs = sf.read(SOUND_DICT[numVi_list[trials]])
                                    sd.play(data, fs)
                                    sd.wait()
                                #playVideo(f"{stim}",Marker,STIM_TIME,board_shim,mywin)
                                playVideo(f"{stim[numVi_list[trials]]}",BLOCK_MARKER[numVi_list[trials]],STIM_TIME,board,mywin)                            #b.hear('A_')
                                drawFixation(FIXATION_TIME,board,mywin)
                                STIM_CHECK += 1
                                print(STIM_CHECK)
                     
                                #จบ 1 trials
                                print("save and send") 
                                drawTextOnScreen('Save & Send',mywin)
                                #freeze to get full data
                                core.wait(7)
                                
                                data = board.get_board_data()

                                if CALIBRATION == True:
                                    #visualize ERD
                                    drawTextOnScreen('Visualize',mywin)
                                    raw = getdata(data,0,n_samples=250,dropEnable=DROPENABLE)
                                    __,epochs_raw_data,__ = getepoch(raw,0,7)
                                    executor.submit(Erd_Plot,epochs_raw_data,trials)
                                    image_path = f'.\{ERD_FOLDER}{PARTICIPANT_ID}\{NAME}_{trials:02d}.png'
                                    #drawERD(image_path,erdWin)
                                    file_name = f'{PARTICIPANT_ID}R{trials+1:02d}'
                                    save_raw(raw,file_name,ONLINE_FOLDER)
                                else:
                                    #sending and predict
                                    raw = getdata(data,0,n_samples=250,dropEnable=DROPENABLE)
                                    file_name = f'{PARTICIPANT_ID}R{trials+1:02d}'
                                    save_raw(raw,file_name,ONLINE_FOLDER)
                                    drawTextOnScreen('Sending',mywin)
                                    temp:ndarray = data.copy()
                                    database = Database(values=temp,names=file_name)
                                    print(database.value.shape)
                                    
                                    #thread 1 pack
                                    executor.submit(database.locked_update,1)
                                    core.wait(0.1)
                                    
                                    #thread 2 send                
                                    drawTextOnScreen('Sending',mywin)
                                    executor.submit(send_raw,database)
                                    
                                    core.wait(15)
                                    drawTextOnScreen('',mywin)
                                    a.hear('A_')
                                throw_data = board.get_board_data()
                            else:
                                if PLAY_SOUND == True:
                                    data, fs = sf.read(SOUND_DICT[numIm_list[trials]])
                                    sd.play(data, fs)
                                    sd.wait()
                                drawTrial(f"{stim[numIm_list[trials]]}",BLOCK_MARKER[numIm_list[trials]],STIM_TIME,board,mywin)
                                drawFixation(FIXATION_TIME,board,mywin)
                                STIM_CHECK += 1
                                print(STIM_CHECK)
                                
                                #จบ 1 trials
                                print("save and send") 
                                drawTextOnScreen('Save & Send',mywin)
                                #freeze to get full data
                                #core.wait(7)
                                
                                data = board.get_board_data()
                                
                                
                                if CALIBRATION == True:
                                    print("Fininsh saving")
                                    #visualize ERD
                                    drawTextOnScreen('Visualize',mywin)
                                    raw = getdata(data,0,n_samples=250,dropEnable=DROPENABLE)
                                    #drawERD(image_path,erdWin)
                                    file_name = f'{PARTICIPANT_ID}R{trials+1:02d}'
                                    save_raw(raw,file_name,ONLINE_FOLDER)
                                    __,epochs_raw_data,__ = getepoch(raw,0,7)
                                    #Erd_Plot(epochs_raw_data,trials)
                                    executor.submit(Erd_Plot,epochs_raw_data,trials)
                                    image_path = f'.\{ERD_FOLDER}{PARTICIPANT_ID}\{NAME}_{trials:02d}.png'
                                else:
                                    #sending and predict
                                    raw = getdata(data,0,n_samples=250,dropEnable=DROPENABLE)
                                    file_name = f'{PARTICIPANT_ID}R{trials+1:02d}'
                                    #save_raw(raw,file_name,ONLINE_FOLDER)
                                    
                                    drawTextOnScreen('Sending',mywin)
                                    temp:list = data.copy()
                                    database = Database(values=temp,names=file_name)
                                    print(database.value.shape)
                                    print(type(database.value))
                                    
                                    #thread 1 pack
                                    executor.submit(database.locked_update,1)
                                    core.wait(0.1)
                                    
                                    #thread 2 send                
                                    drawTextOnScreen('Sending',mywin)
                                    executor.submit(send_raw,database)
                                    
                                    core.wait(10)
                                    drawTextOnScreen('',mywin)
                                    a.hear('A_')
                                throw_data = board.get_board_data()
                            
                drawTextOnScreen('End of experiment, Thank you',mywin)
                stop  = time.time()
                core.wait(5)
                break
            
     
    if board.is_prepared():
            logging.info('Releasing session')
            # stop board to stream
            board.stop_stream()
            board.release_session()
    
    
    
if __name__ == "__main__":
    main()
