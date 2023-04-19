#Configuration for units=pix
HEIGHT: int = 500
WIDTH: int = 500
UNITS = "pix"
HEIGHT_OF_TARGET = 35
HORI_DIVIDER_START = [-850,-75]
HORI_DIVIDER_END = [850,-75]
VER_DIVIDER_1_START = [-300,350]
VER_DIVIDER_1_END = [-300,-350]
VER_DIVIDER_2_START = [300, 350]
VER_DIVIDER_2_END = [300,-350]
DISPLAY_BOX_SIZE = [1700,100]
DISPLAY_BOX_POS = (0,450)

SIZE: int = 100
NUM_BLOCK: int = 5
NUM_TRIAL: int = 5
NUM_SESSION: int = 3
EPOCH_DURATION: float = 5
ITI_DURATION: float = 0.1
CUE_DURATION: float = 0.5
NO_SUBSPELLER: int = 6

FREQS: list = [10, 12 , 15]

POSITIONS: list = [(0,300), (-600, -150), (600, -150)]

AMPLITUDE: float = 1.0

PHASES: list = [0 , 0 , 0]

TARGET_CHARACTERS:list = ["A", "B", "C"]

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "sunsun_20230419"
RECORDING_DIR:str = "three_flicker/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 1
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0}