#Configuration for units=pix
HEIGHT: int = 100
WIDTH: int = 100
UNITS = "pix"
HEIGHT_OF_TARGET = 35
HORI_DIVIDER_START = [-900,0]
HORI_DIVIDER_END = [900,0]
VER_DIVIDER_1_START = [0,500]
ONLINE_VER_DIVIDER_1_START = [0, 400]
VER_DIVIDER_1_END = [0,-500]
DISPLAY_BOX_SIZE = [1700,100]
DISPLAY_BOX_POS = (0,450)

SIZE: int = 100
NUM_BLOCK: int = 6
NUM_TRIAL: int = 2
NUM_SESSION: int = 3
EPOCH_DURATION: float = 2
ITI_DURATION: float = 0.1
CUE_DURATION: float = 1
NO_SUBSPELLER: int = 4

# FREQS: list = [8, 8, 8.6, 8.6, 9, 9, 9.6, 9.6 ]

FREQS: list = [8, 8, 8.6, 8.6, 8, 8, 8.6, 8.6, 9, 9, 9.6, 9.6, 9, 9 ,9.6, 9.6]


# POSITIONS: list = [(-800, 200), (- 400, 200), (400, 200), (800, 200), (-800,-200), (-400, -200), (400, -200), (800, -200)]
POSITIONS: list = [(-800, 300), (- 400, 300), (400, 300), (800, 300), (-800, 100), (- 400, 100), (400, 100), (800, 100),(-800, -100), (- 400, -100), (400, -100), (800, -100), (-800,-300), (-400, -300), (400, -300), (800, -300)]

AMPLITUDE: float = 1.0

# PHASES: list = [0 , 0 , 1.05 , 1.05 , 1.75 , 1.75, 0.80, 0.80]

PHASES: list = [0 , 0 , 1.05 , 1.05, 0 , 0, 1.05 , 1.05 , 1.75 , 1.75, 0.80 , 0.80, 1.75, 1.75, 0.80, 0.80]


# TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H"]
TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

SUBSPELLERS:dict = {"1": ["A", "B", "E", "F"],
                    "2": ["C", "D", "G", "H"],
                    "3": ["I", "J", "M", "N"],
                    "4": ["K", "L", "O", "P"]
                    }

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "sunsun_4target_20230512_v2"
RECORDING_DIR:str = "hybrid/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 30
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0, "J": 10.0, "K": 11.0, "L":12.0, "M": 13.0, "N": 14.0, "O": 15.0, "P":16.0, "trial_start": 99.0}