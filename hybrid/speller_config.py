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
SEQUENCE_POS = ( -140,470)
OUTPUT_POS = (-160,430)

SIZE: int = 100
NUM_BLOCK: int = 10
NUM_TRIAL: int = 3
NUM_SESSION: int = 3
EPOCH_DURATION: float = 2
ITI_DURATION: float = 0.1
CUE_DURATION: float = 1
NO_SUBSPELLER: int = 4
NO_CHARACTER: int = 3
OFFSET_VALUE: int = 1000

#12 targets
# Non Symmetrical positioning
FREQS: list = [8, 8, 8.6, 8, 8.6, 8.6, 9, 9, 9.6, 9, 9.6, 9.6 ]
# Symmetrical  positioning
# FREQS: list = [8, 8, 8, 8.6, 8.6, 8.6, 9, 9, 9, 9.6, 9.6, 9.6 ]
#8 targets
# FREQS: list = [8, 8, 8.6, 8.6, 9, 9, 9.6, 9.6]
#16 targets
# FREQS: list = [8, 8, 8.6, 8.6, 8, 8, 8.6, 8.6, 9, 9, 9.6, 9.6, 9, 9 ,9.6, 9.6]
# FREQS: list = [12.4, 12.4, 13.2, 13.2, 12.4, 12.4, 13.2, 13.2, 14.0, 14.0, 14.6, 14.6, 14.0, 14.0 ,14.6, 14.6]
# FREQS: list = [8, 8, 10, 10, 8, 8, 10, 10, 12, 12, 14, 14, 12, 12 ,14, 14]

#12 targets
# Non symmetrical positioning
POSITIONS: list = [(-850, 300), (-170, 300), (510, 300), (-510, 100), (170, 100), (850, 100), (-850, -100), (-170, -100), (510, -100), (-510, -300), (170, -300), (850, -300)]
# Symmetrical positioning
# POSITIONS: list = [(-850, 200), (-510, 200), (-170, 200), (170, 200), (510, 200), (850, 200), (-850,-200), (-510, -200), (-170, -200), (170, -200), (510, -200), (850, -200)]
#8 targets
# POSITIONS: list = [(-800, 200), (- 400, 200), (400, 200), (800, 200), (-800,-200), (-400, -200), (400, -200), (800, -200)]
#16 targets
# POSITIONS: list = [(-800, 300), (- 400, 300), (400, 300), (800, 300), (-800, 100), (- 400, 100), (400, 100), (800, 100),(-800, -100), (- 400, -100), (400, -100), (800, -100), (-800,-300), (-400, -300), (400, -300), (800, -300)]

AMPLITUDE: float = 1.0

#12 targets
# Non symmetrical positioning
PHASES: list = [0 , 0 , 1.05 , 0,  1.05 , 1.05, 1.75, 1.75 ,0.80,  1.75, 0.80, 0.80]
# Symmetrical positioning
# PHASES: list = [0 , 0 , 0, 1.05,  1.05 , 1.05, 1.75, 1.75 ,1.75, 0.80, 0.80, 0.80]
#8 targets
# PHASES: list = [0 , 0 , 1.05 , 1.05 , 1.75, 1.75, 0.80, 0.80]
#16 targets
# PHASES: list = [0 , 0 , 1.05 , 1.05, 0 , 0, 1.05 , 1.05 , 1.75 , 1.75, 0.80 , 0.80, 1.75, 1.75, 0.80, 0.80]
# PHASES: list = [0 , 0 , 1.40 , 1.40, 0 , 0, 1.40 , 1.40 , 0.80 , 0.80, 1.85 , 1.85, 0.80, 0.80, 1.85, 1.85]
# PHASES: list = [0 , 0 , 1 , 1, 0 , 0, 1 , 1 , 0 , 0, 1 , 1, 0, 0, 1, 1]


#12 targets
TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
#8 targets
# TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H"]
#16 targets
# TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

#16 targets
# SUBSPELLERS:dict = {"1": ["A", "B", "E", "F"],
#                     "2": ["C", "D", "G", "H"],
#                     "3": ["I", "J", "M", "N"],
#                     "4": ["K", "L", "O", "P"]
#                     }
#12 targets
# Non symmerical config
SUBSPELLERS:dict = {"1": ["A", "B", "D"],
                    "2": ["C", "E", "F"],
                    "3": ["G", "H", "J"],
                    "4": ["I", "K", "L"]
                    }
# Symmetrical config
# SUBSPELLERS:dict = {"1": ["A", "B", "C"],
#                     "2": ["D", "E", "F"],
#                     "3": ["G", "H", "I"],
#                     "4": ["J", "K", "L"]
#                     }
#8 targets
# SUBSPELLERS:dict = {"1": ["A", "B"],
#                     "2": ["C", "D"],
#                     "3": ["E", "F"],
#                     "4": ["G", "H"]
#                     }

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "anubinda_20230630_2sec_0.5_overlap_8Hz_12target"
# PARTICIPANT_ID:str="test"
RECORDING_DIR:str = "hybrid/record/final"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 40
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0, "J": 10.0, "K": 11.0, "L":12.0, "M": 13.0, "N": 14.0, "O": 15.0, "P":16.0, "trial_start": 99.0}