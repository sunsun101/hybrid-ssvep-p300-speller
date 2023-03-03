import mne
import matplotlib
matplotlib.use('TkAgg')


# fname = "record\\wan\\wan1.fif"
# fname = "record\\first_recording\\first_recording2.fif"
fname = "sunsun_v2.fif"
raw = mne.io.read_raw_fif(fname, preload = True)
raw.filter(l_freq=1, h_freq=None)
raw.notch_filter(freqs=[50,100])
events = mne.find_events(raw)
raw.plot(events=events, start=5, duration=10, color='gray',
         event_color={1: "r", 2: "r", 3: "r", 4: "r", 5: "r", 6: "r", 7: "r", 8: "r", 9: "r", 10: "r", 11: "r", 12: "r", 13: "r", 14: "r", 15: "r", 16: "r", 17: "r", 18: "r", 19: "r", 20: "r", 21: "r", 22: "r", 23: "r", 24: "r", 25: "r", 26: "r", 27: "r", 28: "r", 29: "r", 30: "r", 31: "r", 32: "r", 33: "r", 34: "r", 35: "r", 36: "r", 37: "r", 38: "r", 39: "r", 40: "r", 41: "r", 42: "r", 43: "r", 44: "r", 45: "r", 99:"g"}, block = True)
         