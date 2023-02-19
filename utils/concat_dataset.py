import os
import mne


def concat_dataset(subjects, num_blocks):
    raws = []
    base_path = "E:\Thesis\HybridSpeller\\record"

    for subject in subjects:
        for i in range(1,num_blocks):
            print(subject)
            path = os.path.join(base_path, subject, f"{subject}{i}.fif")
            print("Here is the path")
            print(path)
            raws.append(mne.io.read_raw_fif(path, preload = True))
    raw = mne.concatenate_raws(raws)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    print(raw.info['ch_names'])
    print(raw.info['sfreq'])
    return raw



if __name__ == "__main__":
    subjects = ['best_recording', 'wan']
    raw = concat_dataset(subjects, 5)
    print(raw.info)