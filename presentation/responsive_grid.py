from psychopy import visual, core
import random
import string

win = visual.Window(size=[800, 600], units='norm', color='black')

alphabets = list(string.ascii_uppercase)
numbers = [str(i) for i in range(10)]
special_chars = ["(", " ", ")", "!", "-", "<<", ".", "?", ","]
alpha_num_special = alphabets + numbers + special_chars

text_objects = []
for i in range(45):
    # random character
    text = random.choice(alpha_num_special)
    text_objects.append(visual.TextStim(win, text=text, color='white', pos=(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)), alignHoriz='center', alignVert='center'))

while True:
    for t in text_objects:
        t.draw()
    win.flip()