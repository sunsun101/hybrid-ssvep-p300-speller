import json
import random

data = {}

import string

alphabets = list(string.ascii_uppercase)
numbers = [str(i) for i in range(10)]
special_chars = ["(", " ", ")", "!", "-", "<<", ".", "?", ","]

alpha_num_special = numbers + alphabets + special_chars

print(len(alpha_num_special))

# Add keys from 1 to 45
for i in range(0,45):
    key = alpha_num_special[i]
    value = float(i)
    data[key] = value

# Write the data to a json file
with open("data.json", "w") as json_file:
    json.dump(data, json_file)
