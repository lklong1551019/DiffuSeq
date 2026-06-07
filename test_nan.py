import torch
import json
import numpy as np

with open('NaNLossInput.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('Raw IDs:'):
            ids_line = lines[i+1]
            break

ids = json.loads(ids_line)
print("Shape of parsed ids:", np.array(ids).shape)
