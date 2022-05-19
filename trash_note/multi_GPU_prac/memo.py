import numpy as np

zeroth_file = "train_data_.npz"

loaded_data = np.load(zeroth_file)

print(loaded_data['data'])
print(len(loaded_data['data']))

print(loaded_data['label'])
print(len(loaded_data['label']))
