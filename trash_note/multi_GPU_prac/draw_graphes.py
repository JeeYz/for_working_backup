

import matplotlib.pyplot as plt
import numpy as np
import librosa


with open("D:\\data_log.txt", "r") as f:
    while True:
        line = f.readline()
        line = line.split()
        if not line: break
        try:
            line = np.array(line, dtype=np.float32)
        except:
            # continue
            print("error is occured..")

        aug_data = librosa.effects.time_stretch(line, 0.6)
        plt.figure()
        plt.plot(range(len(line)), line)

        plt.figure()
        plt.plot(range(len(aug_data)), aug_data)

    plt.show()



