
#%%
import numpy as np
import matplotlib.pyplot as plt
import pyaudio as pa

##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


chunk = 400
sample_format = pa.paInt16
channels = 1
sr = 16000
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000

seconds = 2

p = pa.PyAudio()

# recording
print('Recording')

stream = p.open(format=sample_format, channels=channels, rate=sr,
                frames_per_buffer=chunk, input=True)

data = list()

frames = list()

print("recording start~!!")

for i in range(0, int(sr / chunk * seconds)):
    if i%(sr/chunk)==0:
        print("{sec}".format(sec=(seconds-(chunk/sr)*i)))
    data = stream.read(chunk)
    data = np.frombuffer(data, 'int16')
    frames.extend(data)
   



frames = standardization_func(frames) 

plt.plot(frames)
plt.show()







