
from global_variables import *
import file_processing as fpro
import signal_processing as spro
import test_and_check as tcheck
import augment_processing as augp
import trigger_algorithm as trigal
import modifying_data_and_info as moddi
import none_aug_processing as nonepro
import gen_data_files as gendata
import CW_json_files_decoder as decoder
import CW_signal_processing as cwsig



stack = list()

recording_time = 2
frame_t = 0.025
shift_t = 0.01
chunk = 400
per_sec = sample_rate/chunk
num = 0
start_time, end_time = float(), float()

front_size = 4000
tail_size = 4000
full_size = 20000

voice_trigger = 200
std_trigger = 1.0

pady_size = 20

## global model
test_bool = False

## keyword global model
command_label_num = 6

command_tflite_file = tflite_file

command_interpreter = tf.lite.Interpreter(model_path=command_tflite_file)
command_interpreter.allocate_tensors()
command_input_details = command_interpreter.get_input_details()[0]
command_output_details = command_interpreter.get_output_details()[0]


print("preprocessing done...")


#%%
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)





#%%
def fit_determined_size(data, **kwargs):
    # print(full_size-len(data))
    if len(data) > full_size:
        return data[0:full_size]
    elif len(data) == full_size:
        return data
    else:
        return np.append(data, np.zeros(full_size-len(data)))







#%%
def add_noise_data(data, **kwargs):

    noise_data = np.random.randn(full_size)*0.01
    
    return data+noise_data








#%%
def cut_input_signal(data, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if std_trigger < start:
            start_index = i
            break
        else:
            start_index = 0

    for i,end in enumerate(reversed(mean_val_list)):
        if std_trigger < end:
            end_index = len(mean_val_list)-i-1
            break
        else:
            end_index = len(mean_val_list)

    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0

    temp_tail = gap_frame_shift*end_index+tail_size

    if temp_tail > len(data):
        temp_tail = len(data)-1

    result = data[temp:temp_tail]

    return result





#%%
def cut_input_signal_v2(data, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if std_trigger < start:
            start_index = i
            break
        else:
            start_index = 0

    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0


    result = data[temp:]

    return result



#%%
def cut_input_signal_v3(data, file_pointer, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val = np.mean(np.abs(one_frame_data))

        file_pointer.write(str(mean_val)+"  ")
        if std_trigger < mean_val:
            start_index = i
            break
        else:
            start_index = 0

    file_pointer.write("\n")
    file_pointer.write(str(start_index) + "\n")

    front_size = 5000
    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0

    result = data[temp:]

    # if temp < 0:
    #     abs_temp = np.abs(temp)
    #     result = np.append(np.zeros(abs_temp), data)
    # elif temp == 0:
    #     result = data
    # else:
    #     result = data[front_size:]

    return result





#%%
def receive_data(data, stack):

    global start_time
    global num

    # print(len(data))

    data = np.asarray(data, dtype=np.float32)
    # std_val = standardization_func(data)
    # mean_val = np.mean(np.abs(std_val))
    mean_val = np.mean(np.abs(data))

    # std_val.tolist()
    # stack.extend(std_val)
    stack.extend(data)

    # print(mean_val)

    if len(stack) > sample_rate*(recording_time+1):
        del stack[0:chunk]


    w_file = open("D:\\data_log.txt", "a")
    # w_info_file = open("D:\\data_info_log.txt", "a")

    if float(voice_trigger) <= float(mean_val) or num != 0:
        # print(mean_val)
        num+=1
        # print(num, mean_val)
        if num == 80:
            # print(num, mean_val)

            for one in range(len(stack)):
                w_file.write(str(stack[one]))
                w_file.write("  ")
            
            w_file.write("\n")

            stack = standardization_func(stack)
            start_time = time.time()
            print(len(stack))
            # data = cut_input_signal_v3(stack, w_info_file, frame_size=800, shift_size=400)
            data = cut_input_signal(stack, frame_size=800, shift_size=400)

            data = fit_determined_size(data)
            print(len(data))
            # test_data = add_noise_data(data)

            test_data = np.array([data], dtype=np.float32)

            # plt.plot(range(len(test_data)), test_data)

            # plt.show()

            for one in range(len(test_data[0])):
                w_file.write(str(test_data[0][one]))
                w_file.write("  ")
            
            w_file.write("\n")

            print(len(test_data[0]))
            decoding_command(test_data)
            num = 0
        
    w_file.close()
    # w_info_file.close()

    return





#%%
def send_data(chunk, stream):
    stack = list()
    while True:
        data = stream.read(chunk, exception_on_overflow = False)
        data = np.frombuffer(data, 'int16')
        receive_data(data, stack)
        # print("sending data...")

    return





#%%
def record_voice():

    chunk = 400
    sample_format = pa.paInt16
    channels = 1
    sr = 16000

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(format=sample_format, channels=channels, rate=sr,
                    frames_per_buffer=chunk, input=True)

    data = list()

    send = threading.Thread(target=send_data, args=(chunk, stream))
    decoder = threading.Thread(target=receive_data, args=(data, stack))


    send.start()
    decoder.start()

    while True:
        # print("zot")
        time.sleep(0.1)

    # send.join()
    # decoder.join()


    return



#%%
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            print("%d : %6.2f %% <<< %s"%(i, j*100, command_label_dict[index_num]))
        else:
            print("%d : %6.2f %%"%(i, j*100))

    return







#%%
def decoding_command(test_data):
    global end_time

    print(test_data)

    command_interpreter.set_tensor(command_input_details['index'], test_data)
    command_interpreter.invoke()
    predictions = command_interpreter.get_tensor(command_output_details['index'])

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])
    print(predictions[0])

    print_result(a, predictions)

    end_time = time.time()

    print("decoding time : %f" %(end_time-start_time))

    return





#%%
def main():
    record_voice()








#%%
if __name__=='__main__':
    main()





## endl





