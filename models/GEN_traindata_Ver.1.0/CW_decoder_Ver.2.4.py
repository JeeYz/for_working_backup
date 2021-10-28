
from global_variables import *


stack = list()

num = 0
start_time, end_time = float(), float()

pady_size = 20

## global model
test_bool = False

## keyword global model
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
def return_min_max():
    temp = np.iinfo('int16')
    min_val = np.float32(temp.min)
    max_val = np.float32(temp.max)
    return min_val, max_val


#%%
def normalization_for_block(input_data):
    min_val, max_val = return_min_max()
    temp = (input_data-min_val)/(max_val-min_val)
    temp = np.array(temp, dtype=TRAIN_DATA_TYPE)
    return temp


#%%
def calculate_mean_val(input_data):
    return np.mean(np.abs(input_data))


#%%
def receive_data(data, stack):

    data = np.asarray(data, dtype=TRAIN_DATA_TYPE)
    mean_val = calculate_mean_val(data)
    norm_data = normalization_for_block(data)

    stack.extend(data)

    if len(stack) > TARGET_SAMPLE_RATE*(RECORDING_TIME+1):
        del stack[0:CHUNK_SIZE]

    cond1
    if float(VOICE_TRIGGER) <= float(mean_val) or num != 0:
        num+=1
        temp = 
        if num == 80:

            stack = standardization_func(stack)
            start_time = time.time()
            print(len(stack))

            data = trigal.signal_trigger_algorithm_for_decode(stack)

            print(len(data))

            test_data = np.array([data], dtype=np.float32)

            print(len(test_data[0]))
            decoding_command(test_data)
            num = 0
        
    return


#%%
def send_data(chunk, stream):
    stack = list()
    while True:
        data = stream.read(chunk, exception_on_overflow = False)
        data = np.frombuffer(data, 'int16')
        receive_data(data, stack)

    print("End send thread...")
    raise Exception("전송 스레드가 종료 됩니다.")


#%%
def record_voice():

    sample_format = pa.paInt16

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(
        format=sample_format, 
        channels=NUM_CHANNEL, 
        rate=TARGET_SAMPLE_RATE,
        frames_per_buffer=CHUNK_SIZE, 
        input=True,
    )

    data = list()

    send = threading.Thread(target=send_data, args=(CHUNK_SIZE, stream))
    decoder = threading.Thread(target=receive_data, args=(data, stack))

    send.start()
    decoder.start()

    while True:
        time.sleep(0.1)

    return


#%%
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            print("%d : %6.2f %% <<< %s"%(i, j*100))
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





