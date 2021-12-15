
from os import write
from numpy.core.defchararray import index
from numpy.core.fromnumeric import mean
from global_variables import *

start_time, end_time = float(), float()
pady_size = 20

start_time = None

## global model

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
    temp = (temp-0.5)*2.0
    temp = np.array(temp, dtype=TRAIN_DATA_TYPE)
    return temp


#%%
def calculate_mean_val(input_data):
    return np.mean(np.abs(input_data))


def write_npz(input_data, filepath):
    np.savez(filepath, data=input_data)


#%%
def receive_data(data, stack):
    global start_time

    data = np.asarray(data, dtype=TRAIN_DATA_TYPE)
    # norm_data = normalization_for_block(data)
    # std_data = standardization_func(norm_data)
    mean_val = calculate_mean_val(data)

    stack.extend(data)

    if len(stack) > TARGET_SAMPLE_RATE*(RECORDING_TIME+2):
        del stack[0:CHUNK_SIZE]

    num = GLOBAL_DECODING_DATA.condition_num
    # print(mean_val, '\t', len(stack))

    if float(VOICE_TRIGGER) <= float(mean_val) or num != 0:
    # if float(STD_TRIGGER) <= float(mean_val) or num != 0:
        GLOBAL_DECODING_DATA.add_a_sec_condition()
        num = GLOBAL_DECODING_DATA.condition_num
        if num == RETURN_STACK_SIZE:
            start_time = time.time()
            # write_npz(stack, "D:\\stack_log_for_graph.npz")

            GLOBAL_DECODING_DATA.set_target_data(stack)
            GLOBAL_DECODING_DATA.standardization_data()

            print(len(stack))

            # write_numpy_for_draw_graph([stack])
            
            data = GLOBAL_DECODING_DATA.get_target_data()
            data = trigal.signal_trigger_algorithm_for_decode(data)
            # write_npz(data, "D:\\data_log_for_graph.npz")
            print(len(data))

            test_data = np.array([data], dtype=np.float32)

            print(len(test_data[0]))
            decoding_command(test_data)

            GLOBAL_DECODING_DATA.set_condition_num_zero()  
    
    GLOBAL_DECODING_DATA.set_none_target_data()
    GLOBAL_DECODING_DATA.reset_stack_data()
    # GLOBAL_DECODING_DATA.set_none_stack_data()
        
    return


#%%
def send_data(stream, stack):
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow = False)
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

    send = threading.Thread(
        target=send_data, 
        args=(stream, GLOBAL_DECODING_DATA.stack_data)
    )
    decoder = threading.Thread(
        target=receive_data, 
        args=(data, GLOBAL_DECODING_DATA.stack_data)
    )

    send.start()
    decoder.start()

    # while True:
    #     time.sleep(0.1)

    return


#%%
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            for one in LabelsKorEng:
                if one.value == index_num:
                    label_name = one
                    break
                else:
                    label_name = "None"
            print("%d : %6.2f %% << %d"%(i, j*100, index_num), label_name)
        else:
            print("%d : %6.2f %%"%(i, j*100))

    return


#%%
def write_numpy_for_draw_graph(input_data):
    target_path = CWdata_path+'\\'+'test_npz_data'+'\\'
    filename = str(time.time())
    target_file = target_path+filename
    np.savez(
        target_file,
        data=input_data,
        label = 'No Label',
    )

    return



#%%
def decoding_command(test_data):
    global end_time
    global start_time

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
    start_time = None

    # write_numpy_for_draw_graph(test_data)

    return








#%%
def main():
    record_voice()








#%%
if __name__=='__main__':
    main()





## endl





