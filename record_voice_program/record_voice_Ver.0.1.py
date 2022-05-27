
from cProfile import label
import datetime
from os import write
from numpy import percentile
from numpy.core.defchararray import index
from numpy.core.fromnumeric import mean
from global_variables import *

from tkinter import *

GUI_ROOT = Tk()
print_text = StringVar()
text_gui = Label(GUI_ROOT, textvariable=print_text)
text_gui['fg'] = 'white'
text_gui['bg'] = 'black'
text_gui['font'] = 'Times 50 bold'

default_text = "명령어를 다시 입력하세요."

mic_photo = PhotoImage(file='mic.png')
img_label = Label(GUI_ROOT, image=mic_photo, borderwidth=0)

start_time, end_time = float(), float()
pady_size = 20
threshold = 0.80

start_time = None

global_label_dict = {
    0: ["None", "none"],   
    1: ["선택", 'choice'],
    2: ["클릭", 'click'],
    3: ["닫기", 'close'], 
    4: ["홈", 'home'],
    5: ["종료", 'end'],
    6: ["어둡게", 'darken'],
    7: ["밝게", 'brighten'],
    8: ["음성 명령어", 'voice_command'],
    9: ["촬영", 'picture'],
    10:[ "녹화", 'record'],
    11:[ "정지", 'stop'],
    12:[ "아래로", 'below'],
    13:[ "위로", 'up'],
    14:[ "다음", 'next'],
    15:[ "이전", 'previous'],
    16:[ "재생", 'play'],
    17:[ "되감기", 'rewind'],
    18:[ "빨리감기", 'ff'],
    19:[ "처음", 'init'],
    20:[ "소리 작게", 'volume_down'],
    21:[ "소리 크게", 'volume_up'],
    22:[ "화면 크게", 'screen_big'],
    23:[ "화면 작게", 'screen_small'],
    24:[ "전체 화면", 'full_screen'],
    25:[ "이동", 'move'],
    26:[ "멈춤", 'hold'],
    27:[ "모든 창 보기", 'show_all_windows'],
    28:[ "전화", 'phone'],
    29:[ "통화", 'call'],
    30:[ "수락", 'accept'],
    31:[ "거절", 'reject'],  
}




#%%
def receive_data(data, stack):
    global start_time
    # global file_num
    # global temp_filename

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

        if num == 50:
            print_decoding_screen()

        if num == RETURN_STACK_SIZE:
            start_time = time.time()
            # write_npz(stack, "D:\\stack_log_for_graph.npz")

            stack = np.asarray(stack, dtype=np.int16)

            # temp = temp_filename+str(file_num)+".wav"
            # wavfile.write(temp, TARGET_SAMPLE_RATE, stack)
            # file_num+=1

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

    run_gui()

    # while True:
    #     time.sleep(0.1)

    return


#%%
def print_result(index_num, output_data):

    print(output_data[0][index_num])

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            for one in LabelsKorEng:
                if one.value == index_num:
                    if output_data[0][index_num] > threshold:
                        label_name = one
                        break
                    else:
                        label_name = "None"
                        break
                else:
                    label_name = "None"
            print("%d : %6.2f %% << "%(i, j*100), label_name)
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
def writeLogFile(labelIndex, predictions, runningTime):
    now = datetime.datetime.now()

    if (labelIndex==0):
        label_name = "NONE"
    else:
        for one in LabelsKorEng:
            if one.value == labelIndex:
                label_name = one
                label_name = str(label_name).split(".")[-1]
                break

    percentageLabel = "{:.2f}".format(predictions[0][labelIndex]*100)
    runningTime = "{:.4f}".format(runningTime)
    
    writeLine = \
        "Time : {time}, Label : {label}, Word : {word}, Percentage = {percentage}%, RunTime : {runtime}sec"\
        .format(time=now.replace(microsecond=0), label=labelIndex,
                percentage=percentageLabel, 
                word=label_name, runtime=runningTime)

    with open(outputLogFile, 'a', encoding='utf-8') as f:
        f.write(writeLine)
        f.write("\n")
    


#%%
def draw_graph_raw_signal(data, **kwargs):

    title_name = kwargs['title_name']

    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title(title_name)

    # plt.tight_layout()
    # plt.show()

    return


#%%
def modifying_label(input_index, input_data):
    result_label = 0

    if input_data[0][input_index] < threshold:
        result_label = 0
    else:
        result_label = input_index

    return result_label


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

    mod_index = modifying_label(a, predictions)
    
    print_text.set(text_mapping(mod_index))
    text_gui.pack(pady=pady_size)

    end_time = time.time()

    print("decoding time : %f" %(end_time-start_time))
    runningTime = end_time-start_time
    start_time = None
    
    writeLogFile(a, predictions, runningTime)

    # if mod_index != 0:
    #     time.sleep(1)

    # text_gui.pack_forget()
    # print_second_default()

    ## 수정필요
    # draw_graph_raw_signal(test_data, "result data")

    # write_numpy_for_draw_graph(test_data)

    return

    

#%% text mapping
def text_mapping(input_index):
    return_text = str()
    tail_words = " 명령어가 입력되었습니다."

    if input_index == 0:
        return_text = "명령어를 다시 입력하세요."
    else:
        return_text = global_label_dict[input_index]+tail_words

    return return_text

    

#%%
def insert_command_list():
    command_list = list()

    for i,c in enumerate(global_label_dict):
        if i == 0: continue
        command_list.append(global_label_dict[i])

    new_command_list = '명령어:\n' + '\n'.join(command_list)

    command_label = Label(GUI_ROOT, text=new_command_list)
    command_label['fg'] = 'white'
    command_label['bg'] = 'black'
    command_label['font'] = 'Times 13 bold'
    command_label.pack(side='right')

    return


#%%
def print_default_screen():
    print_text.set("명령어를 입력해주세요.")
    text_gui.pack(pady=pady_size)
    
    img_label.pack(side='bottom')

    return

    
#%%
def print_second_default():
    print_text.set(default_text)
    text_gui.pack(pady=pady_size)
    
    img_label.pack(side='bottom')

    return


#%%
def print_decoding_screen():
    text_gui.pack_forget()
    img_label.pack_forget()

    print_text.set("디코딩중...")    
    text_gui.pack(pady=pady_size)
    
    return



#%% run gui
def run_gui():
    GUI_ROOT.geometry('640x640')
    GUI_ROOT.title('decoder')
    GUI_ROOT.configure(bg='black')
    GUI_ROOT.attributes('-fullscreen', True)

    # print_text.set(default_text)
    # text_gui.pack(pady=pady_size)

    program_name = Label(GUI_ROOT, text = 'PNC Command Recognition System Ver.2.4')
    program_name['fg'] = 'white'
    program_name['bg'] = 'black'
    program_name['font'] = 'Times 20 bold'
    program_name.pack(anchor = 'w', side='bottom')

    version_num = Label(GUI_ROOT, text = 'Ver.2.4')
    version_num['fg'] = 'white'
    version_num['bg'] = 'black'
    version_num['font'] = 'Times 12 bold'
    version_num.pack(anchor = 'e', side='bottom')

    # img_label.pack(side='bottom')

    insert_command_list()
    print_default_screen()

    GUI_ROOT.mainloop()
    
    return





#%%
def main():
    record_voice()
    return








#%%
if __name__=='__main__':
    main()





## endl





