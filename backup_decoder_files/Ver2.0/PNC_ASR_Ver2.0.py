#%%
import threading

import numpy as np
import tensorflow as tf

import time

from tkinter import *
import pyaudio as pa
import psutil

from python_speech_features import logfbank

import cv2
import datetime

stack = list()
trigger_val = 150

sample_rate = 16000
recording_time = 3
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000
voice_size = 2*sample_rate
chunk = 400
per_sec = sample_rate/chunk
num = 0
start_time, end_time = float(), float()
global_flag = 0


keyword_label_dict = {0: 'None',
            1: 'hipnc'}

command_label_dict = {0: 'None',
            1: 'call', # 전화
            2: 'camera', # 카메라
            3: 'picture', # 촬영
            4: 'record', # 녹화
            5: 'stop', # 중지
            6: 'end'} # 종료

command_list = ['전화',
                '카메라',
                '촬영',
                '녹화',
                '중지']

exit_command = '종료'
starting_keyword = '하이피앤씨'


#%%
def text_mapping(input_num):
    if input_num == 1:
        return_text = "전화를 실행합니다..."
    elif input_num == 2:
        return_text = "카메라를 실행합니다..."
    elif input_num == 3:
        return_text = "현재 화면을 촬영합니다..."
    elif input_num == 4:
        return_text = "녹화를 실행합니다..."
    elif input_num == 5:
        return_text = "녹화를 중지합니다..."
    elif input_num == 6:
        return_text = "음성 인식 프로그램을 종료합니다..."
    elif input_num == 100:
        return_text = "시동어를 다시 입력하세요..."
    elif input_num == 10:
        return_text = "명령어를 입력하세요..."
    elif input_num == 0:
        return_text = "시동어를 다시 입력하세요..."
    elif input_num == 1000:
        return_text = "녹화 중입니다."
    elif input_num == 2000:
        return_text = "이미지 파일로 저장합니다."
    elif input_num == 3000:
        return_text = "카메라가 실행 중입니다."
    elif input_num == 4000:
        return_text = "동영상 파일로 저장합니다."
    elif input_num == 5000:
        return_text = "카메라를 종료합니다."
    elif input_num == 6000:
        return_text = "카메라가 실행되지 않았습니다."
    elif input_num == 7000:
        return_text = "현재 녹화 중이지 않습니다."
    elif input_num == 200:
        return_text = "시동어를 입력하세요."
    return return_text


#%%
pady_size = 20

root = Tk()
print_text = StringVar()
label = Label(root, textvariable=print_text)
# mic_photo = PhotoImage(file='mic.png')
# img_label = Label(root, image=mic_photo)
mic_photo = PhotoImage(file='mic.png')
img_label = Label(root, image=mic_photo, borderwidth=0)

## global model
conv_shape = (199, 26, 1)

test_bool = False

## keyword global model
keyword_label_num = 2
if test_bool == True:
    keyword_tflite_file = 'D:\\new_ver_train_data\\keyword_model_parameter.tflite'
elif test_bool == False:
    keyword_tflite_file = 'keyword_model_parameter.tflite'
keyword_interpreter = tf.lite.Interpreter(model_path=keyword_tflite_file)
keyword_interpreter.allocate_tensors()
keyword_input_details = keyword_interpreter.get_input_details()[0]
keyword_output_details = keyword_interpreter.get_output_details()[0]

## global command
command_label_num = 7
if test_bool == True:
    command_tflite_file = 'D:\\new_ver_train_data\\command_model_parameter.tflite'
elif test_bool == False:
    command_tflite_file = 'command_model_parameter.tflite'
command_interpreter = tf.lite.Interpreter(model_path=command_tflite_file)
command_interpreter.allocate_tensors()
command_input_details = command_interpreter.get_input_details()[0]
command_output_details = command_interpreter.get_output_details()[0]

## call confirm
call_label_num = 2
if test_bool == True:
    call_tflite_file = 'D:\\new_ver_train_data\\call_model_parameter.tflite'
elif test_bool == False:
    call_tflite_file = 'call_model_parameter.tflite'
call_interpreter = tf.lite.Interpreter(model_path=call_tflite_file)
call_interpreter.allocate_tensors()
call_input_details = call_interpreter.get_input_details()[0]
call_output_details = call_interpreter.get_output_details()[0]

## camera confirm
camera_label_num = 2
if test_bool == True:
    camera_tflite_file = 'D:\\new_ver_train_data\\camera_model_parameter.tflite'
elif test_bool == False:
    camera_tflite_file = 'camera_model_parameter.tflite'
camera_interpreter = tf.lite.Interpreter(model_path=camera_tflite_file)
camera_interpreter.allocate_tensors()
camera_input_details = camera_interpreter.get_input_details()[0]
camera_output_details = camera_interpreter.get_output_details()[0]

## picture confirm
picture_label_num = 2
if test_bool == True:
    picture_tflite_file = 'D:\\new_ver_train_data\\picture_model_parameter.tflite'
elif test_bool == False:
    picture_tflite_file = 'picture_model_parameter.tflite'
picture_interpreter = tf.lite.Interpreter(model_path=picture_tflite_file)
picture_interpreter.allocate_tensors()
picture_input_details = picture_interpreter.get_input_details()[0]
picture_output_details = picture_interpreter.get_output_details()[0]

## record confirm
record_label_num = 2
if test_bool == True:
    record_tflite_file = 'D:\\new_ver_train_data\\record_model_parameter.tflite'
elif test_bool == False:
    record_tflite_file = 'record_model_parameter.tflite'
record_interpreter = tf.lite.Interpreter(model_path=record_tflite_file)
record_interpreter.allocate_tensors()
record_input_details = record_interpreter.get_input_details()[0]
record_output_details = record_interpreter.get_output_details()[0]

## stop confirm
stop_label_num = 2
if test_bool == True:
    stop_tflite_file = 'D:\\new_ver_train_data\\stop_model_parameter.tflite'
elif test_bool == False:
    stop_tflite_file = 'stop_model_parameter.tflite'
stop_interpreter = tf.lite.Interpreter(model_path=stop_tflite_file)
stop_interpreter.allocate_tensors()
stop_input_details = stop_interpreter.get_input_details()[0]
stop_output_details = stop_interpreter.get_output_details()[0]

## end confirm
end_label_num = 2
if test_bool == True:
    end_tflite_file = 'D:\\new_ver_train_data\\end_model_parameter.tflite'
elif test_bool == False:
    end_tflite_file = 'end_model_parameter.tflite'
end_interpreter = tf.lite.Interpreter(model_path=end_tflite_file)
end_interpreter.allocate_tensors()
end_input_details = end_interpreter.get_input_details()[0]
end_output_details = end_interpreter.get_output_details()[0]


#%% about video
record = False
camera_bool = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
control_para = -1

lock_frame = threading.Lock()
lock_var = threading.Lock()

#%%
def video_thread():

    global record
    global control_para
    global camera_bool

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    record = False
    camera_bool = True

    while True:
        # lock.acquire()
        lock_frame.acquire()
        ret, frame = camera.read()
        lock_frame.release()
        # print(control_para)
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow('test', frame)
        now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        cv2.waitKey(10)

        lock_var.acquire()
        if control_para==0: # press ESC
            print("Escape hit, closing...")
            break

        elif control_para==1:
            img_name = "images\\" + str(now) + '.png'
            cv2.imwrite(img_name, frame)
            print("{} written~!!".format(img_name))
            control_para = -1
            time.sleep(1)
            print_text_to_gui(2000)

        elif control_para==2: # press space
            print("recording start!!")
            record = True
            vid_name = "videos\\" + str(now) + ".avi"
            video = cv2.VideoWriter(vid_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            control_para = -1

        elif control_para==3:
            video.release()
            control_para = -1
            record = False
            time.sleep(1)
            print_text_to_gui(4000)

        elif control_para==4:
            camera_bool = False
            control_para = -1
            break
        lock_var.release()

        if record == True:
            print("now recording...")
            video.write(frame)

    print_text_to_gui(5000)
    camera.release()
    cv2.destroyAllWindows()

    return


#%%
vid_control = threading.Thread(target=video_thread)
def video_control(command_para):

    global control_para
    global vid_control

    if command_para == 2:
        vid_control.start()
    elif command_para == 3:
        control_para = 1
    elif command_para == 4:
        control_para = 2
    elif command_para == 5:
        control_para = 3
    elif command_para == 6:
        control_para = 4
        vid_control.join()

    return


#%%
def print_text_to_gui(input_num):

    mappint_text = text_mapping(input_num)
    print_text.set(mappint_text)
    label.pack(pady=pady_size)

    if input_num == 10:
        img_label.pack(side='bottom')

    elif input_num == 6:
        time.sleep(2)
        kill_this_program()
        root.destroy()

    else:
        img_label.pack_forget()

    return



##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


##
def evaluate_mean_of_frame(data, **kwargs):

    if "frame_time" in kwargs.keys():
        frame_time = kwargs['frame_time']
    if "shift_time" in kwargs.keys():
        shift_time = kwargs['shift_time']
    if "sample_rate" in kwargs.keys():
        sr = kwargs['sample_rate']
    if "buffer_size" in kwargs.keys():
        buf_size = kwargs['buffer_size']
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    if "threshold_value" in kwargs.keys():
        trigger_val = kwargs['threshold_value']

    frame_size = int(sr*frame_time)
    shift_size = int(sr*shift_time)

    num_frames = len(data)//(frame_size-shift_size)+1

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    for i,end in enumerate(reversed(mean_val_list)):
        if trigger_val < end:
            end_index = len(mean_val_list)-i
            break
        else:
            end_index = len(mean_val_list)

    temp = (frame_size-shift_size)*start_index-buf_size
    if temp <= 0:
        temp = 0

    result = data[temp:(frame_size-shift_size)*end_index+buf_size]

    if full_size > len(result):
        result = fit_determined_size(result, full_size=full_size)
        result = add_noise_data(result, full_size=full_size)
    elif full_size == len(result):
        result = add_noise_data(result, full_size=full_size)
    else:
        result = result[:full_size]
        result = add_noise_data(result, full_size=full_size)

    return result


def fit_determined_size(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    result = np.append(data, np.zeros(full_size-len(data)))

    return result


##
def add_noise_data(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']

    noise_data = np.random.randn(full_size)*0.01
    result = data+noise_data

    return result


##
def generate_train_data(data):

    data = np.asarray(data)

    logfb_feat = logfbank(data)
    logfb_feat = standardization_func(logfb_feat)

    train_feats = tf.expand_dims(logfb_feat, -1)

    train_feats = np.float32(train_feats)
    return np.array([train_feats])


##
def receive_data(data, stack):

    global start_time
    global num, global_flag

    mean_val = np.mean(np.abs(data))

    stack.extend(data)

    if len(stack) > sample_rate*(recording_time+1):
        del stack[0:chunk]

    if global_flag == 0:
        if float(trigger_val) <= float(mean_val) or num != 0:
            # print(num, mean_val)
            num+=1
            # print(num, mean_val)
            if num == 80:
                # print(num, mean_val)
                start_time = time.time()
                stack = standardization_func(stack)
                data = evaluate_mean_of_frame(stack, frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*2,
                                            threshold_value=0.5)

                test_data = generate_train_data(data)

                decoding_keyword(test_data)
                num = 0

    elif global_flag ==1:
        num+=1
        if num == 120:
            # print(num, mean_val)
            start_time = time.time()
            stack = standardization_func(stack)
            data = evaluate_mean_of_frame(stack, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*2,
                                        threshold_value=0.5)


            test_data  = generate_train_data(data)

            decoding_command(test_data)
            num = 0

    else:
        print("error!!")
        time.sleep(1000)


    return


##
def send_data(chunk, stream):
    stack = list()
    while True:
        data = stream.read(chunk, exception_on_overflow = False)
        data = np.frombuffer(data, 'int16')
        receive_data(data, stack)

    return

##
def record_voice():

    chunk = 400
    sample_format = pa.paInt16
    channels = 1
    sr = 16000

    seconds = 1

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(format=sample_format, channels=channels, rate=sr,
                    frames_per_buffer=chunk, input=True)

    data = list()

    print_text.set("시동어를 입력하세요...")
    label['fg'] = 'white'
    label['bg'] = 'black'
    label['font'] = 'Times 25 bold'
    label.pack(anchor='center', pady=pady_size)

    send = threading.Thread(target=send_data, args=(chunk, stream))
    decoder = threading.Thread(target=receive_data, args=(data, stack))

    send.start()
    decoder.start()

    run_gui()

    while True:
        time.sleep(0.1)

    send.join()
    decoder.join()

    stream.stop_stream()
    stream.close()

    return



##
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            print("%d : %6.2f %% <<< %s"%(i, j*100, label_dict[index_num]))
        else:
            print("%d : %6.2f %%"%(i, j*100))

    return


##
def decoding_keyword(test_data):
    global end_time, global_flag

    keyword_interpreter.set_tensor(keyword_input_details['index'], test_data)
    keyword_interpreter.invoke()
    predictions = keyword_interpreter.get_tensor(keyword_output_details['index'])

    end_time = time.time()

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])

    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print_text_to_gui(100)
        print("it's not keyword...")
    else:
        global_flag = 1
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print_text_to_gui(10)
        print("please, input command...")

    print("decoding time : %f" %(end_time-start_time))
    return


##
def return_label_number(predictions):
    label_number = 0
    max_temp = 0
    for i,j in enumerate(predictions):
        if i == 0:
            continue
        else:
            if j > max_temp:
                max_temp = j
                label_number = i

    return label_number


##
def make_a_decision(para):
    global camera_bool, record

    if para == 1:
        print_text_to_gui(para)
    elif para == 2:
        if camera_bool==True:
            print_text_to_gui(3000)
        else:
            video_control(para)
            print_text_to_gui(para)
    elif para == 3:
        if camera_bool==False:
            print_text_to_gui(6000)
        else:
            print_text_to_gui(para)
            video_control(para)
    elif para == 4:
        if camera_bool == False:
            print_text_to_gui(6000)
        elif camera_bool == True and record == True:
            print_text_to_gui(1000)
        else:
            video_control(para)
            print_text_to_gui(para)
    elif para == 5:
        if camera_bool == False:
            print_text_to_gui(6000)
        elif camera_bool == True and record == False:
            print_text_to_gui(7000)
        else:
            print_text_to_gui(para)
            video_control(para)
    elif para == 6:
        if camera_bool == False:
            print_text_to_gui(para)
        else:
            print_text_to_gui(5000)
            video_control(para)
            print_text_to_gui(200)

    return


##
def decoding_command(test_data):
    global end_time, global_flag
    global camera_bool, control_para, record

    command_interpreter.set_tensor(command_input_details['index'], test_data)
    command_interpreter.invoke()
    predictions = command_interpreter.get_tensor(command_output_details['index'])

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])
    print(predictions[0])

    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
        if predictions[0][a] != 1.0:
            label_para = return_label_number(predictions[0])
            confirm_command(test_data, label_para)
        else:
            print_text_to_gui(0)
        global_flag = 0
    else:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
################################################### 수정필요
        make_a_decision(a)
################################################### 수정필요
        global_flag = 0


    end_time = time.time()

    print("decoding time : %f" %(end_time-start_time))

    return


##
def confirm_command(test_data, label_para):
    global end_time, global_flag

    if label_para == 1:
        call_interpreter.set_tensor(call_input_details['index'], test_data)
        call_interpreter.invoke()
        predictions = call_interpreter.get_tensor(call_output_details['index'])
    elif label_para == 2:
        camera_interpreter.set_tensor(camera_input_details['index'], test_data)
        camera_interpreter.invoke()
        predictions = camera_interpreter.get_tensor(camera_output_details['index'])
    elif label_para == 3:
        picture_interpreter.set_tensor(picture_input_details['index'], test_data)
        picture_interpreter.invoke()
        predictions = picture_interpreter.get_tensor(picture_output_details['index'])
    elif label_para == 4:
        record_interpreter.set_tensor(record_input_details['index'], test_data)
        record_interpreter.invoke()
        predictions = record_interpreter.get_tensor(record_output_details['index'])
    elif label_para == 5:
        stop_interpreter.set_tensor(stop_input_details['index'], test_data)
        stop_interpreter.invoke()
        predictions = stop_interpreter.get_tensor(stop_output_details['index'])
    elif label_para == 6:
        end_interpreter.set_tensor(end_input_details['index'], test_data)
        end_interpreter.invoke()
        predictions = end_interpreter.get_tensor(end_output_details['index'])

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])


    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
        print_text_to_gui(0)
    else:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[label_para])
        make_a_decision(label_para)

    global_flag = 0

    return


#%%
def kill_this_program():
    process_name = "PNC_ASR_Ver2.0.exe"
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            proc.kill()


#%%
def insert_command_list():

    new_command_list = '명령어:\n' + '\n'.join(command_list)
    new_command_list += '\n\n시동어:\n'+starting_keyword
    new_command_list += '\n\n'+'종료 명령어:'+'\n'+exit_command

    command_label = Label(root, text=new_command_list)
    command_label['fg'] = 'white'
    command_label['bg'] = 'black'
    command_label['font'] = 'Times 10 bold'
    command_label.pack(side='right')

    return


#%%
def run_gui():
    root.geometry('640x640')
    root.title('decoder')
    root.configure(bg='black')
    root.attributes('-fullscreen', True)

    program_name = Label(root, text = 'PNC ASR Client DL Ver.')
    program_name['fg'] = 'white'
    program_name['bg'] = 'black'
    program_name['font'] = 'Times 20 bold'
    program_name.pack(anchor = 'w', side='bottom')

    version_num = Label(root, text = 'Ver.2.0')
    version_num['fg'] = 'white'
    version_num['bg'] = 'black'
    version_num['font'] = 'Times 12 bold'
    version_num.pack(anchor = 'e', side='bottom')

    insert_command_list()

    root.mainloop()

    kill_this_program()



#%%
def main():
    record_voice()








#%%
if __name__=='__main__':
    main()





## endl
