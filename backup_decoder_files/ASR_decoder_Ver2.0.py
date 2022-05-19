# -*- coding: utf-8 -*-

#%%
'''
record audio
send and receive data
'''


#%%
import pyaudio as pa
import wave
from socket import *
import time
import threading
import os
from tkinter import *
import subprocess
import psutil
import sys
import psutil

pady_size = 20

port_num = 5050
ip_address = "127.0.0.1"

test_while_num = 10

chunk = 2**10
sample_format = pa.paInt16
channels = 1
sr = 16000
start_time=time.time()

root = Tk()
print_text = StringVar()
label = Label(root, textvariable=print_text)
# mic_photo = PhotoImage(file='mic.png')
# img_label = Label(root, image=mic_photo)
mic_photo = PhotoImage(file='mic.png')
img_label = Label(root, image=mic_photo, borderwidth=0)


command_line_long = r"C:\\PNC\\ASR\\./online2-tcp-nnet3-decode-faster --feature-type=mfcc --add-pitch=false --min-active=2000 --max-active=4000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 --frame-subsampling-factor=3 --config=C:\\PNC\\ASR\\chain\\tdnn_1a_rvb_online\\conf\\online.conf --mfcc-config=C:\\PNC\\ASR\\chain\\tdnn_1a_rvb_online\\conf\\mfcc.conf C:\\PNC\\ASR\\chain\\tdnn_1a_rvb\\final.mdl C:\\PNC\\ASR\\chain\\tdnn_1a_rvb\\graph\\HCLG.fst C:\\PNC\\ASR\\chain\\tdnn_1a_rvb\\graph\\words.txt"


#%% command mapping
command_list = ['카메라',
                '전화',
                '위치',
                '앨범',
                '실행',
                '뒤로',
                '초기화',
                '촬영',
                '녹화',
                '중지',
                '받아',
                '거부',
                '밝게',
                '어둡게']

exit_command = '종료'
starting_keyword = '하이피앤씨'

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
def text_mapping(input_num):
    if input_num == "00":
        return_text = "명령어를 입력하세요..."
    elif input_num == '0':
        return_text = "카메라를 실행합니다..."
    elif input_num == '1':
        return_text = "전화를 실행합니다..."
    elif input_num == '2':
        return_text = "지도를 실행합니다..."
    elif input_num == '3':
        return_text = "앨범을 실행합니다..."
    elif input_num == '4':
        return_text = "프로그램을 실행합니다..."
    elif input_num == '5':
        return_text = "음성 인식 프로그램을 종료합니다..."
    elif input_num == '6':
        return_text = "뒤로~!!"
    elif input_num == '7':
        return_text = "상태를 초기화합니다..."
    elif input_num == '8':
        return_text = "촬영을 실행합니다..."
    elif input_num == '9':
        return_text = "녹화를 실행합니다..."
    elif input_num == '10':
        return_text = "프로세스를 중지합니다..."
    elif input_num == '11':
        return_text = "전화를 받겠습니다..."
    elif input_num == '12':
        return_text = "명령을 거부합니다..."
    elif input_num == '13':
        return_text = "밝기를 밝게 합니다..."
    elif input_num == '14':
        return_text = "밝기를 어둡게 합니다..."
    elif input_num == '-1':
        return_text = "시동어를 입력하세요..."

    return return_text



#%%
def kill_the_server():
    process_name = "online2-tcp-nnet3-decode-faster.exe"
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            proc.kill()


#%%
def kill_this_program():
    process_name = "PNC_ASR_Client_Ver1.2(GUI).exe"
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            proc.kill()


#%%
def run_gui():
    root.geometry('640x640')
    root.title('decoder')
    root.configure(bg='black')
    root.attributes('-fullscreen', True)

    program_name = Label(root, text = 'PNC ASR Client')
    program_name['fg'] = 'white'
    program_name['bg'] = 'black'
    program_name['font'] = 'Times 20 bold'
    program_name.pack(anchor = 'w', side='bottom')

    version_num = Label(root, text = 'Ver.1.2')
    version_num['fg'] = 'white'
    version_num['bg'] = 'black'
    version_num['font'] = 'Times 12 bold'
    version_num.pack(anchor = 'e', side='bottom')

    insert_command_list()

    root.mainloop()

    kill_this_program()


#%%
def print_text_to_gui(input_text):

    mappint_text = text_mapping(input_text)
    print_text.set(mappint_text)
    label.pack(pady=pady_size)

    if input_text == '00':
        img_label.pack(side='bottom')
        start_time = time.time()
        print(start_time)
    elif input_text == '5':
        time.sleep(2)
        kill_the_server()
        root.destroy()

    else:
        img_label.pack_forget()
        start_time = time.time()

    return


#%%
def run_server():
    subprocess.run(command_line_long, shell=True)
    print("running server")



#%%
def send(sock, p):
    # global
    stream = p.open(format=sample_format, channels=channels, rate=sr,
                        frames_per_buffer=chunk, input=True)

    while True:
        data = stream.read(chunk)
        sock.sendall(data)
        if time.time() - start_time > 8:
            print_text.set("시동어를 입력하세요...")
            label.pack(pady=pady_size)

    stream.stop_stream()
    stream.close()


#%%
def receive(sock):
    global start_time
    while True:
        start_time = time.time()
        data = sock.recv(1024)
        print(start_time)
        print("response time : ", time.time() - start_time)
        print("received data : ", data.decode('utf-8'))
        if data.decode('utf-8') == '\n':
            continue
        print_text_to_gui(data.decode('utf-8'))


#%%
def record_voice():

    p = pa.PyAudio()

    clientsocket = socket(AF_INET, SOCK_STREAM)
    clientsocket.connect((ip_address, port_num))
    #
    print('Recording start')

    # label_0 = Label(root, text='Recording start')
    # label_0.pack()
    print_text.set("시동어를 입력하세요...")
    label['fg'] = 'white'
    label['bg'] = 'black'
    label['font'] = 'Times 25 bold'
    label.pack(anchor='center', pady=pady_size)


    sender = threading.Thread(target=send, args=(clientsocket, p))
    receiver = threading.Thread(target=receive, args=(clientsocket,))

    sender.start()
    receiver.start()

    # run_gui = threading.Thread(target=run_gui, args=())
    # run_gui.start()
    run_gui()

    # while True:
    #     time.sleep(0.1)

    p.terminate()
    clientsocket.close()

    return


#%%
def main():

    run_ser = threading.Thread(target=run_server, args=())
    run_ser.daemon = True
    run_ser.start()

    print("waiting for running server...")
    time.sleep(10)
    print("please press decoding button")
    # print(f.read())

    record_voice()



    print("good bye, world~!! for D.Ritchie")

    while True:
        time.sleep(0.1)


#%%
if __name__=='__main__':
    print('hello, world~!~!')
    main()
    print("good bye, world~!! for D.Ritchie")
