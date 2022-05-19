import cv2
import datetime
import threading
import time

camera_bool = False

img_counter = 0
vid_counter = 0

control_para = 100

lock_frame = threading.Lock()
lock_var = threading.Lock()

def video_control():

    # global record
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
            # lock_var.acquire()
            control_para = 100
            # lock_var.release()
        elif control_para==2: # press space
            print("recoding start!!")
            record = True
            vid_name = str(now) + ".avi"
            video = cv2.VideoWriter(vid_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            # lock_var.acquire()
            control_para = 100
            # lock_var.release()
        elif control_para==3:
            video.release()
            # lock_var.acquire()
            control_para = 100
            record = False
            # lock_var.release()
            break
        lock_var.release()


        if record == True:
            print("now recording...")
            video.write(frame)

    camera.release()
    cv2.destroyAllWindows()

    return


def main():

    global control_para

    vid_control = threading.Thread(target=video_control)
    vid_control.start()

    # lock.acquire()
    while True:
        # frame = get_frame()
        # cv2.imshow('test', frame)
        # now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        #
        # cv2.waitKey(1)
        # lock_var.acquire()
        control_para = int(input('input number : '))
        # print(control_para)
        # lock_var.release()
        # time.sleep(0.001)
    # lock.release()
    vid_control.join()


    return







##
if __name__ == '__main__':
    main()



## endl
