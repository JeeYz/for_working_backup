import cv2
import datetime
import threading
import time

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# capture = cv2.VideoCapture("*.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

cv2.namedWindow('test')

img_counter = 0
vid_counter = 0

# global frame

# control_para = 0

# lock = threading.Condition()

def video_control():

    # global record
    global img_counter, vid_counter, record
    # global control_para
    # control_para = 100
    # ret, frame = camera.read()

    while True:
        # lock.acquire()
        ret, frame = camera.read()

        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow('test', frame)
        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    return


def main():

    # global control_para

    # vid_control = threading.Thread(target=video_control)
    # vid_control.start()
    #
    # # lock.acquire()
    # while True:
    #     # lock.acquire()
    #     # control_para = input('input number : ')
    #     # lock.release()
    #     time.sleep(0.1)
    # # lock.release()
    # vid_control.join()

    global camera, fourcc, record
    while True:
        # lock.acquire()
        ret, frame = camera.read()

        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow('test', frame)
        # now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        cv2.waitKey(1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # if k%256 == 27: # press ESC
        #     print("Escape hit, closing...")
        #     break
        # elif k%256 == 49:
        #     img_name = "opencv_frame_{}.png".format(img_counter)
        #     cv2.imwrite(img_name, frame)
        #     print("{} written~!!".format(img_name))
        #     img_counter+=1
        #     vid_name = "opencv_frame_{}.png".format(img_counter)
        # elif k%256 == 32: # press space
        #
        #     print("녹화 시작")
        #     record = True
        #     video = cv2.VideoWriter(str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        #     # print("{} written~!!".format(vid_name))
        #     img_counter+=1
        # elif k & 0xFF == ord('q'):
        #     video.release()
        #     break

        # if record == True:
        #     print("녹화 중..")
        #     video.write(frame)

    camera.release()
    cv2.destroyAllWindows()


    return







##
if __name__ == '__main__':
    main()



## endl
