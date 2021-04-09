from threading import Thread, Lock
import cv2
import datetime
import time
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

class WebcamVideoStream :
    def __init__(self) :
        self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
        cv2.imshow('webcam', frame)

        now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        k = cv2.waitKey(10)
        if k == 27 :
            break
        elif k == 49:
            img_name = str(now) + ".png"
            cv2.imwrite(img_name, frame)
            print("{} written~!!".format(img_name))
        elif k == 32: # press space

            print("recording start!!")
            record = True
            video = cv2.VideoWriter(str(now) + ".avi", fourcc, 60, (frame.shape[1], frame.shape[0]))
            # print("{} written~!!".format(vid_name))
        elif k & 0xFF == ord('q'):
            video.release()
            break

        if record == True:
            print("now recording...")
            video.write(frame)


    vs.stop()
    cv2.destroyAllWindows()
