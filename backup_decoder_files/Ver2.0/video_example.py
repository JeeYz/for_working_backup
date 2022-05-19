import cv2
import datetime

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# capture = cv2.VideoCapture("*.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

cv2.namedWindow('test')

img_counter = 0
vid_counter = 0

def main():
    global img_counter, vid_counter, record
    while True:
        ret, frame = camera.read()

        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow('test', frame)
        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        k = cv2.waitKey(1)

        if k%256 == 27: # press ESC
            print("Escape hit, closing...")
            break
        elif k%256 == 49:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written~!!".format(img_name))
            img_counter+=1
            vid_name = "opencv_frame_{}.png".format(img_counter)
        elif k%256 == 32: # press space

            print("녹화 시작")
            record = True
            video = cv2.VideoWriter(str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            # print("{} written~!!".format(vid_name))
            img_counter+=1
        elif k & 0xFF == ord('q'):
            video.release()
            break

        if record == True:
            print("녹화 중..")
            video.write(frame)

    camera.release()
    cv2.destroyAllWindows()
    return







##
if __name__ == '__main__':
    main()



## endl
