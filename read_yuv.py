import cv2
import numpy as np

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)
        

    def read_raw(self):
        try:
            raw = self.f.read(int(self.frame_len))
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        
        return ret, bgr


if __name__ == "__main__":
    #filename = "bctest.raw"
    #filename = "binary.raw"
    #filename = "bridge-close_cif.yuv"
    filename = "_onlyY_BW.yuv"
    #filename = "bctest_2.raw"

    filename = input("input YUV filename : ")
    size = (288, 352)
    cap = VideoCaptureYUV(filename, size)


    while 1:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("frame", frame)
            cv2.imwrite('grayYUV.png', frame)
            cv2.waitKey(27)
        else:
            break