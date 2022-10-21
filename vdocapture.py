import cv2
import queue
import threading
from multiprocessing.pool import ThreadPool
from footwear_main import Footwear


q = queue.Queue()
pool = ThreadPool(processes=100)


class VideoCapture:

    def __init__(self,url):
        self.cap = cv2.VideoCapture(url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def play_video(url):

    cap_line = VideoCapture(url)
    while True:
        images = cap_line.read()
        imgs = [None] * 1
        imgs[0] = images
        img0 = imgs.copy()
        # print("start:",datetime.datetime.now())
        face_dt = pool.apply_async(Footwear(images,img0).detect,)
        out = face_dt.get()
        # print("end:", datetime.datetime.now())
        resize = cv2.resize(out, (640, 480))
        cv2.imshow(url, resize)
        cv2.waitKey(1)
if __name__ == '__main__':

    arr = [
        {"name": "rtsp1", "url": r"rtsp://admin:Admin123$@10.11.25.60:554/stream1"},
        # {"name": "rtsp2","url": "0"},
        # {"name": "rtsp3","url": r"D:\videos\face_em.mp4"},
        # {"name": "rtsp4","url": r"rtsp://admin:Admin123$@10.11.25.65:554/stream1"},
        # {"name": "rtsp5", "url": r"rtsp://admin:Admin123$@10.11.25.60:554/stream1"},
        # {"name": "rtsp6", "url": r"rtsp://admin:Admin123$@10.11.25.62:554/stream1"}

    ]
    for i in arr:
        url = i['url']
        name = i["name"]
        t1 = threading.Thread(target=play_video, args=(url,))
        t1.start()



