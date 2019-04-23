import cv2
import imutils
import numpy as np


class VideoGenerator:

    frame: bytes = None
    net = cv2.dnn.readNetFromCaffe('/Users/pavel/PycharmProjects/ServerSecurity/caffemodel/deploy.prototxt.txt',
                                   '/Users/pavel/PycharmProjects/ServerSecurity/caffemodel/res10_300x300_ssd_iter_140000.caffemodel')

    @staticmethod
    def kernel(data: bytes):
        if data is not None:
            print('kernel')
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
            frame = np.array(decoded)
            frame = imutils.resize(frame, width=1000)
            frame = imutils.rotate(frame, 270)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            VideoGenerator.net.setInput(blob)
            detections = VideoGenerator.net.forward()
            for i in range(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence < 0.5:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            return cv2.imencode('.jpg', frame)[1].tobytes()

    @staticmethod
    def set(data: bytes):
        VideoGenerator.frame = data

    @staticmethod
    def get():
        print('get')
        if VideoGenerator.frame is None:
            print('data nul')
            return "data nul"
        else:
            while True:

                print('while')
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + VideoGenerator.kernel(VideoGenerator.frame) + b'\r\n')
