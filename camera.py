'''
    camera.py

    source: https://medium.datadriveninvestor.com/video-streaming-using-flask-and-opencv-c464bf8473d6
'''

import cv2

cv2.samples.addSamplesDataSearchPath(cv2.__path__[0] + '/data')
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile('haarcascade_frontalface_alt2.xml'))
eyes_cascade = cv2.CascadeClassifier(cv2.samples.findFile('haarcascade_eye_tree_eyeglasses.xml'))

ds_factor = 0.6

class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, None, fx = ds_factor, fy = ds_factor, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_rects=eyes_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
        	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        	break
        for (x, y, w, h) in eyes_rects:
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            eye_center = (x + w//2, y + h//2)
            radius = int(round((w + h)*0.25))
            cv2.circle(image, eye_center, radius, (255, 0, 0 ), 4)
            

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
