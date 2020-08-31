from flask import Flask, request, render_template,Response


app = Flask(__name__)

@app.route('/')
def dynamic_page():
   return render_template('index.html')


def gen():
  from imutils.video import VideoStream
  from imutils.video import FPS
  import numpy as np
  import argparse
  import imutils
  import time
  import cv2
  from imutils import contours
  from imutils import perspective
  from scipy.spatial import distance 
  import pandas as pd
  def compute_distance(midpoints,num):
    dist = np.zeros((num,num))

    for i in range(num):
      for j in range(i+1,num):
        if i!=j:
          dst = distance.euclidean(midpoints[i], midpoints[j])
          dist[i][j]=dst
    return dist
  def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
      for j in range(i,num):
        if( (i!=j) & (dist[i][j]<=thresh)):
          p1.append(i)
          p2.append(j)
          d.append(dist[i][j])
    return p1,p2,d
  def change_2_red(img,detections,p1,p2,h,w):
    risky = np.unique(p1+p2)
    for i in risky:
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (x1,y1,x2,y2) = box.astype("int")
      
      cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)  
    return img
  # Add Argument 
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--prototxt", required=True,
      help="path to Caffe 'deploy' prototxt file")
  ap.add_argument("-m", "--model", required=True,
      help="path to Caffe pre-trained model")
  ap.add_argument("-c", "--confidence", type=float, default=0.1,
      help="minimum probability to filter weak detections")
  args = vars(ap.parse_args())
  CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
      "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
      "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
      "sofa", "train", "tvmonitor"]
  COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
  # load our serialized model from disk
  print("[INFO] loading model...")
  net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
  # initialize the video stream, allow the cammera sensor to warmup,
  # and initialize the FPS counter
  print("[INFO] starting video stream...")
  # Enter the IP address correspond to your device of IP Wecam Application 
  vs = VideoStream('sample.mp4').start()
  # time.sleep(2.0)
  fps = FPS().start()

  while True:
      frame=vs.read()
      (h, w) = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
      net.setInput(blob)
      detections = net.forward()
      person_count=0;
      mid=[]
      for i in np.arange(0, detections.shape[2]):
          # extract the confidence (i.e., probability) associated with
          # the prediction
          confidence = detections[0, 0, i, 2]
          # filter out weak detections by ensuring the `confidence` is
          # greater than the minimum confidence
          if confidence > args["confidence"]:
              # extract the index of the class label from the
              # `detections`, then compute the (x, y)-coordinates of
              # the bounding box for the object
              idx = int(detections[0, 0, i, 1])
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
              # draw the prediction on the frame
              if CLASSES[idx]=='person':
                  label = "{}: {:.2f}%".format(CLASSES[idx],
                  confidence * 100)
                  cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0,255,0), 2)
                  y = startY - 15 if startY - 15 > 15 else startY + 15
                  cv2.putText(frame, label, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                  x_mid=int((startX+endX)/2)
                  y_mid=int((startY+ endY)/2)
                  # print(x_mid)
                  # print(y_mid)
                  mid.append([x_mid,y_mid])
                  
    
      dist=compute_distance(mid,len(mid))
      p1,p2,d=find_closest(dist,len(mid),183)
      df = pd.DataFrame({"p1":p1,"p2":p2,"dist":d})
      # print(df)
      
      img = change_2_red(frame,detections,p1,p2,h,w)
      for i in range(0,len(p1)):
            # print(mid[p1[i]])
            # print(mid[p2[i]])

            cv2.line(img,(mid[p1[i]][0],mid[p1[i]][1]),(mid[p2[i]][0],mid[p2[i]][1]),(0,0,0), 2)

            # label = "{}: {:.2f}%".format(d[i])
            cv2.putText(img,str(d[i]), (mid[p1[i]][0],mid[p1[i]][1]+15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255), 2)
            print(d[i])

      cv2.imshow("Frame", img)
      (flag, encodedImage) = cv2.imencode(".jpg", img)


      yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + bytearray(encodedImage.tobytes()) + b'\r\n')
    #   yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    #   yield img
    #   yield b'\r\n\r\n'
      key = cv2.waitKey(20) & 0xFF
      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
          break
      # update the FPS counter
      fps.update()
  fps.stop()
  print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
  # do a bit of cleanup
  cv2.destroyAllWindows()
  vs.stop()
@app.route('/result')
def videoFeed():
 yield Response(gen(),
                mimetype='multipart/x-mixed-replace; boundary=frame')
 yield Response(stream_template('index.html', rows=rows))
 
if __name__ == '__main__':
    app.run(debug=True)