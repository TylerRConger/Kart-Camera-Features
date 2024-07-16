import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


yolo = cv.dnn.readNet("yolov4-obj_best_coneIdentifier.weights",'yolov4-obj.cfg')

classes = []

with open('obj.names', 'r') as f:
    classes = f.read().splitlines()

# Webcam usage
#cap = cv.VideoCapture(0)

# Video Usage
cap = cv.VideoCapture('policeDrivingTest.mp4')

count = 0

while True:
    

    _, image = cap.read()

    if _:

        # Save a jpg of each frame, can be useful for static training later
        # TODO: Give this a file directory so you don't just spam yourself
        # cv.imwrite('frame{:d}.jpg'.format(count), image)


        # Consider frame count for videos. 
        # This is almost entirely unnecessary in real time applications as most recent frames are processed first
        # but in video situations it may be necessary to skip multiple frames in order to match actual video speed
        # Skipping 30 at 30fps is equal to skipping 1 second
        # Also worth considering video quality, 440p seems to work really well and give a quicker speed while also minimizing lose
        count += 30
        cap.set(cv.CAP_PROP_POS_FRAMES, count)

        #image = cv.imread('coneTester.jpg')

        # get the shape
        height, width, _ = image.shape

        # normalize the image
        blob = cv.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True , crop = False )

        yolo.setInput(blob)

        # Get actual layers
        output_layer_names = yolo.getUnconnectedOutLayersNames()
        layerOutput = yolo.forward(output_layer_names)

        boxes = []
        confidences = []
        class_IDs = []

        for output in layerOutput:
            for detection in output:
                # Store all the elements that are useful, first 5 are just bounding info
                scores = detection[5:]
                class_ID= np.argmax(scores)
                confidence = scores[class_ID]

                # 70% confident
                if  confidence > 0.7:
                    # Get the bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_IDs.append(class_ID)


        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # assign a font
        font = cv.FONT_HERSHEY_PLAIN

        # get a unique color and assign
        colors = np.random.uniform(0, 255, size = (len(boxes), 3) )
        
        # Error checking verify the list is not empty before attempting

        for i in indexes:
            x, y, w, h = boxes[i]

            label = str(classes[class_IDs[i]])
            conf = str(round(confidences[i], 2))
            color = colors[i]
            cv.rectangle(image, (x , y), (x + w, y + h), color, 2)

            cv.putText(image, label + " " + conf, (x, y+20), font, 2, (255, 255, 255), 1)
                

        cv.imshow('Final Detection', image)
        key = cv.waitKey(1)
        if key == 27:
                break


cap.release()
cv.destroyAllWindows()
