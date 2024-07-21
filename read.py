import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

CAMERA_FOCAL_LENGTH = None

CONE_ACTUAL_HEIGHT = .9144
CONE_EST_DIST = 30

# Find the focal length of the camera on a known distance image
# Only really needs to be done once
# obj_height - Is the actual height of the object
# measured_distance - The distance between the camera and the object
# height_in_image - Height in pixels of the cone in the image
def getFocalLength(measured_distance, obj_height, height_in_image): 
    global CAMERA_FOCAL_LENGTH
    CAMERA_FOCAL_LENGTH = (height_in_image * measured_distance) / obj_height 

# Get the estimated cone height and return it
def getConeHeight(detectedHeight):
     distance = (CONE_ACTUAL_HEIGHT * CAMERA_FOCAL_LENGTH) / detectedHeight
     return distance

def main():
    # Set the weights file
    yolo = cv.dnn.readNet("backup\yolov4-obj_best.weights",'yolov4-obj.cfg')

    classes = []

    with open('obj.names', 'r') as f:
        classes = f.read().splitlines()

    # Webcam usage
    # cap = cv.VideoCapture(0)

    # Video Usage
    cap = cv.VideoCapture('policeDrivingTest.mp4')

    # Saving the video locally, can also be done through ROS nodes but useful when running this code by itself
    # out = cv.VideoWriter('output.avi', -1, 20.0, (416, 416))

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
            count += 15
            cap.set(cv.CAP_PROP_POS_FRAMES, count)


            # Useful for a single tester image without massively modifying the code
            image = cv.imread('images/val/cone (195).jpg')

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
                    # Change this appropriately for how confident you want to be about identifying an object
                    # I've found 70% works fairly well for our use case, but higher may be better
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

            # If you want each cone to be a unique color use line below, otherwise use array below
            # colors = np.random.uniform(0, 255, size = (len(boxes), 3) )

            # This is green color R G B
            color = [0, 255, 0]
    
            # Error checking verify the list is not empty before attempting

            for i in indexes:
                x, y, w, h = boxes[i]

                label = str(classes[class_IDs[i]])
                conf = str(round(confidences[i], 2))
                # Uncomment if using color randomizer
                #color = colors[i]
                cv.rectangle(image, (x , y), (x + w, y + h), color, 2)

                # Get the pixel height and width of the detected cone
                # This information is kinda useful, gives us another data point, but we rely on LiDAR for most distance information
                coneWidth = w
                coneHeight = h

                # If the focal length isn't set yet, set it up, should only be done once
                if CAMERA_FOCAL_LENGTH == None:
                    getFocalLength(CONE_EST_DIST, CONE_ACTUAL_HEIGHT, coneHeight)

                distance = getConeHeight(coneHeight)

                # If a cone is nearby alert the system that there is an issue
                # Pipe this information to ROS so the LiDAR can confirm or deny close-ness of obstacle
                if distance < 1:
                    print("Watch out cone near!")
    
                # Display distance and confidence
                cv.putText(image, label + " " + conf + ":: Distance " + str(round(distance, 2)), (x, y+20), font, 2, (255, 255, 255), 1)                   

                    
            
            # Change text based on camera angle view
            cv.imshow('Kart View', image)

            # Write the frame to video if wanted
            # out.write(image)
            
            # Write the single image
            cv.imwrite('testResult2.jpg', image)

            # Wait for escape key to be pressed
            key = cv.waitKey(1)
            if key == 27:
                    break


    cap.release()
    cv.destroyAllWindows()

    #out.release()

main()