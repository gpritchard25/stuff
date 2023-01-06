import cv2
import numpy as np
import pyautogui

# Load the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set the input image dimensions
width = 1920
height = 1080

# Initialize the list of class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generate colors for the different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set the confidence threshold for detecting objects
confidence_threshold = 0.5

while True:
    # Take a screenshot of the screen
    image = pyautogui.screenshot()

    # Convert the screenshot to a NumPy array
    image_np = np.array(image)

    # Resize the image to the input dimensions
    image_np = cv2.resize(image_np, (width, height))

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image_np, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # Set the input to the YOLO model
    net.setInput(blob)

    # Get the output layers
    layers = net.getUnconnectedOutLayers()

    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in layers]

    # Get the YOLO detections
    detections = net.forward(output_layers)

    # Initialize the list of bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over the detections
    for output in detections:
        # Loop over the detections in the output
        for detection in output:
            # Get the class ID and confidence of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > confidence_threshold:
                # Get the bounding box for the object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                box = (x, y, w, h)

                # Add the bounding box, confidence, and class ID to the lists
                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence

