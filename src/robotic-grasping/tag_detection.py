import cv2
import numpy as np

# Start capturing video from the default webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a blur to reduce noise before edge detection
    gray = cv2.medianBlur(gray, 7)

    centers = []

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, #inverse ratio of resolution
        minDist=100, #Minimum distance between detected centers
        param1=50, #Upper threshold for the internal Canny edge detector.
        param2=30, #Threshold for center detection
        minRadius=10, #Minimum radius to be detected. If unknown, put zero as default.
        maxRadius=40)

    # If some circles are detected, draw them
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Draw a rectangle at the center of the circle
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            centers.append((x, y))

        if centers:
            # Compute the bounding box
            x_coords, y_coords = zip(*centers)
            x_min, x_max = max(0, min(x_coords)), min(frame.shape[1], max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(frame.shape[0], max(y_coords))
            
            # Draw the bounding rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Crop the image to the bounding box
            if x_min < x_max and y_min < y_max:
                cropped_image = frame[y_min:y_max, x_min:x_max]

                # Display the cropped image if it's valid
                if cropped_image.size > 0:
                    cv2.imshow("Cropped Image", cropped_image)

    # Display the resulting frame
    cv2.imshow("output", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()