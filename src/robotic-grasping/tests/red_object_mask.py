#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the color image to HSV color space
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the HSV range for red color
        lower_red_1 = np.array([0, 120, 70])   # Lower range for red
        upper_red_1 = np.array([10, 255, 255]) # Upper range for red
        lower_red_2 = np.array([170, 120, 70]) # Lower range for red
        upper_red_2 = np.array([180, 255, 255]) # Upper range for red

        # Create a binary mask for the red color
        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        red_mask = mask1 | mask2

        # Apply the red mask to the color image
        red_objects = cv2.bitwise_and(color_image, color_image, mask=red_mask)

        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the red objects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle on the color image

            # Optional: Extract depth information for each bounding box
            object_depth = depth_image[y:y+h, x:x+w]

            # Define depth range for the mask
            depth_threshold_min = object_depth.min() - 50
            depth_threshold_max = object_depth.max() + 50

            # Create the mask based on depth
            depth_mask = np.where((depth_image >= depth_threshold_min) & 
                                  (depth_image <= depth_threshold_max), 255, 0).astype(np.uint8)

            # Combine the red mask and depth mask for this bounding box area
            combined_mask = cv2.bitwise_and(red_mask[y:y+h, x:x+w], depth_mask[y:y+h, x:x+w])

            # Apply the combined mask to the color image for this area
            masked_image = cv2.bitwise_and(color_image[y:y+h, x:x+w], color_image[y:y+h, x:x+w], mask=combined_mask)

            # Replace the original area with the masked area in the main image
            color_image[y:y+h, x:x+w] = masked_image

        # Display the results
        cv2.imshow('Red Object Masked Image with Bounding Boxes', color_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()
