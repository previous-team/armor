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

        # Here you'd run your object detection model to get the bounding box
        # Example bounding box for an object [x1, y1, x2, y2]
        x1, y1, x2, y2 = 100, 100, 200, 200

        # Extract object depth
        object_depth = depth_image[y1:y2, x1:x2]

        # Define depth range for the mask
        depth_threshold_min = object_depth.min() - 50
        depth_threshold_max = object_depth.max() + 50

        # Create the mask based on depth
        mask = np.where((depth_image >= depth_threshold_min) & 
                        (depth_image <= depth_threshold_max), 255, 0).astype(np.uint8)

        # Apply mask to color image
        masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)

        # Display the results
        cv2.imshow('Masked Image', masked_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()

