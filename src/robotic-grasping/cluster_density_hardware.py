import pyrealsense2 as rs
import cv2
import numpy as np

class CameraSubscriber:
    def __init__(self):
        # Initialize the Intel RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the pipeline to stream both depth and color images
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the pipeline
        self.pipeline.start(self.config)

    def get_images(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def calculate_pixel_clutter_density(self, rgb_image, depth_image, window_size=40):
        # Converts the RGB image to grayscale and applies edge detection
        gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred_rgb = cv2.GaussianBlur(gray_rgb, (5, 5), 0)
        edges = cv2.Canny(blurred_rgb, 50, 150)

        # Find all contours once
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize clutter density map
        clutter_density_map = np.zeros_like(gray_rgb, dtype=np.float32)
        half_window = window_size // 2

        # Analyze the local region for clutter
        for y in range(half_window, gray_rgb.shape[0] - half_window):
            for x in range(half_window, gray_rgb.shape[1] - half_window):
                
                # Extract window around the current pixel
                rgb_window = gray_rgb[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                depth_window = depth_image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]

                object_positions = []
                local_clutter_density = 0
                window_area = window_size * window_size

                # Filter contours that are within the current window
                for contour in contours:
                    x_w, y_w, w, h = cv2.boundingRect(contour)
                    if (x_w >= x-half_window and x_w+w <= x+half_window and
                        y_w >= y-half_window and y_w+h <= y+half_window):
                        
                        centroid = (x_w + w // 2, y_w + h // 2)
                        object_positions.append(centroid)

                        # Average depth of the object
                        depth_region = depth_window[y_w:y_w+h, x_w:x_w+w]
                        avg_depth = np.mean(depth_region)
                        
                        object_size = w * h

                        # Calculate proximity factor and contribution to local clutter density
                        for other_centroid in object_positions:
                            if other_centroid == centroid:
                                continue
                            # Calculate 2D distance between centroids
                            distance = np.linalg.norm(np.array(centroid) - np.array(other_centroid))
                            local_clutter_density += (1 / (distance + 1e-5)) * object_size

                clutter_density_map[y, x] = local_clutter_density

        return clutter_density_map, edges

    def display_images(self):
        depth_image, rgb_image = self.get_images()

        if depth_image is not None and rgb_image is not None:
            # Normalize depth image for display
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Depth Image", depth_image_normalized)
            cv2.imshow("RGB Image", rgb_image)

            clutter_density_map, edges = self.calculate_pixel_clutter_density(rgb_image, depth_image)

            if clutter_density_map is not None:
                clutter_density_normalized = cv2.normalize(clutter_density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap = cv2.applyColorMap(clutter_density_normalized, cv2.COLORMAP_JET)
                cv2.imshow("Clutter Density Heatmap", heatmap)

            cv2.imshow("Canny Edges", edges)
            cv2.waitKey(1)

    def run(self):
        while True:
            self.display_images()

if __name__ == '__main__':
    camera_subscriber = CameraSubscriber()
    try:
        camera_subscriber.run()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
