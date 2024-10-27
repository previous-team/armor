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
        if rgb_image is None or depth_image is None:
            return None
        
        # Convert the RGB image to grayscale and apply edge detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the image to detect objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate distances between object centroids and their sizes
        objects = []
        total_area = rgb_image.shape[0] * rgb_image.shape[1]  # Total image area

        for contour in contours:
            # Calculate the bounding box of each object
            x, y, w, h = cv2.boundingRect(contour)
            
            # Remove small objects (noise)
            if w * h < 0.01 * total_area:
                continue

            centroid = (x + w // 2, y + h // 2)

            # Find the corresponding depth of the object by averaging depth pixels within the bounding box
            depth_region = depth_image[y:y+h, x:x+w]
            avg_depth = np.mean(depth_region)

            # Calculate object size (approximated by bounding box area)
            object_size = w * h
            
            # Append object position and depth
            objects.append((centroid, avg_depth, object_size))
        
        # Initialize clutter density map
        clutter_density_map = np.zeros_like(gray, dtype=np.float32)

        # Define window size
        window_size = 5  # Adjust this value as needed

        def calculate_clutter_for_window(x, y):
            clutter_density = 0
            for other_centroid, other_depth, object_size in objects:
                distance = np.linalg.norm(np.array((x, y)) - np.array(other_centroid))
                clutter_density += (1 / (distance + 1e-5)) * object_size
            return clutter_density

        for x in range(0, rgb_image.shape[1], window_size):
            for y in range(0, rgb_image.shape[0], window_size):
                clutter_density = min(calculate_clutter_for_window(x, y), 500)
                clutter_density_map[y:y+window_size, x:x+window_size] = clutter_density

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
