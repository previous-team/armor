import pyrealsense2 as rs
import cv2
import numpy as np

class CameraSubscriber:
    def __init__(self):
        
        output_size = 224
        width=640
        height=480

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)
        
        # Initialize the Intel RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

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
        filtered_depth = self.spatial_filter.process(depth_frame)
        filtered_depth = self.temporal_filter.process(filtered_depth)
        

        if not filtered_depth or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(filtered_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image
    
    def crop(self, img, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        return img
    
    def cropping(self,rgb_image,depth_image):

        rgb_image = self.crop(rgb_image, bottom_right=self.bottom_right, top_left=self.top_left)
        depth_image = self.crop(depth_image, bottom_right=self.bottom_right, top_left=self.top_left)
        
        return rgb_image,depth_image
        
        
        
        


    
    def calculate_pixel_clutter_density(self, rgb_image, depth_image , window_size=10):
        if rgb_image is None or depth_image is None:
            return None
        
        # Convert the RGB image to grayscale and apply edge detection
        # gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        smoothed_image = cv2.GaussianBlur(depth_image, (5, 5), 0)

        
        gray = depth_image.copy()
        gray = np.uint8(gray)
        print("Max:", np.max(gray), "Min:", np.min(gray))

        # blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(gray, 30, 150)

        # Find contours in the image to detect objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Calculate distances between object centroids and their sizes
        objects = []
        total_area = depth_image.shape[0] * depth_image.shape[1]  # Total image area

        self.test_img = depth_image.copy()

        for contour in contours:
            # Calculate the bounding box of each object
            x, y, w, h = cv2.boundingRect(contour)
            
            # Remove small objects (noise)
            # if w * h < 0.01 * total_area:
            #     continue
            # else:
            cv2.rectangle(self.test_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
        window_size = 10  # Adjust this value as needed

        def calculate_clutter_for_window(x, y):
            clutter_density = 0
            for other_centroid, other_depth, object_size in objects:
                distance = np.linalg.norm(np.array((x, y)) - np.array(other_centroid))
                clutter_density += (1 / (distance + 1e-5)) * object_size
            return clutter_density

        for x in range(0, depth_image.shape[1], window_size):
            for y in range(0, depth_image.shape[0], window_size):
                clutter_density = min(calculate_clutter_for_window(x, y), 500)
                clutter_density_map[y:y+window_size, x:x+window_size] = clutter_density

        return clutter_density_map, edges
    

    def display_images(self):
        depth_image, rgb_image = self.get_images()
        rgb_image,depth_image = self.cropping(rgb_image,depth_image)

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
