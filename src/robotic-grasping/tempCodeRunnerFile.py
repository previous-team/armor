    def display_images(self):
        if self.depth_image is not None and self.rgb_image is not None:
            # Display the images
            cv2.imshow("Depth Image", self.depth_image)
            cv2.imshow("RGB Image", self.rgb_image)
            cv2.waitKey(1)