import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
import cv2
from sklearn.cluster import DBSCAN

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=100, width=50):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(float))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        print(grasp_point)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps

'''Function for checking red object'''
def is_target_red(grasp_point, color_image, depth_image):
    # Convert the color image to HSV color space
    color_image = np.copy(color_image)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    lower_red_1 = np.array([0, 120, 70])    # Lower range for red
    upper_red_1 = np.array([10, 255, 255])  # Upper range for red
    lower_red_2 = np.array([170, 120, 70])  # Second lower range for red
    upper_red_2 = np.array([180, 255, 255]) # Second upper range for red

    
    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = mask1 | mask2

    

    red_objects = cv2.bitwise_and(color_image, color_image, mask=red_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Draw bounding boxes around detected red objects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            object_depth = depth_image[y:y+h, x:x+w]
            depth_threshold_min = object_depth.min() - 50
            depth_threshold_max = object_depth.max() + 50
            depth_mask = np.where((depth_image >= depth_threshold_min) &
                                  (depth_image <= depth_threshold_max), 255, 0).astype(np.uint8)
            combined_mask = cv2.bitwise_and(red_mask[y:y+h, x:x+w], depth_mask[y:y+h, x:x+w])
            masked_image = cv2.bitwise_and(color_image[y:y+h, x:x+w], color_image[y:y+h, x:x+w], mask=combined_mask)
            color_image[y:y+h, x:x+w] = masked_image

        return True  

    return False  


'''Function for checking blue object'''
def is_target_blue(grasp_point, color_image, depth_image):
    # Convert the color image to HSV color space
    color_image = np.copy(color_image)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    blue_mask = cv2.inRange(hsv_image,lower_blue,upper_blue)

    blue_objects = cv2.bitwise_and(color_image, color_image, mask=blue_mask)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Draw bounding boxes around detected red objects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            object_depth = depth_image[y:y+h, x:x+w]
            depth_threshold_min = object_depth.min() - 50
            depth_threshold_max = object_depth.max() + 50
            depth_mask = np.where((depth_image >= depth_threshold_min) &
                                  (depth_image <= depth_threshold_max), 255, 0).astype(np.uint8)
            combined_mask = cv2.bitwise_and(blue_mask[y:y+h, x:x+w], depth_mask[y:y+h, x:x+w])
            masked_image = cv2.bitwise_and(color_image[y:y+h, x:x+w], color_image[y:y+h, x:x+w], mask=combined_mask)
            color_image[y:y+h, x:x+w] = masked_image

        return True  

    return False  
'''Function for contour'''
def graspability_check(grasp_image,grasp_point_224):
    grasp_param = None
    gray_image = cv2.cvtColor(grasp_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(grasp_image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return grasp_param,contour_image


def graspability_with_cluster_density(depth_image,grasp_point_224):
    """
        Function for cluster density calculation within the rectangle around the target object
        :param depth_image: Depth image from the camera
        :param grasp_point_224: Grasp point (224x224 image coordinates)
        :param target_radius: Radius to consider points near the grasp point
        :return: cluster density and updated image with clusters marked
        """
    
    grasp_point_depth_value = depth_image[grasp_point_224[1], grasp_point_224[0]]
    print(f"Depth value at grasp_point_224 {grasp_point_224}: {grasp_point_depth_value}")

    grasp_point_depth_value_right = depth_image[grasp_point_224[1]+10, grasp_point_224[0]]
    print(f"Depth value at grasp_point_224 to right :::: {grasp_point_224}: {grasp_point_depth_value_right}")

    grasp_point_depth_value_right = depth_image[grasp_point_224[1]-10, grasp_point_224[0]]
    print(f"Depth value at grasp_point_224 to left ::: {grasp_point_224}: {grasp_point_depth_value_right}")#depth values are zero

    depth_range=(-5, 5)#is this range correct or needs to be sorted
    target_radius=50
    min_depth, max_depth = depth_range
    thresh_image = (depth_image >= min_depth) & (depth_image <= max_depth)

    # _, thresh_image = cv2.threshold(depth_image, 50, 255, cv2.THRESH_BINARY)
    points = np.column_stack(np.where(thresh_image > 0))
    if len(points) == 0:
        print("No valid depth points within the specified range")
        return None, depth_image
    ############maybe this clustering is not working 
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=5).fit(points)
    labels = dbscan.labels_

    # Find clusters (excluding noise points labeled as -1)
    unique_labels = set(labels)
    clusters = [points[labels == label] for label in unique_labels if label != -1]

    # Calculate density of objects near the target grasp point
    cluster_density = 0
    cluster_image = depth_image.copy()
    points_in_radius = []

    for cluster in clusters:
        # Calculate distances from cluster points to the grasp point
        distances = np.linalg.norm(cluster - np.array([grasp_point_224[1], grasp_point_224[0]]), axis=1)
        close_points = cluster[distances <= target_radius]

        # Update density based on proximity to the grasp point
        if len(close_points) > 0:
            points_in_radius.append(close_points)
            cluster_density += len(close_points) / len(cluster)

        # Draw the cluster for visualization
        color = tuple(np.random.randint(0, 255, 3).tolist())
        print(f"Cluster Density::::{cluster_density}")
        # for point in cluster:
        #     cv2.circle(cluster_image, (point[1], point[0]), 1, color, -1)

    # Check if there are multiple clusters close to the target
    grasp_param = True
    print(f'Points in radius:::::::::{len(points_in_radius)}')
    if len(points_in_radius) > 1:
        grasp_param = False  # Multiple objects close to target, not ideal to grasp
    

    return grasp_param, cluster_image


def hardware_detect_grasps(q_img, ang_img, width_img=None, no_grasps=1,rgb_img=None,depth_img=None):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    
    grasps = []
    masked_img = np.zeros_like(rgb_img) if rgb_img is not None else None  # Initialize masked_img
    contour_img = np.zeros_like(rgb_img) if rgb_img is not None else None
    masked_img_depth = np.zeros_like(depth_img) if depth_img is not None else None
    cluster_img = np.zeros_like(depth_img) if depth_img is not None else None
    grasp_param = None
    for grasp_point_array in local_max:
        grasp_point_224 = tuple(grasp_point_array)
        # print(f'grasp point for 224x224: {grasp_point_224}')  
       
        grasp_angle = ang_img[grasp_point_224]
        # print(f'Grasp angle:{grasp_angle}')

        if is_target_blue(grasp_point_224,rgb_img,depth_img):
            print('Red object detected')
            # Create a Grasp object
            g = Grasp(grasp_point_224, grasp_angle)
            if width_img is not None:
                g.length = width_img[grasp_point_224]
                g.width = g.length / 2

            # Convert the Grasp to a GraspRectangle
            grasp_rectangle = g.as_gr


            rr, cc = grasp_rectangle.polygon_coords(masked_img.shape[:2])
            if rgb_img is not None:
                mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
                mask[rr, cc] = 255  #the rectangle area
                masked_img[rr, cc] = rgb_img[rr, cc] #retaining only rectangle area
            rr1,cc1 = grasp_rectangle.polygon_coords(masked_img_depth.shape[:2])
            if depth_img is not None:
                mask = np.zeros(masked_img_depth.shape[:2])
                mask[rr1, cc1] = 255  #the rectangle area
                masked_img_depth[rr1, cc1] = depth_img[rr1, cc1] #retaining only rectangle area
            grasps.append(g)

            # added cuz sometimes masked image becomes none i.e. usually happens when no rectangle is detected
            if masked_img is None:
                masked_img = np.zeros((224, 224, 3), dtype=np.uint8)  # Fallback to a black image
                
            if masked_img_depth is None:
                masked_img_depth = np.zeros((224, 224, 3), dtype=np.uint8)  # Fallback to a black image
                
            # grasp_param,contour_img = graspability_check(masked_img,grasp_point_224)
            grasp_param,cluster_img = graspability_with_cluster_density(masked_img_depth,grasp_point_224)
            # grasp_param,cluster_img = graspability_with_cluster_density(depth_img,grasp_point_224)
            print(f'Graspable:{grasp_param}')

    return grasps,masked_img,cluster_img,grasp_param