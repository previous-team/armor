import numpy as np
from scipy.spatial.transform import Rotation as R

def camera_to_world(camera_coords, camera_pose):
    """
    Convert coordinates from camera frame to world frame.

    :param camera_coords: numpy array of shape (3,), the (x, y, z) coordinates in the camera frame
    :param camera_pose: dictionary with keys 'position' and 'orientation'
                        'position' is a numpy array of shape (3,)
                        'orientation' is a numpy array of shape (4,) for quaternion or (3,) for Euler angles
    :return: numpy array of shape (3,), the (x, y, z) coordinates in the world frame
    """
    # Extract position and orientation
    position = camera_pose['position']
    orientation = camera_pose['orientation']
    
    # Check if orientation is given as quaternion or Euler angles
    if len(orientation) == 4:
        # Quaternion to rotation matrix
        rotation_matrix = R.from_quat(orientation).as_matrix()
    elif len(orientation) == 3:
        # Euler angles to rotation matrix
        rotation_matrix = R.from_euler('xyz', orientation).as_matrix()
    else:
        raise ValueError("Orientation must be a quaternion (4,) or Euler angles (3,)")

    # Create the homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    # Convert camera coordinates to homogeneous coordinates
    camera_coords_homogeneous = np.append(camera_coords, 1)

    # Apply the transformation
    world_coords_homogeneous = np.dot(transformation_matrix, camera_coords_homogeneous)

    # Extract the world coordinates
    world_coords = world_coords_homogeneous[:3]

    return world_coords

# Example usage
camera_coords = np.array([(583.75006)/1000, 84.63112/1000, -1.263151/1000])  # [-84.63112] [1.263151] [583.75006]
# x is z; y is -x; z is -y
camera_pose = {
    'position': np.array([0.6, 0, 0.5]),
    # 'orientation': np.array([0, 0, 0, 1])  # Quaternion (w, x, y, z)
    'orientation': np.array([0.0000001, 0.959932, -3.141591])  # Euler angles (roll, pitch, yaw)
}

world_coords = camera_to_world(camera_coords, camera_pose)
print("World Coordinates:", world_coords)