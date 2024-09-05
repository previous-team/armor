import numpy as np

def camera_to_world(camera_pose, camera_coords):
    # Extract translation and rotation from camera_pose
    tx, ty, tz = camera_pose[:3]
    roll, pitch, yaw = camera_pose[3:]

    # Compute rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R = R_x @ R_y @ R_z

    # Transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    # Convert camera_coords to homogeneous coordinates
    camera_coords_hom = np.append(camera_coords, 1)

    # Apply transformation
    world_coords_hom = T @ camera_coords_hom

    # Return the transformed coordinates (excluding the homogeneous coordinate)
    return world_coords_hom[:3]

# Example usage
camera_pose = np.array([0.6, 0.0, 0.5, 0.0, -2.53073, 1.5708]) # x, y, z, roll, pitch, yaw ## CCW rotation is negative
camera_coords = np.array([18.80/1000, 37.60/1000, 579.25/1000]) # Example coordinates in camera frame
world_coords = camera_to_world(camera_pose, camera_coords)
print("X:", round(world_coords[0], 3), "; Y:", round(world_coords[1], 3), "; Z:", round(world_coords[2], 3), sep="")