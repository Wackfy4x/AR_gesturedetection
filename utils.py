import numpy as np
import cv2

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, aruco_params)

MARKER_LENGTH = 0.05   # taille réelle de ton marqueur imprimé, en mètres
GRAB_RADIUS = 0.05     # distance max (mètres) pour "attraper" un cube

cubes = []              # chaque cube = {"position": np.array([x, y, z])}
grabbed_cube_index = None
was_pinching = False
was_fist = False


def init_camera_intrinsics(frame_width, frame_height):
    """Approximation des paramètres intrinsèques de la caméra (à affiner plus tard avec une vraie calibration)."""
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    return camera_matrix, dist_coeffs


def detect_marker_pose(frame, camera_matrix, dist_coeffs):
    """Détecte le marqueur ArUco. Renvoie (rvec, tvec) ou None si pas de marqueur visible."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return None
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )
    return rvecs[0], tvecs[0]  # pose du premier marqueur trouvé


def project_hand_to_marker_plane(pixel_x, pixel_y, rvec, tvec, camera_matrix):
    """
    Transforme la position 2D (pixel) de la main en un point 3D situé
    sur le plan du marqueur (son "sol" virtuel, z=0 dans son propre repère).
    """
    rot_matrix, _ = cv2.Rodrigues(rvec)
    inv_camera = np.linalg.inv(camera_matrix)

    ray_camera = inv_camera @ np.array([pixel_x, pixel_y, 1.0])
    ray_world = np.linalg.inv(rot_matrix) @ ray_camera
    cam_origin_world = np.linalg.inv(rot_matrix) @ (-tvec.flatten())

    # intersection du rayon avec le plan z=0 (le plan du marqueur)
    t = -cam_origin_world[2] / ray_world[2]
    return cam_origin_world + t * ray_world