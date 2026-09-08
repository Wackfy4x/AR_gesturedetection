import time

import numpy as np
import cv2

PLANE_DEPTH = 0.4      # profondeur fixe (mètres) du plan virtuel devant la caméra, où vivent les cubes
GRAB_RADIUS = 0.05     # distance max (mètres) pour "attraper" ou "copier" un cube
CUBE_SIZE = 0.03       # taille (mètres) d'un cube affiché
DWELL_TIME_SEC = 0.8   # temps de maintien main ouverte requis pour déclencher une copie
COPY_OFFSET = np.array([CUBE_SIZE * 1.5, 0.0, 0.0])  # décalage du cube copié par rapport à l'original

cubes = []              # chaque cube = {"position": np.array([x, y, z])}
grabbed_cube_index = None
was_pinching = False
was_fist = False
dwell_start_time = None
dwell_target_index = None


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


def project_hand_to_fixed_plane(pixel_x, pixel_y, camera_matrix, plane_depth=PLANE_DEPTH):
    """
    Transforme la position 2D (pixel) de la main en un point 3D situé sur un plan
    virtuel fixe, à `plane_depth` mètres devant la caméra (repère caméra = repère monde,
    valable tant que la caméra ne bouge pas).
    """
    inv_camera = np.linalg.inv(camera_matrix)
    ray = inv_camera @ np.array([pixel_x, pixel_y, 1.0])  # ray[2] == 1 par construction
    return ray * plane_depth


def handle_pinch_creation(is_pinching_now, hand_world_pos):
    """Crée un cube uniquement au moment où le pincement commence."""
    global was_pinching

    if is_pinching_now and not was_pinching:
        cubes.append({"position": hand_world_pos.copy()})

    was_pinching = is_pinching_now


def find_closest_cube(world_pos):
    """Renvoie l'index du cube le plus proche, si dans le rayon de préhension."""
    if not cubes:
        return None
    distances = [np.linalg.norm(c["position"] - world_pos) for c in cubes]
    closest = int(np.argmin(distances))
    return closest if distances[closest] < GRAB_RADIUS else None


def handle_fist_move(is_fist_now, hand_world_pos):
    """Attrape le cube le plus proche à la fermeture du poing, le suit tant qu'il reste fermé."""
    global was_fist, grabbed_cube_index

    if is_fist_now and not was_fist:
        grabbed_cube_index = find_closest_cube(hand_world_pos)
    elif is_fist_now and grabbed_cube_index is not None:
        cubes[grabbed_cube_index]["position"] = hand_world_pos.copy()
    elif not is_fist_now:
        grabbed_cube_index = None

    was_fist = is_fist_now


def handle_open_copy(is_open_now, hand_world_pos):
    """Duplique le cube visé après un temps de maintien main ouverte au-dessus de lui.

    Le minuteur se réarme après chaque copie : maintenir la main ouverte
    duplique donc le cube en continu, une fois par DWELL_TIME_SEC.
    """
    global dwell_start_time, dwell_target_index

    if not is_open_now:
        dwell_start_time = None
        dwell_target_index = None
        return

    target = find_closest_cube(hand_world_pos)
    if target is None:
        dwell_start_time = None
        dwell_target_index = None
        return

    if target != dwell_target_index:
        dwell_target_index = target
        dwell_start_time = time.time()
        return

    if time.time() - dwell_start_time >= DWELL_TIME_SEC:
        cubes.append({"position": cubes[target]["position"] + COPY_OFFSET})
        dwell_start_time = time.time()


def _cube_corners(position, size):
    """Coins d'un cube (8x3) centré sur `position`."""
    half = size / 2
    x, y, z = position
    return np.array([
        [x - half, y - half, z - half], [x + half, y - half, z - half],
        [x + half, y + half, z - half], [x - half, y + half, z - half],
        [x - half, y - half, z + half], [x + half, y - half, z + half],
        [x + half, y + half, z + half], [x - half, y + half, z + half],
    ], dtype=np.float64)


_CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


_NO_ROTATION = np.zeros((3, 1))
_NO_TRANSLATION = np.zeros((3, 1))


def draw_cube(frame, position, camera_matrix, dist_coeffs, color=(255, 120, 0)):
    """Projette et dessine l'arête d'un cube (position déjà dans le repère caméra)."""
    corners_3d = _cube_corners(position, CUBE_SIZE)
    img_points, _ = cv2.projectPoints(
        corners_3d, _NO_ROTATION, _NO_TRANSLATION, camera_matrix, dist_coeffs
    )
    img_points = img_points.reshape(-1, 2).astype(int)

    for i, j in _CUBE_EDGES:
        cv2.line(frame, tuple(img_points[i]), tuple(img_points[j]), color, 2)
    return frame


def draw_cubes(frame, camera_matrix, dist_coeffs):
    """Dessine tous les cubes existants ; met en évidence celui en cours de déplacement."""
    for i, cube in enumerate(cubes):
        color = (0, 255, 0) if i == grabbed_cube_index else (255, 120, 0)
        frame = draw_cube(frame, cube["position"], camera_matrix, dist_coeffs, color)
    return frame
