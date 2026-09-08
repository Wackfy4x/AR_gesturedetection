import math
import cv2
import mediapipe as mp
import numpy as np

from utils import GRAB_RADIUS

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


def init_camera():
    """Vérifie que la caméra s'est bien ouverte."""
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")


def read_frame():
    """Lit une seule image du flux. Renvoie (succès, frame)."""
    return cap.read()


def detect_hands(frame):
    """Passe la frame à MediaPipe et renvoie les résultats bruts."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(frame_rgb)


def draw_hands(frame, results):
    """Dessine le squelette de chaque main détectée directement sur la frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


def get_pinch_distance(hand_landmarks):
    """Distance entre le bout du pouce (4) et le bout de l'index (8)."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))


def is_pinching(hand_landmarks, threshold=0.06):
    """True si pouce et index sont suffisamment proches."""
    return get_pinch_distance(hand_landmarks) < threshold


def is_fist(hand_landmarks, threshold=0.15):
    """True si les 4 doigts (hors pouce) sont repliés vers le centre de la main."""
    fingertip_ids = [8, 12, 16, 20]  # index, majeur, annulaire, auriculaire
    wrist = hand_landmarks.landmark[0]

    distances = [
        math.dist(
            (hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y),
            (wrist.x, wrist.y),
        )
        for tip in fingertip_ids
    ]
    return (sum(distances) / len(distances)) < threshold


def interpret_gesture(hand_landmarks):
    """Combine les détecteurs ci-dessus en un geste nommé."""
    if is_pinching(hand_landmarks):
        return "pincement"
    if is_fist(hand_landmarks):
        return "poing"
    return "main ouverte"


def extract_gestures(results):
    """Parcourt toutes les mains détectées et renvoie la liste de leurs gestes."""
    gestures = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gestures.append(interpret_gesture(hand_landmarks))
    return gestures


def draw_gestures(frame, gestures):
    """Affiche le(s) geste(s) détecté(s) en texte sur la frame, à titre de debug."""
    for i, gesture in enumerate(gestures):
        cv2.putText(
            frame, gesture, (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    return frame


def show_frame(frame):
    """Affiche une frame dans une fenêtre."""
    cv2.imshow("Webcam", frame)


def should_quit():
    """Vérifie si l'utilisateur veut quitter (touche 'q')."""
    return cv2.waitKey(1) & 0xFF == ord('q')


def release_camera():
    """Libère la caméra, ferme les fenêtres et le détecteur MediaPipe."""
    cap.release()
    cv2.destroyAllWindows()
    hands.close()



def handle_pinch_creation(is_pinching_now, hand_world_pos):
    """Crée un cube uniquement au moment où le pincement commence."""
    global was_pinching, cubes

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

def run():
    """Boucle principale : orchestre les fonctions ci-dessus."""
    init_camera()

    while True:
        success, frame = read_frame()
        if not success:
            break

        results = detect_hands(frame)
        frame = draw_hands(frame, results)

        gestures = extract_gestures(results)
        frame = draw_gestures(frame, gestures)

        show_frame(frame)

        if should_quit():
            break

    release_camera()


if __name__ == "__main__":
    run()