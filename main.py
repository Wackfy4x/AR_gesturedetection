import math
import time

import cv2
import mediapipe as mp

import utils

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

PALM_LANDMARK = 9  # base du majeur, utilisée comme centre de la paume


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


FOLDED_RATIO = 1.1  # doigt-poignet / taille-paume en dessous duquel un doigt est considéré replié
PINCH_RATIO = 0.4   # pouce-index / taille-paume en dessous duquel on considère un pincement


def _palm_size(hand_landmarks):
    """Distance poignet (0) → base du majeur (9), utilisée comme échelle de la main
    pour rendre les seuils de gestes indépendants de la distance à la caméra."""
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    return math.dist((wrist.x, wrist.y), (middle_mcp.x, middle_mcp.y))


def _finger_folded(hand_landmarks, tip_id, palm_size):
    """True si le bout du doigt `tip_id` est proche du poignet, relativement à la taille de la main."""
    wrist = hand_landmarks.landmark[0]
    tip = hand_landmarks.landmark[tip_id]
    return math.dist((tip.x, tip.y), (wrist.x, wrist.y)) < palm_size * FOLDED_RATIO


def get_pinch_distance(hand_landmarks):
    """Distance entre le bout du pouce (4) et le bout de l'index (8)."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))


def is_pinching(hand_landmarks):
    """True si pouce et index sont proches ET que les 3 autres doigts sont tendus.

    Cette seconde condition évite qu'un poing fermé (où pouce et index finissent
    aussi par se toucher) ne soit confondu avec un pincement.
    """
    palm_size = _palm_size(hand_landmarks)
    pinch_close = get_pinch_distance(hand_landmarks) < palm_size * PINCH_RATIO
    other_fingers_extended = not any(
        _finger_folded(hand_landmarks, tip, palm_size) for tip in (12, 16, 20)
    )
    return pinch_close and other_fingers_extended


def is_fist(hand_landmarks):
    """True si les 4 doigts (hors pouce) sont repliés vers le poignet."""
    fingertip_ids = (8, 12, 16, 20)  # index, majeur, annulaire, auriculaire
    palm_size = _palm_size(hand_landmarks)
    return all(_finger_folded(hand_landmarks, tip, palm_size) for tip in fingertip_ids)


def interpret_gesture(hand_landmarks):
    """Combine les détecteurs ci-dessus en un geste nommé.

    Le poing est testé avant le pincement : en refermant la main, le pouce
    et l'index se frôlent souvent brièvement, ce qui déclencherait un faux
    pincement si le test était fait dans l'autre ordre.
    """
    if is_fist(hand_landmarks):
        return "poing"
    if is_pinching(hand_landmarks):
        return "pincement"
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


STABILITY_FRAMES = 3  # nb de frames consécutives identiques requises pour confirmer un changement de geste

_stable_gesture = "main ouverte"
_pending_gesture = None
_pending_count = 0


def stabilize_gesture(raw_gesture):
    """Filtre les flickers d'une seule frame : un geste n'est confirmé qu'après
    STABILITY_FRAMES lectures consécutives identiques."""
    global _stable_gesture, _pending_gesture, _pending_count

    if raw_gesture == _stable_gesture:
        _pending_gesture = None
        _pending_count = 0
        return _stable_gesture

    if raw_gesture == _pending_gesture:
        _pending_count += 1
    else:
        _pending_gesture = raw_gesture
        _pending_count = 1

    if _pending_count >= STABILITY_FRAMES:
        _stable_gesture = raw_gesture
        _pending_gesture = None
        _pending_count = 0

    return _stable_gesture


PINCH_HOLD_SEC = 0.35  # durée de maintien du pincement requise avant de créer un cube

_previous_stable_gesture = "main ouverte"
_pinch_streak_start = None
_pinch_streak_blocked = False
_pinch_streak_confirmed = False


def should_create_on_pinch(stable_gesture):
    """N'autorise une création de cube que si le pincement fait suite à une main
    ouverte (jamais directement à un poing) ET qu'il est maintenu au moins
    PINCH_HOLD_SEC. Deux flickers similaires se produisent en pratique lors des
    transitions main ouverte <-> poing (les doigts ne se plient/déplient pas tous
    à la même vitesse, et passent brièvement par un état qui se lit comme un
    pincement) : le filtre de stabilité de quelques frames (~100 ms) ne suffit
    pas à les distinguer d'un vrai pincement volontaire, d'où ce délai de maintien.
    Renvoie True une seule fois par pincement maintenu (comme un front montant).
    """
    global _previous_stable_gesture, _pinch_streak_start, _pinch_streak_blocked, _pinch_streak_confirmed

    if stable_gesture != "pincement":
        _pinch_streak_start = None
        _pinch_streak_blocked = False
        _pinch_streak_confirmed = False
        _previous_stable_gesture = stable_gesture
        return False

    if _previous_stable_gesture != "pincement":
        _pinch_streak_start = time.time()
        _pinch_streak_blocked = _previous_stable_gesture == "poing"
        _pinch_streak_confirmed = False

    _previous_stable_gesture = stable_gesture

    if _pinch_streak_blocked or _pinch_streak_confirmed:
        return False

    if time.time() - _pinch_streak_start >= PINCH_HOLD_SEC:
        _pinch_streak_confirmed = True
        return True

    return False


def draw_status(frame, hand_detected):
    """Affiche l'état de détection de la main et le nombre de cubes, à titre de debug."""
    text = f"Main: {'detectee' if hand_detected else 'NON detectee'} | Cubes: {len(utils.cubes)}"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(
        frame, text, (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
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


def run():
    """Boucle principale : orchestre capture, détection et manipulation des cubes."""
    init_camera()
    camera_matrix = None
    dist_coeffs = None

    while True:
        success, frame = read_frame()
        if not success:
            break

        if camera_matrix is None:
            height, width = frame.shape[:2]
            camera_matrix, dist_coeffs = utils.init_camera_intrinsics(width, height)

        results = detect_hands(frame)
        gestures = extract_gestures(results)

        # seule la première main détectée pilote les cubes (état de geste mono-main)
        if results.multi_hand_landmarks:
            primary_hand = results.multi_hand_landmarks[0]
            stable_gesture = stabilize_gesture(gestures[0])
            allow_pinch_create = should_create_on_pinch(stable_gesture)

            palm = primary_hand.landmark[PALM_LANDMARK]
            pixel_x = palm.x * frame.shape[1]
            pixel_y = palm.y * frame.shape[0]
            hand_world_pos = utils.project_hand_to_fixed_plane(pixel_x, pixel_y, camera_matrix)

            utils.handle_pinch_creation(allow_pinch_create, hand_world_pos)
            utils.handle_fist_move(stable_gesture == "poing", hand_world_pos)
            utils.handle_open_copy(stable_gesture == "main ouverte", hand_world_pos)

        frame = utils.draw_cubes(frame, camera_matrix, dist_coeffs)
        frame = draw_hands(frame, results)
        frame = draw_gestures(frame, gestures)
        frame = draw_status(frame, bool(results.multi_hand_landmarks))

        show_frame(frame)

        if should_quit():
            break

    release_camera()


if __name__ == "__main__":
    run()
