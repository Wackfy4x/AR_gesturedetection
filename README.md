# AR Gesture Cubes

Manipulation d'objets 3D en réalité augmentée, contrôlée à la main via une simple webcam. Aucun casque, aucun capteur externe : juste une webcam et un marqueur imprimé.

## Aperçu

Ce projet ancre des cubes virtuels dans l'espace réel grâce à un marqueur ArUco détecté par la caméra, puis permet de les créer et de les déplacer à mains nues grâce à la reconnaissance de gestes.

| Geste | Action |
|---|---|
| Pincement (pouce + index) | Crée un cube à l'emplacement de la main |
| Poing fermé | Attrape le cube le plus proche et le déplace |
| Main ouverte | Relâche le cube tenu |

## Stack technique

- **OpenCV** — capture vidéo et détection du marqueur ArUco (`cv2.aruco`)
- **MediaPipe Hands** — détection des 21 points de repère de la main en temps réel
- **PyOpenGL** — rendu 3D et compositing avec le flux caméra
- **GLFW** — gestion de la fenêtre et du contexte OpenGL
- **NumPy** — calculs géométriques (projection, changement de repère)

## Comment ça marche

1. La webcam capture le flux vidéo en continu.
2. MediaPipe détecte la main et ses 21 points de repère à chaque frame.
3. La position et la forme de la main sont interprétées en gestes (pincement, poing fermé) par calcul de distances géométriques simples entre landmarks.
4. Un marqueur ArUco visible dans le champ de la caméra fournit un repère 3D fixe dans le monde réel (pose calculée via `solvePnP`).
5. La position 2D de la main est projetée dans ce repère 3D par intersection rayon/plan.
6. Les gestes détectés déclenchent la création ou le déplacement de cubes, ancrés dans ce repère et rendus avec PyOpenGL par-dessus le flux caméra.

## Installation

```bash
git clone https://github.com/<ton-nom-utilisateur>/ar-gesture-cubes.git
cd ar-gesture-cubes
pip install -r requirements.txt
```

### Prérequis

- Python 3.10+
- Une webcam
- Un marqueur ArUco imprimé (dictionnaire `DICT_4X4_50`), taille mesurée précisément et renseignée dans la config

## Utilisation

```bash
python main.py
```

Place le marqueur ArUco bien à plat dans le champ de la webcam, puis pince ou ferme le poing au-dessus pour créer et déplacer des cubes.

## Limites connues

- Calibration caméra approximative par défaut (paramètres intrinsèques estimés, pas de calibration par échiquier) — l'ancrage peut légèrement dériver.
- Un seul marqueur pris en charge à la fois.
- Détection de gestes sensible à la distance à la caméra (seuils calibrés empiriquement).
- Un seul type de forme (cube) pour l'instant.

## Pistes d'amélioration

- Calibration caméra propre (`cv2.calibrateCamera`)
- Formes supplémentaires (sphère, cylindre) et geste de sélection de forme
- Redimensionnement à deux mains
- Suivi multi-marqueurs pour une scène plus grande

## Licence

À définir (ex. MIT).
