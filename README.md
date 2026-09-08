# AR Gesture Cubes

Manipulation d'objets 3D en réalité augmentée, contrôlée à la main via une simple webcam. Aucun casque, aucun capteur externe, aucun marqueur à imprimer : juste une webcam fixe.

## Aperçu

Ce projet ancre des cubes virtuels sur un plan fixe devant la webcam, puis permet de les créer, déplacer et copier à mains nues grâce à la reconnaissance de gestes. Il n'utilise aucun marqueur physique : l'ancrage repose sur l'hypothèse que la caméra reste immobile pendant l'utilisation (voir [Limites connues](#limites-connues)).

| Geste | Action |
|---|---|
| Pincement (pouce + index) | Crée un cube à l'emplacement de la main |
| Poing fermé | Attrape le cube le plus proche et le déplace tant que le poing reste fermé ; le relâche dès qu'il s'ouvre |
| Main ouverte maintenue au-dessus d'un cube | Copie le cube ciblé |

## État d'avancement

- ✅ Capture vidéo (OpenCV)
- ✅ Détection des mains et de leurs 21 landmarks (MediaPipe)
- ✅ Détection des gestes (pincement, poing, main ouverte) via `interpret_gesture`
- ✅ Projection main → point 3D sur un plan fixe devant la caméra (`utils.py`), branchée dans la boucle principale de `main.py`
- ✅ Logique de création (`handle_pinch_creation`), déplacement (`handle_fist_move`) et copie (`handle_open_copy`) de cubes, câblée dans `run()`
- ✅ Rendu des cubes : arêtes projetées à chaque frame avec `cv2.projectPoints` et dessinées par-dessus le flux caméra (`utils.draw_cubes`) — solution légère en attendant un éventuel rendu PyOpenGL
- ❌ Rendu PyOpenGL/GLFW plein (texturé, éclairé) — les dépendances sont dans `requirements.txt` mais pas encore utilisées

## Stack technique

- **OpenCV** — capture vidéo et rendu des cubes (`cv2.projectPoints` + `cv2.line`)
- **MediaPipe Hands** — détection des 21 points de repère de la main en temps réel
- **NumPy** — calculs géométriques (projection caméra → plan fixe)
- **PyOpenGL / GLFW** — présents dans les dépendances pour un futur rendu 3D plus riche (texturé, éclairé), non utilisés pour l'instant : le rendu actuel se fait directement en overlay OpenCV

## Comment ça marche

1. La webcam capture le flux vidéo en continu.
2. MediaPipe détecte la main et ses 21 points de repère à chaque frame.
3. La position et la forme de la main sont interprétées en gestes (pincement, poing fermé, main ouverte) par calcul de distances géométriques simples entre landmarks.
4. Un plan virtuel fixe est défini à une profondeur constante (`PLANE_DEPTH`, `utils.py`) devant la caméra. Comme il n'y a pas de marqueur physique, ce plan est simplement le repère caméra lui-même (rotation identité, translation nulle) — il reste donc valide **tant que la caméra ne bouge pas**.
5. Le centre de la paume (landmark 9) de la première main détectée est projeté sur ce plan par un rayon caméra → pixel inversé (`project_hand_to_fixed_plane`).
6. Les gestes détectés déclenchent la création, le déplacement ou la copie de cubes, dont la position 3D est stockée dans ce repère caméra. Chaque cube est ensuite reprojeté à l'écran (`cv2.projectPoints`) et dessiné en fil de fer par-dessus le flux caméra.

## Détail des gestes et interactions

Le geste brut détecté à chaque frame (`interpret_gesture`, `main.py`) est filtré par `stabilize_gesture` avant de piloter les cubes : un changement de geste n'est confirmé qu'après `STABILITY_FRAMES` (3) lectures consécutives identiques. Cela évite qu'un flicker d'une seule frame (ex. pouce et index qui se frôlent brièvement en refermant la main) ne déclenche une action non voulue. Pour la même raison, `interpret_gesture` teste le poing avant le pincement, et `is_pinching` exige que les 3 autres doigts soient tendus (sinon un poing fermé, où pouce et index finissent par se toucher, serait lu comme un pincement).

Une garde supplémentaire (`should_create_on_pinch`) empêche un pincement de créer un cube s'il fait suite directement à un poing : en ouvrant la main après un déplacement, les doigts se déplient à des vitesses différentes et passent brièvement par un état ambigu qui se lit comme un pincement — sans cette garde, relâcher un cube en créait un nouveau au même endroit. Un pincement ne crée un cube que s'il fait suite à une main ouverte confirmée.

Le même problème existe dans l'autre sens : en refermant la main pour attraper un cube (main ouverte → poing), le pouce et l'index se rapprochent souvent avant que les 3 autres doigts ne finissent de se replier, ce qui se lit brièvement comme un pincement légitime (parti d'une main ouverte, donc non bloqué par la garde ci-dessus). Le filtre de stabilité de `STABILITY_FRAMES` (~100 ms) est trop court pour l'en distinguer de façon fiable. `should_create_on_pinch` exige donc en plus que le pincement soit maintenu au moins `PINCH_HOLD_SEC` (0.35 s) avant de créer un cube — un flicker de fermeture rapide expire avant ce délai, alors qu'un pincement volontaire maintenu le dépasse. Conséquence : la création d'un cube a un léger délai perceptible (~0.35 s) par rapport à un pincement instantané.

### Créer — pincement

Un cube est créé au moment où le pincement *commence* (front montant : `is_pinching` passe de `False` à `True`), à la position 3D de la main au moment du pincement. Tenir le pincement ne crée pas de cubes en rafale — il faut relâcher puis pincer à nouveau pour en créer un autre.

### Déplacer — poing fermé

À la fermeture du poing, le cube le plus proche de la main est attrapé s'il se trouve dans le rayon de préhension (`GRAB_RADIUS`, `utils.py`). Tant que le poing reste fermé, ce cube suit la position de la main. Dès que le poing s'ouvre, le cube est automatiquement relâché à sa position courante — il n'y a pas de geste de relâchement dédié.

### Copier — main ouverte

La main ouverte est aussi l'état "neutre" par défaut (ni pincement, ni poing), ce qui pose un problème de distinction : il faut éviter de copier un cube en permanence tant que la main reste ouverte au-dessus. Le déclenchement retenu est un **temps de maintien (dwell time)** :

1. Dès qu'une main ouverte est détectée à moins de `GRAB_RADIUS` d'un cube, un minuteur démarre (`DWELL_TIME_SEC`, ~800 ms).
2. Si la main reste ouverte et proche du même cube jusqu'à l'échéance, le cube est dupliqué (nouveau cube décalé de `COPY_OFFSET` pour rester visible séparément de l'original), et le minuteur se réarme aussitôt.
3. Si la main s'éloigne, change de geste, ou vise un autre cube avant l'échéance, le minuteur est annulé.

Conséquence du point 2 : maintenir la main ouverte au-dessus d'un cube le duplique en continu, une fois par `DWELL_TIME_SEC`.

Implémentée dans `handle_open_copy` (`utils.py`) et branchée dans la boucle principale de `main.py`.

## Installation

```bash
git clone https://github.com/<ton-nom-utilisateur>/ar-gesture-cubes.git
cd ar-gesture-cubes
pip install -r requirements.txt
```

### Prérequis

- Python 3.10+
- Une webcam, posée de façon stable (voir [Limites connues](#limites-connues))

> Une version précédente de ce projet utilisait un marqueur ArUco imprimé comme repère 3D (voir `assets/marker_id0.png`, encore présent mais plus utilisé par le code). Cette approche a été abandonnée au profit d'un plan virtuel fixe, plus simple à utiliser : aucune impression requise.

## Utilisation

```bash
python main.py
```

Place ta main dans le champ de la webcam, à une distance à peu près constante (le plan virtuel est fixé à `PLANE_DEPTH`, ~40 cm par défaut). Pince pour créer un cube, ferme le poing sur un cube pour le déplacer, et maintiens la main ouverte au-dessus d'un cube pour le copier. Quitte avec la touche `q`.

## Limites connues

- **La caméra doit rester immobile** : l'ancrage des cubes n'a aucun repère externe (pas de marqueur, pas de SLAM) — il utilise directement le repère caméra comme repère monde. Bouger la webcam fait dériver tous les cubes existants par rapport à la scène réelle.
- Calibration caméra approximative par défaut (paramètres intrinsèques estimés, pas de calibration par échiquier).
- `PLANE_DEPTH` est une profondeur estimée, pas mesurée : les cubes créés à des distances main-caméra très différentes de cette valeur seront positionnés de façon imprécise (la main reste projetée sur un seul plan de profondeur constante, quelle que soit sa distance réelle à la caméra).
- Seuils de gestes (`FOLDED_RATIO`, `PINCH_RATIO`, `main.py`) calibrés empiriquement ; ils sont désormais relatifs à la taille de la main détectée (`_palm_size`) et donc peu sensibles à la distance à la caméra, mais peuvent encore nécessiter un ajustement selon la morphologie de la main.
- Un seul type de forme (cube) pour l'instant.
- Un seul jeu d'état de geste (pincement/poing/dwell) partagé globalement : seule la première main détectée pilote la création/déplacement/copie, même si `max_num_hands=2`.
- Pas de geste de relâchement dédié pour la copie : le dwell time (main ouverte + proximité) peut déclencher une copie non désirée si la main s'attarde par hasard près d'un cube, et se répète tant qu'elle reste ouverte au-dessus.
- Pas de geste de suppression de cube pour l'instant.
- Rendu en fil de fer simple (`cv2.line`), sans faces pleines, éclairage ni gestion de l'occlusion.

## Pistes d'amélioration

- Calibration caméra propre (`cv2.calibrateCamera`)
- Estimation de profondeur de la main (via la taille apparente de la main, ou `z` MediaPipe) plutôt qu'un `PLANE_DEPTH` constant
- Recalage optionnel par marqueur ArUco (le code existe encore dans l'historique git) pour les cas où la caméra doit pouvoir bouger
- Rendu PyOpenGL/GLFW (faces pleines, éclairage, meilleur compositing avec le flux caméra) en remplacement du fil de fer OpenCV actuel
- Geste de suppression (ex. "poing secoué" ou glisser un cube hors du plan)
- État de geste par main (pour piloter des cubes indépendamment à deux mains, ou redimensionner à deux mains)
- Formes supplémentaires (sphère, cylindre) et geste de sélection de forme

## Licence

À définir (ex. MIT).
