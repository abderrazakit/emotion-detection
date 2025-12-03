import cv2

# --- 1. Initialisation des Classificateurs ---
# Charger le classificateur Haar Cascade pour la détection de visage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Vérification si le classificateur a pu être chargé
if face_cascade.empty():
    print("Erreur: Impossible de charger 'haarcascade_frontalface_default.xml'. Assurez-vous qu'il est dans le répertoire du projet.")
    exit()

# --- 2. Initialisation de la capture vidéo ---
# Le '0' indique d'utiliser la caméra par défaut (la première trouvée).
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit() 

print("Caméra activée. Détection de visage en cours. Appuyez sur 'q' pour fermer.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur: Impossible de recevoir l'image.")
        break

    # Convertir l'image en niveaux de gris (nécessaire pour la détection Haar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner un rectangle autour de chaque visage détecté
    for (x, y, w, h) in faces:
        # (0, 255, 0) est la couleur BGR (vert), 2 est l'épaisseur de la ligne
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher la frame
    cv2.imshow('Webcam - Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. Nettoyage ---
cap.release()
cv2.destroyAllWindows()