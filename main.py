import cv2
# Imports pour le Modèle d'Émotion
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
from tensorflow.keras.models import Model as KerasModel # Ajout pour la compatibilité

# --- TÂCHE 5 : Chargement du Modèle d'Émotion ---

# 1. Charger la structure (JSON)
# On ouvre le fichier JSON qui décrit l'architecture du modèle
try:
    json_file = open('Emotion_Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # On crée le modèle à partir de l'architecture JSON, en spécifiant l'objet Model pour la compatibilité
    emotion_model = model_from_json(loaded_model_json, custom_objects={'Model': KerasModel})

    # 2. Charger les poids (H5) dans le nouveau modèle
    # Le fichier Emotion_h5_file.h5 contient les poids entraînés
    emotion_model.load_weights("Emotion_h5_file.h5")

    # 3. Définir les étiquettes d'émotion
    # Ce dictionnaire associe l'index de prédiction (0 à 6) à une émotion
    emotion_labels = {0: 'Colere', 1: 'Degout', 2: 'Peur', 3: 'Heureux', 4: 'Triste', 5: 'Surprise', 6: 'Neutre'}
    
    print("Modèle d'émotion chargé avec succès.")
    
except Exception as e:
    print(f"Erreur lors du chargement du modèle d'émotion : {e}")

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

print("Caméra activée. Détection d'émotion en cours. Appuyez sur 'q' pour fermer.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur: Impossible de recevoir l'image.")
        break

    # Convertir l'image en niveaux de gris (nécessaire pour la détection Haar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Traitement de chaque visage détecté
    for (x, y, w, h) in faces:
        
        # 1. Dessiner le rectangle autour du visage (pour le visuel)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 2. Extraire et préparer la région d'intérêt (ROI)
        # On extrait la zone du visage en niveaux de gris (gray)
        roi_gray = gray[y:y + h, x:x + w]
        
        # CORRECTION DE LA TAILLE ICI : 48x48 remplacé par 128x128
        cropped_img = cv2.resize(roi_gray, (128, 128)) 
        
        # 3. Prétraitement de l'image pour le modèle Keras
        # La forme attendue est (1, 128, 128, 1)
        cropped_img = np.expand_dims(cropped_img, axis=0) # Ajoute la dimension du lot (batch)
        cropped_img = np.expand_dims(cropped_img, axis=-1) # Ajoute la dimension du canal (grayscale)
        cropped_img = cropped_img / 255.0 # Normalisation des pixels entre 0 et 1
        
        # 4. Faire la prédiction
        try:
            emotion_prediction = emotion_model.predict(cropped_img)
        except NameError:
            emotion_label = "ERREUR MODELE"
        else:
            # Trouver l'indice de l'émotion la plus probable (le plus haut score)
            max_index = int(np.argmax(emotion_prediction))
            
            # Récupérer l'étiquette d'émotion correspondante
            emotion_label = emotion_labels[max_index]
        
        # 5. Afficher le résultat sur l'image
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Afficher la frame
    cv2.imshow('Webcam - Face and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. Nettoyage ---
cap.release()
cv2.destroyAllWindows()