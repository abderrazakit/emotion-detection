import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque  # <--- IMPORT IMPORTANT POUR LA MÉMOIRE

# --- CHARGEMENT DU MODÈLE ---
model_path = os.path.join('models', 'rafdb_mobilenet_best.h5')

print("Loading Model...")
try:
    classifier = load_model(model_path)
    print(f"✅ Modèle chargé depuis : {model_path}")
except:
    print(f"❌ Erreur critique : Le fichier est introuvable ici : {model_path}")
    print("Vérifie que le fichier 'rafdb_mobilenet_best.h5' est bien dans le dossier 'models'.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Attention : Assure-toi que l'ordre correspond bien à ton entraînement (vérifie class_indices si doute)
emotion_labels = ['Surprise', 'Disgust', 'Fear', 'Happy', 'Sad', 'Angry', 'Neutral']

# --- CONFIGURATION STABILISATION ---
# On garde en mémoire les 8 dernières prédictions pour faire une moyenne
emotion_buffer = deque(maxlen=8) 

cap = cv2.VideoCapture(0)
print("Starting Webcam. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Si aucun visage n'est détecté, on vide la mémoire pour ne pas mélanger les anciennes émotions
    if len(faces) == 0:
        emotion_buffer.clear()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        try:
            roi_color = frame[y:y + h, x:x + w]
            # Redimensionnement 224x224 (Standard MobileNet)
            roi = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = preprocess_input(roi)

            # --- PRÉDICTION ---
            prediction = classifier.predict(roi, verbose=0)[0] # Tableau de probabilités

            # --- STABILISATION (Le cœur de la solution) ---
            # 1. On ajoute les probas actuelles à la mémoire
            emotion_buffer.append(prediction)
            
            # 2. On calcule la MOYENNE de toutes les probas en mémoire
            avg_prediction = np.mean(emotion_buffer, axis=0)
            
            # 3. On décide l'émotion sur la base de la MOYENNE
            max_index = np.argmax(avg_prediction)
            label = emotion_labels[max_index]
            confidence = avg_prediction[max_index]

            # --- AFFICHAGE ---
            label_text = f"{label} ({confidence * 100:.0f}%)"
            
            # Change la couleur du texte selon la confiance (Vert si sûr, Rouge si hésitant)
            text_color = (0, 255, 0) if confidence > 0.50 else (0, 0, 255)
            
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        except Exception as e:
            print(f"Erreur: {e}")
            continue

    cv2.imshow('MobileNet Stabilisé', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()