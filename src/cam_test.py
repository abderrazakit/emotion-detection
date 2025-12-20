import cv2
import sys

def start_webcam():
    # 1. Charger le détecteur de visage pré-entraîné (Haar Cascade)
    # OpenCV le fournit par défaut
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2. Ouvrir la webcam (0 est généralement la webcam par défaut)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERREUR: Impossible d'accéder à la webcam.")
        sys.exit()

    print("✅ Webcam active. Appuie sur 'q' pour quitter.")

    while True:
        # Lire une image (frame) de la vidéo
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture du flux vidéo.")
            break

        # 3. Optimisation : Convertir en Gris pour la détection
        # (Haar Cascade travaille mieux en noir et blanc)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Détecter les visages
        # scaleFactor=1.1 : Réduit l'image de 10% à chaque passe pour trouver les gros et petits visages
        # minNeighbors=5 : Qualité de détection (plus c'est haut, moins il y a de faux positifs)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 5. Dessiner un rectangle autour de chaque visage trouvé
        for (x, y, w, h) in faces:
            # Dessine un rectangle vert (BGR : 0, 255, 0) avec une épaisseur de 2px
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # (Pour le Sprint 2, c'est ICI qu'on ajoutera le code pour appeler l'IA d'Achraf)
            # region_of_interest = frame[y:y+h, x:x+w]
            # prediction = model.predict(region_of_interest)

        # 6. Afficher le résultat
        cv2.imshow('Detection Visage - Sprint 1', frame)

        # Quitter si on appuie sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam()