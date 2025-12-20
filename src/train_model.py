import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- CONFIGURATION (Vérifie bien les chemins !) ---
# Selon tes captures, ton dossier s'appelle "DATASET"
BASE_DIR = "data/raw/DATASET" 
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Paramètres RAF-DB
IMG_SIZE = (100, 100) # Taille validée
BATCH_SIZE = 32
EPOCHS = 15           # On laisse tourner 15 fois pour bien apprendre
NUM_CLASSES = 7

def build_model():
    """Construit le modèle MobileNetV2 adapté"""
    # 1. Télécharger le cerveau pré-entraîné de Google (MobileNetV2)
    # include_top=False : On enlève la dernière couche (qui classait des chats/chiens)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # 2. Geler le modèle de base (On ne touche pas à ce qu'il sait déjà)
    base_model.trainable = False 
    
    # 3. Ajouter notre tête personnalisée (Pour nos 7 émotions)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Sécurité anti-par-coeur
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 4. Compilation avec un learning rate faible (0.0001) pour être précis
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print(f"--- Démarrage de l'entraînement sur {os.path.abspath(TRAIN_DIR)} ---")

    # --- PRÉPARATION DES DONNÉES ---
    # Data Augmentation pour le train : on rend l'IA plus robuste
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalisation
        rotation_range=20,      # Rotation légère
        width_shift_range=0.1,  # Décalage
        height_shift_range=0.1,
        horizontal_flip=True,   # Miroir
        fill_mode='nearest'
    )

    # Pour le test : Juste la normalisation (pas de trucage)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Chargement des images d'entraînement...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    print("Chargement des images de test...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )

    # AFFICHE CECI À L'ÉQUIPE : C'est l'ordre officiel des émotions pour Nadjma
    print("⚠️ MAPPING DES CLASSES À NOTER :", train_generator.class_indices)

    # --- ENTRAÎNEMENT ---
    model = build_model()
    print("\nLe modèle apprend... (Cela peut prendre 10-20 min selon ton PC)")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # --- SAUVEGARDE ---
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/emotion_model_rafdb.h5')
    print("\n✅ Modèle sauvegardé : models/emotion_model_rafdb.h5")

    # --- GRAPHIQUE ---
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Précision (Plus haut = Mieux)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Erreur (Plus bas = Mieux)')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()