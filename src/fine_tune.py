import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "data/raw/DATASET" # Assure-toi que c'est le bon chemin (celui qui a marché pour train_model.py)
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS_FINE = 10 

def main():
    print("--- Démarrage du Fine-Tuning ---")

    # 1. Charger le modèle précédent
    model_path = 'models/emotion_model_rafdb.h5'
    if not os.path.exists(model_path):
        print(f"❌ ERREUR: Le fichier {model_path} n'existe pas.")
        return
        
    print(f"Chargement de {model_path}...")
    model = load_model(model_path)

    # 2. Dégeler le modèle intelligemment
    # CORRECTION ICI : On agit directement sur 'model', pas sur une sous-couche
    model.trainable = True # On déverrouille tout d'abord

    # On veut geler les 100 premières couches (les connaissances de base de MobileNet)
    # Et laisser les dernières (spécifiques) apprendre.
    fine_tune_at = 100
    
    print(f"Nombre total de couches : {len(model.layers)}")
    print(f"On gèle les {fine_tune_at} premières couches, on entraîne le reste.")

    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    # 3. Re-compilation (Trés important : Learning rate très bas !)
    model.compile(optimizer=Adam(learning_rate=1e-5), # Très lent et précis
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- PRÉPARATION DONNÉES ---
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True, 
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Lecture des données dans {TRAIN_DIR}...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='rgb'
    )
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='rgb'
    )

    # --- ENTRAÎNEMENT FINAL ---
    print("\n--- GO ! Spécialisation du cerveau IA en cours... ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS_FINE,
        validation_data=validation_generator
    )

    # --- SAUVEGARDE ---
    final_path = 'models/emotion_model_rafdb_final.h5'
    model.save(final_path)
    print(f"\n✅ VICTOIRE ! Modèle optimisé sauvegardé sous : {final_path}")

    # --- GRAPHIQUE ---
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Précision Fine-Tuning')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()