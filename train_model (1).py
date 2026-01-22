"""
Model Training Module for Face Recognition Attendance System
Trains LBPH Face Recognizer using captured face images
"""

import cv2
import os
import numpy as np
from pathlib import Path
import pickle

class FaceTrainer:
    def __init__(self, dataset_path='dataset', trainer_path='trainer'):
        self.dataset_path = dataset_path
        self.trainer_path = trainer_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create trainer directory if it doesn't exist
        Path(self.trainer_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    def get_images_and_labels(self):
        """
        Load all face images from dataset and create labels
        
        Returns:
            faces: List of face images
            ids: List of corresponding person IDs
            name_map: Dictionary mapping ID to name
        """
        faces = []
        ids = []
        name_map = {}
        
        # Check if dataset exists
        if not os.path.exists(self.dataset_path):
            print(f"❌ Error: Dataset path '{self.dataset_path}' not found!")
            return faces, ids, name_map
        
        # Get all person folders
        person_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        if not person_folders:
            print("❌ Error: No person folders found in dataset!")
            return faces, ids, name_map
        
        print(f"\n📂 Found {len(person_folders)} persons in dataset")
        
        # Process each person's folder
        for idx, folder_name in enumerate(person_folders):
            folder_path = os.path.join(self.dataset_path, folder_name)
            
            # Extract person ID and name from folder name (format: ID_Name)
            try:
                person_id, person_name = folder_name.split('_', 1)
                name_map[idx] = {'id': person_id, 'name': person_name}
                print(f"  Loading: {person_name} (ID: {person_id})")
            except ValueError:
                print(f"  ⚠️ Skipping invalid folder: {folder_name}")
                continue
            
            # Load all images from this person's folder
            image_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for image_file in image_files:
                img_path = os.path.join(folder_path, image_file)
                
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Detect face in image
                face_detected = self.face_cascade.detectMultiScale(img, 1.3, 5)
                
                for (x, y, w, h) in face_detected:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(idx)
        
        print(f"\n✅ Loaded {len(faces)} face images from {len(person_folders)} persons")
        return faces, ids, name_map
    
    def train(self):
        """Train the face recognition model"""
        print("=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        # Load images and labels
        faces, ids, name_map = self.get_images_and_labels()
        
        if len(faces) == 0:
            print("❌ Error: No faces found for training!")
            return False
        
        # Train the recognizer
        print("\n🔄 Training face recognition model...")
        self.recognizer.train(faces, np.array(ids))
        
        # Save the trained model
        model_path = os.path.join(self.trainer_path, 'face_model.yml')
        self.recognizer.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Save name mapping
        mapping_path = os.path.join(self.trainer_path, 'name_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(name_map, f)
        print(f"✅ Name mapping saved to: {mapping_path}")
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)
        print(f"Total persons: {len(name_map)}")
        print(f"Total images: {len(faces)}")
        
        return True

def main():
    """Main function to run model training"""
    trainer = FaceTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
