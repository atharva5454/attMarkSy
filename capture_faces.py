"""
Face Capture Module for Attendance System
Captures multiple images of a person for dataset creation
"""

import cv2
import os
from pathlib import Path

class FaceCapture:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create dataset directory if it doesn't exist
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
    
    def capture_images(self, person_name, person_id, num_images=30):
        """
        Capture multiple images of a person for training
        
        Args:
            person_name: Name of the person
            person_id: Unique ID (roll number/employee ID)
            num_images: Number of images to capture (default: 30)
        """
        # Create person-specific folder
        person_folder = os.path.join(self.dataset_path, f"{person_id}_{person_name}")
        Path(person_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return False
        
        count = 0
        print(f"\n📸 Starting face capture for {person_name} (ID: {person_id})")
        print(f"Target: {num_images} images")
        print("Instructions:")
        print("- Look at the camera")
        print("- Move your head slightly in different angles")
        print("- Press 'q' to quit early\n")
        
        while count < num_images:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save face image
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(person_folder, f"{person_id}_{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                
                # Display progress
                cv2.putText(
                    frame,
                    f"Captured: {count}/{num_images}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                print(f"✓ Captured image {count}/{num_images}", end='\r')
            
            # Display frame
            cv2.imshow('Face Capture - Press Q to quit', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Successfully captured {count} images!")
        print(f"📁 Saved to: {person_folder}")
        return True

def main():
    """Main function to run face capture"""
    print("=" * 50)
    print("FACE REGISTRATION SYSTEM")
    print("=" * 50)
    
    # Get user input
    person_name = input("\nEnter person's name: ").strip()
    person_id = input("Enter ID (Roll No/Employee ID): ").strip()
    
    if not person_name or not person_id:
        print("❌ Error: Name and ID cannot be empty!")
        return
    
    # Create capture object and start
    capturer = FaceCapture()
    capturer.capture_images(person_name, person_id, num_images=30)

if __name__ == "__main__":
    main()
