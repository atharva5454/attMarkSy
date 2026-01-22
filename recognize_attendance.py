"""
Face Recognition and Attendance Marking Module
Recognizes faces in real-time and marks attendance
"""

import cv2
import os
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

class AttendanceSystem:
    def __init__(self, trainer_path='trainer', attendance_path='attendance'):
        self.trainer_path = trainer_path
        self.attendance_path = attendance_path
        self.attendance_file = os.path.join(attendance_path, 'attendance.csv')
        
        # Create attendance directory if it doesn't exist
        Path(self.attendance_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize face cascade and recognizer
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load trained model
        self.load_model()
        
        # Track today's attendance to prevent duplicates
        self.today_attendance = set()
        self.load_today_attendance()
    
    def load_model(self):
        """Load the trained face recognition model and name mapping"""
        model_path = os.path.join(self.trainer_path, 'face_model.yml')
        mapping_path = os.path.join(self.trainer_path, 'name_mapping.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Model not found at {model_path}. Please train the model first!"
            )
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"❌ Name mapping not found at {mapping_path}. Please train the model first!"
            )
        
        # Load model and mapping
        self.recognizer.read(model_path)
        with open(mapping_path, 'rb') as f:
            self.name_map = pickle.load(f)
        
        print("✅ Model and mappings loaded successfully")
        print(f"📊 Total registered persons: {len(self.name_map)}")
    
    def load_today_attendance(self):
        """Load attendance already marked today to prevent duplicates"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if os.path.exists(self.attendance_file):
            try:
                df = pd.read_csv(self.attendance_file)
                today_records = df[df['Date'] == today]
                self.today_attendance = set(today_records['ID'].values)
                print(f"📋 {len(self.today_attendance)} attendance(s) already marked today")
            except Exception as e:
                print(f"⚠️ Warning: Could not load today's attendance: {e}")
    
    def mark_attendance(self, person_id, person_name):
        """
        Mark attendance for a person
        
        Args:
            person_id: Person's ID
            person_name: Person's name
        """
        # Check if already marked today
        if person_id in self.today_attendance:
            return False
        
        # Get current date and time
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        
        # Create attendance record
        record = {
            'Name': person_name,
            'ID': person_id,
            'Date': date,
            'Time': time
        }
        
        # Append to CSV file
        df = pd.DataFrame([record])
        
        if os.path.exists(self.attendance_file):
            df.to_csv(self.attendance_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.attendance_file, mode='w', header=True, index=False)
        
        # Add to today's attendance set
        self.today_attendance.add(person_id)
        
        print(f"✅ Attendance marked: {person_name} ({person_id}) at {time}")
        return True
    
    def recognize_faces(self, confidence_threshold=70):
        """
        Start real-time face recognition and attendance marking
        
        Args:
            confidence_threshold: Recognition confidence threshold (lower is better)
        """
        print("\n" + "=" * 50)
        print("ATTENDANCE SYSTEM - FACE RECOGNITION")
        print("=" * 50)
        print("Instructions:")
        print("- Stand in front of the camera")
        print("- Attendance will be marked automatically")
        print("- Press 'q' to quit")
        print("=" * 50 + "\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not access webcam")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face
                label, confidence = self.recognizer.predict(face_roi)
                
                # Check confidence threshold
                if confidence < confidence_threshold:
                    # Face recognized
                    person_info = self.name_map[label]
                    person_name = person_info['name']
                    person_id = person_info['id']
                    
                    # Draw green rectangle and name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display name and confidence
                    text = f"{person_name}"
                    cv2.putText(
                        frame,
                        text,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display ID
                    cv2.putText(
                        frame,
                        f"ID: {person_id}",
                        (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    # Mark attendance (will only mark once per day)
                    self.mark_attendance(person_id, person_name)
                    
                else:
                    # Unknown face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Unknown",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
            
            # Display attendance count
            cv2.putText(
                frame,
                f"Today's Attendance: {len(self.today_attendance)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Face Recognition Attendance - Press Q to quit', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 50)
        print(f"✅ Attendance session ended")
        print(f"📊 Total attendance marked today: {len(self.today_attendance)}")
        print("=" * 50)

def main():
    """Main function to run attendance recognition"""
    try:
        system = AttendanceSystem()
        system.recognize_faces(confidence_threshold=70)
    except FileNotFoundError as e:
        print(e)
        print("\n💡 Run 'python train_model.py' first to train the model")

if __name__ == "__main__":
    main()
