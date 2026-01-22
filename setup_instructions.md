# 🎯 Face Recognition Attendance System - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage Guide](#usage-guide)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Features](#advanced-features)

---

## Prerequisites

### System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Version 3.8 to 3.11
- **Webcam:** Built-in or external USB webcam
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** At least 500MB free space

### Software Requirements
- Python 3.8+
- pip (Python package manager)
- Git (optional, for cloning)

---

## Installation

### Step 1: Install Python
If you don't have Python installed:

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify: `python --version`

**macOS:**
```bash
brew install python@3.10
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Step 2: Create Project Directory
```bash
# Create and navigate to project folder
mkdir face_attendance
cd face_attendance
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies
Create a `requirements.txt` file with the following content:
```
opencv-contrib-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
streamlit==1.28.0
Pillow==10.0.1
```

Install packages:
```bash
pip install -r requirements.txt
```

### Step 5: Create Project Files
Copy all the provided Python files into your project directory:
- `capture_faces.py`
- `train_model.py`
- `recognize_attendance.py`
- `app.py`

---

## Project Structure

After setup, your directory should look like this:

```
face_attendance/
├── dataset/                    # Face images (auto-created)
│   ├── EMP001_John_Doe/
│   │   ├── EMP001_0.jpg
│   │   ├── EMP001_1.jpg
│   │   └── ...
│   └── EMP002_Jane_Smith/
│       └── ...
│
├── trainer/                    # Trained model (auto-created)
│   ├── face_model.yml
│   └── name_mapping.pkl
│
├── attendance/                 # Attendance records (auto-created)
│   └── attendance.csv
│
├── capture_faces.py           # User registration module
├── train_model.py             # Model training module
├── recognize_attendance.py    # Attendance marking module
├── app.py                     # Streamlit web interface
├── requirements.txt           # Dependencies
└── venv/                      # Virtual environment
```

---

## Usage Guide

### Method 1: Using Streamlit Web Interface (Recommended)

**Start the web application:**
```bash
streamlit run app.py
```

The interface will open in your browser at `http://localhost:8501`

**Workflow:**
1. **Register Users** → Add new users with their photos
2. **Train Model** → Train the face recognition model
3. **Mark Attendance** → Start real-time attendance marking
4. **View Attendance** → Check and download attendance records

---

### Method 2: Using Command Line

#### Step 1: Register a New User
```bash
python capture_faces.py
```
- Enter person's name (e.g., "John Doe")
- Enter ID (e.g., "EMP001")
- Look at the camera
- 30 images will be captured automatically
- Press 'Q' to quit early if needed

**Example:**
```
Enter person's name: John Doe
Enter ID (Roll No/Employee ID): EMP001

📸 Starting face capture for John Doe (ID: EMP001)
Target: 30 images
✓ Captured image 30/30
✅ Successfully captured 30 images!
```

#### Step 2: Train the Model
After registering **all users**, train the model:
```bash
python train_model.py
```

**Output:**
```
📂 Found 3 persons in dataset
  Loading: John Doe (ID: EMP001)
  Loading: Jane Smith (ID: EMP002)
  Loading: Bob Wilson (ID: EMP003)

✅ Loaded 90 face images from 3 persons
🔄 Training face recognition model...
✅ Model saved to: trainer/face_model.yml
✅ Name mapping saved to: trainer/name_mapping.pkl

TRAINING COMPLETE!
Total persons: 3
Total images: 90
```

#### Step 3: Mark Attendance
```bash
python recognize_attendance.py
```

- Camera window will open
- Stand in front of camera
- Your name will appear when recognized
- Attendance is marked automatically
- Press 'Q' to quit

**Output:**
```
✅ Model and mappings loaded successfully
📊 Total registered persons: 3
📋 0 attendance(s) already marked today

✅ Attendance marked: John Doe (EMP001) at 09:15:23
✅ Attendance marked: Jane Smith (EMP002) at 09:16:45
```

#### Step 4: View Attendance Records
Check the `attendance/attendance.csv` file:

| Name | ID | Date | Time |
|------|-------|------------|----------|
| John Doe | EMP001 | 2024-01-22 | 09:15:23 |
| Jane Smith | EMP002 | 2024-01-22 | 09:16:45 |
| Bob Wilson | EMP003 | 2024-01-22 | 09:17:12 |

---

## Key Features

### ✅ Automatic Attendance Marking
- Real-time face detection and recognition
- Displays person's name on screen
- Marks attendance with date and time
- Prevents duplicate entries for the same day

### 🔒 Security Features
- Face images stored locally
- Attendance records in CSV format
- One attendance per person per day
- Confidence-based recognition

### 📊 Attendance Management
- CSV export for Excel
- Date-wise filtering
- View attendance history
- Analytics dashboard (Streamlit UI)

### 🎨 User-Friendly Interface
- Web-based UI with Streamlit
- Simple navigation
- Real-time statistics
- Download reports

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'cv2'"
**Solution:**
```bash
pip install opencv-contrib-python==4.8.1.78
```

### Problem: Camera not opening
**Solutions:**
1. Check if another application is using the camera
2. Try changing camera index in code:
   ```python
   cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
   ```
3. Grant camera permissions to Python/Terminal

### Problem: "Face not detected during registration"
**Solutions:**
- Ensure good lighting
- Move closer to the camera
- Remove glasses or caps
- Face the camera directly

### Problem: Low recognition accuracy
**Solutions:**
1. Capture more images (increase to 50)
2. Ensure variety in head angles during capture
3. Lower confidence threshold:
   ```python
   system.recognize_faces(confidence_threshold=80)
   ```
4. Retrain the model with better images

### Problem: "Model not found" error
**Solution:**
Run training first:
```bash
python train_model.py
```

---

## Advanced Configuration

### Adjust Recognition Sensitivity
In `recognize_attendance.py`, modify:
```python
confidence_threshold=70  # Lower = more strict (50-70 recommended)
```

### Change Number of Capture Images
In `capture_faces.py`:
```python
capturer.capture_images(person_name, person_id, num_images=50)
```

### Modify Face Detection Parameters
In any module:
```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,      # Decrease for more detections
    minNeighbors=5,       # Increase for fewer false positives
    minSize=(100, 100)    # Minimum face size
)
```

---

## Performance Tips

1. **Better Accuracy:**
   - Capture images in similar lighting to recognition environment
   - Include various angles (front, slight left, slight right)
   - Ensure clear, unblurred images

2. **Faster Processing:**
   - Use smaller image sizes
   - Reduce video resolution
   - Close other applications

3. **Database Scaling:**
   - For 50+ users, consider switching to SQLite
   - Implement batch training
   - Use multi-threading for real-time detection

---

## CSV File Format

The attendance CSV has the following structure:

```csv
Name,ID,Date,Time
John Doe,EMP001,2024-01-22,09:15:23
Jane Smith,EMP002,2024-01-22,09:16:45
```

Can be opened in Excel, Google Sheets, or any spreadsheet application.

---

## Export to Excel

Using Python:
```python
import pandas as pd

df = pd.read_csv('attendance/attendance.csv')
df.to_excel('attendance.xlsx', index=False)
```

Or use the Streamlit UI's download button to get CSV format.

---

## Next Steps / Enhancements

Ready to add more features? Consider:

1. **Database Integration:**
   - SQLite for better performance
   - MySQL for multi-user systems

2. **Web Dashboard:**
   - Flask/Django for full web app
   - User authentication
   - Admin panel

3. **Cloud Storage:**
   - Firebase for cloud backup
   - Real-time sync across devices

4. **Additional Features:**
   - Email notifications
   - SMS alerts
   - Mask detection
   - Temperature screening
   - Multi-camera support

5. **Mobile App:**
   - Flutter/React Native app
   - Remote attendance marking

---

## Support & Resources

- **OpenCV Documentation:** https://docs.opencv.org/
- **Streamlit Docs:** https://docs.streamlit.io/
- **Python Face Recognition:** https://github.com/ageitgey/face_recognition

---

## License & Credits

This is an educational project built with:
- OpenCV (Computer Vision)
- Python (Programming)
- Streamlit (Web Interface)
- LBPH Face Recognizer (Face Recognition)

---

**🎉 You're all set! Start by running `streamlit run app.py` and enjoy your attendance system!**
