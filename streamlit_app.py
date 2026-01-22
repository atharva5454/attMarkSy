"""
Streamlit Web Interface for Face Recognition Attendance System
User-friendly GUI for managing the attendance system
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📸 Face Recognition Attendance System</h1>', 
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Register User", "Train Model", "Mark Attendance", "View Attendance"]
)

# Initialize session state
if 'registration_success' not in st.session_state:
    st.session_state.registration_success = False

# Helper functions
def check_dataset_exists():
    """Check if dataset directory exists and has data"""
    if not os.path.exists('dataset'):
        return False, 0
    folders = [f for f in os.listdir('dataset') 
               if os.path.isdir(os.path.join('dataset', f))]
    return len(folders) > 0, len(folders)

def check_model_exists():
    """Check if trained model exists"""
    return os.path.exists('trainer/face_model.yml')

def get_attendance_stats():
    """Get attendance statistics"""
    if not os.path.exists('attendance/attendance.csv'):
        return 0, 0, None
    
    df = pd.read_csv('attendance/attendance.csv')
    total_records = len(df)
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = len(df[df['Date'] == today])
    
    return total_records, today_records, df

# Page: Home
if page == "Home":
    st.header("Welcome to Face Recognition Attendance System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 System Status")
        dataset_exists, num_users = check_dataset_exists()
        model_exists = check_model_exists()
        total_att, today_att, _ = get_attendance_stats()
        
        st.metric("Registered Users", num_users)
        st.metric("Model Status", "✅ Trained" if model_exists else "❌ Not Trained")
        st.metric("Today's Attendance", today_att)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        **Step 1:** Register users with their photos  
        **Step 2:** Train the recognition model  
        **Step 3:** Start marking attendance  
        **Step 4:** View attendance records
        """)
    
    with col3:
        st.subheader("ℹ️ Features")
        st.markdown("""
        ✓ Real-time face detection  
        ✓ Automatic attendance marking  
        ✓ Duplicate prevention  
        ✓ CSV export support  
        ✓ Easy to use interface
        """)
    
    # System info
    st.markdown("---")
    st.subheader("📋 System Information")
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.info(f"**Dataset Location:** dataset/")
        st.info(f"**Model Location:** trainer/")
    with info_col2:
        st.info(f"**Attendance File:** attendance/attendance.csv")
        st.info(f"**Total Records:** {total_att}")

# Page: Register User
elif page == "Register User":
    st.header("👤 Register New User")
    
    st.markdown("""
    <div class="info-box">
    <strong>Instructions:</strong><br>
    1. Enter the user's name and ID<br>
    2. Click 'Start Registration'<br>
    3. Look at the camera and move your head slightly<br>
    4. 30 images will be captured automatically
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        person_name = st.text_input("Enter Name", placeholder="John Doe")
    with col2:
        person_id = st.text_input("Enter ID (Roll No/Employee ID)", placeholder="EMP001")
    
    if st.button("🎥 Start Registration", type="primary"):
        if not person_name or not person_id:
            st.error("❌ Please enter both name and ID!")
        else:
            with st.spinner("Opening camera for face capture..."):
                try:
                    # Run capture_faces.py as subprocess
                    from capture_faces import FaceCapture
                    capturer = FaceCapture()
                    success = capturer.capture_images(person_name, person_id, num_images=30)
                    
                    if success:
                        st.success(f"✅ Successfully registered {person_name} (ID: {person_id})")
                        st.balloons()
                    else:
                        st.error("❌ Registration failed. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show registered users
    st.markdown("---")
    st.subheader("📋 Registered Users")
    
    dataset_exists, num_users = check_dataset_exists()
    if dataset_exists:
        folders = [f for f in os.listdir('dataset') 
                   if os.path.isdir(os.path.join('dataset', f))]
        
        user_data = []
        for folder in folders:
            try:
                user_id, user_name = folder.split('_', 1)
                num_images = len([f for f in os.listdir(f'dataset/{folder}') 
                                if f.endswith(('.jpg', '.png'))])
                user_data.append({
                    'ID': user_id,
                    'Name': user_name,
                    'Images': num_images
                })
            except:
                continue
        
        if user_data:
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No users registered yet.")

# Page: Train Model
elif page == "Train Model":
    st.header("🧠 Train Face Recognition Model")
    
    dataset_exists, num_users = check_dataset_exists()
    
    if not dataset_exists:
        st.warning("⚠️ No users found in dataset. Please register users first.")
    else:
        st.success(f"✅ Found {num_users} registered users in dataset")
        
        st.markdown("""
        <div class="info-box">
        <strong>Training Information:</strong><br>
        • Training will process all registered user images<br>
        • This may take 30-60 seconds depending on dataset size<br>
        • You need to train the model after adding new users
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model... Please wait..."):
                try:
                    from train_model import FaceTrainer
                    trainer = FaceTrainer()
                    success = trainer.train()
                    
                    if success:
                        st.success("✅ Model trained successfully!")
                        st.balloons()
                        st.info("💡 You can now start marking attendance")
                    else:
                        st.error("❌ Training failed. Please check the logs.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show model status
    st.markdown("---")
    st.subheader("📊 Model Status")
    
    if check_model_exists():
        st.success("✅ Model is trained and ready")
        
        # Show model info
        if os.path.exists('trainer/name_mapping.pkl'):
            import pickle
            with open('trainer/name_mapping.pkl', 'rb') as f:
                name_map = pickle.load(f)
            st.metric("Persons in Model", len(name_map))
    else:
        st.warning("⚠️ Model not trained yet")

# Page: Mark Attendance
elif page == "Mark Attendance":
    st.header("✅ Mark Attendance")
    
    model_exists = check_model_exists()
    
    if not model_exists:
        st.warning("⚠️ Model not trained yet. Please train the model first.")
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Instructions:</strong><br>
        1. Click 'Start Attendance System'<br>
        2. Look at the camera<br>
        3. Attendance will be marked automatically when face is recognized<br>
        4. Each person can mark attendance only once per day<br>
        5. Press 'Q' to quit the camera window
        </div>
        """, unsafe_allow_html=True)
        
        # Show today's attendance
        _, today_count, df = get_attendance_stats()
        st.metric("Today's Attendance Count", today_count)
        
        if st.button("🎥 Start Attendance System", type="primary"):
            st.info("📹 Camera window will open. Press 'Q' to quit.")
            try:
                from recognize_attendance import AttendanceSystem
                system = AttendanceSystem()
                system.recognize_faces(confidence_threshold=70)
                st.success("✅ Attendance session completed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Page: View Attendance
elif page == "View Attendance":
    st.header("📊 View Attendance Records")
    
    total_att, today_att, df = get_attendance_stats()
    
    if df is None or len(df) == 0:
        st.info("No attendance records found.")
    else:
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", total_att)
        with col2:
            st.metric("Today's Attendance", today_att)
        with col3:
            unique_persons = df['ID'].nunique()
            st.metric("Unique Persons", unique_persons)
        
        # Filter options
        st.markdown("---")
        st.subheader("🔍 Filter Attendance")
        
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Select Date", datetime.now())
        with col2:
            name_filter = st.multiselect(
                "Filter by Name",
                options=df['Name'].unique()
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if date_filter:
            date_str = date_filter.strftime('%Y-%m-%d')
            filtered_df = filtered_df[filtered_df['Date'] == date_str]
        
        if name_filter:
            filtered_df = filtered_df[filtered_df['Name'].isin(name_filter)]
        
        # Display table
        st.markdown("---")
        st.subheader("📋 Attendance Records")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Analytics
        if len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("📈 Analytics")
            
            # Attendance by date
            attendance_by_date = filtered_df.groupby('Date').size().reset_index(name='Count')
            st.bar_chart(attendance_by_date.set_index('Date'))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Face Recognition Attendance System v1.0</p>
    <p>Built with OpenCV, Python, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
