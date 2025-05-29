import cv2
import numpy as np
from tensorflow.keras.models import load_model
import smtplib
from email.message import EmailMessage

# Load the trained emotion detection model
model = load_model(r"C:\Users\maniv\Downloads\emotion_detection_model.h5")

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
label_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# Define emotion categories
positive_emotions = {"happy", "neutral", "surprised"}
negative_emotions = {"angry", "disgusted", "sad", "fearful"}

# Email Credentials (Use an App Password for security)
sender_email = "rexspyro4@gmail.com"
sender_password = "klmq xyob rmkh inxe"  # Replace with your Gmail App Password
receiver_email = "harithaksuresh6@gmail.com"  # Replace with recipient's email

# Track the number of alerts sent
alert_count = 0
max_alerts = 5  # Maximum alerts allowed

def send_email_alert(emotion):
    global alert_count
    if alert_count >= max_alerts:
        print("🚨 Maximum email alerts sent. Stopping program.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    msg = EmailMessage()
    msg.set_content(f"🚨 ALERT: Negative Emotion Detected! ({emotion})......found suspicious🚨🚨🚨🚨🚨")
    msg["Subject"] = "🚨🚨🚨⚠ Warning: Negative Emotion Alert🚨🚨🚨"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        print(f"📧 Sending Email Alert {alert_count+1} of {max_alerts}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        alert_count += 1
        print(f"✅ Email Alert Sent! (Total Sent: {alert_count})")
    except Exception as e:
        print(f"❌ Error Sending Email: {e}")

# Open webcam with DirectShow for better performance
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

previous_faces = []  # Store previous face positions for smoothing
alpha = 0.6  # Smoothing factor
frame_count = 0  # To process alternate frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from webcam!")
        break

    frame_count += 1
    if frame_count % 2 != 0:  # Process every 2nd frame to reduce lag
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("No faces detected!")

    smoothed_faces = []
    for (x, y, w, h) in faces:
        # Smooth bounding box
        if previous_faces:
            px, py, pw, ph = previous_faces[0]
            x = int(alpha * px + (1 - alpha) * x)
            y = int(alpha * py + (1 - alpha) * y)
            w = int(alpha * pw + (1 - alpha) * w)
            h = int(alpha * ph + (1 - alpha) * h)
        smoothed_faces.append((x, y, w, h))

        # Extract face ROI
        face_roi = gray_frame[y:y+h, x:x+w]  

        # Preprocess face for model input
        face_roi = cv2.resize(face_roi, (48, 48))  
        face_roi = face_roi / 255.0  # Normalize
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict the emotion
        prediction = model.predict(face_roi, verbose=0)
        class_index = int(np.argmax(prediction))  # Convert np.int64 to int safely

        # Handle unexpected class index
        if class_index not in label_dict:
            print(f"Warning: Unexpected class index {class_index}, defaulting to 'Unknown'")
            emotion = "Unknown"
            emotion_category = "Unknown"
            color = (255, 255, 255)  # White for unknown emotion
        else:
            emotion = label_dict[class_index]

            # Categorize emotion
            emotion_category = "Positive" if emotion in positive_emotions else "Negative"
            color = (0, 255, 0) if emotion_category == "Positive" else (0, 0, 255)

            # Print detected emotion and its category in terminal
            print(f"Detected Emotion: {emotion} ({emotion_category})")

            # Send email alert if emotion is negative
            if emotion_category == "Negative":
                send_email_alert(emotion)

        # Display emotion and category on screen
        cv2.putText(frame, f"{emotion} ({emotion_category})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Update previous faces for smoothing
    previous_faces = smoothed_faces if smoothed_faces else previous_faces

    # Show video feed
    cv2.imshow('Emotion Classification', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
