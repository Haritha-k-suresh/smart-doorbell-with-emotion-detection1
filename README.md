# 🔔 Smart Doorbell with Emotion Detection

Hi! I'm **Haritha K. Suresh**, and this is my AI-based home security project — a smart doorbell system that detects visitor emotions in real-time using deep learning and computer vision. The system proactively alerts homeowners through email if a **negative emotion** is detected, offering a more intelligent layer of protection than traditional surveillance.

---

## 🧠 Key Features

- 🚪 Emotion-based visitor monitoring at your doorstep  
- 🎯 Real-time facial emotion recognition with CNN  
- 📧 Automatic email alerts on detecting **negative emotions**  
- 🔒 Prevents spamming with alert count limits  
- 🧰 Built with OpenCV, TensorFlow, Keras, and Python  

---

## 💡 Project Summary

The Smart Doorbell system uses a webcam (or an ESP32-CAM module in future versions) to capture visitor images when someone is at the door. It uses a **Convolutional Neural Network (CNN)** trained on facial emotion data to classify emotions into **positive** or **negative** categories.

If a **negative emotion** is detected (like *anger, fear, sadness, or disgust*), the system sends an **email alert** to the homeowner.

---

## 😄 Emotion Categories

| Positive Emotions           | Negative Emotions                      |
|-----------------------------|----------------------------------------|
| Happy, Neutral, Surprised   | Angry, Sad, Fearful, Disgusted         |

---

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- FER-2013 Dataset
- SMTP (for email alerts)

---

## 📂 Folder Structure

