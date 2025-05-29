```markdown
# 🔔 Smart Doorbell with Emotion Detection

An AI-powered smart surveillance system that proactively monitors visitor emotions in real-time using computer vision and deep learning. This intelligent doorbell system alerts homeowners by email if negative emotions like anger, fear, sadness, or disgust are detected—enhancing traditional home security with emotion-aware intelligence.

---

## 💡 Project Highlights

- 🔍 Real-time facial emotion recognition
- ✉️ Sends email alerts when negative emotions are detected
- 🎯 Classifies visitor emotions into positive or negative
- 💬 Displays emotion results with visual overlays using webcam
- 🧠 Trained from scratch on the FER-2013 dataset
- 🔐 Prevents spam by limiting alerts per session

---

## 🧠 Features

| Feature                     | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| Face Detection             | Uses OpenCV Haar Cascade                                      |
| Emotion Classification     | CNN model built using TensorFlow/Keras                        |
| Real-time Monitoring       | Webcam-based facial expression recognition                    |
| Email Notification         | Sends alert with emotion status via SMTP                      |
| Alert Count Control        | Prevents email spam by limiting to 5 alerts per run           |

---

## 🚀 Technologies Used

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy
- FER-2013 Dataset
- Gmail SMTP for email notifications

---

## 📦 Folder Structure

```

📁 smart-doorbell/
├── alertupdated.py            # Main system: emotion detection + alerts
├── model\_training.py          # CNN training script (FER-2013)
├── emotion\_detection\_model.h5 # Trained Keras model file
├── .env                       # Email credentials (excluded in .gitignore)
├── .gitignore                 # Prevents secrets from being committed

````

---

## 🧪 Emotion Classification

The model classifies the following emotions:

| Positive        | Negative               |
|----------------|------------------------|
| Happy, Neutral, Surprised | Angry, Sad, Fearful, Disgusted |

Detected emotions are labeled and color-coded on the video feed:
- 🟢 Green = Positive
- 🔴 Red = Negative

---

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Haritha-k-suresh/smart-doorbell.git
cd smart-doorbell
````

### 2. Set Up the Environment

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File

Add your email credentials (use an **App Password** if using Gmail):

```
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECEIVER_EMAIL=recipient_email@gmail.com
```

### 4. Run the Application

```bash
python alertupdated.py
```

Press `q` to stop the live feed.

---

## 🧠 Model Summary

* Input: 48x48 grayscale images
* Architecture: 4 Conv blocks + BatchNorm + Dropout + Dense layers
* Activation: ReLU (hidden), Softmax (output)
* Optimizer: Adam
* Training Accuracy: 99%
* Validation Accuracy: \~82%

---

## 📊 Sample Output

```shell
Detected Emotion: angry (Negative)
📧 Sending Email Alert 1 of 5...
✅ Email Alert Sent!
```

Emotion detection is overlaid live on the webcam video stream.

---

## 🛡️ Security Notes

* `.env` file is ignored in `.gitignore`
* Do **not** commit your real email/password
* Use [Gmail App Passwords](https://support.google.com/accounts/answer/185833) for secure email login

---

## 🌱 Future Improvements

* Integrate ESP32-CAM for IoT-based deployment
* Add support for **voice emotion detection**
* Optimize for edge devices (on-device inference)
* Enable **multi-camera support** for wider surveillance
* Add cloud dashboard for **alert logging and analytics**

---

## 👤 Author

**Haritha K. Suresh**
B.Tech in Artificial Intelligence and Data Science
📧 [harithaksuresh6@gmail.com](mailto:harithaksuresh6@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/haritha-k-suresh-92a55b275/)

---

## 🏁 License

This project is for academic and non-commercial use only.

```

---
```

