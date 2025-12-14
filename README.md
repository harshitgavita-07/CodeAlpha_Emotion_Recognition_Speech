# CodeAlpha Task 2: Emotion Recognition from Speech

**Machine Learning Internship - CodeAlpha**

## 📋 Project Overview
This project recognizes human emotions (happy, sad, angry, fearful) from speech audio using deep learning and audio signal processing techniques.

## 🎯 Objective
Develop a model that can accurately identify emotions from voice recordings, enabling applications in mental health monitoring, customer service analysis, and human-computer interaction.

## 🛠️ Technologies Used
- **Python 3.x**
- **TensorFlow/Keras** - Deep Learning framework
- **Librosa** - Audio processing and feature extraction
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Model evaluation and preprocessing

## 🎵 Audio Feature Extraction
### MFCC (Mel-Frequency Cepstral Coefficients)
- 40 MFCC features extracted from each audio sample
- Captures the power spectrum of sound
- Essential for speech and audio analysis

### Additional Features
- **Chroma Features** - Represents pitch class distribution
- **Mel Spectrogram** - Time-frequency representation of audio

## 🧠 Model Architecture
### Deep Neural Network
```
Input Layer (180 features)
    ↓
Dense Layer (256 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (4 emotions, Softmax)
```

## 📊 Dataset
- **Primary Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Alternative Datasets:** TESS, EMO-DB
- **Emotions Recognized:** Happy, Sad, Angry, Fearful
- **Format:** WAV audio files
- **Sample Rate:** 48kHz (RAVDESS)

### Dataset Download
```
https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
```

## 🎯 Results
- **Training Accuracy:** ~85-90%
- **Validation Accuracy:** ~80-85%
- **Model Performance:** Effective emotion classification across multiple speakers
- **Best Performance:** Happy and Angry emotions (highest accuracy)

## 📁 Files
- `Emotion_Recognition_Speech.ipynb` - Main Jupyter notebook
- Contains data preprocessing, feature extraction, model training, and evaluation

## 🚀 How to Run

### Option 1: Using Real Audio Dataset
1. Download RAVDESS dataset from Kaggle
2. Upload to Google Colab or mount Google Drive
3. Update the file path in the notebook
4. Run all cells

### Option 2: Using Synthetic Demo
1. Open notebook in Google Colab
2. Run all cells (uses synthetic data for demonstration)
3. View model architecture and training process

## 📊 Key Features
- **Audio Preprocessing:** Noise reduction and normalization
- **Feature Engineering:** MFCC, Chroma, Mel-spectrogram extraction
- **Deep Learning:** Multi-layer neural network with dropout regularization
- **Model Evaluation:** Accuracy, Confusion Matrix, Classification Report
- **Visualization:** Training history, emotion distribution, confusion matrix

## 🔬 Technical Highlights
- Real-time audio feature extraction pipeline
- Robust preprocessing for varying audio qualities
- Data augmentation for improved generalization
- Comprehensive model evaluation metrics

## 🎓 Learning Outcomes
- Audio signal processing techniques
- Feature extraction from time-series data
- Deep learning for classification tasks
- Handling multi-modal data (audio)
- Model optimization and evaluation

## 💡 Applications
- **Mental Health:** Emotion monitoring for therapy
- **Customer Service:** Sentiment analysis in call centers
- **Education:** Engagement tracking in e-learning
- **Entertainment:** Emotion-responsive gaming
- **Healthcare:** Early detection of emotional disorders

## 🔮 Future Improvements
- Real-time emotion detection from microphone input
- Multi-language support
- Integration with video emotion recognition
- Mobile app deployment
- Speaker-independent emotion recognition

## 👨‍💻 Author
**Harshit Gavita**  
CodeAlpha Machine Learning Intern

## 📞 Contact
- GitHub: [@harshitgavita-07](https://github.com/harshitgavita-07)
- LinkedIn: [www.linkedin.com/in/harshit-gavita-bb90b3202]

## 🙏 Acknowledgments
Special thanks to **@CodeAlpha** for providing this incredible learning opportunity and comprehensive mentorship in Machine Learning and AI.

---

**Part of CodeAlpha Machine Learning Internship Program**

*Building AI solutions for a better tomorrow* 🚀
