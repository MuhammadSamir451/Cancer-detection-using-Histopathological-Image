# 🧬 Cancer Detection Using CNN

## 📌 Objective  
The aim of this project was to build a Convolutional Neural Network (CNN) that can classify histopathological images of cancer into different categories.  

---

## 📂 Dataset & Preprocessing  
- Dataset consisted of three categories of histopathological images.  
- All images were resized to **128×128 pixels** and normalized to the range **[0,1]**.  
- Data was split into training and testing sets.  

---

## 🏗️ Model Architecture  
- CNN with **3 convolutional layers** followed by max pooling.  
- Fully connected dense layer with dropout to prevent overfitting.  
- Output layer with **softmax activation** for multi-class classification.  

---

## ⚙️ Training Setup  
- Optimizer: **Adam**  
- Loss Function: **Categorical Crossentropy**  
- Epochs: **10**  
- Batch Size: **32**  

---

## 📊 Results  
- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~92%  
- **Test Accuracy:** ~90%  
- **Test Loss:** ~0.30  

---

## 🔍 Observations  
- The CNN achieved strong accuracy across training and test datasets, indicating effective feature learning.  
- Some misclassifications were observed in cancer subtypes with visually similar patterns.  
- Dropout helped reduce overfitting, as training and validation curves were closely aligned.  

---

## 🧠 What is CNN?  
A **Convolutional Neural Network (CNN)** is a type of deep learning model designed for analyzing visual data (images).  
It works by:  
- Extracting patterns (edges, textures, shapes) through **convolutional layers**.  
- Reducing dimensionality and highlighting important features using **pooling layers**.  
- Classifying images with **fully connected layers** at the end.  

CNNs are widely used in **image classification, medical imaging, facial recognition, and object detection**.  

---

## 🏗️ What is VGG16?  
**VGG16** is a popular deep CNN model developed by Oxford’s Visual Geometry Group (VGG).  
- It consists of **16 layers** (13 convolutional + 3 fully connected).  
- Uses small **3×3 filters** throughout, stacked to capture complex patterns.  
- Known for its **simplicity and high accuracy**, making it a common choice for transfer learning in medical imaging.  

While our project uses a basic CNN, models like **VGG16** can further improve accuracy when applied to large datasets.  

---

## ✅ Conclusion  
The CNN model demonstrated that even a relatively simple architecture can achieve high performance in cancer detection tasks.  
Future improvements could include experimenting with **VGG16, ResNet, or EfficientNet** and applying **data augmentation** to improve generalization.  

- note **if you want to jump directly to the link skipping the code running here it is 👉 http://127.0.0.1:7860**
