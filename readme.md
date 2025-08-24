🧬 Cancer Detection Using CNN

I recently worked on a project where I built a Convolutional Neural Network (CNN) to classify histopathological images of cancer into different categories.

📂 Dataset & Preprocessing

3 categories of histopathological images

Resized to 128×128 & normalized

Split into training and testing sets

🏗️ Model

3 convolutional + pooling layers

Dense layer with dropout to prevent overfitting

Softmax output for multi-class classification

⚙️ Training Setup

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 10 | Batch Size: 32

📊 Results
✅ Training Accuracy: ~95%
✅ Validation Accuracy: ~92%
✅ Test Accuracy: ~90%
📉 Test Loss: ~0.30

🔍 Observations

CNN effectively captured cancer patterns

Dropout helped reduce overfitting

Misclassifications happened mostly in visually similar cancer subtypes

🏗️ VGG16 Test
I also experimented with VGG16, and the performance difference was only about 2–3% better compared to my custom CNN. While VGG16 is more powerful, this showed that even a relatively simple CNN can achieve strong results in medical imaging tasks.

✅ Conclusion
Deep learning shows great potential for healthcare applications. Future improvements could include data augmentation and testing models like ResNet or EfficientNet for further gains.
