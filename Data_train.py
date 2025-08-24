import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset Path (POINT TO PARENT FOLDER, not a single class)
# -----------------------------
dataset_path = r"C:\Users\hp\OneDrive\Desktop\Projects\Project 2 (Cancer Detection Using Histopathological Images)\data"

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,   # normalize pixel values
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Confirm dataset loading
print("✅ Classes:", train_gen.class_indices)
print("✅ Training samples:", train_gen.samples)
print("✅ Validation samples:", val_gen.samples)

# -----------------------------
# 3. Simple CNN Model (3 conv layers)
# -----------------------------
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

# -----------------------------
# 4. Train CNN Model (10 epochs)
# -----------------------------
checkpoint_cnn = ModelCheckpoint("best_cnn_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

history_cnn = cnn_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint_cnn]
)

cnn_model.save(r"C:\Users\hp\OneDrive\Desktop\Projects\Project 2 (Cancer Detection Using Histopathological Images)\models\lung_cancer_cnn.h5")
print("✅ CNN model saved as lung_cancer_cnn.h5")

# -----------------------------
# 5. Evaluate CNN Model
# -----------------------------
loss, acc = cnn_model.evaluate(val_gen)
print(f"Simple CNN - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Plot Accuracy & Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_cnn.history['accuracy'], label='Train Acc')
plt.plot(history_cnn.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("CNN Accuracy")

plt.subplot(1,2,2)
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")
plt.show()

# -----------------------------
# 6. BONUS: Transfer Learning (VGG16, 10 epochs)
# -----------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128,128,3))
for layer in base_model.layers:
    layer.trainable = False  # freeze base layers

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

vgg_model = Model(inputs=base_model.input, outputs=output)

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(vgg_model.summary())

checkpoint_vgg = ModelCheckpoint("best_vgg16_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

history_vgg = vgg_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint_vgg]
)

vgg_model.save(r"C:\Users\hp\OneDrive\Desktop\Projects\Project 2 (Cancer Detection Using Histopathological Images)\models\lung_cancer_vgg16.h5")
print("✅ VGG16 Transfer Learning model saved as lung_cancer_vgg16.h5")

# -----------------------------
# 7. Evaluate VGG16 Model
# -----------------------------
loss_vgg, acc_vgg = vgg_model.evaluate(val_gen)
print(f"VGG16 Transfer Learning - Loss: {loss_vgg:.4f}, Accuracy: {acc_vgg:.4f}")

# Plot Accuracy & Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_vgg.history['accuracy'], label='Train Acc')
plt.plot(history_vgg.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("VGG16 Accuracy")

plt.subplot(1,2,2)
plt.plot(history_vgg.history['loss'], label='Train Loss')
plt.plot(history_vgg.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("VGG16 Loss")
plt.show()
