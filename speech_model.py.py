
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import BertTokenizer, TFBertModel
from opensmile_preprocessing import opensmile_Functionals, emo_labels
from sklearn.metrics import accuracy_score, f1_score
import json

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


with open('modified_shemo.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

transcripts = [info['transcript'] for info in data.values()]
audio_paths = [info['path'] for info in data.values()]
emotions = [info['emotion'] for info in data.values()]


X_mfcc, y = opensmile_Functionals()
X_mfcc = np.expand_dims(X_mfcc, axis=-1)

min_len = len(X_mfcc)
X_mfcc = X_mfcc[:min_len]

y = y[:min_len]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(emo_labels)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
    X_mfcc, y_categorical, test_size=0.2, random_state=seed_value
)

input_shape_mfcc = (166, 1)

def build_multimodal_model(input_shape_mfcc, num_classes):
    input_mfcc = tf.keras.layers.Input(shape=input_shape_mfcc)
 
    # MFCC branch
    x_mfcc = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(input_mfcc)
    x_mfcc = tf.keras.layers.BatchNormalization()(x_mfcc)
    x_mfcc = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x_mfcc)
    x_mfcc = tf.keras.layers.BatchNormalization()(x_mfcc)
    x_mfcc = tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu')(x_mfcc)
    x_mfcc = tf.keras.layers.BatchNormalization()(x_mfcc)
    x_mfcc = tf.keras.layers.MaxPooling1D(pool_size=2)(x_mfcc)
    x_mfcc = tf.keras.layers.Flatten()(x_mfcc)
    x_mfcc = tf.keras.layers.Dropout(0.5)(x_mfcc)

    
    # Fully connected layers
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_mfcc)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_mfcc], outputs=output)

    return model

model = build_multimodal_model(input_shape_mfcc, num_classes)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit([X_mfcc_train], y_train, validation_data=([X_mfcc_test], y_test), 
                    epochs=50, batch_size=32, callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate([X_mfcc_test], y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Predict the probabilities for the test set
y_pred_probs = model.predict([X_mfcc_test])
# Convert the probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate precision, recall, and f1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


y_pred = model.predict([X_mfcc_test])

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

weighted_accuracy = accuracy_score(y_true_classes, y_pred_classes, sample_weight=None)

unweighted_accuracy = accuracy_score(y_true_classes, y_pred_classes)

weighted_f1_score = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f'Weighted Accuracy (WA): {weighted_accuracy}')
print(f'Unweighted Accuracy (UA): {unweighted_accuracy}')
print(f'Weighted F1-Score (WF1): {weighted_f1_score}')

model.save('multimodal_emotion_recognition_model.keras')
