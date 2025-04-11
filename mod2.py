import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

model_path = r'C:\Users\Admin\Desktop\garbage segregation\garbage_dataset\train\custom_garbage_model.h5'

model = load_model(model_path)

class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))

    predictions = model.predict(preprocessed_frame)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]

    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Garbage Segregation', frame)

    # Allow user to correct the prediction
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):  # Press 'c' to correct the prediction
        new_class_idx = int(input(f"Enter correct class index (0-9): "))
        if 0 <= new_class_idx < len(class_names):
            predicted_class_idx = new_class_idx
            predicted_class = class_names[predicted_class_idx]
            print(f"Updated prediction to: {predicted_class}")

cap.release()
cv2.destroyAllWindows()
