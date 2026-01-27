import onnxruntime as ort
import numpy as np
from keras_image_helper import create_preprocessor

url = 'https://raw.githubusercontent.com/Uthmanee/machine-learning-zoomcamp-capstone2/master/testImage.jpeg'
classes = ['NORMAL', 'PNEUMONIA']


def preprocess_pytorch(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)


preprocessor = create_preprocessor(preprocess_pytorch, target_size=(224, 224))

X = preprocessor.from_url(url)

onnx_model_path = "finalChestXrayModel2.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

result = session.run([output_name], {input_name: X})
result

scores = result[0][0].tolist()
predictions = dict(zip(classes, scores))

# Get predicted class
pred_idx = int(np.argmax(scores))
predicted_class = classes[pred_idx]

print("Scores:", predictions)
print("Prediction:", predicted_class)
