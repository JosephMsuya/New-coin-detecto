import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

# Load model
interpreter = tflite.Interpreter(model_path="coin_500_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera or use an image
cap = cv2.VideoCapture(0)  # Use 0 for camera, or change to image path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    img = cv2.resize(frame, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Check if it's a 500 coin
    if output >= 0.5:
        print("✅ 500 TZS coin detected!")
    else:
        print("❌ Not a 500 TZS coin")

    # Show camera (optional)
    cv2.imshow('Coin Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
