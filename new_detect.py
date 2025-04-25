import argparse
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    input_data = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details['index'], input_data)

def get_output(interpreter, threshold):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]
    results = []

    for i in range(len(scores)):
        if scores[i] > threshold:
            results.append((int(classes[i]), scores[i], boxes[i]))
    return results

def run(model_path, label_path, camera_id, width, height, threshold, num_threads):
    # Load model
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    labels = load_labels(label_path)
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_counter = 0
    fps = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (input_width, input_height))
        set_input_tensor(interpreter, image_resized)

        # Inference
        interpreter.invoke()
        results = get_output(interpreter, threshold)

        # Visualization
        h, w, _ = frame.shape
        for class_id, score, box in results:
            y_min = int(box[0] * h)
            x_min = int(box[1] * w)
            y_max = int(box[2] * h)
            x_max = int(box[3] * w)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f'{labels[class_id]}: {int(score * 100)}%'
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS
        fps_counter += 1
        if fps_counter >= 10:
            end_time = time.time()
            fps = fps_counter / (end_time - start_time)
            start_time = end_time
            fps_counter = 0
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, default='model.tflite')
    parser.add_argument('--labels', required=False, default='labels.txt')
    parser.add_argument('--cameraId', type=int, default=0)
    parser.add_argument('--frameWidth', type=int, default=640)
    parser.add_argument('--frameHeight', type=int, default=480)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--numThreads', type=int, default=1)
    args = parser.parse_args()

    run(args.model, args.labels, args.cameraId, args.frameWidth, args.frameHeight, args.threshold, args.numThreads)
