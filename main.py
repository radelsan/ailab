from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\Радель\PycharmProjects\electricityAI\dataset"

def train_model():
    model = YOLO(r'C:\Users\Радель\PycharmProjects\electricityAI\runs\detect\train9\weights\best.pt')
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=10)

def export_model():
    model = YOLO(r'C:\Users\Радель\PycharmProjects\electricityAI\runs\detect\train9\weights\best.pt')
    model.export(format='onnx', imgsz=640)

def run_model(image_path):
    session = ort.InferenceSession(os.path.join(dataset_path, r'C:\Users\Радель\PycharmProjects\electricityAI\runs\detect\train9\weights\best.onnx'))

    image = Image.open(image_path).resize((640, 640))
    input_array = np.array(image).astype(np.float32)
    input_array = np.transpose(input_array, (2, 0, 1))
    input_array = np.expand_dims(input_array, axis=0)

    outputs = session.run(None, {session.get_inputs()[0].name: input_array})
    return outputs

def process_results(results):
    boxes = results[0][0]
    scores = results[0][-1]

    boxes = np.array(boxes)
    scores = np.array(scores)

    return boxes, scores

def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)

        order = order[np.where(iou <= threshold)[0] + 1]

    return keep

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value

def average_precision(precision, recall):
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def display_image_with_bboxes(image_path, boxes, scores, threshold):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    scores = np.array(scores).flatten()
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(box[0], box[1], f'{scores[i]:.2f}', color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # train_model()  # Обучение модели
    # export_model()  # Экспорт модели

    if os.path.exists(os.path.join(dataset_path, r'C:\Users\Радель\PycharmProjects\electricityAI\runs\detect\train9\weights\best.onnx')):
        print("Файл .onnx найден.")
    else:
        print("Файл .onnx не найден.")

    image_path = r"C:\Users\Радель\PycharmProjects\electricityAI\dataset\test\images\1.jpg"
    results = run_model(image_path)

    print("Структура результатов:", results)

    if isinstance(results, list) and len(results) > 0:
        boxes, scores = process_results(results)

        threshold = 0.5
        keep_indices = np.where(scores > threshold)[0]
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]

        print("Оставшиеся bbox после фильтрации:", filtered_boxes)
        print("Оценки оставшихся bbox:", filtered_scores)

        if isinstance(results, list) and len(results) > 0:
            boxes, scores = process_results(results)

            threshold = 0.5
            keep_indices = np.where(scores > threshold)[0]
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]

            print("Оставшиеся bbox после фильтрации:", filtered_boxes)
            print("Оценки оставшихся bbox:", filtered_scores)

            # display_image_with_bboxes(image_path, filtered_boxes, filtered_scores, threshold)

            if len(filtered_boxes) >= 2:
                iou_value = iou(filtered_boxes[0], filtered_boxes[1])
                print("IOU между двумя bbox:", iou_value)

            precision = np.array([0.9, 0.8, 0.7])
            recall = np.array([0.1, 0.5, 0.9])
            ap = average_precision(precision, recall)
            print("Average Precision:", ap)
        else:
            print("Ошибка: результаты не являются списком или пусты.")
