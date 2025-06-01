import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Load pre-trained model (using MobileNet SSD for example)
        self.net = cv2.dnn.readNetFromCaffe(
            'MobileNetSSD_deploy.prototxt', 
            'MobileNetSSD_deploy.caffemodel'
        )
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect_objects(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            0.007843, 
            (300, 300), 
            127.5
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Minimum confidence threshold
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                results.append({
                    "class": self.classes[idx],
                    "confidence": float(confidence),
                    "box": {
                        "x": int(startX),
                        "y": int(startY),
                        "width": int(endX - startX),
                        "height": int(endY - startY)
                    }
                })
        
        return results
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            color = self.colors[self.classes.index(detection['class'])]
            
            # Draw rectangle
            cv2.rectangle(
                frame, 
                (detection['box']['x'], detection['box']['y']),
                (detection['box']['x'] + detection['box']['width'], 
                 detection['box']['y'] + detection['box']['height']),
                color, 
                2
            )
            
            # Draw label
            cv2.putText(
                frame, 
                label, 
                (detection['box']['x'], detection['box']['y'] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return frame
