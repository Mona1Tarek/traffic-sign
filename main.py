import cv2
import numpy as np
import onnxruntime as ort

# Load the optimized ONNX model
session = ort.InferenceSession("best_optimized.onnx", providers=["CPUExecutionProvider"])

# Get the input name and shape from the model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Class name mapping (translated to English)
class_name_mapping = {
    'Att-STOP': 'STOP',
    'Att-cedez le passage': 'Give Way',
    'Att-danger': 'Danger',
    'Att-eboulement': 'Landslide',
    'Att-intersection ou vous etes prioritaire': 'Intersection, You Have Priority',
    'Att-passage animaux sauvages': 'Wildlife Crossing',
    'Att-passage enfants': 'Children Crossing',
    'Att-passage pietons': 'Pedestrian Crossing',
    'Att-ralentisseurs': 'Speed Bumps',
    'Att-rond point': 'Roundabout',
    'Att-route a double voie': 'Dual Carriageway',
    'Att-route glissante': 'Slippery Road',
    'Att-route prioritaire': 'Priority Road',
    'Att-succession de virages': 'Curves Ahead',
    'Att-travaux': 'Roadworks',
    'Att-virage a droite': 'Right Turn',
    'Att-virage a gauche': 'Left Turn',
    'Indic-autoroute': 'Motorway',
    'Indic-circulation a sens unique': 'One-way Traffic',
    'Indic-parking': 'Parking',
    'Indic-passage pietons': 'Pedestrian Crossing',
    'Indic-station bus': 'Bus Stop',
    'Inter-arret et stationnement': 'No Stopping or Parking',
    'Inter-de depasser': 'No Overtaking',
    'Inter-de faire demi-tour': 'No U-turn',
    'Inter-de tourner a droite': 'No Right Turn',
    'Inter-de tourner a gauche': 'No Left Turn',
    'Inter-sens': 'No Entry',
    'Inter-stationnement': 'No Parking',
    'Inter-vitesse limitee a -100km-h-': 'Speed Limit 100 km/h',
    'Inter-vitesse limitee a -120km-h-': 'Speed Limit 120 km/h',
    'Inter-vitesse limitee a -20km-h-': 'Speed Limit 20 km/h',
    'Inter-vitesse limitee a -30km-h-': 'Speed Limit 30 km/h',
    'Inter-vitesse limitee a -40km-h-': 'Speed Limit 40 km/h',
    'Inter-vitesse limitee a -50km-h-': 'Speed Limit 50 km/h',
    'Inter-vitesse limitee a -60km-h-': 'Speed Limit 60 km/h',
    'Inter-vitesse limitee a -80km-h-': 'Speed Limit 80 km/h',
    'Oblig-continuez a droite': 'Keep Right',
    'Oblig-continuez a gauche': 'Keep Left',
    'Oblig-continuez tout droit': 'Go Straight',
    'Oblig-continuez tout droit ou tournez a droite': 'Go Straight or Turn Right',
    'Oblig-continuez tout droit ou tournez a gauche': 'Go Straight or Turn Left',
    'Oblig-tournez a droite': 'Turn Right',
    'Oblig-tournez a gauche': 'Turn Left',
    'Oblig-tournez le rond point': 'Turn at Roundabout'
}

# Get the list of class names in the correct order
class_names = list(class_name_mapping.keys())
num_classes = len(class_names)

def get_detected_sign_name(class_id, confidence):
    """Returns the name of the detected sign if confidence > 0.5"""
    if confidence > 0.5:
        # Get the original class name
        original_class_name = class_names[class_id]
        
        # Get the English class name from the mapping
        english_class_name = class_name_mapping.get(original_class_name, original_class_name)
        return english_class_name
    return None

def detect_traffic_signs_from_video():
    # Open the webcam
    IP = '192.168.1.18'
    cap = cv2.VideoCapture(f"http://{IP}:8080/?action=stream_0")
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get original dimensions
        height, width = frame.shape[:2]

        # Preprocess the frame
        try:
            input_image = cv2.resize(frame, (640, 640))  # Resize to model's input size
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            input_image = input_image.astype(np.float32) / 255.0
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)
        except Exception:
            continue

        # Run inference
        try:
            outputs = session.run(None, {input_name: input_image})[0]
        except Exception:
            continue

        # Define confidence threshold
        confidence_threshold = 0.5

        # List to store detected sign names
        detected_sign_names = []

        # Process detections
        for i in range(outputs.shape[2]):
            row = outputs[0, :, i]
            
            # Extract coordinates and scores
            x, y, w, h = row[:4]
            class_scores = row[4:4+num_classes]
            
            # Find the best class
            confidence = np.max(class_scores)
            class_id = np.argmax(class_scores)

            # If confidence is above threshold, process detection
            if confidence > confidence_threshold:
                # Convert YOLO format to pixel coordinates
                x1 = max(0, int((x - w / 2) * width / 640))
                y1 = max(0, int((y - h / 2) * height / 640))
                x2 = min(width, int((x + w / 2) * width / 640))
                y2 = min(height, int((y + h / 2) * height / 640))

                # Get the detected sign's name
                sign_name = get_detected_sign_name(class_id, confidence)
                if sign_name:
                    detected_sign_names.append(sign_name)
                    # Draw the bounding box and label
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    # label = f"{sign_name} ({confidence:.2f})"
                    # cv2.putText(frame, label, (x1, y1 - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # # Display the frame
        # cv2.imshow('Detected Traffic Signs', frame)

        # Print detected signs in the list (or use it elsewhere)
        if detected_sign_names:
            print(f"Detected signs: {', '.join(detected_sign_names)}")

        # # Exit on 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the capture object and close all windows
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_traffic_signs_from_video()
