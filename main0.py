import cv2
import numpy as np
import onnxruntime as ort
import time

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

def process_video(video_path):
    """
    Process a video file to detect traffic signs
    
    Args:
        video_path (str): Path to the input video file
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Initialize timing variables
    frame_count = 0
    total_time = 0
    preprocessing_time = 0
    inference_time = 0
    detection_time = 0

    while True:
        # Start timing for this frame
        frame_start_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"\nProcessing frame {frame_count}")

        # Time preprocessing
        preprocess_start = time.time()
        try:
            input_image = cv2.resize(frame, (640, 640))  # Resize to model's input size
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            input_image = input_image.astype(np.float32) / 255.0
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            continue
        preprocess_time = (time.time() - preprocess_start) * 1000
        preprocessing_time += preprocess_time

        # Time inference
        inference_start = time.time()
        try:
            outputs = session.run(None, {input_name: input_image})[0]
        except Exception as e:
            print(f"Error running inference: {e}")
            continue
        inference_time_ms = (time.time() - inference_start) * 1000
        inference_time += inference_time_ms

        # Define confidence threshold
        confidence_threshold = 0.5

        # List to store detected sign names
        detected_sign_names = []

        # Time detection processing
        detection_start = time.time()
        # Process detections
        for i in range(outputs.shape[2]):
            row = outputs[0, :, i]
            
            # Extract coordinates and scores
            class_scores = row[4:4+num_classes]
            
            # Find the best class
            confidence = np.max(class_scores)
            class_id = np.argmax(class_scores)

            # If confidence is above threshold, process detection
            if confidence > confidence_threshold:
                # Get the detected sign's name
                sign_name = get_detected_sign_name(class_id, confidence)
                if sign_name:
                    detected_sign_names.append(sign_name)
        
        detection_time_ms = (time.time() - detection_start) * 1000
        detection_time += detection_time_ms

        # Calculate total frame time
        frame_time = (time.time() - frame_start_time) * 1000
        total_time += frame_time

        # Print timing information for this frame
        print(f"Frame timing (ms):")
        print(f"  Preprocessing: {preprocess_time:.2f}")
        print(f"  Inference: {inference_time_ms:.2f}")
        print(f"  Detection processing: {detection_time_ms:.2f}")
        print(f"  Total frame time: {frame_time:.2f}")

        # Print detected signs
        if detected_sign_names:
            print(f"Detected signs: {', '.join(detected_sign_names)}")

    # Print final statistics
    if frame_count > 0:
        print("\nFinal Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average times per frame (ms):")
        print(f"  Preprocessing: {preprocessing_time/frame_count:.2f}")
        print(f"  Inference: {inference_time/frame_count:.2f}")
        print(f"  Detection processing: {detection_time/frame_count:.2f}")
        print(f"  Total: {total_time/frame_count:.2f}")
        print(f"Total processing time: {total_time/1000:.2f} seconds")

    # Release resources
    cap.release()

if __name__ == '__main__':
    # Define video path
    video_path = "videos/vidd1.mp4"
    
    # Process the video
    process_video(video_path) 