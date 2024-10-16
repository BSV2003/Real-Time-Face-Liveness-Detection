import cv2
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 5
WARNING_SOUND_PATH = "E:\B.E\SEM 6\Mini Project\warning_sound.mp3"
REFLECTION_THRESHOLD = 0.3  # Adjust as needed

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play warning sound
def play_warning_sound():
    playsound(WARNING_SOUND_PATH)

# Function to detect reflections around the detected face
def detect_reflection(roi_color):
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(roi_gray)
    std_intensity = np.std(roi_gray)
    reflection_score = std_intensity / mean_intensity
    return reflection_score > REFLECTION_THRESHOLD

# Estimate distance from the size of the face and convert to meters
def estimate_distance(face_width):
    # Known width of the face (assumed average adult face width)
    known_face_width = 0.15  # in meters (15 cm)
    # Focal length of the camera (placeholder value)
    focal_length = 600  # Adjust this value according to your camera and environment
    # Calculate distance based on face width
    distance = (known_face_width * focal_length) / face_width
    return distance

# Adjust thresholds based on estimated distance
def adjust_thresholds_by_distance(distance):
    # Adjust EAR and reflection thresholds based on distance
    ear_threshold = EAR_THRESHOLD + (distance * 0.01)
    reflection_threshold = REFLECTION_THRESHOLD + (distance * 0.01)
    return ear_threshold, reflection_threshold

# Main function
def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Performance metrics
    TP, FP, TN, FN = 0, 0, 0, 0

    # Variables to track distances and maximum, minimum distances
    distances = []
    max_distance = 0
    min_distance = float('inf')  # Initialize to a very high value

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Estimate distance based on face width
            distance = estimate_distance(w)
            
            # Track distances and update maximum and minimum distances
            distances.append(distance)
            max_distance = max(max_distance, distance)
            min_distance = min(min_distance, distance)
            
            # Adjust thresholds based on distance
            ear_threshold, reflection_threshold = adjust_thresholds_by_distance(distance)
            
            # Define region of interest (ROI) for gray and color frames
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
            
            blink_detected = False
            warning_sound_played = False

            # Process each detected eye
            for (ex, ey, ew, eh) in eyes:
                eye_region = roi_gray[ey:ey + eh, ex:ex + ew]
                ear = calculate_ear(eye_region)
                
                # Check EAR against adjusted threshold
                if ear < ear_threshold:
                    blink_detected = True
                    play_warning_sound()
                    warning_sound_played = True
                    break

            # Check for reflections
            reflection_detected = detect_reflection(roi_color)

            # Determine if face is real or fake
            is_fake = blink_detected and reflection_detected

            # Display results and calculate performance metrics
            if is_fake:
                cv2.putText(frame, "FAKE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                FP += 1
            else:
                cv2.putText(frame, "REAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                TP += 1

            if warning_sound_played:
                if is_fake:
                    TN += 1
                else:
                    FN += 1
            else:
                if not is_fake:
                    TP += 1
                else:
                    FP += 1

            # Print the estimated distance for each detected face
            print(f"Distance from webcam to face: {distance:.2f} meters")

        # Display the frame
        cv2.imshow("Liveliness Detection", frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate performance metrics
    total_predictions = TP + FP + TN + FN
    accuracy = (TP + TN) / total_predictions * 100 if total_predictions > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate confusion matrix
    conf_matrix = np.array([[TP, FP], [FN, TN]])

    # Print performance metrics
    print("Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")

    # Convert confusion matrix to percentages
    conf_matrix_percent = conf_matrix / total_predictions * 100

    # Plot confusion matrix
    labels = ["REAL", "FAKE"]
    ax = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})

    # Add labels TP, FP, FN, TN
    ax.text(0.5, 0.3, "TP", horizontalalignment='center', verticalalignment='center', fontsize=12, color='white')
    ax.text(1.5, 0.3, "FP", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    ax.text(0.5, 1.3, "FN", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    ax.text(1.5, 1.3, "TN", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Haar Cascade: Confusion Matrix (Percentage)")
    plt.show()

    # Calculate and print the average, maximum, and minimum distances in meters
    if distances:
        average_distance = np.mean(distances)
        print(f"Average distance from webcam to faces: {average_distance:.2f} meters")
        print(f"Maximum distance from webcam to faces: {max_distance:.2f} meters")
        print(f"Minimum distance from webcam to faces: {min_distance:.2f} meters")

        # Optimal distance at which faces can be detected properly and correctly
        optimal_distance = (max_distance + min_distance) / 2
        print(f"Optimal distance for face detection: {optimal_distance:.2f} meters")

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Execute the main function
if __name__ == "__main__":
    main()
