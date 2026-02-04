import cv2
import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
import time

# --- 1. SETUP RASPBERRY PI GPIO ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define Servo Pins (Based on MG995 and MG996R)
CHUTE_SERVO_PIN = 17 # MG995: Rotates the base to the correct bin
TRAPDOOR_PIN = 18    # MG996R: Opens the flap

GPIO.setup(CHUTE_SERVO_PIN, GPIO.OUT)
GPIO.setup(TRAPDOOR_PIN, GPIO.OUT)

# Initialize PWM (Pulse Width Modulation) for 50Hz
chute_servo = GPIO.PWM(CHUTE_SERVO_PIN, 50) 
trapdoor = GPIO.PWM(TRAPDOOR_PIN, 50)

chute_servo.start(0)
trapdoor.start(0)

# Helper function to move servos
def set_servo_angle(servo, angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0) # Stop sending signal to prevent jitter

# --- 2. LOAD TENSORFLOW MODEL ---
# (Assumes your model is exported as a SavedModel in 'saved_model/my_waste_model')
print("Loading TensorFlow Model...")
model = tf.saved_model.load('saved_model/my_waste_model')
infer = model.signatures["serving_default"]

# Define Class Names
class_names = ["Plastic", "Paper", "Metal"]

# --- 3. INITIALIZE CAMERA ---
cap = cv2.VideoCapture(0) # Pi Camera

# Ensure trapdoor is closed at start (Angle 0)
set_servo_angle(trapdoor, 0)

print("Pi AI Waste Sorter Ready. Insert trash...")

while True:
    success, img = cap.read()
    if not success:
        break

    # Pre-process image for TensorFlow (Resize to match your training input, usually 224x224 or 320x320)
    input_img = cv2.resize(img, (224, 224))
    input_tensor = tf.convert_to_tensor(input_img, dtype=tf.float32)
    input_tensor = input_tensor[tf.newaxis, ...] # Add batch dimension

    # Run Inference
    predictions = infer(input_tensor)
    
    # Extract highest confidence score and class ID
    # Note: Output keys vary by model, common keys are 'output_0' or 'predictions'
    pred_key = list(predictions.keys())[0] 
    output = predictions[pred_key].numpy()[0]
    
    class_id = np.argmax(output)
    confidence = output[class_id]

    # --- 4. SORTING LOGIC ---
    # Only sort if AI is 75% confident
    if confidence > 0.75: 
        detected_class = class_names[class_id]
        print(f"{detected_class} Detected! (Confidence: {confidence:.2f})")

        if class_id == 0: # PLASTIC
            print("Rotating to Bin 1...")
            set_servo_angle(chute_servo, 30) # Rotate to Plastic Bin

        elif class_id == 1: # PAPER
            print("Rotating to Bin 2...")
            set_servo_angle(chute_servo, 90) # Rotate to Paper Bin

        elif class_id == 2: # METAL
            print("Rotating to Bin 3...")
            set_servo_angle(chute_servo, 150) # Rotate to Metal Bin

        # --- DROP THE TRASH ---
        time.sleep(1) # Wait for chute to align
        print("Opening trapdoor...")
        set_servo_angle(trapdoor, 90) # Open flap
        time.sleep(2) # Wait for trash to fall
        set_servo_angle(trapdoor, 0)  # Close flap
        time.sleep(1) # Reset before next detection

    # Display window (Optional, useful for debugging)
    cv2.putText(img, f"Looking for waste...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Pi Waste Scanner", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup GPIO on exit
cap.release()
chute_servo.stop()
trapdoor.stop()
GPIO.cleanup()