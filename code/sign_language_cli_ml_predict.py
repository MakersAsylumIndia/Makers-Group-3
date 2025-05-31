import serial
import serial.tools.list_ports
import threading
import time
import json
import string # Not strictly needed here, but often included
import os
import statistics # For mean (could be useful for pre-filtering, but model handles main logic)
import numpy as np
import joblib # For loading the model, scaler, label_encoder

# --- Configuration from Training Script (MUST MATCH) ---
MODEL_FILE = "sign_language_knn_model.joblib"
SCALER_FILE = "sign_language_scaler.joblib"
LABEL_ENCODER_FILE = "sign_language_label_encoder.joblib"
NUM_SENSORS = 5 # Must match your data collection and training

# --- CLI Configuration ---
SERIAL_BAUD_RATE = 115200
# How long to sample sensor data for one prediction attempt
PREDICTION_SAMPLE_DURATION_SEC = 1.5 # Shorter for quicker feedback
SAMPLES_PER_SECOND_EXPECTED = 10 # Approx. ESP32 sends 1000ms/100ms delay

# --- Global Variables ---
ser = None
is_reading_serial = False
serial_thread = None
sensor_data = [0] * NUM_SENSORS # Stores the LATEST instantaneous sensor data

# ML Model Components
ml_model = None
ml_scaler = None
ml_label_encoder = None

# For continuous display
stop_display_event = threading.Event()
display_thread = None

# For sampling during prediction
sensor_data_history = []
history_lock = threading.Lock()

# --- Helper Functions ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Model and Pipeline Loading ---
def load_trained_pipeline():
    global ml_model, ml_scaler, ml_label_encoder
    model_loaded, scaler_loaded, label_encoder_loaded = True, True, True

    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found. Please train the model first.")
        model_loaded = False
    if not os.path.exists(SCALER_FILE):
        print(f"Error: Scaler file '{SCALER_FILE}' not found. Please train the model first.")
        scaler_loaded = False
    if not os.path.exists(LABEL_ENCODER_FILE):
        print(f"Error: Label Encoder file '{LABEL_ENCODER_FILE}' not found. Please train the model first.")
        label_encoder_loaded = False

    if not (model_loaded and scaler_loaded and label_encoder_loaded):
        return False

    try:
        ml_model = joblib.load(MODEL_FILE)
        ml_scaler = joblib.load(SCALER_FILE)
        ml_label_encoder = joblib.load(LABEL_ENCODER_FILE)
        print("ML Model, Scaler, and Label Encoder loaded successfully.")
        # You can print model details if you want, e.g., ml_model.get_params()
        return True
    except Exception as e:
        print(f"Error loading ML pipeline components: {e}")
        ml_model, ml_scaler, ml_label_encoder = None, None, None
        return False

# --- Serial Communication & Data Reading (Similar to your other CLI) ---
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return None
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
    while True:
        try:
            choice = input(f"Select port number (0-{len(ports)-1}) or 'c' to cancel: ")
            if choice.lower() == 'c':
                return None
            port_index = int(choice)
            if 0 <= port_index < len(ports):
                return ports[port_index].device
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def connect_serial(port_name=None):
    global ser, is_reading_serial, serial_thread, sensor_data, sensor_data_history
    if ser and ser.is_open:
        print("Already connected.")
        return True
    if not port_name:
        port_name = list_serial_ports()
        if not port_name:
            return False
    try:
        ser = serial.Serial(port_name, SERIAL_BAUD_RATE, timeout=1)
        time.sleep(2)
        if ser.is_open:
            ser.flushInput()
            sensor_data = [0] * NUM_SENSORS
            with history_lock:
                sensor_data_history.clear()
            is_reading_serial = True
            serial_thread = threading.Thread(target=read_from_serial_continuously, daemon=True)
            serial_thread.start()
            print(f"Successfully connected to {port_name}.")
            return True
        else:
            print(f"Failed to open port {port_name}.")
            ser = None
            return False
    except serial.SerialException as e:
        print(f"Serial connection error on {port_name}: {e}")
        ser = None
        return False

def disconnect_serial():
    global ser, is_reading_serial, serial_thread
    if not ser or not ser.is_open:
        return
    is_reading_serial = False
    if serial_thread and serial_thread.is_alive():
        serial_thread.join(timeout=1.0)
        if serial_thread.is_alive():
            print("Warning: Serial thread did not stop in time.")
    if ser:
        try:
            ser.close()
        except Exception as e:
            print(f"Error closing serial port: {e}")
    ser = None

def read_from_serial_continuously():
    global sensor_data, is_reading_serial, sensor_data_history
    # Adjust max_history_len if needed, PREDICTION_SAMPLE_DURATION_SEC is key here
    max_history_len = int(SAMPLES_PER_SECOND_EXPECTED * (PREDICTION_SAMPLE_DURATION_SEC + 2))

    while is_reading_serial and ser and ser.is_open:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == NUM_SENSORS:
                        try:
                            current_sample = [int(p.strip()) for p in parts]
                            sensor_data = current_sample
                            with history_lock:
                                sensor_data_history.append(current_sample)
                                if len(sensor_data_history) > max_history_len:
                                    sensor_data_history.pop(0)
                        except ValueError:
                            pass # Silently ignore parsing errors
        except serial.SerialException:
            is_reading_serial = False
            break
        except Exception:
            is_reading_serial = False
            break
        time.sleep(0.005)
    if is_reading_serial: # Ensure flag is false if loop exits unexpectedly
        is_reading_serial = False

# --- Live Sensor Data Display ---
def print_sensor_data_continuously():
    global sensor_data, stop_display_event
    stop_display_event.clear()
    last_displayed_data_str = ""
    try:
        while not stop_display_event.is_set():
            current_data_str = ""
            if not (ser and ser.is_open and is_reading_serial):
                current_data_str = "Sensor Stream: [ESP32 Not Connected or Not Reading]"
            else:
                data_to_display = list(sensor_data)
                if isinstance(data_to_display, list) and \
                   len(data_to_display) == NUM_SENSORS and \
                   all(isinstance(x, int) for x in data_to_display):
                    current_data_str = f"Sensor Stream: {data_to_display}"
                else:
                    current_data_str = f"Sensor Stream: [Waiting for valid data ({data_to_display})...]"

            if current_data_str != last_displayed_data_str:
                print(f"\r{current_data_str}{' ' * 25}", end="", flush=True)
                last_displayed_data_str = current_data_str
            time.sleep(0.1)
    except Exception as e:
        print(f"\nError in display thread: {e}")
    finally:
        print("\r" + " " * (len(last_displayed_data_str) + 30) + "\r", end="")

# --- Gesture Prediction ---
def collect_and_average_samples_for_prediction(duration_sec=PREDICTION_SAMPLE_DURATION_SEC):
    global sensor_data_history
    
    print(f"\nPrepare to make the gesture. Sampling for prediction will begin in 1 second...")
    time.sleep(1)
    
    with history_lock:
        sensor_data_history.clear() # Get fresh samples
        
    print(f"Sampling for {duration_sec} seconds... Hold the pose!")
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        if not is_reading_serial:
            print("Warning: Serial reading stopped during sampling for prediction.")
            break
        time.sleep(0.05)

    with history_lock:
        collected_readings_snapshot = list(sensor_data_history)
    
    if not collected_readings_snapshot:
        print("Warning: No samples collected for prediction.")
        return None

    print(f"Collected {len(collected_readings_snapshot)} samples for prediction.")
    
    # Average the samples (same logic as your training data prep)
    if not collected_readings_snapshot:
        return None
    avg_readings = [0] * NUM_SENSORS
    all_sensors_had_no_valid_data = True
    for i in range(NUM_SENSORS):
        sensor_values_for_sensor_i = [s[i] for s in collected_readings_snapshot if isinstance(s, list) and len(s) == NUM_SENSORS]
        if sensor_values_for_sensor_i:
            try:
                avg_readings[i] = int(statistics.mean(sensor_values_for_sensor_i))
                all_sensors_had_no_valid_data = False
            except statistics.StatisticsError: # Should not happen if list is not empty
                 avg_readings[i] = 0 # Default if error
        else:
            avg_readings[i] = 0
    
    if all_sensors_had_no_valid_data:
        print("Error: Could not calculate valid averages from collected samples for prediction.")
        return None
        
    return avg_readings


def predict_live_gesture():
    global ml_model, ml_scaler, ml_label_encoder
    if not all([ml_model, ml_scaler, ml_label_encoder]):
        print("Error: ML model, scaler, or label encoder not loaded. Cannot predict.")
        print("Please ensure the trained model files exist and run `loadmodel` command if needed.")
        return

    if not (ser and ser.is_open and is_reading_serial):
        print("Error: ESP32 not connected or not actively reading. Cannot predict.")
        return

    manage_display_thread_pause()

    current_averaged_readings = collect_and_average_samples_for_prediction()

    if not current_averaged_readings:
        print("Failed to get valid averaged readings for prediction.")
        manage_display_thread_resume()
        return
    
    print(f"Averaged sensor readings for prediction: {current_averaged_readings}")

    try:
        readings_np = np.array(current_averaged_readings).reshape(1, -1)
        readings_scaled = ml_scaler.transform(readings_np)

        prediction_encoded = ml_model.predict(readings_scaled)
        prediction_proba = ml_model.predict_proba(readings_scaled)

        predicted_letter = ml_label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100 # As percentage

        print(f"\n>>> Predicted Gesture: {predicted_letter} (Confidence: {confidence:.2f}%) <<<")
        
        # Optional: show probabilities for all classes
        classes = ml_label_encoder.classes_
        print("Probabilities per class:")
        for i, class_label in enumerate(classes):
            print(f"  {class_label}: {prediction_proba[0][i]*100:.2f}%")

    except Exception as e:
        print(f"Error during ML prediction: {e}")
    
    print("") # Newline for readability
    manage_display_thread_resume()


# --- Main Application Loop ---
def print_help():
    print("\nAvailable commands:")
    print("  connect [port]  - Connect to ESP32")
    print("  disconnect      - Disconnect from ESP32")
    print("  predict         - Sample from ESP32 and predict current gesture using ML model")
    print("  loadmodel       - (Re)load the trained ML model, scaler, and label encoder")
    print("  clear           - Clear screen")
    print("  help            - Show help")
    print("  quit / exit     - Exit")
    print("-" * 40)

def manage_display_thread_pause():
    global display_thread, stop_display_event
    if display_thread and display_thread.is_alive():
        stop_display_event.set()
        display_thread.join(timeout=0.3)
        print("\r" + " " * 80 + "\r", end="")

def manage_display_thread_resume():
    global display_thread, stop_display_event, ser, is_reading_serial
    if ser and ser.is_open and is_reading_serial:
        if not (display_thread and display_thread.is_alive()):
            stop_display_event.clear()
            display_thread = threading.Thread(target=print_sensor_data_continuously, daemon=True)
            display_thread.start()

def main():
    global display_thread, ser, is_reading_serial
    
    # Attempt to load the ML pipeline at startup
    if not load_trained_pipeline():
        print("Warning: ML model pipeline could not be loaded on startup.")
        print("You may need to train your model or ensure model files are present.")
        print("The 'predict' command will not work until the model is loaded successfully (use 'loadmodel').")

    clear_screen()
    print("Sign Language Gesture Predictor (CLI - ML Version)")
    print_help()

    try:
        while True:
            if ser and ser.is_open and is_reading_serial and \
               not (display_thread and display_thread.is_alive()) and \
               not stop_display_event.is_set():
                 manage_display_thread_resume()

            if display_thread and display_thread.is_alive():
                print("\r")

            command_input = input("Enter command: ").strip().lower()
            parts = command_input.split()
            if not parts:
                continue
            cmd = parts[0]

            # Pause display for most commands. 'predict' handles its own.
            if cmd not in ["predict", "disconnect", "quit", "exit"]:
                manage_display_thread_pause()

            if cmd == "connect":
                port_arg = parts[1] if len(parts) > 1 else None
                if connect_serial(port_arg):
                    manage_display_thread_resume()
            elif cmd == "disconnect":
                manage_display_thread_pause()
                disconnect_serial()
                print("Disconnected from ESP32.")
            elif cmd == "predict":
                predict_live_gesture() # Handles its own display pause/resume
            elif cmd == "loadmodel":
                print("Attempting to reload ML model pipeline...")
                if load_trained_pipeline():
                    print("Reload successful.")
                else:
                    print("Reload failed. Check file paths and model integrity.")
            elif cmd == "clear":
                clear_screen()
                print_help()
            elif cmd == "help":
                print_help()
            elif cmd == "quit" or cmd == "exit":
                print("Exiting application...")
                break
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for options.")

            # Resume display if not quitting/exiting, and if not handled by specific commands
            if cmd not in ["quit", "exit", "disconnect", "predict", "connect"]:
                manage_display_thread_resume()

    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
    finally:
        print("Cleaning up...")
        if display_thread and display_thread.is_alive():
            stop_display_event.set()
            display_thread.join(timeout=1.0)
        if ser and ser.is_open:
             disconnect_serial()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    # Ensure scikit-learn and joblib are installed:
    # pip install scikit-learn joblib numpy pyserial
    main()
