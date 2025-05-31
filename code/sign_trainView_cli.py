import serial
import serial.tools.list_ports
import threading
import time
import json
import string
import os
import statistics # For mean

# --- Configuration ---
SERIAL_BAUD_RATE = 115200
NUM_SENSORS = 5
MAPPINGS_FILE = "sign_language_mappings_cli.json"
READING_SIMILARITY_THRESHOLD = 5
SAMPLE_DURATION_SEC = 10.0 # Duration for saving new gestures
IDENTIFY_SAMPLE_DURATION_SEC = 5.0 # Shorter duration for quick identification
SAMPLES_PER_SECOND_EXPECTED = 10

# --- Global Variables ---
ser = None
is_reading_serial = False
serial_thread = None
sensor_data = [0] * NUM_SENSORS
letter_mappings = {}

stop_display_event = threading.Event()
display_thread = None

sensor_data_history = []
history_lock = threading.Lock()

# --- Helper Functions ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def are_readings_too_similar(readings1, readings2, threshold=READING_SIMILARITY_THRESHOLD):
    if not readings1 or not readings2 or len(readings1) != len(readings2) or len(readings1) != NUM_SENSORS:
        return False
    try:
        for i in range(len(readings1)):
            if abs(int(readings1[i]) - int(readings2[i])) > threshold:
                return False
        return True
    except (ValueError, TypeError) as e:
        print(f"Debug: Error in similarity check with values {readings1}, {readings2}: {e}")
        return False

# --- Serial Communication & Data Reading ---
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
        print(f"Attempting to connect to {port_name}...")
        time.sleep(2)
        if ser.is_open:
            ser.flushInput()
            print(f"Successfully opened port {port_name}. Flushing input buffer.")
            sensor_data = [0] * NUM_SENSORS
            with history_lock:
                sensor_data_history.clear()
            is_reading_serial = True
            serial_thread = threading.Thread(target=read_from_serial_continuously, daemon=True)
            serial_thread.start()
            print(f"Serial reading thread started for {port_name}.")
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
    max_history_len = int(SAMPLES_PER_SECOND_EXPECTED * (max(SAMPLE_DURATION_SEC, IDENTIFY_SAMPLE_DURATION_SEC) + 2)) # ensure enough history for longest sample

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
                            # print(f"DEBUG Warning: Could not parse sensor values from line: '{line}'")
                            pass
                    # else:
                        # if line:
                            # print(f"DEBUG Warning: Line did not contain {NUM_SENSORS} parts: '{line}' ({len(parts)} parts)")
        except serial.SerialException:
            # print("\nDEBUG SerialException in read thread. Auto-disconnecting...")
            is_reading_serial = False
            break
        except Exception as e:
            # print(f"\nDEBUG Unexpected error in serial read thread: {e}")
            is_reading_serial = False
            break
        time.sleep(0.005)
    if is_reading_serial:
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
                print(f"\r{current_data_str}{' ' * 25}", end="", flush=True) # Increased padding
                last_displayed_data_str = current_data_str
            time.sleep(0.1)
    except Exception as e:
        print(f"\nError in display thread: {e}")
    finally:
        print("\r" + " " * (len(last_displayed_data_str) + 30) + "\r", end="") # Increased padding

# --- Gesture Management (Sampling and Averaging) ---
def collect_samples_for_gesture(duration_sec=None, prompt_message_verb="Sampling"):
    global sensor_data_history
    
    if duration_sec is None:
        duration_sec = SAMPLE_DURATION_SEC

    print(f"\nPrepare to make the gesture. {prompt_message_verb} will begin in 1 second...")
    time.sleep(1)
    
    with history_lock:
        sensor_data_history.clear()
        
    print(f"{prompt_message_verb} for {duration_sec} seconds... Hold the pose!")
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        if not is_reading_serial:
            print("Warning: Serial reading stopped during sampling.")
            break
        time.sleep(0.05)

    with history_lock:
        collected_readings_snapshot = list(sensor_data_history)
    
    if not collected_readings_snapshot:
        print(f"Warning: No samples collected during the {duration_sec}s period. ESP32 might not be sending data, data format incorrect, or serial reading stopped.")
        return None

    print(f"Collected {len(collected_readings_snapshot)} samples over {duration_sec}s.")
    return collected_readings_snapshot

def average_samples(samples_list):
    if not samples_list:
        return None
    avg_readings = [0] * NUM_SENSORS
    all_sensors_had_no_valid_data = True
    for i in range(NUM_SENSORS):
        sensor_values_for_sensor_i = []
        for sample in samples_list:
            if isinstance(sample, list) and len(sample) == NUM_SENSORS:
                try:
                    sensor_values_for_sensor_i.append(int(sample[i]))
                except (IndexError, ValueError):
                    pass
        if sensor_values_for_sensor_i:
            avg_readings[i] = int(statistics.mean(sensor_values_for_sensor_i))
            all_sensors_had_no_valid_data = False
        else:
            avg_readings[i] = 0
    if all_sensors_had_no_valid_data:
        print("Error: Could not calculate valid averages for ANY sensor.")
        return None
    return avg_readings

def save_gesture(letter):
    global letter_mappings
    if not (ser and ser.is_open and is_reading_serial):
        print("Error: ESP32 not connected or not actively reading. Cannot save gesture.")
        return
    letter = letter.upper()
    if not (len(letter) == 1 and 'A' <= letter <= 'Z'):
        print("Error: Invalid letter. Please use A-Z.")
        return
    manage_display_thread_pause()
    samples = collect_samples_for_gesture() # Uses default SAMPLE_DURATION_SEC
    if not samples:
        print("Failed to collect samples. Gesture not saved.")
        manage_display_thread_resume()
        return
    averaged_readings = average_samples(samples)
    if not averaged_readings or not all(isinstance(x, int) for x in averaged_readings):
        print(f"Failed to average samples or result invalid: {averaged_readings}. Gesture not saved.")
        manage_display_thread_resume()
        return
    for existing_letter, readings in letter_mappings.items():
        if are_readings_too_similar(readings, averaged_readings):
            if existing_letter == letter:
                print(f"Warning: Readings similar to existing for '{letter}'. Overwriting.")
                break
            else:
                print(f"Error: Readings ({averaged_readings}) too similar to letter '{existing_letter}' ({readings}).")
                manage_display_thread_resume()
                return
    letter_mappings[letter] = averaged_readings
    print(f"Gesture for letter '{letter}' saved with readings: {averaged_readings}\n")
    manage_display_thread_resume()

def identify_current_gesture():
    global letter_mappings, display_thread
    if not letter_mappings:
        print("No gestures saved in memory. Please save some gestures or load from file first.")
        return

    if not (ser and ser.is_open and is_reading_serial):
        print("Error: ESP32 not connected or not actively reading. Cannot identify gesture.")
        return

    manage_display_thread_pause()

    print("Identifying current gesture...")
    samples = collect_samples_for_gesture(
        duration_sec=IDENTIFY_SAMPLE_DURATION_SEC,
        prompt_message_verb="Identifying"
    )

    if not samples:
        print("Failed to collect samples for identification. Gesture not identified.")
        manage_display_thread_resume()
        return

    current_averaged_readings = average_samples(samples)
    if not current_averaged_readings:
        print("Failed to average current samples. Gesture not identified.")
        manage_display_thread_resume()
        return
    if not all(isinstance(x, int) for x in current_averaged_readings):
        print(f"Averaged samples for identification are invalid: {current_averaged_readings}. Gesture not identified.")
        manage_display_thread_resume()
        return

    print(f"Current averaged readings for identification: {current_averaged_readings}")

    possible_matches = []
    for letter, saved_readings in letter_mappings.items():
        if are_readings_too_similar(current_averaged_readings, saved_readings):
            possible_matches.append(letter)

    if not possible_matches:
        print("Unknown gesture. No saved gesture is similar enough to the current pose.")
    elif len(possible_matches) == 1:
        print(f"Identified gesture: {possible_matches[0]}")
    else:
        print(f"Ambiguous gesture. The current pose is similar to: {', '.join(sorted(possible_matches))}")
        print("Consider making your saved gestures more distinct or adjusting READING_SIMILARITY_THRESHOLD.")
    
    print("") # Newline for readability
    manage_display_thread_resume()

# --- File I/O ---
def show_gesture(letter):
    letter = letter.upper()
    if letter in letter_mappings:
        print(f"Gesture for '{letter}': {letter_mappings[letter]}")
    else:
        print(f"No gesture saved for letter '{letter}'.")

def list_gestures():
    if not letter_mappings:
        print("No gestures saved yet.")
        return
    print("\nSaved Gestures (Averaged Values):")
    for letter, readings in sorted(letter_mappings.items()):
        print(f"  {letter}: {readings}")
    print("")

def save_mappings_to_file(filepath=MAPPINGS_FILE):
    global letter_mappings
    try:
        with open(filepath, 'w') as f:
            json.dump(letter_mappings, f, indent=4)
        print(f"Mappings successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving mappings to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")

def load_mappings_from_file(filepath=MAPPINGS_FILE):
    global letter_mappings
    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                valid_mappings = {}
                incompat_num_sensors_found = False
                for k, v in loaded_data.items():
                    if isinstance(k, str) and len(k) == 1 and 'A' <= k.upper() <= 'Z' and \
                       isinstance(v, list) and all(isinstance(x, int) for x in v):
                        if len(v) == NUM_SENSORS:
                            valid_mappings[k.upper()] = v
                        else:
                            incompat_num_sensors_found = True
                    else:
                        print(f"Warning: Invalid mapping data in file for key '{k}', value '{v}', skipping.")
                if incompat_num_sensors_found:
                    print(f"Warning: Some mappings in '{filepath}' skipped (NUM_SENSORS mismatch).")
                letter_mappings = valid_mappings
                print(f"Mappings loaded from {filepath}. {len(letter_mappings)} valid gestures found.")
            else:
                print(f"Error: Data in {filepath} not dict. Initializing empty.")
                letter_mappings = {}
    except FileNotFoundError:
        print(f"Mappings file '{filepath}' not found. Starting empty.")
        letter_mappings = {}
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' invalid JSON. Initializing empty.")
        letter_mappings = {}
    except IOError as e:
        print(f"Error loading mappings from {filepath}: {e}")
        letter_mappings = {}

# --- Main Application Loop ---
def print_help():
    print("\nAvailable commands:")
    print("  connect [port]  - Connect to ESP32")
    print("  disconnect      - Disconnect from ESP32")
    print("  save <LETTER>   - Sample and save gesture for the letter")
    print("  identify        - Try to identify the current gesture")
    print("  show <LETTER>   - Show saved readings for a letter")
    print("  list            - List all saved gestures")
    print("  savefile        - Save current gestures to file")
    print("  loadfile        - Load gestures from file")
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
    load_mappings_from_file()
    clear_screen()
    print("Sign Language Gesture Trainer (CLI) - Averaging Version")
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

            # Pause display for most commands. 'save' and 'identify' handle their own.
            if cmd not in ["save", "identify", "disconnect", "quit", "exit"]:
                manage_display_thread_pause()

            if cmd == "connect":
                port_arg = parts[1] if len(parts) > 1 else None
                if connect_serial(port_arg):
                    manage_display_thread_resume()
            elif cmd == "disconnect":
                manage_display_thread_pause()
                disconnect_serial()
                print("Disconnected from ESP32.")
            elif cmd == "save":
                if len(parts) > 1:
                    save_gesture(parts[1]) # Handles its own display pause/resume
                else:
                    print("Usage: save <LETTER>")
                    manage_display_thread_resume() # Resume if command was malformed
            elif cmd == "identify":
                identify_current_gesture() # Handles its own display pause/resume
            elif cmd == "show":
                if len(parts) > 1:
                    show_gesture(parts[1])
                else:
                    print("Usage: show <LETTER>")
            elif cmd == "list":
                list_gestures()
            elif cmd == "savefile":
                save_mappings_to_file()
            elif cmd == "loadfile":
                load_mappings_from_file()
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
            if cmd not in ["quit", "exit", "disconnect", "save", "identify", "connect"]:
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
    main()
