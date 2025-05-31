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
# How close averaged sensor values can be to be considered "the same" as another gesture
READING_SIMILARITY_THRESHOLD = 5 # Lowered from 30 to 5 for stricter similarity
SAMPLE_DURATION_SEC = 10.0 # Increased from 2 to 10 seconds for longer sampling
# Approx. how many samples per second ESP32 sends (1000ms / 100ms delay = 10)
SAMPLES_PER_SECOND_EXPECTED = 10

# --- Global Variables ---
ser = None
is_reading_serial = False
serial_thread = None
sensor_data = [0] * NUM_SENSORS # Stores the LATEST instantaneous sensor data
letter_mappings = {}  # Store { 'A': [avg_val1, avg_val2,...], 'B': [...] }

# For continuous display
stop_display_event = threading.Event()
display_thread = None # Will hold the display thread object

# For sampling during save_gesture
sensor_data_history = [] # Store list of recent readings for sampling
history_lock = threading.Lock() # To protect access to sensor_data_history

# --- Helper Functions ---
def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def are_readings_too_similar(readings1, readings2, threshold=READING_SIMILARITY_THRESHOLD):
    """Checks if two sets of sensor readings are too similar."""
    if not readings1 or not readings2 or len(readings1) != len(readings2) or len(readings1) != NUM_SENSORS:
        # print(f"Debug: Similarity check - invalid input: {readings1}, {readings2}")
        return False # Should not happen with consistent NUM_SENSORS and valid data
    try:
        for i in range(len(readings1)):
            # Ensure comparison is between numbers
            if abs(int(readings1[i]) - int(readings2[i])) > threshold:
                return False # At least one sensor is different enough
        return True # All sensors are within the threshold
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
        time.sleep(2) # Allow time for connection to establish and ESP32 to reset
        if ser.is_open:
            ser.flushInput() # Clear any old data in the buffer
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
        # print("Not connected.") # Can be noisy if called during cleanup
        return

    is_reading_serial = False # Signal thread to stop
    if serial_thread and serial_thread.is_alive():
        # print("Waiting for serial thread to stop...")
        serial_thread.join(timeout=1.0) # Reduced timeout, was 1.5
        if serial_thread.is_alive():
            print("Warning: Serial thread did not stop in time.")
    if ser:
        try:
            ser.close()
        except Exception as e:
            print(f"Error closing serial port: {e}")
    ser = None
    # print("Disconnected from ESP32.") # Moved to main context if successful

def read_from_serial_continuously():
    global sensor_data, is_reading_serial, sensor_data_history
    max_history_len = int(SAMPLES_PER_SECOND_EXPECTED * (SAMPLE_DURATION_SEC + 2))

    while is_reading_serial and ser and ser.is_open:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip() # Add errors='ignore'
                if line:
                    # print(f"Raw serial line: '{line}'") # Debug: uncomment to see raw lines
                    parts = line.split(',')
                    if len(parts) == NUM_SENSORS:
                        try:
                            current_sample = [int(p.strip()) for p in parts] # Add p.strip()
                            sensor_data = current_sample

                            with history_lock:
                                sensor_data_history.append(current_sample)
                                if len(sensor_data_history) > max_history_len:
                                    sensor_data_history.pop(0)
                        except ValueError:
                            # print(f"Warning: Could not parse sensor values from line: '{line}'") # Debug: uncomment for parsing errors
                            pass # Silently ignore parsing errors for individual lines
                    # else: # Debug: uncomment to see lines not matching NUM_SENSORS
                        # if line: # Avoid printing for empty lines after strip
                        #    print(f"Warning: Line did not contain {NUM_SENSORS} parts: '{line}' ({len(parts)} parts)")
        except serial.SerialException:
            print("\nSerial error during read. Auto-disconnecting...")
            is_reading_serial = False
            break
        except Exception as e: # Catch broader exceptions
            print(f"\nUnexpected error in serial read thread: {e}")
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
            if not (ser and ser.is_open and is_reading_serial):
                current_data_str = "Sensor Stream: [ESP32 Not Connected or Not Reading]"
            else:
                # Make a local copy for thread safety, though sensor_data updates are mostly atomic for lists of ints
                data_to_display = list(sensor_data)
                if isinstance(data_to_display, list) and len(data_to_display) == NUM_SENSORS and all(isinstance(x, int) for x in data_to_display):
                    current_data_str = f"Sensor Stream: {data_to_display}"
                else:
                    current_data_str = f"Sensor Stream: [Waiting for valid data ({data_to_display})...]"

            if current_data_str != last_displayed_data_str:
                print(f"\r{current_data_str}{' ' * 20}", end="", flush=True)
                last_displayed_data_str = current_data_str
            time.sleep(0.1)
    except Exception as e:
        print(f"\nError in display thread: {e}")
    finally:
        print("\r" + " " * (len(last_displayed_data_str) + 10) + "\r", end="")


# --- Gesture Management (Sampling and Averaging) ---
def collect_samples_for_gesture():
    global sensor_data_history
    
    print(f"\nPrepare to make the gesture. Sampling will begin in 1 second...")
    time.sleep(1)
    
    with history_lock:
        sensor_data_history.clear() # Clear history just before sampling
        
    print(f"Sampling for {SAMPLE_DURATION_SEC} seconds... Hold the pose!")
    start_time = time.time()
    
    while time.time() - start_time < SAMPLE_DURATION_SEC:
        if not is_reading_serial: # Check if serial reading stopped unexpectedly
            print("Warning: Serial reading stopped during sampling.")
            break
        time.sleep(0.05)

    with history_lock:
        collected_readings_snapshot = list(sensor_data_history) # Make a copy
    
    if not collected_readings_snapshot:
        print("Warning: No samples collected during the period. ESP32 might not be sending data, data format incorrect, or serial reading stopped.")
        return None

    print(f"Collected {len(collected_readings_snapshot)} samples.")
    return collected_readings_snapshot

def average_samples(samples_list):
    if not samples_list:
        print("Debug: average_samples received empty samples_list.")
        return None
        
    avg_readings = [0] * NUM_SENSORS
    all_sensors_had_no_valid_data = True # Track if ANY sensor gets valid data
    
    for i in range(NUM_SENSORS):
        sensor_values_for_sensor_i = []
        for sample_idx, sample in enumerate(samples_list):
            if isinstance(sample, list) and len(sample) == NUM_SENSORS:
                try:
                    sensor_values_for_sensor_i.append(int(sample[i]))
                except IndexError:
                    # print(f"Debug: IndexError for sensor {i}, sample {sample_idx}: {sample}") # More detailed
                    pass # Should not happen if len(sample) == NUM_SENSORS
                except ValueError:
                    # print(f"Debug: ValueError for sensor {i}, sample {sample_idx} value '{sample[i]}': {sample}")
                    pass # Skip non-integer values
            # else: # Debug: uncomment to see malformed samples
                # print(f"Debug: Malformed sample structure or length skipped: {sample}")

        if sensor_values_for_sensor_i:
            avg_readings[i] = int(statistics.mean(sensor_values_for_sensor_i))
            all_sensors_had_no_valid_data = False # At least one sensor had processable data
        else:
            # This warning can be noisy if one sensor is consistently bad.
            # print(f"Warning: No valid data for sensor {i} to average. Defaulting to 0 for this sensor.")
            avg_readings[i] = 0 # Default to 0 if no data for this sensor

    if all_sensors_had_no_valid_data:
        print("Error: Could not calculate valid averages for ANY sensor. All sensor columns lacked parseable data.")
        return None
        
    return avg_readings

def save_gesture(letter):
    global letter_mappings, display_thread
    if not (ser and ser.is_open and is_reading_serial):
        print("Error: ESP32 not connected or not actively reading. Cannot save gesture.")
        return

    letter = letter.upper()
    if not (len(letter) == 1 and 'A' <= letter <= 'Z'):
        print("Error: Invalid letter. Please use A-Z.")
        return

    manage_display_thread_pause() # Pause display

    samples = collect_samples_for_gesture()
    if not samples:
        print("Failed to collect samples for the gesture. Gesture not saved.")
        manage_display_thread_resume() # Resume display
        return

    averaged_readings = average_samples(samples)
    if not averaged_readings: # average_samples now returns None on critical failure
        print("Failed to average samples (e.g., no valid data). Gesture not saved.")
        manage_display_thread_resume() # Resume display
        return
    # Additional check, though average_samples should return list of ints or None
    if not all(isinstance(x, int) for x in averaged_readings):
        print(f"Failed to average samples, result is not all integers: {averaged_readings}. Gesture not saved.")
        manage_display_thread_resume() # Resume display
        return

    for existing_letter, readings in letter_mappings.items():
        if are_readings_too_similar(readings, averaged_readings):
            if existing_letter == letter:
                print(f"Warning: These averaged readings are very similar to existing for '{letter}'. Overwriting.")
                break
            else:
                print(f"Error: Averaged readings ({averaged_readings}) are too similar to those for letter '{existing_letter}' ({readings}).")
                print(f"Please make a more distinct gesture or adjust READING_SIMILARITY_THRESHOLD.")
                manage_display_thread_resume() # Resume display
                return

    print(f"Debug: Adding to letter_mappings: '{letter}': {averaged_readings}") # DEBUG
    letter_mappings[letter] = averaged_readings
    print(f"Gesture for letter '{letter}' saved with averaged readings: {averaged_readings}\n")
    print(f"Debug: letter_mappings is now: {letter_mappings}") # DEBUG

    manage_display_thread_resume() # Resume display

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
    # --- Added Debug Print ---
    print(f"Debug save_mappings_to_file: Attempting to save letter_mappings: {letter_mappings}")
    try:
        with open(filepath, 'w') as f:
            json.dump(letter_mappings, f, indent=4)
        print(f"Mappings successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving mappings to {filepath}: {e}")
    except Exception as e: # Catch any other potential errors during dump
        print(f"An unexpected error occurred during saving to file: {e}")


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
                            # print(f"Warning: Mapping for '{k}' in file has {len(v)} sensors, current NUM_SENSORS is {NUM_SENSORS}. Skipping.")
                            incompat_num_sensors_found = True
                    else:
                        print(f"Warning: Invalid mapping data found in file for key '{k}', value '{v}', skipping.")
                
                if incompat_num_sensors_found:
                    print(f"Warning: Some mappings in '{filepath}' were skipped due to NUM_SENSORS mismatch (expected {NUM_SENSORS}).")

                letter_mappings = valid_mappings
                print(f"Mappings loaded from {filepath}. {len(letter_mappings)} valid gestures found.")
            else:
                print(f"Error: Data in {filepath} is not in the expected dictionary format. Initializing empty mappings.")
                letter_mappings = {}
    except FileNotFoundError:
        print(f"Mappings file '{filepath}' not found. Starting with empty mappings.")
        letter_mappings = {}
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' contains invalid JSON. Initializing empty mappings.")
        letter_mappings = {}
    except IOError as e:
        print(f"Error loading mappings from {filepath}: {e}")
        letter_mappings = {}

# --- Main Application Loop ---
def print_help():
    print("\nAvailable commands:")
    print("  connect [port]  - Connect to ESP32 (prompts if port not given)")
    print("  disconnect      - Disconnect from ESP32")
    print("  save <LETTER>   - Sample and save current gesture for the letter (e.g., save A)")
    print("  show <LETTER>   - Show saved readings for a letter (e.g., show A)")
    print("  list            - List all saved gestures")
    print("  savefile        - Save current gestures to file")
    print("  loadfile        - Load gestures from file (overwrites current unsaved changes)")
    print("  clear           - Clear the screen")
    print("  help            - Show this help message")
    print("  quit / exit     - Exit the application")
    print("-" * 40)

def manage_display_thread_pause():
    global display_thread, stop_display_event
    if display_thread and display_thread.is_alive():
        stop_display_event.set()
        display_thread.join(timeout=0.2)
        print("\r" + " " * 70 + "\r", end="") # Clear potential sensor display line

def manage_display_thread_resume():
    global display_thread, stop_display_event, ser, is_reading_serial
    if ser and ser.is_open and is_reading_serial: # Only resume if connected and supposed to be reading
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
            # Ensure display thread is running if connected and not explicitly paused by a command
            if ser and ser.is_open and is_reading_serial and \
               not (display_thread and display_thread.is_alive()) and \
               not stop_display_event.is_set():
                 manage_display_thread_resume()

            if display_thread and display_thread.is_alive():
                print("\r") # Move to next line for input to avoid overwriting display

            command_input = input("Enter command: ").strip().lower()
            parts = command_input.split()
            if not parts:
                continue
            cmd = parts[0]

            # For most commands, pause display. 'save' handles its own.
            # 'disconnect' also implies display should stop naturally.
            if cmd not in ["save", "disconnect", "quit", "exit"]:
                manage_display_thread_pause()

            if cmd == "connect":
                port_arg = parts[1] if len(parts) > 1 else None
                if connect_serial(port_arg): # connect_serial sets is_reading_serial
                    manage_display_thread_resume()
            elif cmd == "disconnect":
                manage_display_thread_pause() # Explicitly pause display before disconnect
                disconnect_serial() # This sets is_reading_serial to False
                print("Disconnected from ESP32.")
            elif cmd == "save":
                if len(parts) > 1:
                    save_gesture(parts[1]) # save_gesture handles its own display pause/resume
                else:
                    print("Usage: save <LETTER>")
                    manage_display_thread_resume() # Resume if command was malformed
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
                load_mappings_from_file() # Potentially overwrites current letter_mappings
            elif cmd == "clear":
                clear_screen()
                print_help() # Show help again after clearing
            elif cmd == "help":
                print_help()
            elif cmd == "quit" or cmd == "exit":
                print("Exiting application...")
                break
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for options.")

            # Resume display if not quitting/exiting, and if not handled by 'save' or 'connect'/'disconnect'
            if cmd not in ["quit", "exit", "disconnect", "save", "connect"]:
                manage_display_thread_resume()

    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
    finally:
        print("Cleaning up...")
        if display_thread and display_thread.is_alive():
            stop_display_event.set()
            display_thread.join(timeout=1.0) # Increased timeout slightly
        if ser and ser.is_open: # Ensure disconnect is called if connected
             disconnect_serial()
             print("Disconnected from ESP32 during cleanup.")
        # save_mappings_to_file() # Optional: auto-save on exit
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
