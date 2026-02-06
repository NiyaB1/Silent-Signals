import time
import csv
import os
from pynput import keyboard, mouse
from datetime import datetime

OUTPUT_FILE = "data/interaction_log.csv"

keyboard_events = []
mouse_events = []

def on_key_press(key):
    keyboard_events.append((time.time(), "key_press"))

def on_key_release(key):
    keyboard_events.append((time.time(), "key_release"))

def on_move(x, y):
    mouse_events.append((time.time(), "mouse_move"))

def on_click(x, y, button, pressed):
    if pressed:
        mouse_events.append((time.time(), "mouse_click"))

def on_scroll(x, y, dx, dy):
    direction = "scroll_up" if dy > 0 else "scroll_down"
    mouse_events.append((time.time(), direction))

def save_events():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "event", "source"])

        for t, e in keyboard_events:
            writer.writerow([t, e, "keyboard"])

        for t, e in mouse_events:
            writer.writerow([t, e, "mouse"])

    print("Events saved.")

def main():
    start_time = datetime.now()
    print(f"Logging started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Logging started. Press Ctrl+C to stop.")

    keyboard_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release
    )
    mouse_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll
    )

    keyboard_listener.start()
    mouse_listener.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        end_time = datetime.now()
        print(f"\nLogging stopped at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Stopping listeners...")
        keyboard_listener.stop()
        mouse_listener.stop()
        save_events()
        print("Logging stopped cleanly.")

if __name__ == "__main__":
    main()