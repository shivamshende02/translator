import pyautogui
import tkinter as tk

def start_dictation():
    """Simulates the Windows + H key combination to start voice typing."""
    try:
        pyautogui.hotkey("win", "h")  # Activate Windows dictation tool
        print("Voice typing started. Speak now!")
    except Exception as e:
        print(f"Error starting dictation: {e}")

def stop_dictation():
    """Simulates the Escape key to stop voice typing."""
    try:
        pyautogui.press("esc")  # Stops Windows dictation tool
        print("Voice typing stopped.")
    except Exception as e:
        print(f"Error stopping dictation: {e}")

def capture_text():
    """Captures the text entered into the text box."""
    user_text = text_box.get("1.0", tk.END).strip()  # Get text from the text box
    if user_text:
        print("\nCaptured Text:")
        print(user_text)
    else:
        print("\nNo text to capture. Please dictate some text first.")
    text_box.focus_set()

# Create the GUI
root = tk.Tk()
root.title("Windows Voice Typing Tool")
root.geometry("600x600")

# Instructions
instructions = tk.Label(root, text="This tool activates Windows Voice Typing.\nUse the buttons below to start and stop dictation.")
instructions.pack(pady=10)

# Text box to display typed text
text_box = tk.Text(root, width=50, height=10)
text_box.pack(pady=20)
text_box.focus_set()

# Button to start dictation
start_button = tk.Button(root, text="Start Voice Typing", command=start_dictation, bg="green", fg="white", width=20)
start_button.pack(pady=5)

# Button to stop dictation
stop_button = tk.Button(root, text="Stop Voice Typing", command=stop_dictation, bg="red", fg="white", width=20)
stop_button.pack(pady=5)

# Button to capture text from the text box
capture_button = tk.Button(root, text="Capture Text", command=capture_text, width=20)
capture_button.pack(pady=10)

# Run the GUI application
root.mainloop()
