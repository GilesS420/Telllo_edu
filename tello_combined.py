# Import necessary libraries
from djitellopy import Tello
from drone import keypress as kp
import cv2
import time
import numpy as np

# Initialize the custom keypress module
kp.init()

# Create a Tello drone object and connect to it
me = Tello()
me.connect()

# Print the Tello drone's battery level
print("Battery level:", me.get_battery())

# Start the video stream from the Tello drone
me.streamon()

# Initialize downvision for the bottom camera
downvision_enabled = True
me.send_command_with_return("downvision 1")
print("Downvision enabled")

# Get the frame reader
frame_reader = me.get_frame_read()

# Global variable to store the current frame
img = None

# Function to get control inputs based on keyboard key presses
def getKeyboardInput():
    global img
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50  # Increased from 20 for faster flight

    # ZQSD: Movement control (AZERTY keyboard layout)
    # Z: Forward
    if kp.getKey("z"):
        fb = speed
    # S: Backward
    elif kp.getKey("s"):
        fb = -speed

    # Q: Left
    if kp.getKey("q"):
        lr = -speed
    # D: Right
    elif kp.getKey("d"):
        lr = speed

    # Arrow keys: Height adjustment
    # UP: Move up
    if kp.getKey("UP"):
        ud = speed
    # DOWN: Move down
    elif kp.getKey("DOWN"):
        ud = -speed

    # A/E: Rotation control
    # A: Rotate left (counter-clockwise)
    if kp.getKey("a"):
        yv = -speed
    # E: Rotate right (clockwise)
    elif kp.getKey("e"):
        yv = speed

    # SPACE or T: Take off
    if kp.getKey("space") or kp.getKey("t"):
        print("🚁 Opdracht: Opstijgen!")
        try:
            me.takeoff()
            print("✅ Drone stijgt op")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Fout bij opstijgen: {e}")
        time.sleep(0.5)

    # P: Land the drone
    if kp.getKey("p"):
        me.land()
        time.sleep(3)

    # F: Capture image
    if kp.getKey("f"):
        # Stop the drone for a sharp image
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)  # Brief pause for stabilization
        cv2.imwrite(f'capture_{time.time()}.jpg', img)
        print(f"📸 Image captured: capture_{time.time()}.jpg")
        time.sleep(0.3)

    # V: Toggle downvision (bottom camera)
    if kp.getKey("v"):
        global downvision_enabled
        downvision_enabled = not downvision_enabled
        me.send_command_with_return(f"downvision {1 if downvision_enabled else 0}")
        print(f"Downvision {'enabled' if downvision_enabled else 'disabled'}")
        time.sleep(0.3)

    # Return the control inputs as a list [lr, fb, ud, yv]
    return [lr, fb, ud, yv]

# Main execution loop
print("\n" + "="*40)
print("TELLO DRONE CONTROL - KEYBINDS")
print("="*40)
print("\n🎮 MOVEMENT (ZQSD):")
print("  Z     → Forward")
print("  S     → Backward")
print("  Q     → Left")
print("  D     → Right")
print("\n⬆️  HOOGTE (Arrow Keys):")
print("  ↑     → Omhoog (Up)")
print("  ↓     → Omlaag (Down)")
print("  (Use for height control while flying)")
print("\n🔄 ROTATIE:")
print("  A     → Draaien links")
print("  E     → Draaien rechts")
print("\n✈️  ACTIES:")
print("  SPACE/T → Opstijgen")
print("  P       → Landen")
print("  F       → Foto maken")
print("  V       → Downvision aan/uit")
print("  ESC     → Afsluiten")
print("="*40 + "\n")

try:
    while True:
        # Get control inputs based on keyboard key presses
        vals = getKeyboardInput()

        # Send the control inputs to the Tello drone
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        # Capture a frame from the drone's video stream
        img = frame_reader.frame
        
        # Resize the captured image to a larger size for better visibility
        img = cv2.resize(img, (720, 480))

        # Convert BGR to RGB for proper color display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add status information on the image
        battery = me.get_battery()
        cv2.putText(img_rgb, f"Battery: {battery}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_rgb, f"Downvision: {'ON' if downvision_enabled else 'OFF'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_rgb, "Front Camera", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display the image with status
        cv2.imshow("Tello Drone Control", img_rgb)

        # Check for ESC key press to exit
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # ESC
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Clean up
    print("Cleaning up...")
    me.land()
    time.sleep(2)
    me.streamoff()
    cv2.destroyAllWindows()
    print("Drone landed and disconnected")
