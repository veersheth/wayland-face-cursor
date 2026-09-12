import time
import wayland_automation as wa

mouse = wa.Mouse()

print("Testing Keyboard automation in 3 seconds")
print("Click inside a text box")
time.sleep(3)

print("Typing...")
wa.typewrite("Hello this is your computer here", interval=0.05)

# print("Pressing Enter key")
# wa.press("enter")

print("Testing mouse automation...")
print("Moving to coordinates 1000 x 700")

try:
    # mouse.click(1000, 800, "left")
    mouse.click(1000, 700, "right")    
    # mouse.swipe(0, 500, 1000, 500, speed=0.1)
except Exception as e:
    print(f"Error: {e}")
