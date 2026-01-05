import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize pyautogui with safety features
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05  # Short pause between actions

# Game control mappings (adjust these based on your browser zoom level)
GAME_AREA = (100, 200)  # Top-left corner of game area on screen
ACCELERATE_KEY = 'up'    # or 'w' depending on game controls
BRAKE_KEY = 'down'       # or 's'
TILT_LEFT_KEY = 'left'   # or 'a'
TILT_RIGHT_KEY = 'right' # or 'd'

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # Finger tip landmarks
    count = 0
    
    # Thumb (compare x-coordinates)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1
    
    # Other fingers (compare y-coordinates)
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    
    return count

def control_game(finger_count, last_state):
    new_state = None
    
    if finger_count == 1:  # Index finger - accelerate
        pyautogui.keyDown(ACCELERATE_KEY)
        pyautogui.keyUp(BRAKE_KEY)
        new_state = 'accelerate'
    elif finger_count == 2:  # Index + middle - brake
        pyautogui.keyDown(BRAKE_KEY)
        pyautogui.keyUp(ACCELERATE_KEY)
        new_state = 'brake'
    elif finger_count == 3:  # Tilt left
        pyautogui.keyDown(TILT_LEFT_KEY)
        pyautogui.keyUp(TILT_RIGHT_KEY)
        new_state = 'left'
    elif finger_count == 4:  # Tilt right
        pyautogui.keyDown(TILT_RIGHT_KEY)
        pyautogui.keyUp(TILT_LEFT_KEY)
        new_state = 'right'
    else:  # No action
        pyautogui.keyUp(ACCELERATE_KEY)
        pyautogui.keyUp(BRAKE_KEY)
        pyautogui.keyUp(TILT_LEFT_KEY)
        pyautogui.keyUp(TILT_RIGHT_KEY)
        new_state = 'none'
    
    # Only print if state changed
    if new_state != last_state:
        print(f"Action: {new_state}")
    
    return new_state

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    last_state = None
    
    try:
        print("Starting gesture control. Make sure:")
        print("1. Game is in focus (click on the game window)")
        print("2. Browser zoom is at 100%")
        print("3. Hands are visible to camera")
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    finger_count = count_fingers(hand_landmarks)
                    last_state = control_game(finger_count, last_state)
                    
                    # Display finger count
                    cv2.putText(img, f"Fingers: {finger_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                last_state = control_game(0, last_state)
                cv2.putText(img, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Gesture Control", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
                
    finally:
        # Release all keys and resources
        pyautogui.keyUp(ACCELERATE_KEY)
        pyautogui.keyUp(BRAKE_KEY)
        pyautogui.keyUp(TILT_LEFT_KEY)
        pyautogui.keyUp(TILT_RIGHT_KEY)
        cap.release()
        cv2.destroyAllWindows()
        print("Gesture control stopped")

if __name__ == "__main__":
    main()