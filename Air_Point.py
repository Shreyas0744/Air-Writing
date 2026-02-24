import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

# --- Configuration ---
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
ROI_MARGIN_X = 100   
ROI_MARGIN_Y = 80    
PINCH_RATIO_ON = 0.25
PINCH_RATIO_OFF = 0.40
PINCH_DEBOUNCE_S = 0.2
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully!")

# --- UI Button Class ---
class Button:
    def __init__(self, x, y, w, h, text, color, action_type, action_value):
        self.rect = (x, y, w, h)
        self.text = text
        self.color = color # Display color
        self.action_type = action_type # 'color', 'tool', 'clear'
        self.action_value = action_value # (r,g,b) or size or None

    def draw(self, img, is_hovered):
        x, y, w, h = self.rect
        # Shadow/Border
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 3)
        # Fill
        fill_color = self.color
        if is_hovered:
            # Lighten click feedback
            fill_color = tuple(min(c + 50, 255) for c in self.color)
        
        cv2.rectangle(img, (x+3, y+3), (x + w - 3, y + h - 3), fill_color, -1)
        
        # Text
        text_color = (255, 255, 255)
        # If color is very bright (like Yellow/White), use black text
        if sum(self.color) > 400: text_color = (0, 0, 0)
        
        cv2.putText(img, self.text, (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

# --- Adaptive Kalman Filter Class ---
class HandKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
        # Initial parameters (will be adjusted dynamically)
        self.process_noise_base = 1e-4 # Very smooth for static/slow
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise_base
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1 
        
        self.prediction = np.zeros((2, 1), np.float32)
        self.initialized = False

    def predict(self, x, y, speed):
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        
        # ADAPTIVE SMOOTHING:
        # Movement Speed (px/frame) determines Process Noise
        # Slow (< 5px): High smoothing (Noise ~ 1e-4)
        # Fast (> 50px): Low smoothing (Noise ~ 1e-1)
        
        # Scale speed to 0.0-1.0 range (clamped at 50px)
        speed_factor = np.clip(speed / 50.0, 0, 1.0)
        
        # Non-linear ramp: square the factor to stay smooth at low speeds
        dynamic_noise = self.process_noise_base + (0.1 * (speed_factor ** 2))
        
        # Ensure float32 for OpenCV
        self.kf.processNoiseCov = (np.eye(4, dtype=np.float32) * dynamic_noise).astype(np.float32)
        
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])

class AirPainter:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, # STATEFUL TRACKING (Perfect Detection)
            num_hands=2,  # Enable 2 hands
            min_hand_detection_confidence=0.6, # Lower = easier to detect initially
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5       # Lower = sticks to hand better
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.is_drawing = False # Track state for hysteresis
        self.kf = HandKalmanFilter() # Initialize Kalman Filter
        self.start_time = time.time() # For timestamp calc
        
        # Track previous raw position to calculate speed for KF
        self.prev_raw_x = 0
        self.prev_raw_y = 0

        # Gesture History
        self.history_size = 30
        self.history = [] # list of (x, y) coordinates of index finger tip
        self.gesture_cooldown = 0
        self.current_gesture = "None"

        # Tool State
        self.brush_color = (0, 255, 0) # Default Green
        self.brush_size = 8
        self.eraser_mode = False
        self.tool_mode = 'free' # 'free', 'line', 'rect', 'circle'
        self.shape_start = None # (x, y)
        self.release_start_time = 0 # Debounce timer
        
        # UI Setup
        self.custom_gesture_template = None
        self.recording_start_time = 0
        self.gesture_threshold = 0.12 # Increased sensitivity (more forgiving)
        self.gesture_error_smooth = 1.0 # Initialize with high error

        # UI Setup
        self.buttons = []
        # Colors: BGR
        self.buttons.append(Button(20, 10, 80, 50, "Red", (0, 0, 255), 'color', (0, 0, 255)))
        self.buttons.append(Button(110, 10, 80, 50, "Green", (0, 255, 0), 'color', (0, 255, 0)))
        self.buttons.append(Button(200, 10, 80, 50, "Blue", (255, 0, 0), 'color', (255, 0, 0)))
        self.buttons.append(Button(290, 10, 80, 50, "Yelo", (0, 255, 255), 'color', (0, 255, 255)))
        
        self.buttons.append(Button(400, 10, 80, 50, "Eraser", (0, 0, 0), 'tool', 'eraser'))
        self.buttons.append(Button(490, 10, 80, 50, "Free", (100, 100, 100), 'tool', 'free'))
        
        # Shapes
        self.buttons.append(Button(580, 10, 80, 50, "Line", (150, 150, 150), 'tool', 'line'))
        self.buttons.append(Button(670, 10, 80, 50, "Rect", (150, 150, 150), 'tool', 'rect'))
        self.buttons.append(Button(760, 10, 80, 50, "Circ", (150, 150, 150), 'tool', 'circle'))
        
        self.buttons.append(Button(850, 10, 80, 50, "Clear", (50, 50, 200), 'clear', None))
        self.buttons.append(Button(940, 10, 100, 50, "New Gest", (200, 100, 50), 'record_gesture', None))
        self.buttons.append(Button(1050, 10, 80, 50, "Reset", (100, 100, 100), 'reset_gesture', None))
        self.buttons.append(Button(1140, 10, 120, 50, "Gestures", (150, 150, 0), 'toggle_gestures', None))

        self.show_gestures_help = False
        self.active_trigger_gesture = "Pinc"
        self.gesture_buttons = [] # For the selection menu
        
        gestures_list = ["Open palm", "Fist", "Thumbs up", "Two fingers", "Pinc"]
        for i, g in enumerate(gestures_list):
            # Layout the gesture selection buttons vertically on the right
            self.gesture_buttons.append(Button(980, 150 + i*60, 250, 50, g, (100, 100, 100), 'set_trigger', g))

    def map_coordinates(self, x, y, width, height):
        """Maps coordinates from a smaller ROI to the full screen."""
        # Clamp coordinates to the ROI
        x = np.clip(x, ROI_MARGIN_X, width - ROI_MARGIN_X)
        y = np.clip(y, ROI_MARGIN_Y, height - ROI_MARGIN_Y)
        
        # Normalize to 0.0 - 1.0 within the ROI
        norm_x = (x - ROI_MARGIN_X) / (width - 2 * ROI_MARGIN_X)
        norm_y = (y - ROI_MARGIN_Y) / (height - 2 * ROI_MARGIN_Y)
        
        # Map to full screen resolution
        return int(norm_x * width), int(norm_y * height)

    def normalize_landmarks(self, landmarks):
        """Normalizes landmarks to be invariant to scale and translation."""
        # Convert to numpy array
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # 1. Translation Invariance: Center around wrist (index 0)
        wrist = coords[0]
        coords -= wrist
        
        # 2. Scale Invariance: Scale by distance from Wrist to Middle Finger MCP (Index 9)
        # Using MCP instead of tip for stability against finger curling
        scale = np.linalg.norm(coords[9]) 
        if scale < 1e-6: scale = 1.0 # Prevent div by zero
        coords /= scale
        
        return coords

    def is_finger_extended(self, hand_landmarks, finger_index):
        """
        Determines if a finger is extended.
        Indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
        """
        # For non-thumb fingers, check if tip is above PIP/MCP
        if finger_index == 4: # Thumb
            # Compare thumb tip with thumb IP (index 3) or MCP (index 2)
            # Depending on handedness (horizontal relative position)
            # For simplicity, use distance from wrist comparison or MCP
            return np.linalg.norm(np.array([hand_landmarks[4].x, hand_landmarks[4].y]) - np.array([hand_landmarks[0].x, hand_landmarks[0].y])) > \
                   np.linalg.norm(np.array([hand_landmarks[3].x, hand_landmarks[3].y]) - np.array([hand_landmarks[0].x, hand_landmarks[0].y]))
        
        # Other fingers: Check if tip y is significantly higher (smaller value) than intermediate joint
        # idx-2 is the joint below tip (PIP)
        return hand_landmarks[finger_index].y < hand_landmarks[finger_index - 2].y

    def detect_gesture(self, hand_landmarks):
        """Returns the name of the detected gesture."""
        extended = [
            self.is_finger_extended(hand_landmarks, 4),  # Thumb
            self.is_finger_extended(hand_landmarks, 8),  # Index
            self.is_finger_extended(hand_landmarks, 12), # Middle
            self.is_finger_extended(hand_landmarks, 16), # Ring
            self.is_finger_extended(hand_landmarks, 20)  # Pinky
        ]
        
        # Static Gestures
        if all(extended): return "Open palm"
        if not any(extended): return "Fist"
        if extended[0] and not any(extended[1:]): return "Thumbs up"
        if not extended[0] and extended[1] and extended[2] and not extended[3] and not extended[4]: return "Two fingers"
        
        # Pinch (already handled by distance, but for consistency):
        # We'll use the existing pinch ratio logic in the main loop for "Pinch"
        
        # Dynamic Gestures (Swipe, Circle) are handled in the main loop using history
        return None

    def get_gesture_error(self, landmarks):
        if self.custom_gesture_template is None:
            return 1.0 # Max error
        
        current_gesture = self.normalize_landmarks(landmarks)
        
        # Calculate Mean Squared Error
        mse = np.mean(np.square(current_gesture - self.custom_gesture_template))
        return mse

    def process_ui(self, frame, cx, cy, is_pinching):
        # UI Background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (220, 220, 220), -1)
        
        clicked_btn = None
        for btn in self.buttons:
            bx, by, bw, bh = btn.rect
            is_hovered = (bx < cx < bx + bw) and (by < cy < by + bh)
            
            btn.draw(frame, is_hovered)
            
            if is_hovered and is_pinching:
                clicked_btn = btn
        
        if clicked_btn:
            if clicked_btn.action_type == 'color':
                self.brush_color = clicked_btn.action_value
                self.brush_size = 8
                self.eraser_mode = False
            elif clicked_btn.action_type == 'tool':
                if clicked_btn.action_value == 'eraser':
                    self.eraser_mode = True
                    self.tool_mode = 'free'
                    self.brush_color = (0, 0, 0)
                    self.brush_size = 30
                else: # Free, Line, Rect, Circle
                    self.eraser_mode = False
                    self.tool_mode = clicked_btn.action_value
                    self.brush_size = 8
                    self.brush_color = (0, 255, 0) 
            elif clicked_btn.action_type == 'clear':
                self.canvas = np.zeros_like(self.canvas)
            elif clicked_btn.action_type == 'record_gesture':
                self.recording_start_time = time.time()
                self.custom_gesture_template = None
            elif clicked_btn.action_type == 'reset_gesture':
                self.custom_gesture_template = None
                self.gesture_error_smooth = 1.0
            elif clicked_btn.action_type == 'toggle_gestures':
                self.show_gestures_help = not self.show_gestures_help
            elif clicked_btn.action_type == 'set_trigger':
                self.active_trigger_gesture = clicked_btn.action_value
                # Keep menu open after selection or close it? Let's keep it open for now
                # self.show_gestures_help = False 
                
        # If menu is open, process those buttons too
        if self.show_gestures_help:
            for btn in self.gesture_buttons:
                bx, by, bw, bh = btn.rect
                is_hovered = (bx < cx < bx + bw) and (by < cy < by + bh)
                
                # Special styling for active trigger
                original_color = btn.color
                if btn.action_value == self.active_trigger_gesture:
                    btn.color = (0, 255, 255) # Highlight Yellow/Gold
                
                btn.draw(frame, is_hovered)
                btn.color = original_color # Restore for next frame
                
                if is_hovered and is_pinching:
                    self.active_trigger_gesture = btn.action_value
                    clicked_btn = btn
                    
        return clicked_btn is not None

    def process(self, frame):
        h, w, c = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), np.uint8)

        frame = cv2.flip(frame, 1) # Mirror first
        
        # Draw ROI Boundary (Inactive for UI, but shows reachable area)
        cv2.rectangle(frame, (ROI_MARGIN_X, ROI_MARGIN_Y), (w - ROI_MARGIN_X, h - ROI_MARGIN_Y), (0, 255, 255), 1)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Calculate timestamp in ms
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        # Use detect_for_video for stateful tracking
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        ui_drawn = False

        # --- Recording Logic ---
        if self.recording_start_time > 0:
            elapsed = time.time() - self.recording_start_time
            remaining = 3.0 - elapsed
            
            if remaining > 0:
                cv2.putText(frame, f"Recording in {int(np.ceil(remaining))}...", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                self.recording_start_time = 0
                # Capture NOW if hands are present
                if detection_result.hand_landmarks:
                    # Find the "Left" (User Right) hand preferably, or just first hand
                    target_hand = None
                    for i, handedness in enumerate(detection_result.handedness):
                        if handedness[0].category_name == "Left":
                            target_hand = detection_result.hand_landmarks[i]
                            break
                    if target_hand is None: target_hand = detection_result.hand_landmarks[0]
                    
                    self.custom_gesture_template = self.normalize_landmarks(target_hand)
                    print("Custom Gesture Saved!")
                    cv2.putText(frame, "Gesture Saved!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[i][0].category_name
                
                idx_tip = hand_landmarks[INDEX_FINGER_TIP]
                thm_tip = hand_landmarks[THUMB_TIP]
                
                ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
                tx, ty = int(thm_tip.x * w), int(thm_tip.y * h)

                # Visualize landmarks
                cv2.circle(frame, (ix, iy), 3, (255, 0, 0), -1)
                cv2.circle(frame, (tx, ty), 3, (255, 0, 0), -1)

                if handedness == "Left":  # DRAWING HAND (User's Right Hand)
                    # Calculate raw_cx, raw_cy as the average of index and thumb tips
                    raw_cx = (ix + tx) // 2
                    raw_cy = (iy + ty) // 2
                    
                    # Map coordinates for better edge reaching
                    mapped_x, mapped_y = self.map_coordinates(raw_cx, raw_cy, w, h)
                    
                    # Calculate RAW speed (before smoothing)
                    if self.prev_raw_x == 0: self.prev_raw_x, self.prev_raw_y = mapped_x, mapped_y
                    speed = np.hypot(mapped_x - self.prev_raw_x, mapped_y - self.prev_raw_y)
                    self.prev_raw_x, self.prev_raw_y = mapped_x, mapped_y

                    # Adaptive Prediction
                    cx, cy = self.kf.predict(mapped_x, mapped_y, speed)
                    
                    # Update History for dynamic gestures
                    self.history.append((cx, cy))
                    if len(self.history) > self.history_size:
                        self.history.pop(0)

                    # --- Dynamic Gesture Detection ---
                    detected_dynamic = None
                    if len(self.history) == self.history_size:
                        # Swipe Detection
                        dx = self.history[-1][0] - self.history[0][0]
                        dy = self.history[-1][1] - self.history[0][1]
                        dist = np.hypot(dx, dy)
                        
                        if dist > 150: # Minimum distance for swipe
                            if abs(dx) > abs(dy) * 2: # Mostly horizontal
                                detected_dynamic = "Swipe right" if dx > 0 else "Swipe left"
                        
                        # Circle Detection
                        if detected_dynamic is None:
                            # Check if first and last are close
                            end_dist = np.hypot(self.history[-1][0] - self.history[0][0], 
                                                self.history[-1][1] - self.history[0][1])
                            if end_dist < 80:
                                # Check total path length
                                path_len = sum(np.hypot(self.history[i][0]-self.history[i-1][0], 
                                                        self.history[i][1]-self.history[i-1][1]) 
                                               for i in range(1, len(self.history)))
                                if path_len > 300: # Significant movement
                                    # Check bounding box size to ensure it's not a back-and-forth
                                    min_x = min(p[0] for p in self.history)
                                    max_x = max(p[0] for p in self.history)
                                    min_y = min(p[1] for p in self.history)
                                    max_y = max(p[1] for p in self.history)
                                    if (max_x - min_x) > 100 and (max_y - min_y) > 100:
                                        detected_dynamic = "Circle gesture"

                    # --- Gesture Priority ---
                    static_gesture = self.detect_gesture(hand_landmarks)
                    
                    # Pinch Check
                    distance = np.hypot(ix - tx, iy - ty)
                    wx, wy = int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h)
                    mx, my = int(hand_landmarks[9].x * w), int(hand_landmarks[9].y * h)
                    scale = np.hypot(wx - mx, wy - my)
                    if scale < 1.0: scale = 1.0
                    pinch_ratio = distance / scale
                    
                    is_pinching_now = pinch_ratio < PINCH_RATIO_ON
                    
                    if is_pinching_now: 
                        self.current_gesture = "Pinc" # Standardizing name as requested
                    elif detected_dynamic:
                        self.current_gesture = detected_dynamic
                        self.history = [] # Clear history to avoid multi-triggers
                    elif static_gesture:
                        self.current_gesture = static_gesture
                    else:
                        self.current_gesture = "None"

                    # Display Current Gesture
                    cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 680), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    # Draw UI and check interaction
                    # For UI interaction, we ALWAYS use pinch, because it's the standard click
                    ui_clicked = self.process_ui(frame, cx, cy, is_pinching_now)
                    ui_drawn = True

                    if ui_clicked:
                        # Feedback
                        cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
                        self.is_drawing = False 
                        self.release_start_time = 0
                    else:
                        # Debounce Logic for Drawing State
                        
                        # Determine "Active" state based on Selected Trigger
                        is_active_gesture = (self.current_gesture == self.active_trigger_gesture)
                        
                        # Apply Hysteresis / Debounce
                        if not self.is_drawing:
                            if is_active_gesture:
                                self.is_drawing = True
                                self.release_start_time = 0
                        else:
                            # Currently Drawing
                            should_release = not is_active_gesture

                            if should_release:
                                # Candidate for release
                                if self.release_start_time == 0:
                                    self.release_start_time = time.time()
                                elif (time.time() - self.release_start_time) > PINCH_DEBOUNCE_S:
                                    self.is_drawing = False
                                    self.release_start_time = 0
                            else:
                                self.release_start_time = 0
                        
                        draw_color = self.brush_color
                        if self.eraser_mode: draw_color = (0, 0, 0)
                        
                        if self.is_drawing:
                            if self.tool_mode == 'free':
                                if self.prev_x == 0 and self.prev_y == 0:
                                    self.prev_x, self.prev_y = cx, cy
                                
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), draw_color, self.brush_size)
                                self.prev_x, self.prev_y = cx, cy
                                
                                # Active Cursor (Filled Circle)
                                cursor_color = draw_color if not self.eraser_mode else (200, 200, 200)
                                cv2.circle(frame, (cx, cy), 5, cursor_color, -1)
                                cv2.putText(frame, "Drawing", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            else: # Shape Mode: Preview
                                if self.shape_start is None:
                                    self.shape_start = (cx, cy)
                                
                                sx, sy = self.shape_start
                                if self.tool_mode == 'line':
                                    cv2.line(frame, (sx, sy), (cx, cy), draw_color, self.brush_size)
                                elif self.tool_mode == 'rect':
                                    cv2.rectangle(frame, (sx, sy), (cx, cy), draw_color, self.brush_size)
                                elif self.tool_mode == 'circle':
                                    radius = int(np.hypot(cx - sx, cy - sy))
                                    cv2.circle(frame, (sx, sy), radius, draw_color, self.brush_size)
                                
                                cv2.putText(frame, "Release to Draw", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        else: # Not Pinching
                            self.prev_x, self.prev_y = 0, 0
                            
                            # Shape Commit
                            if self.shape_start is not None and self.tool_mode != 'free':
                                sx, sy = self.shape_start
                                if self.tool_mode == 'line':
                                    cv2.line(self.canvas, (sx, sy), (cx, cy), draw_color, self.brush_size)
                                elif self.tool_mode == 'rect':
                                    cv2.rectangle(self.canvas, (sx, sy), (cx, cy), draw_color, self.brush_size)
                                elif self.tool_mode == 'circle':
                                    radius = int(np.hypot(cx - sx, cy - sy))
                                    cv2.circle(self.canvas, (sx, sy), radius, draw_color, self.brush_size)
                                self.shape_start = None

                            # Hover Cursor (Crosshair for precision)
                            cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 1)
                            cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 1)
                            cv2.putText(frame, f"Hover ({self.tool_mode})", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif handedness == "Right": # CONTROL HAND (User's Left Hand)
                    # Clear Gesture: Thumb and Index FAR apart (> 150)
                    dist_l = np.hypot(ix - tx, iy - ty)
                    
                    if dist_l > 150: # Open Palmish
                        cv2.putText(frame, "Cleared!", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.canvas = np.zeros((h, w, 3), np.uint8)
                    else:
                        cv2.putText(frame, "Control Hand", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if not ui_drawn:
             self.process_ui(frame, -100, -100, False)

        # Gestures Info (Always show active trigger)
        cv2.putText(frame, f"Trigger: {self.active_trigger_gesture}", (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Merge Canvas
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv_mask)
        frame = cv2.bitwise_or(frame, self.canvas)

        return frame

# --- Main Execution ---
painter = AirPainter()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Air Writing Project", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Air Writing Project", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = painter.process(img)
    cv2.imshow("Air Writing Project", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): painter.canvas = None

cap.release()
cv2.destroyAllWindows()