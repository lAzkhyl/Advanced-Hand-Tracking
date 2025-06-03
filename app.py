import argparse
import cv2 as cv
import mediapipe as mp
import math
import time

class OneEuroFilter:
    """Smoothing filter for noisy real-time signals"""
    def __init__(self, freq=30, mincutoff=1.0, beta=0.1, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def lowpass_filter(self, x, x_prev, alpha):
        """Basic low-pass filter implementation"""
        return alpha * x + (1 - alpha) * x_prev
        
    def __call__(self, x, t):
        """Apply filter to new value with timestamp"""
        if self.x_prev is None:  # Initialize on first call
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
            
        dt = t - self.t_prev
        if dt <= 0:  # Skip invalid timestamps
            return self.x_prev
            
        # Derivative smoothing
        alpha_d = math.exp(-dt * 2 * math.pi * self.dcutoff)
        dx = (x - self.x_prev) / dt
        dx_hat = self.lowpass_filter(dx, self.dx_prev, alpha_d)
        
        # Adaptive cutoff based on derivative
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = math.exp(-dt * 2 * math.pi * cutoff)
        x_hat = self.lowpass_filter(x, self.x_prev, alpha)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def draw_landmarks(image, landmark_point, scale_factor=1.0):
    """Draw hand landmarks with dynamic scaling"""
    if not landmark_point:
        return image
    
    # Predefined hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Calculate dynamic sizes based on scale factor
    line_thickness_black = max(1, int(3 * scale_factor))
    line_thickness_white = max(1, int(1 * scale_factor))
    circle_radius_fingertip = max(1, int(6 * scale_factor))
    circle_radius_other = max(1, int(4 * scale_factor))
    circle_border_thickness = max(1, int(1 * scale_factor))
    
    # Draw connections
    for start_idx, end_idx in connections:
        start = tuple(landmark_point[start_idx])
        end = tuple(landmark_point[end_idx])
        cv.line(image, start, end, (0, 0, 0), line_thickness_black)
        cv.line(image, start, end, (255, 255, 255), line_thickness_white)
    
    # Draw landmarks
    for i, point in enumerate(landmark_point):
        # Select appropriate radius
        if i in [4, 8, 12, 16, 20]:  # Fingertips
            radius = circle_radius_fingertip
        else:
            radius = circle_radius_other
            
        # Draw black filled circle
        cv.circle(image, tuple(point), radius, (0, 0, 0), -1)
        # Draw white border
        cv.circle(image, tuple(point), radius, (255, 255, 255), circle_border_thickness)
    
    return image

def main():
    """Main hand tracking application"""
    args = get_args()
    
    # Video capture setup
    cap = cv.VideoCapture(args.device)
    if not cap.isOpened():
        return
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    # MediaPipe hands initialization
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    # Initialize filters for each landmark (x and y separately)
    filters = [{'x': OneEuroFilter(), 'y': OneEuroFilter()} for _ in range(21)]
    
    # Tracking state variables
    last_valid_landmarks = None
    no_detection_count = 0
    MAX_NO_DETECTION = 10
    DEAD_ZONE_THRESHOLD = 2.0  # Minimum movement threshold (pixels)
    
    # Scaling reference variables
    ref_width = None  # Reference palm width
    scale_factor = 1.0  # Current scale factor

    while True:
        if cv.waitKey(1) == 27:  # ESC to exit
            break
        
        # Read and flip frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        debug_image = frame.copy()  # More efficient than deepcopy for numpy arrays
        h, w = debug_image.shape[:2]
        
        # Process frame with MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            no_detection_count = 0
            
            # Process first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmarks = []
            
            # Extract and normalize landmarks
            for landmark in hand_landmarks.landmark:
                current_landmarks.append([
                    min(int(landmark.x * w), w - 1),
                    min(int(landmark.y * h), h - 1)
                ])
            
            # Apply smoothing filters
            timestamp = time.time()
            filtered_landmarks = []
            for i, (x, y) in enumerate(current_landmarks):
                fx = filters[i]['x'](x / w, timestamp) * w
                fy = filters[i]['y'](y / h, timestamp) * h
                filtered_landmarks.append([int(fx), int(fy)])
            
            # Calculate palm width (distance between wrist and middle finger base)
            x0, y0 = filtered_landmarks[0]
            x9, y9 = filtered_landmarks[9]
            current_palm_width = math.hypot(x9 - x0, y9 - y0)
            
            # Initialize reference width if not set
            if ref_width is None:
                ref_width = current_palm_width
            
            # Calculate scale factor (clamped between 0.5 and 2.0)
            scale_factor = current_palm_width / ref_width
            scale_factor = max(0.5, min(2.0, scale_factor))
            
            # Dead zone handling
            if last_valid_landmarks:
                total_movement = sum(
                    math.hypot(fx - lx, fy - ly)
                    for (fx, fy), (lx, ly) in zip(filtered_landmarks, last_valid_landmarks)
                )
                avg_movement = total_movement / 21
                
                if avg_movement < DEAD_ZONE_THRESHOLD:
                    filtered_landmarks = last_valid_landmarks
            
            # Update tracking state
            last_valid_landmarks = filtered_landmarks
            debug_image = draw_landmarks(debug_image, filtered_landmarks, scale_factor)
        else:
            # Use last valid landmarks if available
            no_detection_count += 1
            if last_valid_landmarks and no_detection_count <= MAX_NO_DETECTION:
                debug_image = draw_landmarks(debug_image, last_valid_landmarks, scale_factor)
            else:
                last_valid_landmarks = None
                ref_width = None  # Reset reference when hand is lost
        
        cv.imshow('Advanced Hand Tracking', debug_image)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()