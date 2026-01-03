import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import threading
import json
from datetime import datetime
import os

class RaceCarTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Race Car Tracker")
        self.root.geometry("1400x800")
        
        # Variables
        self.video_path = None
        self.cap = None
        self.model = None
        self.crossed_data = []
        self.frame_rate = 30
        self.current_frame = 0
        self.car_crossed_frames = {'red': None, 'blue': None}
        self.total_frames = 0
        self.is_seeking = False  # Flag to prevent progress bar updates during seeking
        
        # Car tracking
        self.red_car_id = None
        self.blue_car_id = None
        self.tracked_objects = {}
        
        # Finish line tracking
        self.finish_line_points = []
        self.drawing_line = False
        self.finish_line = None  # (x1, y1, x2, y2)
        self.crossing_data = []
        self.last_positions = {'red': None, 'blue': None}
        self.last_lap_times = {'red': None, 'blue': None}
        self.car_last_side = {'red': None, 'blue': None}  # Track last detected side
        self.car_crossed_flag = {'red': False, 'blue': False}  # Has car crossed in current pass?
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top frame for controls
        control_frame = tk.Frame(self.root, bg='#2c3e50', padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(control_frame, text="Upload Video", command=self.upload_video,
                 bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Load YOLO Model", command=self.load_model,
                 bg='#2ecc71', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Start Tracking", command=self.start_tracking,
                 bg='#9b59b6', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Draw Finish Line", command=self.setup_finish_line,
                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", 
                                    bg='#2c3e50', fg='white', font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for video
        video_frame = tk.Frame(content_frame, bg='black')
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(video_frame, text="Video Display", 
                bg='black', fg='white', font=('Arial', 14, 'bold')).pack()
        
        self.canvas = tk.Canvas(video_frame, bg='black')
        self.canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Progress bar frame
        progress_frame = tk.Frame(video_frame, bg='#2c3e50')
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time label
        self.time_label = tk.Label(progress_frame, text="00:00 / 00:00", 
                                  bg='#2c3e50', fg='white', font=('Arial', 10))
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_scale = tk.Scale(progress_frame, from_=0, to=100, 
                                      orient=tk.HORIZONTAL, bg='#34495e', 
                                      fg='white', highlightthickness=0,
                                      command=self.on_progress_seek)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Right frame for crossings list
        right_frame = tk.Frame(content_frame, bg='#2c3e50', padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        tk.Label(right_frame, text="Race Timings", 
                bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack()
        
        # Scrollable list for crossings
        scrollbar = tk.Scrollbar(right_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.crossings_listbox = tk.Listbox(right_frame, bg='#34495e', fg='white',
                                           yscrollcommand=scrollbar.set, width=40,
                                           font=('Arial', 9))
        self.crossings_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.crossings_listbox.yview)
        
        # Export button
        tk.Button(right_frame, text="Export Timings", command=self.export_timings,
                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                 padx=10, pady=5).pack(pady=5)
        
    def on_canvas_click(self, event):
        """Handle canvas click for drawing finish line"""
        if self.drawing_line:
            self.finish_line_points.append((event.x, event.y))
            
            if len(self.finish_line_points) == 1:
                self.status_label.config(text="Click second point to complete the line")
            elif len(self.finish_line_points) == 2:
                self.finish_line = (self.finish_line_points[0][0], self.finish_line_points[0][1],
                                   self.finish_line_points[1][0], self.finish_line_points[1][1])
                self.drawing_line = False
                self.finish_line_points = []
                self.status_label.config(text="Finish line set. Ready to track.")
                self.display_frame()
    
    def setup_finish_line(self):
        """Setup mode for drawing finish line"""
        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="Please load a video first")
            return
        
        self.drawing_line = True
        self.finish_line_points = []
        self.status_label.config(text="Click first point on canvas to draw finish line")
    
    def point_to_line_side(self, point, line):
        """Determine which side of the line a point is on"""
        x, y = point
        x1, y1, x2, y2 = line
        
        # Line equation: (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1 = 0
        # Result > 0 = one side, Result < 0 = other side
        result = (y2 - y1) * x - (x2 - x1) * y + (x2 - x1) * y1 - (y2 - y1) * x1
        return 1 if result > 0 else -1 if result < 0 else 0
    
    def bbox_crosses_line(self, bbox, line, prev_bbox=None):
        """Check if bounding box crosses the finish line"""
        if line is None or prev_bbox is None:
            return False
        
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = prev_bbox
        
        # Get the center point and corners of current and previous bbox
        curr_centers = [
            ((x1 + x2) / 2, (y1 + y2) / 2),  # center
            (x1, y1), (x2, y1), (x1, y2), (x2, y2)  # corners
        ]
        
        prev_centers = [
            ((px1 + px2) / 2, (py1 + py2) / 2),
            (px1, py1), (px2, py1), (px1, py2), (px2, py2)
        ]
        
        # Check if any point crossed the line
        for curr_pt, prev_pt in zip(curr_centers, prev_centers):
            curr_side = self.point_to_line_side(curr_pt, line)
            prev_side = self.point_to_line_side(prev_pt, line)
            
            # If sides are different (not 0), then a crossing occurred
            if curr_side != 0 and prev_side != 0 and curr_side != prev_side:
                return True
        
        return False
    
    def get_bbox_line_side(self, bbox, line):
        """Determine which side of the line a bounding box center is on"""
        if line is None or bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self.point_to_line_side(center, line)
    
    def record_crossing(self, car_color, frame_num, current_time):
        """Record a crossing event"""
        lap_time = None
        
        if self.last_lap_times[car_color] is not None:
            lap_time = current_time - self.last_lap_times[car_color]
        
        self.last_lap_times[car_color] = current_time
        
        crossing_entry = {
            'car_color': car_color,
            'frame': frame_num,
            'time': f"{int(current_time // 60):02d}:{int(current_time % 60):02d}",
            'timestamp': current_time,
            'lap_time': f"{lap_time:.2f}s" if lap_time else "First crossing"
        }
        
        self.crossing_data.append(crossing_entry)
        
        # Update listbox
        lap_str = f" (Lap: {crossing_entry['lap_time']})" if lap_time else ""
        self.crossings_listbox.insert(tk.END, f"{car_color.upper()} - {crossing_entry['time']}{lap_str}")
    
    def export_timings(self):
        """Export crossing data to JSON"""
        if not self.crossing_data:
            messagebox.showwarning("No Data", "No crossing data to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"race_timings_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.crossing_data, f, indent=2)
        
        messagebox.showinfo("Success", f"Timings exported to {filename}")
        self.status_label.config(text=f"Exported to {filename}")
        
    def draw_finish_line_on_frame(self, frame):
        """Draw the finish line on the frame"""
        if self.finish_line:
            x1, y1, x2, y2 = self.finish_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "FINISH LINE", 
                       (int((x1 + x2) / 2) - 50, int((y1 + y2) / 2) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
        
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            
            # Get and set video dimensions - maintain original quality
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.canvas.config(width=video_width, height=video_height)
            
            # Update progress bar range
            self.progress_scale.config(to=self.total_frames if self.total_frames > 0 else 100)
            self.progress_scale.set(0)
            
            self.status_label.config(text=f"Video loaded: {file_path.split('/')[-1]}")
            self.display_frame()
            
    def load_model(self):
        try:
            # Load final trained model
            self.model = YOLO('final.pt')
            self.status_label.config(text="YOLO model loaded successfully")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
            
    def display_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Draw finish line if set
                frame = self.draw_finish_line_on_frame(frame)
                
                # Convert to RGB without resizing - maintain original quality
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
            
    def on_progress_seek(self, value):
        """Handle progress bar seek"""
        if not self.is_seeking and self.cap and self.cap.isOpened():
            frame_num = int(float(value))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            self.display_frame()
            self.update_progress_label()
        
    def update_progress_label(self):
        """Update the time label with current and total time"""
        if self.frame_rate > 0:
            current_time = self.current_frame / self.frame_rate
            total_time = self.total_frames / self.frame_rate
            
            curr_min = int(current_time // 60)
            curr_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.time_label.config(text=f"{curr_min:02d}:{curr_sec:02d} / {total_min:02d}:{total_sec:02d}")
        
    def identify_car_color(self, frame, bbox):
        """Identify if car is red or blue based on dominant color"""
        x1, y1, x2, y2 = map(int, bbox)
        car_region = frame[y1:y2, x1:x2]
        
        if car_region.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
        
        # Red color range (in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        red_pixels = np.sum(red_mask > 0)
        blue_pixels = np.sum(blue_mask > 0)
        
        # Lowered threshold from 100 to 30 to catch more detections
        if red_pixels > blue_pixels and red_pixels > 30:
            return 'red'
        elif blue_pixels > red_pixels and blue_pixels > 30:
            return 'blue'
        return None
    
    def start_tracking(self):
        if not self.video_path or not self.model:
            self.status_label.config(text="Please load video and model first")
            return
        
        if not self.finish_line:
            self.status_label.config(text="Please draw a finish line first")
            return
        
        # Reset crossing data
        self.crossing_data = []
        self.last_positions = {'red': None, 'blue': None}
        self.last_lap_times = {'red': None, 'blue': None}
        self.car_last_side = {'red': None, 'blue': None}
        self.car_crossed_flag = {'red': False, 'blue': False}
        self.crossings_listbox.delete(0, tk.END)
        
        threading.Thread(target=self.track_video, daemon=True).start()
    
    def detect_at_multiple_scales(self, frame):
        """Run YOLO detection at multiple image scales to catch small objects"""
        all_boxes = []
        
        # Original scale - detect only red_rc_car (0) and blue_rc_car (1)
        results = self.model(frame, classes=[0, 1], conf=0.7)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                all_boxes.append((box.xyxy[0].cpu().numpy(), box.conf[0].cpu().numpy(), class_id))
        
        # Upscaled (2x) - helps detect small distant cars
        frame_2x = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        results_2x = self.model(frame_2x, classes=[0, 1], conf=0.5)
        for result in results_2x:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                # Scale back to original size
                bbox = bbox / 2
                class_id = int(box.cls[0].cpu().numpy())
                all_boxes.append((bbox, box.conf[0].cpu().numpy(), class_id))
        
        return all_boxes
    
    def remove_duplicate_boxes(self, boxes, iou_threshold=0.3):
        """Remove duplicate boxes detected at different scales using IOU"""
        if not boxes:
            return []
        
        # Sort by confidence score descending
        boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
        keep = []
        
        for i, (bbox1, conf1, class_id1) in enumerate(boxes):
            is_duplicate = False
            for bbox2, conf2, class_id2 in keep:
                iou = self.calculate_iou(bbox1, bbox2)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append((bbox1, conf1, class_id1))
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
        
    def track_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        prev_positions = {}
        vehicle_id_counter = 0
        detection_count = 0
        
        # Track which vehicles have crossed (to avoid double-counting)
        vehicles_crossed = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / self.frame_rate
            
            # Use original frame without resizing - maintain quality
            display_frame = frame.copy()
            
            # Draw finish line on display
            display_frame = self.draw_finish_line_on_frame(display_frame)
            
            # Run YOLO detection at multiple scales to catch small distant cars
            all_boxes = self.detect_at_multiple_scales(frame)
            
            # Remove duplicate detections (same object detected at different scales)
            all_boxes = self.remove_duplicate_boxes(all_boxes)
            
            # Process detections and match with previous frames
            current_detections = {}
            display_count = 0
            
            for bbox, conf, class_id in all_boxes:
                if conf > 0.3:  # Lower threshold since we have multi-scale detection
                    display_count += 1
                    # Use original bbox coordinates - no scaling needed
                    scaled_bbox = bbox
                    
                    # Determine car type and color
                    car_type = "RED CAR" if class_id == 0 else "BLUE CAR"
                    car_color = 'red' if class_id == 0 else 'blue'
                    box_color = (0, 0, 255) if class_id == 0 else (255, 0, 0)  # Red BGR or Blue BGR
                    
                    # Draw bounding box for all detections
                    cv2.rectangle(display_frame, 
                                (int(scaled_bbox[0]), int(scaled_bbox[1])),
                                (int(scaled_bbox[2]), int(scaled_bbox[3])),
                                box_color, 2)
                    cv2.putText(display_frame, f"{car_type} ({conf:.2f})",
                              (int(scaled_bbox[0]), int(scaled_bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    # Find matching vehicle from previous frame (closest match)
                    best_match_id = None
                    best_distance = 100  # Increased from 50 to allow more flexible matching
                    
                    current_center = (
                        (scaled_bbox[0] + scaled_bbox[2]) / 2,
                        (scaled_bbox[1] + scaled_bbox[3]) / 2
                    )
                    
                    for prev_id, prev_bbox in prev_positions.items():
                        prev_center = (
                            (prev_bbox[0] + prev_bbox[2]) / 2,
                            (prev_bbox[1] + prev_bbox[3]) / 2
                        )
                        distance = ((current_center[0] - prev_center[0])**2 + 
                                  (current_center[1] - prev_center[1])**2)**0.5
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match_id = prev_id
                    
                    # Assign ID (either matching previous or new)
                    if best_match_id is not None:
                        vehicle_id = best_match_id
                    else:
                        vehicle_id = f"vehicle_{vehicle_id_counter}_{car_color}"
                        vehicle_id_counter += 1
                    
                    # Check for finish line crossing
                    if best_match_id is not None and best_match_id in prev_positions:
                        prev_bbox = prev_positions[best_match_id]
                        if self.bbox_crosses_line(scaled_bbox, self.finish_line, prev_bbox):
                            # Record crossing only once per vehicle per crossing event
                            crossing_id = f"{vehicle_id}_{frame_count}"
                            if crossing_id not in vehicles_crossed:
                                self.record_crossing(car_color, frame_count, current_time)
                                vehicles_crossed.add(crossing_id)
                    
                    # Always add to current detections to continue tracking
                    current_detections[vehicle_id] = scaled_bbox
                    
                    # Check for finish line crossing based on car color
                    curr_side = self.get_bbox_line_side(scaled_bbox, self.finish_line)
                    
                    # Only process if car center is clearly on one side (not on the line)
                    if curr_side != 0:
                        # First time seeing this car color
                        if self.car_last_side[car_color] is None:
                            self.car_last_side[car_color] = curr_side
                        else:
                            prev_side = self.car_last_side[car_color]
                            
                            # If car switched sides
                            if curr_side != prev_side:
                                # Only record crossing if we haven't already crossed in this pass
                                if not self.car_crossed_flag[car_color]:
                                    self.record_crossing(car_color, frame_count, current_time)
                                    self.car_crossed_flag[car_color] = True
                                
                                # Update side
                                self.car_last_side[car_color] = curr_side
                            else:
                                # Same side as before - if we had crossed, reset the flag
                                # to allow another crossing in the opposite direction
                                if self.car_crossed_flag[car_color]:
                                    self.car_crossed_flag[car_color] = False
            
            # Update previous positions for next frame
            prev_positions = current_detections
            
            # Display frame
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk
            
            # Update progress bar
            self.current_frame = frame_count
            self.root.after(0, self.update_progress_bar_display, frame_count, display_count)
            self.root.update()
            
        cap.release()
        self.status_label.config(text="Tracking complete")
        
    def update_progress_bar_display(self, frame_count, display_count):
        """Update progress bar and status during playback"""
        self.is_seeking = True
        self.progress_scale.set(frame_count)
        self.is_seeking = False
        self.update_progress_label()
        self.status_label.config(text=f"Tracking... Frame: {frame_count} | Detections: {display_count}")
        
        



if __name__ == "__main__":
    root = tk.Tk()
    app = RaceCarTracker(root)
    root.mainloop()