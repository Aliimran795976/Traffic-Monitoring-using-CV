import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import json
import matplotlib.pyplot as plt
import numpy as np


model = YOLO('yolov8n.pt')

class TrafficMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Monitor")
        self.root.configure(bg="#1c1d1f")
        self.root.geometry("800x600")  # Set window size
        self.root.resizable(True, True)  # Allow resizing of the window

        # Store current frame and analysis data
        self.current_frame = None
        self.density_data = []
        self.vehicle_count_data = []
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        
        # Heatmap data
        self.heatmap_enabled = False
        self.heatmap_data = None
        self.heatmap_alpha = 0.5
        
        # Video writer
        self.out = None
        self.output_filename = None
        
        # Analysis status
        self.is_analyzing = False
        self.is_replaying = False

        # Color scheme
        self.bg_color = '#1c1d1f'
        self.button_color = '#242930'
        self.button_hover_color = '#2f3d4f'
        self.text_color = 'white'
        self.highlight_color = '#2f3d4f'  


        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.main_frame, bg='black', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create frame for buttons
        self.button_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Status bar (above buttons)
        self.status_frame = tk.Frame(self.button_frame, bg=self.button_color, height=20)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(5, 5))
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(self.status_frame, text="Ready", bg=self.button_color, fg=self.text_color, font=('Helvetica', 9))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Create a sub-frame for buttons with proper background
        button_container = tk.Frame(self.button_frame, bg=self.bg_color)
        button_container.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Upload video button
        self.upload_button = tk.Button(button_container, text="Upload Video", command=self.upload_video,
                                     bg=self.button_color, fg=self.text_color, relief='flat',
                                     bd=0, highlightthickness=0, font=('Helvetica', 12))
        self.upload_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Start Analysis button
        self.start_button = tk.Button(button_container, text="Start Analysis", command=self.start_analysis,
                                    bg=self.button_color, fg=self.text_color, relief='flat',
                                    bd=0, highlightthickness=0, font=('Helvetica', 12), state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Replay Output button
        self.replay_button = tk.Button(button_container, text="Replay Output", command=self.replay_output,
                                   bg=self.button_color, fg=self.text_color, relief='flat',
                                   bd=0, highlightthickness=0, font=('Helvetica', 12), state=tk.DISABLED)
        self.replay_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Use Camera button
        self.camera_button = tk.Button(button_container, text="Use Camera", command=self.use_camera,
                                     bg=self.button_color, fg=self.text_color, relief='flat',
                                     bd=0, highlightthickness=0, font=('Helvetica', 12))
        self.camera_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Heatmap Toggle button
        self.heatmap_button = tk.Button(button_container, text="Enable Heatmap", command=self.toggle_heatmap,
                                    bg=self.button_color, fg=self.text_color, relief='flat',
                                    bd=0, highlightthickness=0, font=('Helvetica', 12))
        self.heatmap_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Create metrics frame with proper background
        metrics_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)

        # Traffic metrics display
        self.density_label = tk.Label(metrics_frame, text="Density: N/A", bg=self.bg_color, fg=self.text_color, font=('Helvetica', 12))
        self.density_label.pack(side=tk.LEFT, padx=10)
        self.density_label.pack_forget()

        self.congestion_label = tk.Label(metrics_frame, text="Congestion: N/A", bg=self.bg_color, fg=self.text_color, font=('Helvetica', 12))
        self.congestion_label.pack(side=tk.LEFT, padx=10)
        self.congestion_label.pack_forget()

        # Bind hover events for all buttons
        for button in [self.upload_button, self.start_button, self.replay_button, self.camera_button, self.heatmap_button]:
            button.bind("<Enter>", lambda e, btn=button: self.on_hover(btn))
            button.bind("<Leave>", lambda e, btn=button: self.on_leave(btn))

        # Instructions for video upload (shown within canvas when no video loaded)
        self.instruction_label = tk.Label(self.canvas, text="Please upload a video to analyze.", bg=self.bg_color, fg=self.text_color, font=('Helvetica', 12))
        self.instruction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Video file
        self.video_path = None
        self.cap = None
        self.processing = False

        # Button hover effect
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

        # Check camera availability at startup
        self.check_camera()

    def on_hover(self, button):
        button.config(bg=self.highlight_color)

    def on_leave(self, button):
        button.config(bg=self.button_color)

    def upload_video(self):
        """Handle video upload"""
        # Reset state if there was a previous analysis
        if self.cap is not None or self.out is not None:
            self.reset_analysis_state()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file!")
                self.reset_analysis_state()
                return
            
            # Update UI
            self.instruction_label.config(text="Video loaded. Click 'Start Analysis' to begin.")
            self.start_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")

    def use_camera(self):
        self.cap = cv2.VideoCapture(0)  # 0 for inbuilt camera
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the camera.")
            return

        self.processing = True
        self.process_video()

    def check_camera(self):
        """Check if camera is available and update button state"""
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
        
        if not camera_available:
            self.camera_button.config(state=tk.DISABLED)
            self.status_label.config(text="No camera detected")
        else:
            self.camera_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready")

    def display_frame(self, frame):
        # Store the current frame for resize events
        self.current_frame = frame.copy()
        
        # Get current canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not properly initialized yet
            canvas_width = 640
            canvas_height = 480

        # Convert frame to image format for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        
        # Calculate aspect ratio
        img_ratio = frame_image.size[0] / frame_image.size[1]
        canvas_ratio = canvas_width / canvas_height

        if canvas_ratio > img_ratio:
            # Canvas is wider than image
            new_height = canvas_height
            new_width = int(new_height * img_ratio)
        else:
            # Canvas is taller than image
            new_width = canvas_width
            new_height = int(new_width / img_ratio)

        # Resize image to fit canvas while maintaining aspect ratio
        frame_image = frame_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=frame_image)  # Keep reference to avoid garbage collection

        # Clear previous image and create new one centered
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.photo)

    def start_analysis(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded!")
            return
            
        # Setup video writer
        input_filename = os.path.basename(self.video_path)
        self.output_filename = f"output_{input_filename}"
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_filename, fourcc, fps, (width, height))
        
        self.processing = True
        self.is_analyzing = True
        self.is_replaying = False
        
        # Update status
        self.status_label.config(text="Analyzing video...")
        
        # Remove the instruction label and start processing
        self.instruction_label.place_forget()
        self.process_video()

        # Disable start button and enable replay button
        self.start_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.camera_button.config(state=tk.DISABLED)

        # Show the density and congestion labels
        self.density_label.pack()
        self.congestion_label.pack()

    def stop_analysis(self):
        self.processing = False
        self.is_analyzing = False
        
        # Generate and save graph before releasing video writer
        if len(self.density_data) > 0:
            self.generate_graph()
        
        # Release video writer and save the video
        if self.out is not None:
            self.out.release()
            self.status_label.config(text=f"Analysis complete. Saved as {self.output_filename}")
            self.out = None
            # Enable replay button
            self.replay_button.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Analysis stopped")
        
        # Enable buttons
        self.start_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.camera_button.config(state=tk.NORMAL)

        # Hide the density and congestion labels
        self.density_label.pack_forget()
        self.congestion_label.pack_forget()

    def replay_output(self):
        if not self.output_filename or not os.path.exists(self.output_filename):
            messagebox.showerror("Error", "No output video available!")
            return
            
        self.is_replaying = True
        self.processing = True
        
        # Open the output video
        self.cap = cv2.VideoCapture(self.output_filename)
        
        # Update status
        self.status_label.config(text="Replaying output video...")
        
        # Disable buttons during replay
        self.replay_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.camera_button.config(state=tk.DISABLED)
        
        # Start replay
        self.replay_video()

    def replay_video(self):
        while self.processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Display frame
            self.display_frame(frame)
            
            # Control playback speed
            cv2.waitKey(25)
            
            # Update the window
            self.root.update_idletasks()
            self.root.update()
            
        # When replay ends
        self.cap.release()
        self.processing = False
        self.is_replaying = False
        
        # Re-enable buttons
        self.replay_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.camera_button.config(state=tk.NORMAL)
        
        # Update status
        self.status_label.config(text="Ready")

    def process_video(self):
        while self.processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Initialize heatmap data if needed
            if self.heatmap_enabled and self.heatmap_data is None:
                self.heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)
            
            # Run YOLO object detection on original frame
            results = model(frame)
            detections = results[0].boxes.data

            # Update heatmap with new detections
            self.update_heatmap(detections, frame_width, frame_height)

            # Apply heatmap overlay if enabled
            if self.heatmap_enabled:
                frame = self.apply_heatmap_overlay(frame.copy())

            # Update vehicle type counts from detections
            for *xyxy, conf, cls in detections:
                label = model.names[int(cls)]
                if label in self.vehicle_types:
                    self.vehicle_types[label] += 1

            # Process detections and update metrics
            density, congestion, vehicle_counts, light_status = self.estimate_traffic_conditions(detections, frame_width, frame_height)
            
            # Store data for graphing
            total_vehicles = sum(vehicle_counts.values())
            self.density_data.append(density)
            self.vehicle_count_data.append(total_vehicles)

            # Update GUI with traffic data
            self.density_label.config(text=f"Density: {density:.4f}")
            self.congestion_label.config(text=f"Congestion: {congestion}")

            # Draw bounding boxes and labels on display frame
            scale_x = frame.shape[1] / frame_width
            scale_y = frame.shape[0] / frame_height
            
            for *xyxy, conf, cls in detections:
                label = model.names[int(cls)]
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    # Scale coordinates to display size
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)
                    
                    # Ensure coordinates stay within frame bounds
                    x1_scaled = max(0, min(x1_scaled, frame.shape[1] - 1))
                    y1_scaled = max(0, min(y1_scaled, frame.shape[0] - 1))
                    x2_scaled = max(0, min(x2_scaled, frame.shape[1] - 1))
                    y2_scaled = max(0, min(y2_scaled, frame.shape[0] - 1))
                    
                    cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
                    label_text = f'{label} {conf:.2f}'
                    # Ensure label stays within frame
                    label_y = max(30, y1_scaled - 10)
                    cv2.putText(frame, label_text, (x1_scaled, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Overlay metrics on display frame
            self.overlay_metrics(frame, density, congestion, vehicle_counts)

            # Save original frame with overlays to output video
            if self.out is not None:
                save_frame = frame.copy()
                self.overlay_metrics(save_frame, density, congestion, vehicle_counts)
                for *xyxy, conf, cls in detections:
                    label = model.names[int(cls)]
                    if label in ['car', 'truck', 'bus', 'motorcycle']:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f'{label} {conf:.2f}'
                        cv2.putText(save_frame, label_text, (x1, max(30, y1 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.out.write(save_frame)

            # Convert frame to image format for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_image)
            
            # Calculate centering position
            x_pos = (self.canvas.winfo_width() - frame.shape[1]) // 2
            y_pos = (self.canvas.winfo_height() - frame.shape[0]) // 2
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=frame_tk)
            self.canvas.image = frame_tk

            self.root.update_idletasks()
            self.root.update()

        if not self.is_replaying:
            self.stop_analysis()

    def estimate_traffic_conditions(self, detections, frame_width, frame_height):
        """Estimate traffic conditions based on detected vehicles"""
        vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
        vehicle_count = 0
        vehicle_type_counts = {v_type: 0 for v_type in vehicle_types}
        
        # Calculate total area of detected vehicles
        total_vehicle_area = 0
        frame_area = frame_width * frame_height
        
        for *xyxy, conf, cls in detections:
            label = model.names[int(cls)]
            if label in vehicle_types and conf > 0.3:  # Only count confident detections
                vehicle_count += 1
                vehicle_type_counts[label] += 1
                
                # Calculate area of vehicle bounding box
                x1, y1, x2, y2 = map(int, xyxy)
                vehicle_area = (x2 - x1) * (y2 - y1)
                total_vehicle_area += vehicle_area

        # Calculate density as ratio of vehicle area to frame area
        density = (total_vehicle_area / frame_area) if frame_area > 0 else 0
        
        # Determine congestion level based on density and vehicle count
        if density > 0.15 or vehicle_count > 8:  # Adjust thresholds based on your needs
            congestion_level = "High"
        elif density > 0.08 or vehicle_count > 4:
            congestion_level = "Medium"
        else:
            congestion_level = "Low"

        return density, congestion_level, vehicle_type_counts, "Green"

    def overlay_metrics(self, frame, density, congestion, vehicle_counts):
        """Overlay traffic metrics on the frame"""
        # Add semi-transparent overlay background
        overlay = frame.copy()
        # Make overlay background wider
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Increased opacity
        
        # Display metrics with better formatting and wider spacing
        cv2.putText(frame, f"Density: {density:.4f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Congestion: {congestion}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display vehicle counts with more space
        y_pos = 100
        count_text = "Vehicles: " + ", ".join([f"{k}: {v}" for k, v in vehicle_counts.items() if v > 0])
        cv2.putText(frame, count_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def toggle_heatmap(self):
        """Toggle heatmap overlay on/off"""
        self.heatmap_enabled = not self.heatmap_enabled
        self.heatmap_button.config(text="Disable Heatmap" if self.heatmap_enabled else "Enable Heatmap")
        
        # Initialize heatmap data if it's being enabled
        if self.heatmap_enabled and self.heatmap_data is None and self.cap is not None:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.heatmap_data = np.zeros((frame_height, frame_width), dtype=np.float32)

    def update_heatmap(self, detections, frame_width, frame_height):
        """Update heatmap data with new detections"""
        if not self.heatmap_enabled or self.heatmap_data is None:
            return

        # Create temporary heatmap for new detections
        temp_heatmap = np.zeros_like(self.heatmap_data)
        
        for *xyxy, conf, cls in detections:
            label = model.names[int(cls)]
            if label in self.vehicle_types:
                x1, y1, x2, y2 = map(int, xyxy)
                # Create a Gaussian kernel for smoother heatmap
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                sigma = max(x2 - x1, y2 - y1) // 4  # Adjust sigma based on detection size
                
                # Create Gaussian kernel
                x = np.arange(0, frame_width, 1, float)
                y = np.arange(0, frame_height, 1, float)[:, np.newaxis]
                x0 = center_x
                y0 = center_y
                gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
                
                # Add to temporary heatmap
                temp_heatmap += gaussian

        # Update main heatmap with decay
        decay_factor = 0.95  # Adjust this value to control how quickly old detections fade
        self.heatmap_data = self.heatmap_data * decay_factor + temp_heatmap

    def apply_heatmap_overlay(self, frame):
        """Apply heatmap overlay to the frame"""
        if not self.heatmap_enabled or self.heatmap_data is None:
            return frame

        # Normalize heatmap data
        heatmap = cv2.normalize(self.heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        
        # Apply color map
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize heatmap if frame dimensions don't match
        if heatmap_colored.shape[:2] != frame.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
        
        # Blend heatmap with original frame
        overlay = cv2.addWeighted(frame, 1, heatmap_colored, self.heatmap_alpha, 0)
        return overlay

    def generate_graph(self):
        """Generate and save graphs showing density vs vehicle count relationship and vehicle type distribution"""
        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(20, 8))
        
        # Density vs Vehicle Count plot
        ax1 = fig.add_subplot(121)
        ax1.scatter(self.vehicle_count_data, self.density_data, alpha=0.5)
        ax1.plot(self.vehicle_count_data, np.poly1d(np.polyfit(self.vehicle_count_data, self.density_data, 1))(self.vehicle_count_data), 
                color='red', linestyle='--', label='Trend Line')
        
        ax1.set_xlabel('Vehicle Count')
        ax1.set_ylabel('Density')
        ax1.set_title('Traffic Density vs Vehicle Count')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Vehicle Types Pie Chart
        ax2 = fig.add_subplot(122)
        # Filter out vehicle types with zero count
        vehicle_data = {k: v for k, v in self.vehicle_types.items() if v > 0}
        if vehicle_data:  # Only create pie chart if we have data
            labels = list(vehicle_data.keys())
            sizes = list(vehicle_data.values())
            # Custom colors for each vehicle type
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            ax2.set_title('Distribution of Vehicle Types')
        
        plt.tight_layout()
        
        # Save graphs
        graph_filename = os.path.splitext(self.output_filename)[0] + '_analysis.png'
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Update status to include graph filename
        self.status_label.config(text=f"Analysis complete. Video saved as {self.output_filename}, analysis graphs saved as {os.path.basename(graph_filename)}")

    def on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            # Update canvas size
            self.canvas.config(width=event.width - 10, height=event.height - 100)
            self.root.update_idletasks()

    def reset_analysis_state(self):
        """Reset all analysis-related state variables for a new video"""
        # Reset data storage
        self.density_data = []
        self.vehicle_count_data = []
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        
        # Reset heatmap
        self.heatmap_data = None
        self.heatmap_enabled = False
        self.heatmap_button.config(text="Enable Heatmap")
        
        # Reset video related variables
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        if self.out is not None:
            self.out.release()
        self.out = None
        self.output_filename = None
        
        # Reset processing flags
        self.processing = False
        self.is_analyzing = False
        self.is_replaying = False
        
        # Reset display
        self.canvas.delete("all")
        self.instruction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.instruction_label.config(text="Please upload a video to analyze.")
        
        # Reset labels
        self.density_label.config(text="Density: N/A")
        self.congestion_label.config(text="Congestion: N/A")
        self.status_label.config(text="Ready")
        
        # Reset button states
        self.upload_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        self.camera_button.config(state=tk.NORMAL)
        self.heatmap_button.config(state=tk.NORMAL)

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficMonitorApp(root)
    root.mainloop()
