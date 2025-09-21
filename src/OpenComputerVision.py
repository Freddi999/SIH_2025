import cv2
import numpy as np
from collections import defaultdict
import argparse
import time

class VehicleCounter:
    def __init__(self, video_path, detection_line_position=0.5, min_area=3000):
        """
        Initialize vehicle counter
        
        Args:
            video_path: Path to video file or 0 for webcam
            detection_line_position: Position of counting line (0.0 to 1.0, relative to frame height)
            min_area: Minimum contour area to consider as vehicle
        """
        self.video_path = video_path
        self.detection_line_position = detection_line_position
        self.min_area = min_area
        
        # Counters
        self.vehicles_in = 0
        self.vehicles_out = 0
        
        # Tracking
        self.vehicle_tracks = {}
        self.next_vehicle_id = 1
        self.max_track_length = 30
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def get_centroid(self, contour):
        """Get centroid of contour"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def track_vehicles(self, centroids, frame_height):
        """Track vehicles and count when they cross the line"""
        detection_line_y = int(frame_height * self.detection_line_position)
        current_tracks = {}
        
        # Update existing tracks
        for vehicle_id, track_data in self.vehicle_tracks.items():
            last_position = track_data['positions'][-1]
            min_distance = float('inf')
            best_match = None
            
            # Find closest centroid to continue track
            for i, centroid in enumerate(centroids):
                if centroid is None:
                    continue
                distance = np.sqrt((centroid[0] - last_position[0])**2 + 
                                 (centroid[1] - last_position[1])**2)
                if distance < min_distance and distance < 100:  # Max movement threshold
                    min_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Continue existing track
                new_position = centroids[best_match]
                track_data['positions'].append(new_position)
                
                # Keep only recent positions
                if len(track_data['positions']) > self.max_track_length:
                    track_data['positions'] = track_data['positions'][-self.max_track_length:]
                
                # Check for line crossing
                self.check_line_crossing(vehicle_id, track_data, detection_line_y)
                
                current_tracks[vehicle_id] = track_data
                centroids[best_match] = None  # Mark as used
        
        # Create new tracks for unmatched centroids
        for centroid in centroids:
            if centroid is not None:
                current_tracks[self.next_vehicle_id] = {
                    'positions': [centroid],
                    'counted': False
                }
                self.next_vehicle_id += 1
        
        self.vehicle_tracks = current_tracks
    
    def check_line_crossing(self, vehicle_id, track_data, detection_line_y):
        """Check if vehicle crossed the detection line"""
        if track_data['counted'] or len(track_data['positions']) < 2:
            return
        
        # Get last two positions
        prev_y = track_data['positions'][-2][1]
        curr_y = track_data['positions'][-1][1]
        
        # Check if crossed line
        if prev_y < detection_line_y and curr_y > detection_line_y:
            # Vehicle going down (IN)
            self.vehicles_in += 1
            track_data['counted'] = True
        elif prev_y > detection_line_y and curr_y < detection_line_y:
            # Vehicle going up (OUT)
            self.vehicles_out += 1
            track_data['counted'] = True
    
    def process_frame(self, frame):
        """Process single frame for vehicle detection"""
        height, width = frame.shape[:2]
        detection_line_y = int(height * self.detection_line_position)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadow pixels have value 127)
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Dilate to fill gaps
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and get centroids
        valid_contours = []
        centroids = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                valid_contours.append(contour)
                centroid = self.get_centroid(contour)
                centroids.append(centroid)
        
        # Track vehicles
        self.track_vehicles(centroids, height)
        
        return fg_mask, valid_contours
    
    def draw_info(self, frame, fg_mask, contours):
        """Draw detection info on frame"""
        height, width = frame.shape[:2]
        detection_line_y = int(height * self.detection_line_position)
        
        # Draw detection line
        cv2.line(frame, (0, detection_line_y), (width, detection_line_y), (0, 255, 255), 3)
        cv2.putText(frame, "DETECTION LINE", (10, detection_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw contours and tracking info
        for vehicle_id, track_data in self.vehicle_tracks.items():
            positions = track_data['positions']
            if len(positions) > 0:
                # Draw current position
                current_pos = positions[-1]
                cv2.circle(frame, current_pos, 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{vehicle_id}", 
                           (current_pos[0] + 10, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw trail
                if len(positions) > 1:
                    points = np.array(positions, np.int32)
                    cv2.polylines(frame, [points], False, (255, 0, 0), 2)
        
        # Draw vehicle contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # Draw counters
        counter_text = f"IN: {self.vehicles_in}  OUT: {self.vehicles_out}  TOTAL: {self.vehicles_in + self.vehicles_out}"
        cv2.rectangle(frame, (10, 10), (500, 60), (0, 0, 0), -1)
        cv2.putText(frame, counter_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add direction indicators
        cv2.putText(frame, "IN (Downward)", (20, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "OUT (Upward)", (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def run(self, output_path=None):
        """Run the vehicle counter"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                fg_mask, contours = self.process_frame(frame)
                
                # Draw information
                output_frame = self.draw_info(frame, fg_mask, contours)
                
                # Write frame if recording
                if writer:
                    writer.write(output_frame)
                
                # Display
                cv2.imshow('Vehicle Counter', output_frame)
                cv2.imshow('Foreground Mask', fg_mask)
                
                # Show processing stats every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames | "
                          f"Processing FPS: {processing_fps:.1f} | "
                          f"IN: {self.vehicles_in} | OUT: {self.vehicles_out}")
                
                # Exit on ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):  # Reset counters
                    self.vehicles_in = 0
                    self.vehicles_out = 0
                    self.vehicle_tracks.clear()
                    print("Counters reset!")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Final stats
            total_time = time.time() - start_time
            print(f"\nFinal Results:")
            print(f"Vehicles IN: {self.vehicles_in}")
            print(f"Vehicles OUT: {self.vehicles_out}")
            print(f"Total vehicles: {self.vehicles_in + self.vehicles_out}")
            print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
            print(f"Average processing FPS: {frame_count / total_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Highway Vehicle Counter')
    parser.add_argument('--input', '-i', default=0, 
                       help='Input video file path (or 0 for webcam)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--line-position', '-l', type=float, default=0.5,
                       help='Detection line position (0.0 to 1.0)')
    parser.add_argument('--min-area', '-a', type=int, default=3000,
                       help='Minimum vehicle area in pixels')
    
    args = parser.parse_args()
    
    # Convert input to int if it's '0' (webcam)
    try:
        video_input = int(args.input)
    except ValueError:
        video_input = args.input
    
    print("Highway Vehicle Counter")
    print("======================")
    print("Controls:")
    print("- ESC: Exit")
    print("- R: Reset counters")
    print("- Close window to exit")
    print()
    
    # Create and run counter
    counter = VehicleCounter(
        video_path=video_input,
        detection_line_position=args.line_position,
        min_area=args.min_area
    )
    
    counter.run(output_path=args.output)


if __name__ == "__main__":
    main()