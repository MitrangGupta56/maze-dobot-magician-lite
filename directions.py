import cv2
import numpy as np
from collections import deque
import heapq
import matplotlib.pyplot as plt
import json
import os

class MazeSolver:
    def __init__(self, image_path, transform_json_path=None, margin_percent=0.05, extra_padding=30, end_effector_radius=15):
        """Initialize the maze solver with an image path
        
        Args:
            image_path: Path to maze image
            transform_json_path: Path to transform_data.json (optional, will auto-detect if None)
            margin_percent: Percentage of image size to use as margin (default 5%)
            extra_padding: Additional padding in pixels for safety (default 30px)
            end_effector_radius: Radius of the robot's end effector in pixels (default 15px)
        """
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        self.start = None
        self.end = None
        self.path = []
        self.directions = []
        
        # Store end effector size for clearance calculations
        self.end_effector_radius = end_effector_radius
        
        # Auto-detect JSON path if not provided
        if transform_json_path is None:
            base_dir = os.path.dirname(image_path)
            transform_json_path = os.path.join(base_dir, "transform_data.json")
        
        self.transform_json_path = transform_json_path
        self.transform_data = None
        self.maze_bounds = None
        self.use_json_maze_data = False
        
        # Load transform data if available
        if os.path.exists(transform_json_path):
            with open(transform_json_path, 'r') as f:
                self.transform_data = json.load(f)
            print(f"[Init] ✓ Loaded transform data from JSON")
            
            # Check if maze structure data is available
            if 'maze_structure' in self.transform_data and self.transform_data['maze_structure']:
                self.use_json_maze_data = True
                print(f"[Init] ✓ Using maze structure from JSON")
            
            # Load maze bounds if available
            if 'maze_bounds' in self.transform_data and self.transform_data['maze_bounds']:
                self.maze_bounds = self.transform_data['maze_bounds']
                print(f"[Init] ✓ Maze bounds loaded from JSON")
        else:
            print(f"[Init] ⚠ No JSON found at {transform_json_path}, will use fallback detection")
        
        # Dynamic margin based on image size (5% of smaller dimension) + extra padding
        min_dimension = min(self.height, self.width)
        base_margin = int(min_dimension * margin_percent)
        self.margin = base_margin + extra_padding
        
        # Wall thickness estimation
        self.wall_thickness = self.estimate_wall_thickness()
        
        # Path planning parameters
        self.CENTER_BIAS = 40.0  # Strong pull to corridor center
        self.MIN_CLEARANCE = max(end_effector_radius, 12)  # Minimum clearance from walls
        
        # Create binary mask and distance transform
        self.create_corridor_mask()
        
        print(f"[Init] Image size: {self.width}x{self.height}")
        print(f"[Init] Margin: {base_margin}px + {extra_padding}px padding = {self.margin}px total")
        print(f"[Init] End effector radius: {end_effector_radius}px")
        print(f"[Init] Wall thickness: ~{self.wall_thickness}px")
        print(f"[Init] Min clearance: {self.MIN_CLEARANCE}px")
        
    def estimate_wall_thickness(self):
        """Fast wall thickness estimation"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Sample a small region in the center
        cy, cx = self.height // 2, self.width // 2
        sample_size = 100
        y1, y2 = max(0, cy - sample_size), min(self.height, cy + sample_size)
        x1, x2 = max(0, cx - sample_size), min(self.width, cx + sample_size)
        sample = binary[y1:y2, x1:x2]
        
        # Find horizontal wall segments
        horizontal_projection = np.sum(sample, axis=1)
        non_zero = horizontal_projection > 0
        
        if np.any(non_zero):
            wall_runs = []
            in_wall = False
            current_run = 0
            
            for pixel in non_zero:
                if pixel:
                    current_run += 1
                    in_wall = True
                elif in_wall:
                    wall_runs.append(current_run)
                    current_run = 0
                    in_wall = False
            
            if wall_runs:
                thickness = int(np.median(wall_runs))
                return max(5, min(thickness, 30))
        
        return 10  # Default
    
    def create_corridor_mask(self):
        """Create binary mask and distance transform for corridor centering"""
        # If JSON has maze structure, use it
        if self.use_json_maze_data and self.maze_bounds:
            print(f"[Init] Creating mask from JSON maze structure...")
            
            # Create mask from image
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Apply maze bounds to focus on maze area
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            x, y = self.maze_bounds['top_left']
            w, h = self.maze_bounds['width'], self.maze_bounds['height']
            mask[y:y+h, x:x+w] = binary[y:y+h, x:x+w]
            
            self.corridor_mask = mask
            
            print(f"[Init] ✓ Using JSON-defined maze bounds: {w}x{h} at ({x},{y})")
        else:
            # Fallback: analyze entire image
            print(f"[Init] Creating mask from full image analysis...")
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.corridor_mask = binary
        
        # Distance transform: each pixel shows distance to nearest wall
        self.distance_transform = cv2.distanceTransform(self.corridor_mask, cv2.DIST_L2, 5).astype(np.float32)
        
        # Normalize distance transform for A* cost calculation
        dmax = float(self.distance_transform.max()) if float(self.distance_transform.max()) > 0 else 1.0
        self.dist_norm = self.distance_transform / (dmax + 1e-6)
        
        print(f"[Init] Distance transform created for corridor centering")
        
    def apply_inner_boundary(self):
        """Create an inner boundary box with guaranteed padding"""
        # If we have maze bounds from JSON, use those instead
        if self.maze_bounds:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            x, y = self.maze_bounds['top_left']
            w, h = self.maze_bounds['width'], self.maze_bounds['height']
            
            # Add some internal margin
            margin = 10
            cv2.rectangle(mask, 
                         (x + margin, y + margin), 
                         (x + w - margin, y + h - margin), 
                         255, -1)
            
            if len(self.image.shape) == 3:
                self.working_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            else:
                self.working_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            
            print(f"[Boundary] Applied JSON-based boundary")
        else:
            # Fallback to original margin-based approach
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            
            cv2.rectangle(mask, 
                         (self.margin, self.margin), 
                         (self.width - self.margin, self.height - self.margin), 
                         255, -1)
            
            if len(self.image.shape) == 3:
                self.working_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            else:
                self.working_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            
            print(f"[Boundary] Applied {self.margin}px inner margin (includes safety padding)")
    
    def detect_markers_from_json(self, start_color='red'):
        """Detect markers from JSON data (PRIMARY METHOD)
        
        Args:
            start_color: 'red' to go from red to green, 'green' to go from green to red
        """
        if not self.transform_data or 'circles' not in self.transform_data:
            print("[JSON] No circle data in JSON, falling back to image detection")
            return False
        
        circles = self.transform_data['circles']
        
        if len(circles) != 2:
            print(f"[JSON] Expected 2 circles, found {len(circles)}")
            return False
        
        # Find red and green circles
        red_circle = None
        green_circle = None
        
        for circle in circles:
            if circle['color'] == 'red':
                red_circle = circle
            elif circle['color'] == 'green':
                green_circle = circle
        
        if not red_circle or not green_circle:
            print("[JSON] Missing red or green circle in JSON data")
            return False
        
        # Extract centers (already in camera coordinates)
        red_center = tuple(red_circle['center'])  # (x, y)
        green_center = tuple(green_circle['center'])  # (x, y)
        
        # Convert to (y, x) format for internal use
        red_center_yx = (red_center[1], red_center[0])
        green_center_yx = (green_center[1], green_center[0])
        
        # Set start/end based on user preference
        if start_color.lower() == 'red':
            self.start = red_center_yx
            self.end = green_center_yx
            print(f"[JSON] RED → GREEN: Start{self.start} → End{self.end}")
        else:
            self.start = green_center_yx
            self.end = red_center_yx
            print(f"[JSON] GREEN → RED: Start{self.start} → End{self.end}")
        
        return True
    
    def detect_circles_fallback(self):
        """FALLBACK: Detect two largest circles as start/end markers"""
        print("[Fallback] Detecting circles...")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle_list = []
            
            for circle in circles[0, :]:
                x, y, r = circle
                if self.is_walkable_simple(y, x):
                    circle_list.append((y, x, r))
            
            if len(circle_list) >= 2:
                circle_list.sort(key=lambda c: (c[1], c[0]))
                
                self.start = (circle_list[0][0], circle_list[0][1])
                self.end = (circle_list[-1][0], circle_list[-1][1])
                
                print(f"[Fallback] Circles detected: Start{self.start} → End{self.end}")
                return True
        
        print("[Fallback] Circle detection failed")
        return False
    
    def detect_markers_color_fallback(self, start_color='red'):
        """FALLBACK: Detect red (start) and green (end) markers with direction control
        
        Args:
            start_color: 'red' to go from red to green, 'green' to go from green to red
        """
        print("[Fallback] Using HSV color detection...")
        self.apply_inner_boundary()
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Red color detection
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Green color detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find red marker
        red_center = None
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 50:
                M = cv2.moments(largest_red)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_center = (cy, cx)
        
        # Find green marker
        green_center = None
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 50:
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    green_center = (cy, cx)
        
        # Set start/end based on user preference
        if red_center is not None and green_center is not None:
            if start_color.lower() == 'red':
                self.start = red_center
                self.end = green_center
                print(f"[Fallback] RED → GREEN: Start{self.start} → End{self.end}")
            else:
                self.start = green_center
                self.end = red_center
                print(f"[Fallback] GREEN → RED: Start{self.start} → End{self.end}")
            
            return True
        
        # FALLBACK 2: Try circle detection
        print("[Warning] Red/Green markers not found, trying circle detection...")
        return self.detect_circles_fallback()
    
    def detect_markers(self, start_color='red'):
        """Detect markers with priority: JSON → Color detection → Circle detection
        
        Args:
            start_color: 'red' to go from red to green, 'green' to go from green to red
        """
        # Try JSON first (most reliable)
        if self.detect_markers_from_json(start_color):
            return True
        
        # Fallback to color detection
        if self.detect_markers_color_fallback(start_color):
            return True
        
        # Final fallback to circle detection
        return self.detect_circles_fallback()
    
    def is_walkable_simple(self, y, x):
        """Simple walkability check for marker detection"""
        if x < self.margin or x >= self.width - self.margin:
            return False
        if y < self.margin or y >= self.height - self.margin:
            return False
        
        pixel = self.image[y, x]
        brightness = np.mean(pixel)
        return brightness > 120
    
    def erode_by_px(self, mask255, px):
        """Erode a binary mask by specified pixels"""
        if px <= 0:
            return (mask255 > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
        return cv2.erode((mask255 > 0).astype(np.uint8) * 255, k, iterations=1)
    
    def skeletonize(self, bin255):
        """Morphological skeletonization to extract corridor centerlines"""
        img = (bin255 > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        
        return skel
    
    def nearest_nonzero(self, img, pt):
        """Find nearest non-zero pixel to given point"""
        x, y = map(int, pt)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and img[y, x] > 0:
            return (x, y)
        
        H, W = img.shape
        for r in range(1, max(H, W), 2):
            xmin, xmax = max(0, x-r), min(W-1, x+r)
            ymin, ymax = max(0, y-r), min(H-1, y+r)
            roi = img[ymin:ymax+1, xmin:xmax+1]
            nz = cv2.findNonZero(roi)
            if nz is not None:
                nz = nz.reshape(-1, 2)
                d2 = (nz[:, 0] + xmin - x)**2 + (nz[:, 1] + ymin - y)**2
                idx = int(np.argmin(d2))
                return (int(nz[idx, 0] + xmin), int(nz[idx, 1] + ymin))
        
        return (x, y)
    
    def bfs_on_mask(self, binary255, start, end, eight=True):
        """BFS pathfinding on binary mask"""
        h, w = binary255.shape
        sx, sy = start
        ex, ey = end
        
        if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
            return []
        if binary255[sy, sx] == 0 or binary255[ey, ex] == 0:
            return []
        
        visited = np.zeros((h, w), np.uint8)
        parent = np.full((h, w, 2), -1, dtype=np.int32)
        dq = deque([(sx, sy)])
        visited[sy, sx] = 1
        
        N4 = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        N8 = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        neigh = N8 if eight else N4
        
        while dq:
            x, y = dq.popleft()
            if (x, y) == (ex, ey):
                break
            
            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0 and binary255[ny, nx] > 0:
                    visited[ny, nx] = 1
                    parent[ny, nx] = [x, y]
                    dq.append((nx, ny))
        
        if parent[ey, ex, 0] == -1 and (ex, ey) != (sx, sy):
            return []
        
        path = [(ex, ey)]
        x, y = ex, ey
        while (x, y) != (sx, sy):
            px, py = parent[y, x]
            if px == -1:
                break
            path.append((int(px), int(py)))
            x, y = int(px), int(py)
        
        path.append((sx, sy))
        path.reverse()
        return path
    
    def astar_centered(self, allow255, dist_norm, start, end, len_w=1.0, center_w=None):
        """Center-biased A* pathfinding algorithm"""
        if center_w is None:
            center_w = self.CENTER_BIAS
        
        h, w = allow255.shape
        sx, sy = start
        ex, ey = end
        
        if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
            return []
        if allow255[sy, sx] == 0 or allow255[ey, ex] == 0:
            return []
        
        INF = 1e12
        g = np.full((h, w), INF, dtype=np.float32)
        parent = np.full((h, w, 2), -1, dtype=np.int32)
        open_heap = []
        
        def hfun(x, y):
            return float(abs(x - ex) + abs(y - ey))
        
        g[sy, sx] = 0.0
        heapq.heappush(open_heap, (hfun(sx, sy), 0.0, sx, sy))
        closed = np.zeros((h, w), np.uint8)
        
        # 4-connected movement
        for_pop = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        
        while open_heap:
            f_curr, g_curr, x, y = heapq.heappop(open_heap)
            
            if closed[y, x]:
                continue
            closed[y, x] = 1
            
            if (x, y) == (ex, ey):
                break
            
            for dx, dy in for_pop:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < w and 0 <= ny < h and allow255[ny, nx] > 0 and not closed[ny, nx]:
                    # Cost function: length + center bias penalty
                    step_cost = len_w + center_w * (1.0 - float(dist_norm[ny, nx]))
                    tentative = g_curr + step_cost
                    
                    if tentative < g[ny, nx]:
                        g[ny, nx] = tentative
                        parent[ny, nx] = [x, y]
                        f = tentative + hfun(nx, ny)
                        heapq.heappush(open_heap, (f, tentative, nx, ny))
        
        if parent[ey, ex, 0] == -1 and (ex, ey) != (sx, sy):
            return []
        
        path = [(ex, ey)]
        x, y = ex, ey
        while (x, y) != (sx, sy):
            px, py = parent[y, x]
            if px == -1:
                break
            path.append((int(px), int(py)))
            x, y = int(px), int(py)
        
        path.append((sx, sy))
        path.reverse()
        return path
    
    def adjust_start_end_to_perimeter(self):
        """Adjust start and end points to the perimeter of the circles based on path direction"""
        if not self.start or not self.end or len(self.path) < 2:
            return
        
        # Calculate direction vector from start to end
        start_y, start_x = self.start
        end_y, end_x = self.end
        
        # Direction from start to first path point
        first_path_point = self.path[1]  # Second point in path
        dx_start = first_path_point[1] - start_x
        dy_start = first_path_point[0] - start_y
        
        # Normalize direction
        dist_start = np.sqrt(dx_start**2 + dy_start**2)
        if dist_start > 0:
            dx_start /= dist_start
            dy_start /= dist_start
        
        # Direction to last path point
        last_path_point = self.path[-2]  # Second to last point
        dx_end = end_x - last_path_point[1]
        dy_end = end_y - last_path_point[0]
        
        # Normalize direction
        dist_end = np.sqrt(dx_end**2 + dy_end**2)
        if dist_end > 0:
            dx_end /= dist_end
            dy_end /= dist_end
        
        # Estimate circle radius (use end_effector_radius as approximation)
        circle_radius = self.end_effector_radius
        
        # Adjust start point to perimeter
        new_start_x = int(start_x + dx_start * circle_radius)
        new_start_y = int(start_y + dy_start * circle_radius)
        self.start = (new_start_y, new_start_x)
        
        # Adjust end point to perimeter (coming FROM the direction)
        new_end_x = int(end_x - dx_end * circle_radius)
        new_end_y = int(end_y - dy_end * circle_radius)
        self.end = (new_end_y, new_end_x)
        
        # Update path with adjusted start/end
        self.path[0] = self.start
        self.path[-1] = self.end
        
        print(f"[Adjust] Start adjusted to perimeter: {self.start}")
        print(f"[Adjust] End adjusted to perimeter: {self.end}")
    
    def solve_with_progressive_fallback(self):
        """Advanced path solver with progressive fallback strategy
        
        Tries in order:
        1. Center-biased A* with auto-margin
        2. Skeleton BFS (for tight spaces)
        3. Direct BFS (guaranteed path if one exists)
        """
        if not self.start or not self.end:
            print("[Error] Start or end position not set!")
            return False
        
        print(f"[Path] Planning from {self.start} to {self.end}")
        print(f"[Path] Using center-biased A* with progressive fallback")
        
        # Convert start/end from (y,x) to (x,y) for compatibility
        start_xy = (self.start[1], self.start[0])
        end_xy = (self.end[1], self.end[0])
        
        # Calculate auto-margin based on corridor width
        dist_tmp = self.distance_transform.copy()
        skel_tmp = self.skeletonize(self.corridor_mask)
        vals = dist_tmp[skel_tmp > 0]
        r_med = float(np.median(vals)) if vals.size > 0 else 3.0
        
        AUTO_MARGIN_RATIO = 0.55
        MIN_MARGIN_PX = 2
        base_auto = max(MIN_MARGIN_PX, int(round(AUTO_MARGIN_RATIO * r_med)))
        
        # Ensure minimum clearance
        m_floor = max(0, int(self.MIN_CLEARANCE))
        margin_px = max(base_auto, m_floor)
        
        print(f"[Path] Auto-margin: {margin_px}px (median corridor radius: {r_med:.1f}px)")
        
        found_path = None
        safe_used = None
        method_used = None
        
        # Progressive fallback: try reducing margin until path found
        for m in range(int(margin_px), int(m_floor) - 1, -1):
            safe = self.erode_by_px(self.corridor_mask, m)
            if cv2.countNonZero(safe) == 0:
                continue
            
            # Snap start/end to safe zone
            s = self.nearest_nonzero(safe, start_xy)
            e = self.nearest_nonzero(safe, end_xy)
            
            # Try 1: Center-biased A*
            path = self.astar_centered(safe, self.dist_norm, s, e, center_w=self.CENTER_BIAS)
            if path:
                found_path = path
                safe_used = safe
                method_used = f"A* (margin={m}px)"
                break
            
            # Try 2: Skeleton BFS (for narrow corridors)
            sk = self.skeletonize(safe)
            if cv2.countNonZero(sk) > 0:
                s2 = self.nearest_nonzero(sk, s)
                e2 = self.nearest_nonzero(sk, e)
                path = self.bfs_on_mask(sk, s2, e2, eight=True)
                if path:
                    found_path = path
                    safe_used = safe
                    method_used = f"Skeleton-BFS (margin={m}px)"
                    break
            
            # Try 3: Direct BFS
            path = self.bfs_on_mask(safe, s, e, eight=False)
            if path:
                found_path = path
                safe_used = safe
                method_used = f"Direct-BFS (margin={m}px)"
                break
        
        # Second pass: relax clearance if still no path
        if found_path is None:
            print("[Path] First pass failed, relaxing clearance requirements...")
            for m in range(int(min(m_floor - 1, max(margin_px - 1, 0))), -1, -1):
                safe = self.erode_by_px(self.corridor_mask, m)
                if cv2.countNonZero(safe) == 0:
                    continue
                
                s = self.nearest_nonzero(safe, start_xy)
                e = self.nearest_nonzero(safe, end_xy)
                
                path = self.astar_centered(safe, self.dist_norm, s, e, center_w=self.CENTER_BIAS)
                if not path:
                    sk = self.skeletonize(safe)
                    if cv2.countNonZero(sk) > 0:
                        s2 = self.nearest_nonzero(sk, s)
                        e2 = self.nearest_nonzero(sk, e)
                        path = self.bfs_on_mask(sk, s2, e2, eight=False)
                if not path:
                    path = self.bfs_on_mask(safe, s, e, eight=False)
                
                if path:
                    found_path = path
                    safe_used = safe
                    method_used = f"Relaxed-{method_used or 'BFS'} (margin={m}px)"
                    break
        
        if found_path is None:
            print("✗ No path found after all fallback attempts!")
            return False
        
        # Convert path from (x,y) back to (y,x) format
        self.path = [(y, x) for (x, y) in found_path]
        
        print(f"✓ Path found using: {method_used}")
        print(f"✓ Path length: {len(self.path)} waypoints")
        
        # Adjust start/end to circle perimeter based on path direction
        self.adjust_start_end_to_perimeter()
        
        return True
    
    def simplify_by_direction(self, points):
        """Simplify path by removing collinear points"""
        if len(points) <= 2:
            return points
        
        out = [points[0]]
        dxp = points[1][0] - points[0][0]
        dyp = points[1][1] - points[0][1]
        
        for i in range(2, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            
            if (dx, dy) != (dxp, dyp):
                out.append(points[i-1])
            
            dxp, dyp = dx, dy
        
        out.append(points[-1])
        return out
    
    def simplify_path_constrained(self, min_waypoints=10, max_waypoints=50):
        """Path simplification with constraints"""
        if len(self.path) < 3:
            return
        
        print(f"[Path] Simplifying to {min_waypoints}-{max_waypoints} waypoints...")
        
        # First pass: remove collinear points
        self.path = self.simplify_by_direction(self.path)
        
        # Enforce maximum waypoints by sampling
        if len(self.path) > max_waypoints:
            print(f"[Path] Sampling from {len(self.path)} to {max_waypoints} waypoints...")
            step = len(self.path) / (max_waypoints - 1)
            final = [self.path[0]]
            for idx in range(1, max_waypoints - 1):
                final.append(self.path[int(idx * step)])
            final.append(self.path[-1])
            self.path = final
        
        # Enforce minimum waypoints by interpolation
        if len(self.path) < min_waypoints:
            print(f"[Path] Interpolating from {len(self.path)} to {min_waypoints} waypoints...")
            points_to_add = min_waypoints - len(self.path)
            
            # Find longest segments
            segments = []
            for i in range(len(self.path) - 1):
                y1, x1 = self.path[i]
                y2, x2 = self.path[i + 1]
                length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                segments.append((length, i))
            
            segments.sort(reverse=True)
            
            # Add midpoints to longest segments
            new_points = {}
            for j in range(min(points_to_add, len(segments))):
                seg_idx = segments[j][1]
                y1, x1 = self.path[seg_idx]
                y2, x2 = self.path[seg_idx + 1]
                mid_y = (y1 + y2) // 2
                mid_x = (x1 + x2) // 2
                
                if seg_idx not in new_points:
                    new_points[seg_idx] = []
                new_points[seg_idx].append((mid_y, mid_x))
            
            # Rebuild path
            final = []
            for i in range(len(self.path)):
                final.append(self.path[i])
                if i in new_points:
                    final.extend(new_points[i])
            
            self.path = final
        
        print(f"[Path] Final waypoint count: {len(self.path)}")
    
    def generate_directions(self, min_waypoints=10, max_waypoints=50):
        """Generate step-by-step directions from the path"""
        if len(self.path) < 2:
            return []
        
        self.simplify_path_constrained(min_waypoints=min_waypoints, max_waypoints=max_waypoints)
        
        directions = []
        
        for i in range(len(self.path) - 1):
            current = self.path[i]
            next_pos = self.path[i + 1]
            
            dy = next_pos[0] - current[0]
            dx = next_pos[1] - current[1]
            
            distance = int(np.sqrt(dx**2 + dy**2))
            
            if abs(dx) < 5 and abs(dy) > 0:
                direction = "DOWN" if dy > 0 else "UP"
                steps = abs(dy)
            elif abs(dy) < 5 and abs(dx) > 0:
                direction = "RIGHT" if dx > 0 else "LEFT"
                steps = abs(dx)
            else:
                if dx > 0 and dy > 0:
                    direction = "DOWN-RIGHT"
                elif dx > 0 and dy < 0:
                    direction = "UP-RIGHT"
                elif dx < 0 and dy > 0:
                    direction = "DOWN-LEFT"
                else:
                    direction = "UP-LEFT"
                steps = distance
            
            if steps > 0:
                directions.append(f"Move {direction} {steps} step{'s' if steps > 1 else ''}")
        
        self.directions = directions
        return directions
    
    def draw_solution(self):
        """Draw the solution path on the image"""
        solution_image = self.original_image.copy()
        
        # Draw maze bounds if available from JSON
        if self.maze_bounds:
            tl = tuple(self.maze_bounds['top_left'])
            tr = tuple(self.maze_bounds['top_right'])
            br = tuple(self.maze_bounds['bottom_right'])
            bl = tuple(self.maze_bounds['bottom_left'])
            
            cv2.line(solution_image, tl, tr, (128, 128, 128), 2)
            cv2.line(solution_image, tr, br, (128, 128, 128), 2)
            cv2.line(solution_image, br, bl, (128, 128, 128), 2)
            cv2.line(solution_image, bl, tl, (128, 128, 128), 2)
        else:
            # Draw inner boundary (fallback)
            cv2.rectangle(solution_image, 
                         (self.margin, self.margin),
                         (self.width - self.margin, self.height - self.margin),
                         (128, 128, 128), 2)
        
        # Draw the path
        line_thickness = max(4, self.wall_thickness // 3)
        for i in range(len(self.path) - 1):
            y1, x1 = self.path[i]
            y2, x2 = self.path[i + 1]
            cv2.line(solution_image, (x1, y1), (x2, y2), (255, 0, 255), line_thickness)
        
        # Draw waypoint markers
        marker_size = max(5, self.wall_thickness // 2)
        for i, (y, x) in enumerate(self.path):
            cv2.circle(solution_image, (x, y), marker_size, (255, 255, 0), -1)
            cv2.putText(solution_image, str(i), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw start and end
        if self.start:
            cv2.circle(solution_image, (self.start[1], self.start[0]), marker_size * 2, (0, 0, 255), -1)
            cv2.putText(solution_image, "START", (self.start[1] + 15, self.start[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        if self.end:
            cv2.circle(solution_image, (self.end[1], self.end[0]), marker_size * 2, (0, 255, 0), -1)
            cv2.putText(solution_image, "END", (self.end[1] + 15, self.end[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return solution_image
    
    def save_solution(self, output_path="maze_solved.png"):
        """Save the solved maze image"""
        solution_image = self.draw_solution()
        cv2.imwrite(output_path, solution_image)
        print(f"✓ Solution saved: {output_path}")
        return output_path
    
    def print_directions(self):
        """Print all directions"""
        print("\n" + "="*50)
        print("STEP-BY-STEP DIRECTIONS:")
        print("="*50)
        for i, direction in enumerate(self.directions, 1):
            print(f"{i}. {direction}")
        print("="*50)
        print(f"Total waypoints: {len(self.path)}")
        print(f"Total moves: {len(self.directions)}")
        print("="*50 + "\n")


def solve_maze(image_path, output_path="maze_solved.png", start_color='red', 
               min_waypoints=10, max_waypoints=50, end_effector_radius=15,
               transform_json_path=None):
    """Main function to solve a maze using advanced path planning algorithms
    
    Uses progressive fallback strategy:
    1. JSON-based detection (most reliable, uses pre-detected markers and maze bounds)
    2. Center-biased A* (keeps path in corridor center, away from walls)
    3. Skeleton BFS (for tight spaces where A* fails)
    4. Direct BFS (guaranteed path if one exists)
    
    Args:
        image_path: Path to maze image
        output_path: Path for solved image
        start_color: 'red' for red→green, 'green' for green→red
        min_waypoints: Minimum number of waypoints (default 10)
        max_waypoints: Maximum number of waypoints (default 50)
        end_effector_radius: Radius of robot's end effector in pixels (default 15px)
        transform_json_path: Path to transform_data.json (auto-detected if None)
    """
    print(f"\n[Solver] Loading: {image_path}")
    print(f"[Solver] Waypoint constraints: {min_waypoints}-{max_waypoints}")
    print(f"[Solver] End effector radius: {end_effector_radius}px")
    print(f"[Solver] Detection: JSON → Color → Circle (priority order)")
    print(f"[Solver] Algorithm: Center-biased A* with progressive fallback")
    
    solver = MazeSolver(image_path, transform_json_path=transform_json_path, 
                       end_effector_radius=end_effector_radius)
    
    print("\n[Solver] Detecting markers...")
    if not solver.detect_markers(start_color=start_color):
        print("✗ Could not detect markers!")
        return None
    
    print("\n[Solver] Planning optimal path...")
    if not solver.solve_with_progressive_fallback():
        print("✗ No solution found!")
        return None
    
    print("\n[Solver] Generating directions...")
    solver.generate_directions(min_waypoints=min_waypoints, max_waypoints=max_waypoints)
    solver.print_directions()
    
    print("[Solver] Saving solution...")
    solver.save_solution(output_path)
    
    return solver


if __name__ == "__main__":
    maze_path = "/home/mdg/Documents/RS1/midsem2/rectified_maze.jpg"
    json_path = "/home/mdg/Documents/RS1/midsem2/transform_data.json"
    
    solver = solve_maze(
        maze_path, 
        "/home/mdg/Documents/RS1/midsem2/maze_solved_advanced.png",
        start_color='red',
        min_waypoints=10,
        max_waypoints=50,
        end_effector_radius=15,
        transform_json_path=json_path
    )
