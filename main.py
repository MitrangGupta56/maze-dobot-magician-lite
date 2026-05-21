import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from directions import MazeSolver

# ==========================================
# Configuration
# ==========================================
GROQ_API_KEY = "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"
MAZE_IMAGE_PATH = "/home/mdg/Documents/RS1/midsem2/rectified_maze.jpg"
SOLVED_IMAGE_PATH = "/home/mdg/Documents/RS1/midsem2/maze_solved.png"
TRANSFORM_PATH = "/home/mdg/Documents/RS1/midsem2/transform_data.json"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ==========================================
# DOBOT CONFIGURATION
# ==========================================
DOBOT_PORT = '/dev/ttyACM0'

CAMERA_POSITION = {
    'x': 250.0,
    'y': 0,
    'z': 150.0,
    'r': 0.0
}

Z_SOLVE = -50.0

# ==========================================
# CALIBRATION DATA (UPDATED)
# ==========================================
CALIBRATION_POINTS = [
    {'pixel': (338, 227), 'dobot': (301.2, -11.4)},
    {'pixel': (430, 101), 'dobot': (358.8, -54.6)},
    {'pixel': (70, 430), 'dobot': (205.2, 112.8)},
    {'pixel': (431, 433), 'dobot': (203.6, -55.2)},
    {'pixel': (102, 179), 'dobot': (323.5, 96.1)},
]

# ==========================================
# AI ORCHESTRATOR (GROQ LLM)
# ==========================================
class AIOrchestrator:
    """Uses Groq LLM to intelligently orchestrate the entire maze solving pipeline"""
    
    def __init__(self, api_key):
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )
        self.conversation_history = []
        self.system_prompt = """You are an AI orchestrator for a robotic maze solving system using Dobot Magician Lite. 

Your responsibilities:
1. Analyze maze detection quality and grid size detection
2. Validate pathfinding solutions for efficiency and correctness
3. Monitor Dobot execution and predict mechanical issues
4. Provide concise, actionable recommendations for failures
5. Generate brief technical summaries

Be concise (2-3 sentences max), technical, and action-oriented. Focus on what matters."""
        
        self.conversation_history.append(SystemMessage(content=self.system_prompt))
    
    def ask(self, query, context=None):
        """Ask the LLM for analysis"""
        if context:
            full_query = f"{query}\n\nData: {json.dumps(context, indent=2)}"
        else:
            full_query = query
        
        self.conversation_history.append(HumanMessage(content=full_query))
        
        try:
            response = self.llm.invoke(self.conversation_history)
            self.conversation_history.append(response)
            return response.content
        except Exception as e:
            return f"AI Error: {str(e)}"
    
    def analyze_detection(self, transform_data):
        """Analyze detection results"""
        circles_found = len(transform_data.get('circles', [])) if transform_data else 0
        bounds_found = transform_data.get('maze_bounds') is not None if transform_data else False
        
        query = f"""Detection complete. Circles: {circles_found}/2, Maze bounds: {'found' if bounds_found else 'not found'}. 
Is this sufficient? Any concerns before solving?"""
        
        return self.ask(query, {
            "circles": circles_found,
            "bounds": bounds_found,
            "has_structure": transform_data.get('maze_structure') is not None if transform_data else False
        })
    
    def analyze_solution(self, solution_data):
        """Analyze the maze solution"""
        if not solution_data["success"]:
            query = f"""Solver failed: {solution_data.get('error')}. 
What's the likely cause and fix?"""
            return self.ask(query)
        
        efficiency = solution_data['total_steps'] / max(solution_data['total_moves'], 1)
        
        query = f"""Solution found: {solution_data['total_steps']} waypoints, {solution_data['total_moves']} moves.
Efficiency ratio: {efficiency:.2f}. Is this optimal?"""
        
        return self.ask(query, {
            "waypoints": solution_data['total_steps'],
            "moves": solution_data['total_moves'],
            "efficiency": efficiency
        })
    
    def predict_execution_time(self, waypoint_count):
        """Predict execution time"""
        estimated_time = waypoint_count * 0.35  # Optimized: 0.35s per waypoint
        
        query = f"""Executing {waypoint_count} waypoints. Estimated time: {estimated_time:.1f}s. 
Any speed optimization suggestions?"""
        
        return self.ask(query, {"waypoints": waypoint_count, "time_estimate": estimated_time})
    
    def analyze_failure(self, error_msg):
        """Analyze failure and provide recommendations"""
        query = f"""Execution failed: {error_msg}. 
Quick diagnosis and recommended fix?"""
        
        return self.ask(query)
    
    def generate_final_report(self, execution_result, start_time):
        """Generate final mission report"""
        elapsed = time.time() - start_time
        
        query = f"""Mission complete. Success: {execution_result['success']}. 
Time: {elapsed:.1f}s. Waypoints: {execution_result.get('waypoints_executed', 0)}.
Brief summary and one key improvement?"""
        
        return self.ask(query, {
            "success": execution_result['success'],
            "time": elapsed,
            "waypoints": execution_result.get('waypoints_executed', 0)
        })


# ==========================================
# COORDINATE CONVERTER
# ==========================================
class CoordinateConverter:
    """Converts raw camera coordinates to Dobot coordinates (no homography needed)"""
    
    def __init__(self, calibration_points):
        self.calibration_points = calibration_points
        self.avg_r = 14.25
        
        self.calculate_transformation()
    
    def calculate_transformation(self):
        """Calculate affine transformation from camera pixels to Dobot coordinates"""
        pixel_coords = np.array([
            [p['pixel'][0], p['pixel'][1], 1] for p in self.calibration_points
        ])
        
        dobot_x = np.array([p['dobot'][0] for p in self.calibration_points])
        dobot_y = np.array([p['dobot'][1] for p in self.calibration_points])
        
        coeffs_x, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_x, rcond=None)
        self.a11, self.a12, self.b1 = coeffs_x
        
        coeffs_y, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_y, rcond=None)
        self.a21, self.a22, self.b2 = coeffs_y
    
    def camera_to_dobot(self, cam_x, cam_y, z_height=None):
        """Convert camera pixel coordinates directly to Dobot coordinates"""
        if z_height is None:
            z_height = Z_SOLVE
        
        dobot_x = self.a11 * cam_x + self.a12 * cam_y + self.b1
        dobot_y = self.a21 * cam_x + self.a22 * cam_y + self.b2
        
        return dobot_x, dobot_y, z_height, self.avg_r


# ==========================================
# DOBOT CONTROLLER (OPTIMIZED)
# ==========================================
class DobotController:
    """Optimized Dobot controller with faster movement"""
    
    def __init__(self, use_simulation=False):
        self.use_simulation = use_simulation
        self.device = None
        self.coord_converter = None
        
        if not use_simulation:
            try:
                from pydobot import Dobot
                print(f"[Dobot] Connecting to {DOBOT_PORT}...")
                self.device = Dobot(port=DOBOT_PORT)
                time.sleep(1)
                
                # Set faster velocity and acceleration
                if hasattr(self.device, 'set_speed'):
                    self.device.set_speed(velocity=200, acceleration=200)
                
                print("[Dobot] ✓ Connected!")
                self.go_home()
            except Exception as e:
                print(f"[Dobot] ✗ Connection failed: {e}")
                print("[Dobot] Using simulation mode")
                self.use_simulation = True
        else:
            print("[Dobot] Simulation mode")
    
    def set_coordinate_converter(self, converter):
        """Set coordinate converter"""
        self.coord_converter = converter
    
    def move_to(self, x, y, z, r=None, wait=0.15):
        """Move Dobot (optimized with shorter wait)"""
        if r is None:
            r = 14.25
        
        if not self.use_simulation and self.device:
            self.device.move_to(x, y, z, r, wait=False)  # Non-blocking
            time.sleep(wait)  # Reduced wait time
        else:
            time.sleep(0.05)  # Simulation delay
    
    def go_camera_position(self):
        """Move to camera position"""
        self.move_to(
            CAMERA_POSITION['x'],
            CAMERA_POSITION['y'],
            CAMERA_POSITION['z'],
            CAMERA_POSITION['r'],
            wait=0.5
        )
    
    def go_home(self):
        """Return to home"""
        self.go_camera_position()
    
    def move_to_camera_pixel(self, cam_x, cam_y):
        """Move to camera pixel position (no rectification needed)"""
        if not self.coord_converter:
            print("[Dobot] Error: No coordinate converter")
            return
        
        dobot_x, dobot_y, z, r = self.coord_converter.camera_to_dobot(cam_x, cam_y)
        self.move_to(dobot_x, dobot_y, z, r)

    def close(self):
        """Close connection"""
        if self.device:
            self.device.close()


# ==========================================
# DETECTION
# ==========================================
def run_detection():
    """Run detection.py"""
    try:
        print("\n[Detection] Starting camera...")
        
        process = subprocess.Popen(
            ["python3", "detection.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        initial_exists = os.path.exists(MAZE_IMAGE_PATH)
        initial_mtime = os.path.getmtime(MAZE_IMAGE_PATH) if initial_exists else 0
        
        timeout = 120
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(MAZE_IMAGE_PATH):
                current_mtime = os.path.getmtime(MAZE_IMAGE_PATH)
                
                if current_mtime > initial_mtime:
                    time.sleep(0.5)
                    process.terminate()
                    time.sleep(0.2)
                    process.kill()
                    
                    print(f"[Detection] ✓ Image captured!")
                    return True
            
            time.sleep(0.1)
        
        process.terminate()
        process.kill()
        print("[Detection] ✗ Timeout")
        return False
            
    except Exception as e:
        print(f"[Detection] ✗ Error: {str(e)}")
        return False


# ==========================================
# MAZE SOLVING
# ==========================================
def get_maze_solution(start_color='red', display=True, min_waypoints=10, max_waypoints=50, end_effector_radius=15):
    """Solve the maze with user-specified direction
    
    Args:
        start_color: 'red' for red→green, 'green' for green→red
        display: Show solution visualization
        min_waypoints: Minimum waypoints for path
        max_waypoints: Maximum waypoints for path
        end_effector_radius: Robot end effector size in pixels
    """
    try:
        print("\n[Solver] Analyzing maze...")
        print(f"[Solver] Path direction: {start_color.upper()} → {'GREEN' if start_color=='red' else 'RED'}")
        
        if not os.path.exists(MAZE_IMAGE_PATH):
            return {"success": False, "error": "Image not found"}
        
        # Initialize solver with transform JSON path
        solver = MazeSolver(
            MAZE_IMAGE_PATH, 
            transform_json_path=TRANSFORM_PATH,
            end_effector_radius=end_effector_radius
        )
        
        print("[Solver] Detecting markers...")
        if not solver.detect_markers(start_color=start_color):
            return {"success": False, "error": "Markers not detected"}
        
        print(f"[Solver] Start: {solver.start}, End: {solver.end}")
        
        print("[Solver] Finding path...")
        if not solver.solve_with_progressive_fallback():
            return {"success": False, "error": "No solution found"}
        
        solver.generate_directions(min_waypoints=min_waypoints, max_waypoints=max_waypoints)
        
        if display:
            solution_img = solver.draw_solution()
            cv2.imshow("Solution - Press any key", solution_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        solver.save_solution(SOLVED_IMAGE_PATH)

        return {
            "success": True,
            "total_steps": len(solver.path),
            "total_moves": len(solver.directions),
            "directions": solver.directions,
            "path": solver.path,
            "start": solver.start,
            "end": solver.end,
            "solved_image": SOLVED_IMAGE_PATH
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Error: {str(e)}"}


# ==========================================
# EXECUTION (OPTIMIZED)
# ==========================================
def execute_maze_solving(solution_data, dobot, ai):
    """Execute maze solving with optimized movement"""
    try:
        if not solution_data["success"]:
            return {"success": False, "error": "Invalid solution"}
        
        path = solution_data["path"]
        
        print(f"\n[Execute] Starting FAST execution ({len(path)} waypoints)")
        print("[Execute] Movement optimized for speed...")
        
        # Move through waypoints rapidly
        # Path format from directions.py: (y, x) tuples
        for i, (y, x) in enumerate(path):
            if i % 10 == 0:  # Progress update every 10 waypoints
                progress = (i / len(path)) * 100
                print(f"  Progress: {progress:.0f}% ({i}/{len(path)})")
            
            # Convert (y, x) to (x, y) for camera coordinates
            dobot.move_to_camera_pixel(x, y)
        
        print(f"\n[Execute] ✓ Complete!")
        print("[Execute] Returning home...")
        dobot.go_camera_position()
        
        return {
            "success": True,
            "waypoints_executed": len(path),
            "start_position": solution_data["start"],
            "end_position": solution_data["end"]
        }
        
    except Exception as e:
        print(f"\n[Execute] ✗ Error: {str(e)}")
        
        # AI analyzes failure
        print("\n[AI] Analyzing failure...")
        analysis = ai.analyze_failure(str(e))
        print(f"[AI] {analysis}")
        
        import traceback
        traceback.print_exc()
        dobot.go_camera_position()
        return {"success": False, "error": str(e)}


# ==========================================
# MAIN PIPELINE WITH AI
# ==========================================
def run_complete_pipeline(use_simulation=False):
    """Complete AI-orchestrated pipeline"""
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" DOBOT MAZE SOLVER v2.0 - AI-ORCHESTRATED")
    print(" Raw Camera Detection (No Perspective Transform)")
    print(" Powered by Groq LLM + Optimized Execution")
    print("="*70)
    
    # Initialize AI
    print("\n[AI] Initializing Groq orchestrator...")
    ai = AIOrchestrator(GROQ_API_KEY)
    print("[AI] ✓ Ready\n")
    
    dobot = DobotController(use_simulation=use_simulation)
    
    try:
        # STEP 1: Position
        print("\n" + "="*70)
        print("[STEP 1/5] POSITIONING")
        print("="*70)
        dobot.go_camera_position()
        time.sleep(2)
        print("✓ Ready")
        
        # STEP 2: Detection
        print("\n" + "="*70)
        print("[STEP 2/5] DETECTION")
        print("="*70)
        input("Press ENTER to capture...")
        
        if not run_detection():
            print("\n✗ Detection failed")
            return
        
        # Load transform data
        transform_data = None
        if os.path.exists(TRANSFORM_PATH):
            with open(TRANSFORM_PATH, 'r') as f:
                transform_data = json.load(f)
            
            circles_found = len(transform_data.get('circles', []))
            bounds_found = transform_data.get('maze_bounds') is not None
            
            print(f"✓ Circles detected: {circles_found}/2")
            print(f"✓ Maze bounds: {'Found' if bounds_found else 'Not found'}")
            
            if transform_data.get('maze_structure'):
                print(f"✓ Maze structure extracted")
        
        # AI analyzes detection
        print("\n[AI] Analyzing detection...")
        detection_analysis = ai.analyze_detection(transform_data)
        print(f"[AI] {detection_analysis}")
        
        # Setup coordinates (simplified - no homography)
        coord_converter = CoordinateConverter(CALIBRATION_POINTS)
        dobot.set_coordinate_converter(coord_converter)
        print("✓ Coordinates configured")
        
        # STEP 3: Path Direction Selection
        print("\n" + "="*70)
        print("[STEP 3/5] PATH DIRECTION SELECTION")
        print("="*70)
        print("Choose path direction:")
        print("  1. RED → GREEN")
        print("  2. GREEN → RED")
        
        while True:
            direction_choice = input("\nEnter choice (1 or 2): ").strip()
            if direction_choice == '1':
                start_color = 'red'
                print("✓ Selected: RED → GREEN")
                break
            elif direction_choice == '2':
                start_color = 'green'
                print("✓ Selected: GREEN → RED")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        # STEP 4: Solve
        print("\n" + "="*70)
        print("[STEP 4/5] SOLVING")
        print("="*70)
        solution_data = get_maze_solution(
            start_color=start_color, 
            display=True,
            min_waypoints=10,
            max_waypoints=50,
            end_effector_radius=15
        )
        
        if not solution_data["success"]:
            print(f"\n✗ Failed: {solution_data['error']}")
            
            # AI analyzes failure
            print("\n[AI] Analyzing failure...")
            failure_analysis = ai.analyze_solution(solution_data)
            print(f"[AI] {failure_analysis}")
            return
        
        print(f"✓ Solution found!")
        print(f"  Waypoints: {solution_data['total_steps']}")
        print(f"  Moves: {solution_data['total_moves']}")
        
        # AI analyzes solution
        print("\n[AI] Analyzing solution...")
        solution_analysis = ai.analyze_solution(solution_data)
        print(f"[AI] {solution_analysis}")
        
        # AI predicts execution time
        print("\n[AI] Predicting execution...")
        time_prediction = ai.predict_execution_time(solution_data['total_steps'])
        print(f"[AI] {time_prediction}")
        
        # STEP 5: Execute
        print("\n" + "="*70)
        print("[STEP 5/5] EXECUTION")
        print("="*70)
        input("Press ENTER to execute (FAST MODE)...")
        
        exec_result = execute_maze_solving(solution_data, dobot, ai)
        
        # AI final report
        print("\n" + "="*70)
        print("MISSION REPORT")
        print("="*70)
        final_report = ai.generate_final_report(exec_result, start_time)
        print(f"\n{final_report}\n")
        
        if exec_result["success"]:
            elapsed = time.time() - start_time
            print("="*70)
            print("✓✓✓ MISSION COMPLETE ✓✓✓")
            print("="*70)
            print(f"Path: {start_color.upper()} → {'GREEN' if start_color=='red' else 'RED'}")
            print(f"Time: {elapsed:.1f}s")
            print(f"Waypoints: {exec_result['waypoints_executed']}")
            print(f"Speed: {exec_result['waypoints_executed']/elapsed:.1f} waypoints/sec")
            print("="*70)
        else:
            print(f"\n✗ Failed: {exec_result['error']}")
    
    finally:
        dobot.close()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          DOBOT MAZE SOLVER v2.0                          ║
║     AI-Orchestrated Autonomous Navigation                ║
║     Raw Camera Detection (No Transform)                  ║
║              Powered by Groq LLM                         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("Select mode:")
    print("1. Run with real Dobot (AI-orchestrated)")
    print("2. Run in simulation mode (AI-orchestrated)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        run_complete_pipeline(use_simulation=False)
    elif choice == '2':
        run_complete_pipeline(use_simulation=True)
    else:
        print("Invalid choice")
