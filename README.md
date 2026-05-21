Dobot Maze Solver v2.0
AI-orchestrated autonomous maze navigation using a Dobot Magician Lite robot arm, overhead camera, and Groq LLM.

Overview
The system captures a physical maze via camera, detects colored markers (red/green circles) as start/end points, computes an optimal path using center-biased A* with progressive fallback algorithms, and physically traces the solution using a Dobot Magician Lite arm. A Groq LLM (LLaMA 3.3 70B) acts as an orchestrator throughout the pipeline — analyzing detection quality, validating solutions, predicting execution time, and diagnosing failures.

File Structure
├── main.py          # Pipeline orchestrator: AI, Dobot control, coordinate conversion
├── detection.py     # Camera capture, HSV circle detection, maze boundary detection
├── directions.py    # MazeSolver class: A* pathfinding, waypoint generation

Hardware Requirements

Dobot Magician Lite (connected via /dev/ttyACM0)
Webcam (default index 2 in detection.py)
Physical maze: white paths on dark background
Red circle = start/end marker
Green circle = start/end marker


Software Dependencies
bashpip install opencv-python numpy langchain-groq langchain-core pydobot matplotlib

Configuration
All constants are defined at the top of each file.
main.py
ConstantDefaultDescriptionGROQ_API_KEYgsk_...Your Groq API keyDOBOT_PORT/dev/ttyACM0Serial port for DobotMAZE_IMAGE_PATH.../rectified_maze.jpgWhere captured image is savedSOLVED_IMAGE_PATH.../maze_solved.pngWhere solution overlay is savedTRANSFORM_PATH.../transform_data.jsonWhere detection metadata is savedZ_SOLVE-50.0Z-height (mm) during maze tracingCAMERA_POSITIONx=250, y=0, z=150Dobot resting/photo positionCALIBRATION_POINTS5 pixel↔Dobot pairsAffine transform calibration
detection.py
ConstantDefaultDescriptionCAM_INDEX2Camera device indexSAVE_PATH.../rectified_maze.jpgOutput image pathMAZE_DATA_PATH.../transform_data.jsonOutput JSON path

Running the System
bashpython3 main.py
Select mode at prompt:

1 — Real Dobot (AI-orchestrated)
2 — Simulation mode (AI-orchestrated, no hardware)

Pipeline Steps
STEP 1/5 — Positioning      Move Dobot to camera overhead position
STEP 2/5 — Detection        Launch detection.py; press SPACE/C to capture when ready
STEP 3/5 — Direction        Choose RED→GREEN or GREEN→RED
STEP 4/5 — Solving          A* pathfinding; solution displayed and saved
STEP 5/5 — Execution        Press ENTER; Dobot traces the path

Module Details
detection.py
Runs as a standalone camera feed. Detects:

Colored circles via HSV masking (red wraps around 0°/180°, green centered ~60°)
Maze boundary via Otsu thresholding + largest contour bounding rect
Walkable mask via combined adaptive + Otsu thresholding

Outputs on capture:

rectified_maze.jpg — raw camera frame
rectified_maze_mask.png — binary walkable mask
transform_data.json — circles, maze bounds, maze structure metadata

Controls while running:

SPACE or C — capture (only if both circles and bounds detected)
Q or ESC — quit

directions.py — MazeSolver
Loads the maze image and JSON, then:

Marker detection — reads circle positions from JSON; falls back to HSV color detection, then Hough circles
Corridor mask — binary mask restricted to maze bounds; distance transform for wall clearance scoring
Pathfinding — progressive fallback:

Center-biased A* (prefers corridor centers, penalizes wall proximity)
Skeleton BFS (medial axis traversal for tight passages)
Direct BFS (guaranteed path if any exists)


Waypoint simplification — constrained to [min_waypoints, max_waypoints] range
Direction generation — cardinal/diagonal step-by-step strings (e.g., Move DOWN-RIGHT 42 steps)

main.py
Ties everything together:

AIOrchestrator — wraps Groq ChatGroq with a stateful conversation; called at detection, solving, pre-execution, failure, and final report stages
CoordinateConverter — least-squares affine fit from 5 calibration pixel↔Dobot pairs; camera_to_dobot(cam_x, cam_y) → (dobot_x, dobot_y, z, r)
DobotController — wraps pydobot; falls back to simulation if connection fails; non-blocking moves with configurable wait
run_detection() — spawns detection.py as subprocess; monitors file modification time; 120s timeout
get_maze_solution() — instantiates MazeSolver, runs solve pipeline, returns path dict
execute_maze_solving() — iterates waypoints, calls move_to_camera_pixel() per step


Calibration
The affine pixel→Dobot mapping uses 5 hardcoded point pairs in CALIBRATION_POINTS. To recalibrate:

Place the Dobot at known XY positions over the maze
Note the corresponding pixel coordinates in the camera frame
Update the list in main.py:

pythonCALIBRATION_POINTS = [
    {'pixel': (px, py), 'dobot': (dobot_x, dobot_y)},
    ...  # minimum 3 points; 5+ recommended
]

Output Files
FileDescriptionrectified_maze.jpgRaw captured framerectified_maze_mask.pngBinary walkable mask (255=path, 0=wall)transform_data.jsonDetection metadata (circles, bounds, structure)maze_solved.pngSolution path overlaid on maze image

Simulation Mode
Running in simulation (mode 2) skips all Dobot hardware calls and replaces time.sleep(0.05) delays for each move. All detection, solving, AI orchestration, and coordinate conve
