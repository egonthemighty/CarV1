# AI Agent Project Guide - CarV1 Self-Driving RC Car

**Last Updated:** January 4, 2026  
**Repository:** https://github.com/egonthemighty/CarV1  
**Project Status:** Optimized for fast testing iterations

---

## 🚨 CRITICAL CURRENT STATE

### What Just Happened (Session Summary)
1. **Applied Performance Optimizations**: Implemented Colab recommendations to reduce training time and credit usage
   - Render camera view directly at target resolution (84×84) - eliminates redundant cv2.resize
   - Increased n_steps to 4096 and batch_size to 128 for more efficient gradient updates
   - Reduced total_timesteps to 10,000 for rapid testing
2. **Training Configuration**: Set for ultra-fast testing (10k steps = ~30 seconds on GPU)
3. **Production Training**: Increase total_timesteps to 500k+ once testing confirms setup is working

### Current Configuration (GitHub - Latest)
- **Policy:** CnnPolicy (Convolutional Neural Network)
- **Observation:** 84×84 grayscale camera images (use_raw_pixels=True)
- **Training Steps:** 10,000 (TESTING ONLY - use 500k+ for real training)
- **n_steps:** 4096 (optimized for efficiency)
- **batch_size:** 128 (optimized for GPU utilization)
- **Environment:** FirstPersonLineFollowEnv with curved track

### ⚡ Training Timestep Guidelines
**ALWAYS USE MINIMAL TIMESTEPS FOR TESTING:**
- **10k steps**: Quick sanity check (~30 seconds on GPU) - verify training runs
- **50k steps**: Basic metrics and video generation (~2-3 minutes)
- **500k+ steps**: Production training for actual model quality
- **Monitor `time/fps` in TensorBoard** - higher is better, target >1000 FPS

### What Needs to Happen Next
1. Upload `CarV1_Colab_Training.ipynb` to Google Colab
2. Run all cells (will clone latest code from GitHub)
3. Training takes ~30 seconds with 10k steps
4. Download model ZIP and place in `training output/`
5. Test with vision-based environment settings

---

## 📁 Project Structure

```
CarV1/
├── env/
│   ├── first_person_env.py          # Main environment (672 lines)
│   ├── camera_line_follow_env.py    # Legacy camera env
│   ├── car_env.py                   # Original top-down env
│   └── __init__.py                  # Gymnasium registration
│
├── config/
│   └── config.py                    # Training configuration
│
├── tests/
│   ├── train_gradient.py            # Google Colab training script
│   ├── test_rover_trained.py        # Model testing/evaluation
│   ├── generate_rover_video.py      # Single episode video generation
│   ├── generate_learning_comparison.py  # 2×2 comparison video
│   └── capture_view.py              # Still image capture
│
├── references/
│   └── hardware_integration.md      # Hardware specifications
│
├── training output/
│   ├── ppo_carv1_final/            # Latest model (feature-based - wrong!)
│   ├── old/                         # Archived models
│   └── videos/                      # Generated visualizations
│
├── CarV1_Colab_Training.ipynb       # Google Colab training notebook
└── venv/                            # Python virtual environment
```

---

## 🔧 Key Files Deep Dive

### 1. env/first_person_env.py (PRIMARY ENVIRONMENT)

**Purpose:** Gymnasium environment matching real Raspberry Pi Camera perspective

**Hardware Configuration (lines 14-21):**
```python
CAR_LENGTH = 0.61          # 24" car with bumpers
WHEELBASE_LENGTH = 0.33     # 13" hub-to-hub
WHEELBASE_WIDTH = 0.244     # 9.625" wheel track
CAMERA_HEIGHT = 0.254       # 10" above ground
CAMERA_X_OFFSET = -0.165    # 6.5" from front hub
CAMERA_PITCH = -11.8        # degrees down (focused 48" ahead)
CAMERA_FOV = 60.0          # degrees horizontal FOV
```

**Critical Parameters:**
- `use_raw_pixels` (line 46): Controls observation type
  - `True`: Returns 84×84 grayscale camera images (for CnnPolicy)
  - `False`: Returns 4 geometric features (for MlpPolicy - "cheating")
- `camera_resolution` (line 47): Tuple (width, height) for camera images

**Observation Space (lines 96-109):**
- **Vision Mode:** Box(0, 255, (84, 84, 1), dtype=uint8)
- **Feature Mode:** Box([-1.5, -1.5, -π, 0], [1.5, 1.5, π, max_speed], dtype=float32)

**Action Space (lines 112-116):** Box([-1, -1], [1, 1], dtype=float32)
- [0]: Steering (-1=full left, +1=full right)
- [1]: Throttle (-1=full reverse, +1=full forward)

**Key Methods:**
- `_render_camera_view()` (line 467): Generates camera images for Rover
- `_draw_first_person_view()` (line 410): Clean road rendering
- `_draw_rover_overlays()` (line 582): Human visualization (hood, collision indicators)
- `_draw_depth_bands()` (line 603): Motion visualization bands
- `_extract_features()` (line 277): Returns geometric features (bypassed in vision mode)
- `_get_track_curvature()` (line 263): Sine-wave curved track generation

**Track Generation:**
- Sine wave: amplitude=0.3 rad/m, frequency=0.05 rad/m
- Creates smooth, realistic curves
- Variable width: 0.6m (configurable)

### 2. config/config.py

**Current Settings:**
```python
TRAINING_CONFIG = {
    'policy_type': 'CnnPolicy',        # CNN for images, MLP for features
    'total_timesteps': 500_000,        # Reduced for troubleshooting
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    # ... (other PPO hyperparameters)
}
```

### 3. tests/train_gradient.py

**Google Colab Training Script**

**Critical Configuration (lines 24-27):**
```python
env_kwargs = {
    "render_mode": None,
    "use_raw_pixels": True,           # Enable vision-based training
    "camera_resolution": (84, 84)     # Match CnnPolicy input
}
```

**Training Flow:**
1. Creates FirstPersonLineFollowEnv with env_kwargs
2. Wraps in DummyVecEnv for vectorization
3. Loads config from config.py
4. Initializes PPO with CnnPolicy
5. Sets up callbacks (checkpoint every 10k steps, evaluation)
6. Trains for specified timesteps
7. Saves final model

### 4. Test Scripts

**generate_rover_video.py:**
- Loads trained model
- Creates environment with `use_raw_pixels=True, camera_resolution=(84,84)`
- Runs single episode
- Generates 800×600 @ 30fps MP4 with telemetry overlay

**generate_learning_comparison.py:**
- Loads 4 checkpoint models (100k, 250k, 350k, 500k)
- Creates 4 parallel environments
- Runs synchronized episodes
- Generates 2×2 grid video showing learning progression

**test_rover_trained.py:**
- Evaluates model performance
- Reports: total reward, final offset, steps completed
- Useful for quick performance checks

---

## 🎯 Training Pipeline

### Google Colab Workflow

1. **Upload Notebook:**
   - File: `CarV1_Colab_Training.ipynb`
   - Upload to Google Colab

2. **Colab Cells Execute:**
   ```bash
   # Cell 1: Clone latest code
   !git clone https://github.com/egonthemighty/CarV1.git
   %cd CarV1
   
   # Cell 2: Install dependencies
   !pip install gymnasium stable-baselines3 pygame opencv-python
   
   # Cell 3: Run training
   !python tests/train_gradient.py
   
   # Cell 4: Archive results
   !zip -r ppo_carv1_final.zip training\ output/carv1_models/*
   ```

3. **Download Model:**
   - Download `ppo_carv1_final.zip` from Colab
   - Extract to `training output/` locally

4. **Test Locally:**
   ```powershell
   cd "F:\Programming\python stuff\customGym\CarV1"
   .\venv\Scripts\python.exe tests/test_rover_trained.py
   ```

### Training Time Estimates (T4 GPU)
- 100k steps: ~30 seconds
- 500k steps: ~2-5 minutes
- 1M steps: ~5-10 minutes
- 2M steps: ~10-20 minutes

---

## 🐛 Known Issues & Solutions

### Issue 1: Model Uses Wrong Observation Space
**Symptom:** `ValueError: Unexpected observation shape (84, 84, 1) for Box environment, please use (4,)`

**Cause:** Model was trained with MlpPolicy (features) but environment created with CnnPolicy (images)

**Solution:**
- Ensure Colab training uses latest GitHub code
- Verify `use_raw_pixels=True` in train_gradient.py
- Check model's `data` file confirms CnnPolicy

### Issue 2: White Lines Not Visible in Videos
**Symptom:** Perfect performance but no road markings in visualization

**Cause:** Rover using geometric features, not camera vision ("cheating")

**Solution:**
- Already fixed in latest code
- Re-run training with vision-based configuration

### Issue 3: Colab Uses Cached Code
**Symptom:** Training runs but produces feature-based model

**Solution:**
- Clear Colab runtime before training
- Verify git clone shows latest commit (70da1e2 or newer)
- Check train_gradient.py shows `use_raw_pixels=True`

---

## 📊 Performance Expectations

### Feature-Based Models (Historical - "Cheating")
- **Straight Track:** 4518 reward, -0.012m offset, 2000 steps
- **Curved Track:** 4591 reward, -0.001m offset, 2000 steps
- Perfect performance using GPS-like geometric features

### Vision-Based Models (Target - Realistic)
- **Expected:** Lower initial performance (CNN must learn edge detection)
- **Goal:** Stable curved track following from raw pixels
- **Success Criteria:** 
  - Completes 1000+ steps without off-track
  - Maintains ±0.1m lateral offset
  - White lines visible and tracked in videos

---

## 🔬 Hardware Specifications

**Actor:** "Rover" - Autonomous RC Car

**Physical Measurements:**
- Total length: 24" (including bumpers)
- Wheelbase: 13" (hub-to-hub)
- Wheel track: 9.625" (left-to-right)

**Camera Setup:**
- Position: 6.5" from front hub, centered laterally
- Height: 10" above ground
- Pitch: 11.8° down (focused 48" ahead)
- FOV: 60° horizontal

**Track:**
- Two white ropes on gray surface
- Width: 0.6m (configurable 0.3-0.9m)
- Layout: Sine-wave curves for realistic driving

**Target Hardware:**
- Raspberry Pi 4 (2GB+ recommended)
- Pi Camera Module v2
- PWM servo for steering
- ESC for throttle control

---

## 💻 Development Environment

### Python Setup
- **Version:** 3.11.3
- **Environment:** Virtual environment at `.\venv\`
- **Activation:** `.\venv\Scripts\Activate.ps1`
- **Python Path:** `.\venv\Scripts\python.exe`

### Key Dependencies
```
gymnasium>=0.29.0
stable-baselines3>=2.7.1
torch>=2.9.1
pygame>=2.6.1
opencv-python>=4.12.0
numpy
```

### Git Repository
- **Remote:** https://github.com/egonthemighty/CarV1
- **Branch:** main
- **Latest Commit:** 70da1e2 (Update test scripts for vision-based model paths)

---

## 🚀 Quick Start for AI Agents

### To Run Training:
```python
# 1. Verify configuration
# Read config/config.py - should show CnnPolicy, 500k timesteps
# Read tests/train_gradient.py - should show use_raw_pixels=True

# 2. Upload to Colab
# Open CarV1_Colab_Training.ipynb in Google Colab
# Execute all cells
# Download resulting ZIP

# 3. Test locally
.\venv\Scripts\python.exe tests/test_rover_trained.py
```

### To Generate Videos:
```python
# Single episode
.\venv\Scripts\python.exe tests/generate_rover_video.py

# Learning progression (2×2 grid)
.\venv\Scripts\python.exe tests/generate_learning_comparison.py
```

### To Capture Still Images:
```python
.\venv\Scripts\python.exe tests/capture_view.py
```

---

## 📝 Important Rules & Patterns

### 1. Always Match Environment Configuration
When loading a trained model, the environment MUST match training configuration:
- If trained with `use_raw_pixels=True`, testing must also use `use_raw_pixels=True`
- Camera resolution must match (84×84)
- Policy type must match observation space

### 2. Check Model Metadata
Before using a model, verify its training configuration:
```python
# Read training output/ppo_carv1_final/data
# Look for policy_class and observation space shape
```

### 3. Git Workflow
- Always commit configuration changes before Colab training
- Push to GitHub so Colab clones latest code
- Verify Colab shows correct commit hash after clone

### 4. Training Artifacts
- Models saved to `training output/carv1_models/`
- Checkpoints every 10k steps in `checkpoints/` subfolder
- Final model: `ppo_camera_line_follow_final.zip`
- Tensorboard logs in `logs/`

---

## 🎓 Learning Resources

### Understanding the "Cheating" Problem
- **Feature-Based (MlpPolicy):** Rover receives perfect geometric information
  - `[left_line_offset, right_line_offset, heading, speed]`
  - Like having GPS coordinates of track boundaries
  - Perfect performance but won't work on real hardware
  
- **Vision-Based (CnnPolicy):** Rover receives raw camera pixels
  - 84×84 grayscale image showing sky, ground, white ropes
  - Must learn edge detection and pattern recognition
  - Harder to train but works with real Pi Camera

### Why 84×84 Resolution?
- Standard size for Atari DQN/PPO research
- Balances:
  - Computational efficiency (7,056 pixels vs 640×480 = 307,200 pixels)
  - Sufficient detail for line detection
  - Fast inference on Raspberry Pi

### Sine-Wave Track Generation
```python
# Formula (in env/first_person_env.py line 263)
curvature = amplitude * sin(frequency * distance)
# amplitude = 0.3 rad/m (max curve sharpness)
# frequency = 0.05 rad/m (curve wavelength ~125m)
```

---

## 🔮 Next Steps (Priority Order)

1. **IMMEDIATE:** Run vision-based training on Google Colab
2. **HIGH:** Test vision-based model performance
3. **HIGH:** Generate comparison videos (feature vs vision)
4. **MEDIUM:** Tune CNN architecture if performance inadequate
5. **MEDIUM:** Increase training duration if needed (1M-2M steps)
6. **LATER:** Raspberry Pi deployment code
7. **LATER:** Real hardware testing

---

## 📞 Technical Details for Debugging

### Common PowerShell Commands
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Python with venv
.\venv\Scripts\python.exe script.py

# Git status
git status

# Commit and push
git add .
git commit -m "message"
git push
```

### File Paths (Windows)
- **Workspace:** `F:\Programming\python stuff\customGym\CarV1`
- **Python:** `C:/Users/Juan O'Shea/AppData/Local/Programs/Python/Python311/python.exe`
- **Venv Python:** `.\venv\Scripts\python.exe`

### Environment Testing
```python
# Quick test environment
from env.first_person_env import FirstPersonLineFollowEnv

# Vision-based
env = FirstPersonLineFollowEnv(
    render_mode="human",
    use_raw_pixels=True,
    camera_resolution=(84, 84)
)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (84, 84, 1)

# Feature-based
env = FirstPersonLineFollowEnv(render_mode="human", use_raw_pixels=False)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (4,)
```

---

## 🤖 AI Agent Interaction Tips

### When Resuming Work:
1. Check git status first
2. Read this file for current state
3. Verify latest code is on GitHub
4. Check for uncommitted changes
5. Review `training output/` for latest models

### When Training:
1. Confirm configuration in config.py
2. Verify train_gradient.py has correct env_kwargs
3. Commit and push before Colab execution
4. Monitor Colab output for errors
5. Download and test results immediately

### When Debugging:
1. Check observation space shape matches policy type
2. Verify use_raw_pixels flag consistency
3. Read model metadata in `data` file
4. Test environment independently before model loading
5. Generate videos to visually verify behavior

### Communication Style:
- Be concise (user prefers brief responses)
- Show code over describing actions
- Report results with metrics
- Ask for clarification when ambiguous
- Avoid excessive explanations

---

**End of AI Agent Guide**
