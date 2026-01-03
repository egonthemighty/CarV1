# Hardware Integration Notes

## System Architecture

```
Simulation (Training)          Real Hardware (Deployment)
┌─────────────────────┐       ┌──────────────────────┐
│  Gymnasium Env      │       │  Raspberry Pi        │
│  ┌───────────────┐  │       │  ┌────────────────┐  │
│  │ Observations  │  │       │  │ Pi Camera      │  │
│  │ - Sensors (8) │  │───────│  │ - RGB Image    │  │
│  │ - Position    │  │  vs   │  │ - 640x480      │  │
│  │ - Velocity    │  │       │  └────────────────┘  │
│  └───────────────┘  │       │                      │
│         ↓           │       │         ↓            │
│  ┌───────────────┐  │       │  ┌────────────────┐  │
│  │ RL Model      │  │       │  │ RL Model       │  │
│  │ PPO           │  │───────│  │ (Optimized)    │  │
│  └───────────────┘  │       │  └────────────────┘  │
│         ↓           │       │         ↓            │
│  ┌───────────────┐  │       │  ┌────────────────┐  │
│  │ Actions       │  │       │  │ PWM Signals    │  │
│  │ - Steering    │  │───────│  │ - Servo        │  │
│  │ - Throttle    │  │       │  │ - ESC          │  │
│  └───────────────┘  │       │  └────────────────┘  │
└─────────────────────┘       └──────────────────────┘
```

## Hardware Specifications

### RC Car Platform (1:8 Scale)
- **Chassis**: 1:8 scale RC car
- **Width**: ~1 foot (~30cm)
- **Length**: ~50-60cm (approximate for 1:8 scale)
- **Wheelbase**: ~30-35cm
- **Weight**: 2-4kg (with electronics)

### Compute Module
- **Board**: Raspberry Pi (4B recommended for inference speed)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: MicroSD card, 32GB+
- **OS**: Raspberry Pi OS or Ubuntu

### Track Environment
- **Boundaries**: Two white ropes on ground
- **Track Width**: Variable, 30cm (vehicle width) to 90cm (3 feet)
- **Track Layout**: Closed loop (manipulable on the fly)
- **Track Appearance**: Infinite road from vehicle perspective
- **Challenge**: Width variations before turns (potential confusion)

### Vision System
- **Camera**: Raspberry Pi Camera Module v2 or HQ Camera
- **Resolution**: 640x480 (training), adjustable for inference
- **Frame Rate**: 30 FPS minimum
- **Field of View**: Standard ~62° diagonal
- **Target Detection**: White rope lines on ground (contrast-based)

### Control System

#### Steering
- **Type**: Servo motor (standard RC servo)
- **Signal**: PWM (50Hz typical)
- **Range**: 1000-2000μs pulse width
- **Neutral**: 1500μs (center position)
- **Mapping**: Model output [-1, 1] → [1000, 2000]μs

#### Throttle/Motor
- **Type**: Brushless motor with ESC (Electronic Speed Controller)
- **Signal**: PWM (50Hz typical)
- **Range**: 1000-2000μs pulse width
- **Neutral**: 1500μs (stop)
- **Forward**: 1500-2000μs
- **Reverse**: 1000-1500μs (if enabled)
- **Mapping**: Model output [-1, 1] → [1000, 2000]μs

### Power System
- **Battery**: LiPo 2S-3S (7.4V-11.1V)
- **Capacity**: 3000-5000mAh
- **Pi Power**: 5V regulated (from BEC or separate regulator)
- **Motor Power**: Direct from battery via ESC

## PWM Signal Mapping

### Model Output to PWM Conversion

```python
def model_to_pwm(action_value, center=1500, range_us=500):
    """
    Convert model action [-1, 1] to PWM microseconds.
    
    Args:
        action_value: Model output in range [-1, 1]
        center: Center/neutral PWM value (microseconds)
        range_us: Range in microseconds from center
    
    Returns:
        PWM pulse width in microseconds
    """
    pwm_us = center + (action_value * range_us)
    return int(np.clip(pwm_us, center - range_us, center + range_us))

# Example usage:
steering_action = 0.5   # Model output (turn right)
steering_pwm = model_to_pwm(steering_action)  # 1750μs

throttle_action = 0.3   # Model output (forward)
throttle_pwm = model_to_pwm(throttle_action)  # 1650μs
```

### Raspberry Pi PWM Implementation

Using `pigpio` library (recommended for precise PWM):
```python
import pigpio

# GPIO pins
STEERING_PIN = 17  # BCM pin numbering
THROTTLE_PIN = 18

# Initialize pigpio
pi = pigpio.pi()

# Set servo pulse widths
pi.set_servo_pulsewidth(STEERING_PIN, steering_pwm)
pi.set_servo_pulsewidth(THROTTLE_PIN, throttle_pwm)
```

## Camera Integration

### Image Preprocessing Pipeline

```python
import cv2
import numpy as np

def preprocess_camera_image(image, target_size=(84, 84)):
    """
    Preprocess Pi Camera image for model input.
    
    Args:
        image: Raw camera image (RGB)
        target_size: Target image dimensions
    
    Returns:
        Preprocessed image ready for model
    """
    # Convert to grayscale (optional, reduces data)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions if needed
    # processed = np.expand_dims(normalized, axis=(0, -1))
    
    return normalized
```

### Camera Configuration
```python
from picamera2 import Picamera2

# Initialize camera
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(config)
camera.start()

# Capture frame
frame = camera.capture_array()
```

## Sim-to-Real Transfer Strategy

### Challenge
- Simulation uses abstract distance sensors
- Real car uses camera vision
- Different observation spaces

### Solution: Camera-Based Simulation (Selected Approach)

**Simulation Design:**
- Render top-down view with two white lines (rope boundaries)
- Variable track width (30-90cm in simulation)
- Curved and straight sections
- Track width variations before turns (to train handling)
- Camera view: First-person from car position

**Key Features to Simulate:**
1. **White rope detection**: High contrast lines on darker ground
2. **Variable width**: Random width changes (30-90cm)
3. **Curves**: Various turn radii and angles
4. **Width transitions**: Widening before turns (challenge scenario)
5. **Infinite track**: Loop that appears continuous

**Implementation:**
- Use Pygame to render track with white lines
- Extract camera view from rendered scene
- Convert to numpy array as observation
- Preprocess to match Pi Camera format

## Model Optimization for Raspberry Pi

### Quantization
```python
# PyTorch quantization for faster inference
import torch

# Load trained model
model = PPO.load("carv1_final_model")

# Extract policy network
policy = model.policy

# Quantize to INT8
quantized_model = torch.quantization.quantize_dynamic(
    policy, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model, "carv1_quantized.pth")
```

### Performance Targets
- **Inference Time**: <33ms (30 FPS)
- **Latency**: <50ms end-to-end (camera → action)
- **CPU Usage**: <80% (leave headroom for other processes)

## Deployment Workflow

1. **Train in Simulation**
   - Use current Gymnasium environment
   - Train PPO model on distance sensors OR camera images
   - Achieve target performance metrics

2. **Export Model**
   - Save model in PyTorch format
   - Optimize/quantize for Raspberry Pi
   - Test inference speed on Pi

3. **Develop Hardware Interface**
   - Implement PWM control code
   - Implement camera capture code
   - Test hardware components individually

4. **Integration**
   - Combine model + hardware interface
   - Create inference loop
   - Add safety checks (collision detection, limits)

5. **Testing**
   - Controlled environment testing
   - Gradual speed increase
   - Emergency stop mechanism
   - Data collection for analysis

6. **Fine-Tuning**
   - Collect real-world performance data
   - Adjust PWM mappings if needed
   - Fine-tune model on real data (optional)
   - Iterate and improve

## Safety Considerations

### Hardware Safety
- [ ] Emergency stop mechanism (physical switch)
- [ ] Battery voltage monitoring
- [ ] Motor current limiting
- [ ] Thermal monitoring (Pi and ESC)
- [ ] Mechanical kill switch

### Software Safety
- [ ] Watchdog timer (reset if frozen)
- [ ] Action limits (max steering/throttle)
- [ ] Collision detection and stop
- [ ] Timeout mechanisms
- [ ] Graceful degradation on errors

### Testing Safety
- [ ] Test in enclosed area
- [ ] Start with reduced speed
- [ ] Remote kill switch capability
- [ ] Clear obstacle-free space
- [ ] Multiple observers during tests

## Development Tools Needed

### Hardware Tools
- Multimeter (voltage/current measurement)
- Oscilloscope (optional, for PWM verification)
- USB keyboard/mouse for Pi setup
- HDMI display for Pi debugging
- Screwdrivers and hex keys

### Software Tools
- SSH access to Raspberry Pi
- VNC for remote desktop (optional)
- Serial console for debugging
- Git for version control
- Python 3.8+ on Raspberry Pi

### Test Equipment
- Test track or marked course
- Cones or markers for boundaries
- Camera for recording tests
- Measuring tape for calibration

## Next Steps for Hardware Integration

1. [ ] Decide on sim-to-real approach (camera vs sensors)
2. [ ] Modify simulation if using camera-based approach
3. [ ] Set up Raspberry Pi with required libraries
4. [ ] Test PWM output on Pi (with oscilloscope/multimeter)
5. [ ] Test camera capture and preprocessing
6. [ ] Create hardware interface module
7. [ ] Test model inference speed on Pi
8. [ ] Integrate all components
9. [ ] Conduct initial hardware tests
10. [ ] Iterate and refine
