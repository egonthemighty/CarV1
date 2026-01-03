# Environment Redesign: Camera-Based Line Following

## Overview
Complete redesign of the gymnasium environment to match real-world hardware setup.

## Current vs. New Design

### Current Environment (v1)
- ❌ Abstract distance sensors (8 rays)
- ❌ Position/velocity observations
- ❌ Empty environment (just walls)
- ❌ Not compatible with real hardware

### New Environment (v2 - Camera-Based)
- ✅ Camera image observations (RGB/grayscale)
- ✅ White rope boundary lines
- ✅ Variable track width (30-90cm)
- ✅ Realistic for sim-to-real transfer

## Specifications

### Vehicle
- **Width**: 30cm (1 foot)
- **Camera Mount**: Front-facing, angled down to see track
- **View Distance**: ~1-2 meters ahead

### Track
- **Boundaries**: Two white ropes on dark/contrasting ground
- **Width Range**: 30cm (tight) to 90cm (wide)
- **Width Transitions**: Gradual and abrupt (especially before turns)
- **Layout**: Closed loop, manipulable (appears infinite)
- **Sections**: Straight, curved (various radii), width transitions

### Training Challenges
1. **Variable Width**: Adapt to changing track width
2. **Turn Detection**: Recognize curves despite width changes
3. **Width Before Turns**: Handle confusing pre-turn widening
4. **Centering**: Stay centered when track is wide
5. **Edge Detection**: Distinguish rope from background

## Observation Space Design

### Option 1: Raw Camera Image (Realistic)
```python
observation_space = spaces.Box(
    low=0, 
    high=255, 
    shape=(84, 84, 1),  # Grayscale, downsampled
    dtype=np.uint8
)
```

**Pros:**
- Realistic, matches real hardware
- Forces model to learn visual features
- Better sim-to-real transfer

**Cons:**
- Slower training (more complex input)
- Requires CNN architecture
- More data needed

### Option 2: Preprocessed Features (Hybrid)
```python
observation_space = spaces.Box(
    low=np.array([0, 0, -1, -1, 0, 0, 0]),  # [left_line_dist, right_line_dist, left_angle, right_angle, width, curvature, speed]
    high=np.array([90, 90, 1, 1, 90, 1, 100]),
    dtype=np.float32
)
```

**Pros:**
- Faster training
- Works with MLP (current architecture)
- Easier to debug

**Cons:**
- Less realistic
- Feature extraction must work on real images
- May not transfer as well

### Recommended: Start with Option 2, Transition to Option 1

## Environment Architecture

### Class Structure
```python
class CameraLineFollowEnv(gym.Env):
    def __init__(
        self,
        window_size=(800, 600),
        camera_resolution=(84, 84),
        use_raw_pixels=False,  # False for features, True for raw images
        track_width_range=(30, 90),  # cm
        track_complexity=1,  # 1-5, affects curves and transitions
        render_mode=None
    ):
        # Initialize track generator
        # Initialize car physics
        # Set up observation/action spaces
```

### Track Generation
```python
class TrackGenerator:
    def generate_track(self, complexity=1, length=100):
        """
        Generate procedural track with white rope boundaries.
        
        Returns:
            List of track segments with positions and widths
        """
        segments = []
        
        # Mix of straight and curved sections
        # Variable widths (30-90cm)
        # Width transitions before curves (challenge)
        
        return segments
```

### Vision Processing
```python
def get_camera_observation(self):
    """Extract camera view from rendered scene."""
    # Get car position and angle
    # Render track view from car's perspective
    # Extract region of interest
    # Preprocess (resize, normalize)
    return camera_view

def extract_line_features(self, image):
    """Extract line position features from camera image."""
    # Edge detection
    # Find white lines
    # Calculate distances and angles
    # Estimate track width and curvature
    return features
```

## Reward Function

### Components
```python
def calculate_reward(self):
    reward = 0.0
    
    # 1. Stay on track (primary objective)
    if off_track:
        reward -= 10.0
        
    # 2. Center positioning (when track is wide)
    deviation_from_center = abs(position - track_center)
    reward -= deviation_from_center * 0.01
    
    # 3. Forward progress
    reward += speed * 0.1
    
    # 4. Smooth driving (penalize jerky movements)
    reward -= abs(steering_change) * 0.05
    reward -= abs(throttle_change) * 0.02
    
    # 5. Bonus for staying on track
    if on_track:
        reward += 0.5
    
    return reward
```

### Success Criteria
- Stay between ropes for 1000 steps (no crashes)
- Maintain average speed > 30cm/s
- Handle width transitions smoothly
- Navigate curves without leaving track

## Implementation Plan

### Phase 1: Basic Track Following
1. Create simple straight track with parallel white lines
2. Fixed width (60cm)
3. Feature-based observations (Option 2)
4. Train basic line following

### Phase 2: Add Curves
1. Add curved track sections
2. Variable curve radii
3. Train on mixed straight/curved tracks

### Phase 3: Variable Width
1. Implement variable track width (30-90cm)
2. Random width changes
3. Train to handle width variations

### Phase 4: Width Transitions (Challenge)
1. Add width changes before turns
2. Abrupt and gradual transitions
3. Train to distinguish widening from track end

### Phase 5: Raw Image Input
1. Switch to raw pixel observations (Option 1)
2. Implement CNN policy
3. Retrain on visual input
4. Prepare for real hardware deployment

## Testing Strategy

### Simulation Tests
- [ ] Track generation creates valid paths
- [ ] Car stays on track with random actions
- [ ] Line detection works from camera view
- [ ] Reward function provides good signal
- [ ] Model learns basic line following

### Sim-to-Real Validation
- [ ] Test on various lighting conditions (simulation)
- [ ] Test with rope texture variations
- [ ] Test with background variations
- [ ] Compare sim/real camera views
- [ ] Validate feature extraction on real images

## Model Architecture

### For Feature-Based (Phase 1-4)
```python
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

### For Raw Images (Phase 5)
```python
policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(pi=[128], vf=[128])]
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
```

## Next Steps

1. **Design track rendering system** (Pygame with white lines)
2. **Implement feature extraction** (line detection from rendered scene)
3. **Create new environment class** (CameraLineFollowEnv)
4. **Test with random agent** (verify environment works)
5. **Train Phase 1 model** (straight track, fixed width)
6. **Iterate through phases** (curves, variable width, transitions)
7. **Switch to raw images** (Phase 5)
8. **Prepare for hardware** (Raspberry Pi deployment)

## Questions to Consider

1. **Camera angle**: How steep should camera be angled? (affects view distance)
2. **Image preprocessing**: Grayscale only or keep RGB? (rope might have color)
3. **Track complexity**: Start simple or train on complex from beginning?
4. **Speed**: Fix speed initially or let model control?
5. **Background**: Uniform or varied? (affects visual learning)
