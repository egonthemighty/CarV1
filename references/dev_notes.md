# Development Tasks and Notes

## Hardware Specifications (Version 1)

### Physical Platform
- **Scale**: 1:8 RC car chassis (~1 foot / 30cm wide)
- **Compute**: Raspberry Pi (inference)
- **Camera**: Raspberry Pi Camera Module (vision-based navigation)

### Track Environment
- **Boundaries**: Two white ropes on ground
- **Track Width**: Variable 30cm (min) to 90cm (max)
- **Layout**: Manipulable closed loop (appears infinite)
- **Challenge**: Width variations especially before turns

### Inputs (Sensors)
- Raspberry Pi Camera (primary vision sensor)
  - Detects white rope boundaries
  - Processes track position and curvature
- *Future*: Additional sensors (IMU, encoders, ultrasonic, etc.)

### Outputs (Controls)
- **Steering**: Left/Right (PWM signal)
- **Throttle**: Forward/Backward (PWM signal)

### Control Interface
- Pulse-Width Modulation (PWM) for both steering and throttle
- Will require mapping from model output [-1, 1] to PWM duty cycle

### Sim-to-Real Considerations
- Current simulation uses abstract sensors (8 distance sensors)
- Real hardware uses camera vision
- Need to bridge gap between simulation and reality:
  - Option 1: Train with camera images in simulation
  - Option 2: Use distance sensors on real car to match simulation
  - Option 3: Transfer learning from sim to real with domain adaptation

## Current Status
Project infrastructure complete. Ready for development and training.

## Immediate Next Steps

### 1. Environment Setup
- [x] Create virtual environment: `python -m venv venv`
- [x] Activate: `.\venv\Scripts\activate` (Windows)
- [x] Install dependencies: `pip install -r requirements.txt`

**⚠️ IMPORTANT - Always use venv's python directly:**
```bash
.\venv\Scripts\python.exe tests/train_camera.py
.\venv\Scripts\python.exe -m pip install <package>
```

### 2. Validation
- [ ] Run environment tests: `python tests/test_env.py`
- [ ] Visual check: `python tests/visualize_env.py`
- [ ] Fix any issues discovered

### 3. Initial Training
- [ ] Review config settings in `config/config.py`
- [ ] Start first training run: `python tests/train.py`
- [ ] Monitor with TensorBoard: `tensorboard --logdir debug_output/tensorboard`

## Known Limitations

### Current Implementation
1. **Simple Physics**: Basic Newtonian physics, no tire model
2. **No Track**: Just walls, no actual track to follow
3. **Basic Sensors**: Ray-casting only checks walls
4. **No Obstacles**: Empty environment besides walls

### Environment Features to Add

#### Priority 1 (Core Functionality - Camera-Based)
- [ ] **Rebuild environment with camera observations** (image input)
- [ ] **Render white rope boundaries** (simulate real track)
- [ ] **Variable track width** (30-90cm random variations)
- [ ] **Track with curves and straights**
- [ ] **Reward function**: Stay centered between ropes
- [ ] **Penalty for leaving track** (crossing rope boundaries)
- [ ] **Width variation before turns** (challenge scenario)
- [ ] **PWM signal output mapping** (model → hardware)

#### Priority 2 (Enhanced Realism)
- [ ] More realistic car physics (friction model)
- [ ] Different track layouts
- [ ] Configurable difficulty levels
- [ ] Collision with track boundaries (not instant termination)
- [ ] **Vision-based sensor simulation** (match Pi Camera)
- [ ] **1:8 scale physics parameters**

#### Priority 3 (Advanced Features & Hardware Integration)
- [ ] Multiple car types
- [ ] Dynamic obstacles
- [ ] Weather conditions affecting physics
- [ ] Multi-agent racing
- [ ] **Raspberry Pi deployment code**
- [ ] **PWM hardware interface layer**
- [ ] **Camera preprocessing pipeline**
- [ ] **Real-time inference optimization**

### Training Improvements
- [ ] Implement curriculum learning
- [ ] Add more evaluation metrics
- [ ] Create visualization of learned behavior
- [ ] Implement early stopping based on performance
- [ ] Add model comparison tools

### Code Quality
- [ ] Add unit tests for physics
- [ ] Add integration tests
- [ ] Improve documentation
- [ ] Add type hints
- [ ] Code formatting with Black

## Experiments to Try

### Reward Function Variations
1. **Speed-focused**: Higher weight on velocity
2. **Safety-focused**: Higher penalty for wall proximity
3. **Efficiency-focused**: Strong time penalties
4. **Smooth-driving**: Penalize jerky movements

### Architecture Experiments
1. Different network sizes (64-64 vs 256-256)
2. Different activation functions
3. CNN-based policy (if adding vision)
4. Recurrent policy (LSTM) for temporal dependencies

### Hyperparameter Search
- Learning rate: [1e-5, 3e-4, 1e-3]
- Batch size: [32, 64, 128, 256]
- Number of steps: [512, 1024, 2048, 4096]
- Entropy coefficient: [0.0, 0.001, 0.01, 0.1]

## Performance Benchmarks

### Target Metrics
- [ ] Achieve 100+ average episode reward
- [ ] Complete 1000 steps without collision
- [ ] Maintain speed > 50 units/sec
- [ ] Train to competence in < 500k timesteps

### Current Results
*To be filled after initial training*

## Debug Notes

### Common Issues
*Document issues encountered and solutions*

### Performance Metrics
*Track training times, convergence rates, etc.*

## Ideas for Extensions

### Short Term
- Add different car colors/skins
- Implement speed display in render
- Add mini-map showing full trajectory
- Export episodes as videos
- **PWM calibration tool**
- **Camera image preprocessing utilities**

### Medium Term
- Create procedural track generator
- Add split times and lap records
- Implement ghost car (previous best)
- Multi-agent competitive racing
- **Vision-based observation space**
- **Raspberry Pi model optimization (quantization)**
- **Hardware-in-the-loop testing**

### Long Term
- 3D rendering with PyBullet or similar
- VR support for human testing
- **Real-world sim-to-real transfer**
- **Integration with 1:8 scale RC car**
- **Field testing and validation**
- **Camera-based obstacle detection**

## Resources and References

### Papers
- PPO: https://arxiv.org/abs/1707.06347
- DQN: https://arxiv.org/abs/1312.5602
- SAC: https://arxiv.org/abs/1801.01290

### Similar Projects
- Gym Racing: https://github.com/gym-racing
- CarRacing-v0: Built-in Gym environment
- CARLA Simulator: https://carla.org/

### Tutorials
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- RL Course: https://spinningup.openai.com/

## Cloud Training Setup
**Google Colab** (Primary Platform - Recommended)
- Free Tesla T4 GPU available
- Training time: 5-10 minutes for 1M timesteps (vs 30-60 min on CPU)
- See: `CarV1_Colab_Training.ipynb` for complete workflow
- Just upload and click "Run All"!

**Local Training** (Fallback)
- Use: `.\venv\Scripts\python.exe tests/train_camera.py`
- CPU only, slower but works

## Meeting Notes / Decisions
*Space for tracking design decisions and changes*

### 2026-01-03 - Training Setup
- ✅ Switched from Paperspace Gradient to Google Colab
- ✅ Created complete Colab notebook for GPU training
- ✅ Colab provides free GPU, easier setup, better UX

### 2026-01-03 - CRITICAL: Camera Perspective Issue ⚠️
**PROBLEM IDENTIFIED**: Major mismatch between training and deployment!
- **Current training**: TOP-DOWN camera view (bird's eye)
- **Real hardware**: FORWARD-FACING camera (first-person from car)
- **Impact**: Model won't work on real car - completely different visual input

**Solutions**:
1. **Retrain with first-person view** (RECOMMENDED) ⭐
   - Update CameraLineFollowEnv to render from car's perspective
   - Camera sees road ahead at ~30-45° angle, not full track
   - Matches real Pi Camera mounting position
   
2. **Mount camera for top-down** (NOT PRACTICAL)
   - Would need camera on tall pole/gimbal
   - Doesn't match intended hardware setup
   
3. **Transfer learning** (FALLBACK)
   - Fine-tune trained model on real camera footage
   - Requires physical track and training runs

**ACTION REQUIRED**: Update environment rendering before next training run!

---

**Last Updated**: 2026-01-03
**Project Start**: 2026-01-03
**Status**: Camera perspective needs redesign before hardware deployment
