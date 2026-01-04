# CarV1 Current Status

**Date:** January 4, 2026  
**Time:** Session End (Pre-Reboot)

---

## ⚠️ IMMEDIATE ATTENTION REQUIRED

### Vision-Based Training NOT YET EXECUTED
The model in `training output/ppo_carv1_final/` is **FEATURE-BASED (wrong!)**

**What Happened:**
1. Discovered previous models were "cheating" with geometric features
2. Updated code to use CnnPolicy with camera images
3. Committed and pushed changes to GitHub (commit 70da1e2)
4. Reduced training to 500k timesteps for faster iteration
5. User ran training on Colab BUT it used old cached code
6. Downloaded model is still using MlpPolicy (4 features, not images)

**Next Action Required:**
Upload `CarV1_Colab_Training.ipynb` to Google Colab and run again to get vision-based model.

---

## 📊 Git Status

**Branch:** main  
**Remote:** https://github.com/egonthemighty/CarV1  
**Latest Commit:** 70da1e2 - "Update test scripts for vision-based model paths"  
**Status:** Clean, all changes committed and pushed

**Recent Commits:**
- `70da1e2` - Update test scripts for vision-based model paths
- `5425427` - Reduce training timesteps to 500k for troubleshooting
- `4b6ad02` - Switch to vision-based training with camera images

---

## 🎯 Configuration State

### config/config.py
```python
'policy_type': 'CnnPolicy'           # ✅ Correct
'total_timesteps': 500_000           # ✅ Reduced for troubleshooting
```

### tests/train_gradient.py
```python
env_kwargs = {
    "render_mode": None,
    "use_raw_pixels": True,          # ✅ Correct
    "camera_resolution": (84, 84)    # ✅ Correct
}
```

### tests/generate_rover_video.py
```python
model_path = "training output/ppo_carv1_final.zip"    # ✅ Updated
env = FirstPersonLineFollowEnv(
    render_mode="rgb_array",
    use_raw_pixels=True,                               # ✅ Added
    camera_resolution=(84, 84)                         # ✅ Added
)
```

### tests/generate_learning_comparison.py
```python
# Updated to use checkpoints at 100k, 250k, 350k, 500k
checkpoint_steps = [100000, 250000, 350000, 500000]
# Updated model path to ppo_carv1_final
```

---

## 📁 File System State

### Clean
- All source code committed and pushed
- No uncommitted changes in critical files

### Ignored (Not Tracked)
```
training output/ppo_carv1_final/        # Latest model (feature-based - wrong)
training output/ppo_carv1_final.zip     # ZIP version of above
training output/old/                    # Archived old models
training output/videos/                 # Generated visualization videos
```

### Modified But Not Committed
```
training output/videos/rover_episode_1.mp4    # Latest test video
```

---

## 🔄 What Works Right Now

### Environment
- FirstPersonLineFollowEnv fully functional
- Curved track generation working
- Vision mode (use_raw_pixels=True) implemented
- Feature mode (use_raw_pixels=False) working

### Testing Scripts
- Video generation scripts updated
- Test scripts updated for new paths
- All scripts functional

### Training Pipeline
- Configuration ready for vision-based training
- Colab notebook ready (needs execution)
- Checkpoint system configured

---

## ❌ What Doesn't Work

### Current Model (ppo_carv1_final)
- Trained with MlpPolicy (features) instead of CnnPolicy (images)
- Cannot be loaded with vision-based environment
- Observation space mismatch: expects (4,) but gets (84, 84, 1)

### Video Generation
- Attempted to generate video with current model
- Failed due to observation space mismatch
- Need vision-based model to proceed

---

## 📋 Task Checklist

### Completed ✅
- [x] Identify "cheating" problem (feature-based training)
- [x] Update config.py to CnnPolicy
- [x] Update train_gradient.py with use_raw_pixels=True
- [x] Update test scripts for new model paths
- [x] Reduce timesteps to 500k for troubleshooting
- [x] Commit all changes to git
- [x] Push to GitHub
- [x] Create comprehensive AI agent documentation

### Pending ⏳
- [ ] Run vision-based training on Google Colab
- [ ] Download vision-based model
- [ ] Test vision-based model performance
- [ ] Generate single episode video
- [ ] Generate 2×2 comparison video (100k/250k/350k/500k)
- [ ] Compare performance: feature-based vs vision-based
- [ ] Decide if training duration needs increase (500k → 1M or 2M)

### Future 🔮
- [ ] Tune CNN architecture if needed
- [ ] Implement Raspberry Pi deployment code
- [ ] Real hardware testing

---

## 💡 Key Insights from Session

### Discovery: The "Cheating" Problem
**Observation:** User noticed white lines weren't visible in generated videos despite perfect performance.

**Investigation:** Revealed Rover was using `_extract_features()` returning perfect geometric data:
```python
[left_line_offset, right_line_offset, heading, speed]
```

**Impact:** Model wouldn't work with real Pi Camera (no GPS-like features available).

**Solution:** Switched to true camera vision with 84×84 grayscale images.

### Why Vision-Based is Harder
- **Feature-based:** Direct geometric information (GPS-like)
  - Easy to learn (linear relationships)
  - Perfect accuracy possible
  - Unrealistic for hardware

- **Vision-based:** Raw pixel data
  - Must learn edge detection from scratch
  - CNN must discover white line patterns
  - More training data needed (500k-2M steps)
  - Realistic for Pi Camera deployment

---

## 🎓 Lessons for Future Sessions

1. **Always verify what the agent actually observes**
   - Don't assume configuration matches reality
   - Check observation shape and content
   - Validate with visualizations

2. **Git-Colab-Test workflow critical**
   - Commit → Push → Colab Clone → Train → Download → Test
   - Verify Colab uses latest commit
   - Check model metadata before testing

3. **Video analysis is powerful debugging**
   - User spotted the problem by watching videos
   - Visual verification caught what metrics didn't show
   - Generate videos early and often

4. **Simulation-to-reality gap matters**
   - Perfect sim performance can hide problems
   - Match training observations to hardware sensors
   - Test realism of training setup

---

## 📞 Quick Reference

### File Locations
- **Environment:** `env/first_person_env.py`
- **Config:** `config/config.py`
- **Training Script:** `tests/train_gradient.py`
- **Colab Notebook:** `CarV1_Colab_Training.ipynb`

### Key Values
- **Observation Shape (Vision):** (84, 84, 1) uint8
- **Observation Shape (Features):** (4,) float32
- **Action Shape:** (2,) float32 [steering, throttle]
- **Training Steps:** 500,000 (~2-5 min on T4 GPU)

### Commands
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run script
.\venv\Scripts\python.exe tests/test_rover_trained.py

# Git operations
git status
git add .
git commit -m "message"
git push
```

---

## 🔐 Final System State

**Git:** Clean, latest code pushed (70da1e2)  
**Config:** Vision-based training ready (CnnPolicy, use_raw_pixels=True)  
**Model:** Feature-based (wrong) - needs replacement  
**Next Step:** Execute Colab training to get vision-based model  
**Documentation:** Complete (AI_AGENT_README.md + this file)  

**Safe to reboot. All work preserved.**

---

**End of Status Report**
