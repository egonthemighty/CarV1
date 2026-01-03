# 🚀 Quick Start: Training on Google Colab

## Option 1: Using GitHub (Easiest)

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for Colab training"
   git push
   ```

2. Open [Google Colab](https://colab.research.google.com/)

3. Upload `CarV1_Colab_Training.ipynb`

4. Enable GPU: **Runtime → Change runtime type → GPU**

5. In the notebook, update cell 8 with your repo URL and uncomment the git clone lines

6. Run all cells!

## Option 2: Using ZIP Upload

1. Run `create_colab_package.bat` to create `CarV1.zip`

2. Open [Google Colab](https://colab.research.google.com/)

3. Upload `CarV1_Colab_Training.ipynb`

4. Enable GPU: **Runtime → Change runtime type → GPU**

5. Run cells 1-5, then upload `CarV1.zip` when prompted in cell 10

6. Continue running remaining cells!

## What Gets Packaged

The ZIP includes:
- ✅ `env/` - Your custom environment
- ✅ `tests/` - Training scripts
- ✅ `config/` - Configuration files
- ✅ `utils/` - Helper utilities
- ✅ `references/` - Documentation
- ✅ `requirements.txt` - Dependencies

The ZIP excludes:
- ❌ `venv/` - Virtual environment (not needed)
- ❌ `.git/` - Git history (too large)
- ❌ `__pycache__/` - Python cache files
- ❌ `models/` - Old trained models

## Training Time

| Device | Time for 1M steps |
|--------|-------------------|
| Colab GPU (T4) | 5-10 minutes |
| Local CPU | 30-60 minutes |

## After Training

Download `carv1_models.zip` from the last cell - it contains:
- `best_model/` - Best performing checkpoint
- `ppo_camera_line_follow_final.zip` - Final trained model
- `checkpoints/` - Intermediate saves

Extract locally and test with:
```bash
python tests/visualize_camera_env.py
```

## Troubleshooting

**"No GPU detected"**: Enable GPU in Runtime → Change runtime type

**"Files missing"**: Make sure you're in the CarV1 directory after uploading

**"Import errors"**: Cell 5 installs packages - wait for it to complete

**Need more training**: Edit `config/config.py` and increase `total_timesteps` to 5M or 10M
