# AW19 Quick Start Guide

## 🚀 Installation (5 minutes)

### 1. Install Dependencies
```bash
cd AW19
pip install -r requirements.txt
```

### 2. Install GPU Support (Optional - Recommended)
For NVIDIA GPUs with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation
```bash
python test_system.py
```

## ▶️ Starting the System

### Option A: Automatic (Windows)
Double-click: `start.bat`

### Option B: Manual
```bash
# Terminal 1: Main API
python main_api.py

# Terminal 2: Screen Capture
python screen_capture.py

# Terminal 3: Debug Viewer (optional)
python tk_debugging_client.py
```

## 🎯 First Use

1. **Screen Capture Window** will appear
   - Adjust sliders to select screen region
   - Click "Start Capture"
   - Red rectangle shows capture area

2. **Main API** will show:
   ```
   Device: cuda  ✓ GPU detected
   Device: cpu   ⚠ CPU only
   ```

3. **Debug Viewer** (optional):
   - Shows pipeline results in real-time
   - RGB, Pose, Mask, Depth tabs

## 📊 Quick Test

### Test ArUco Detection
1. Open this image in a browser: https://chev.me/arucogen/
2. Select: Dictionary: 4x4_50, ID: 0, Size: 200mm
3. Point screen capture at the marker
4. Check Debug Viewer → "Detected Markers" tab

### Test Model Loading
```bash
curl http://localhost:5000/models
```

### Test Pose Estimation
```bash
# Select model
curl -X POST http://localhost:5000/select_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "ball.ply"}'

# Request pose (will use pipeline mask & disparity)
curl -X POST http://localhost:5000/avp_pose \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_frame": "<base64>",
    "camera_matrix": [[800, 0, 400], [0, 800, 300], [0, 0, 1]]
  }'
```

## 🔧 Common Issues

### "CUDA not available"
- Install CUDA-enabled PyTorch (see step 2 above)
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### "Cannot connect to API"
- Ensure Main API is running
- Check: `curl http://localhost:5000/health`

### "Models not found"
- Verify `models/` folder has `.ply` files
- Check: `ls models/`

### Slow Performance
- Enable GPU support
- Reduce capture FPS in screen_capture.py
- Use smaller depth model in `computer_vision_pipeline.py`

## 📈 Performance Expectations

### With GPU (NVIDIA RTX 3060)
- Frame processing: ~20-30ms
- Depth estimation: ~50-100ms
- Total: ~30 FPS possible

### CPU Only (Intel i7)
- Frame processing: ~20-30ms
- Depth estimation: ~2-5 seconds ⚠️
- Total: ~1-2 FPS (without depth)

## 🎓 Next Steps

1. **Read README.md** for full documentation
2. **Explore endpoints** at http://localhost:5000/
3. **Integrate with AVP** using `/avp_pose` endpoint (or `/pose` when forwarding to a real API)
4. **Optimize** by tuning `computer_vision_pipeline.py` and `app_config.py`

## 📝 Key Files

- `main_api.py` - Main server (port 5000), integrated pose relay/mock
- `computer_vision_pipeline.py` - CV pipeline (GPU optimized)
- `screen_capture.py` - Capture UI
- `tk_debugging_client.py` - Debug viewer
- `app_config.py` - Central configuration (hosts, ports, defaults)
- `test_system.py` - System tests

## 💡 Tips

1. **GPU is CRITICAL** for depth estimation
2. **Start small** - test without depth first
3. **Use pipeline mask/disparity** - faster than sending your own
4. **Monitor stats** - GET /stats shows pipeline performance
5. **Debug viewer** - essential for tuning HSV parameters

## ❓ Help

- Check logs in terminal windows
- Run: `python test_system.py`
- Review: README.md for detailed info
- API docs: http://localhost:5000/ (when running)

---

**Ready to go!** 🎉
Start with: `python test_system.py`
