# How to Use VPU (Video Processing Unit) on Rockchip Platform

## 📋 Overview
This guide explains how to use Rockchip VPU hardware acceleration for video decoding using GStreamer on Ubuntu Firefly systems.

## ⚠️ Important Requirements

### 1. **Administrator Permission Required**
VPU hardware requires **sudo** privileges to access `/dev/mpp_service`:
```bash
# Check VPU device permissions
ls -la /dev/mpp_service
# Output: crw-rw---- 1 root video 241, 0 Nov  5 06:41 /dev/mpp_service

# Current user groups
groups
# Output: firefly adm disk dialout cdrom sudo audio dip video plugdev...
```

### 2. **Environment Variables for Display**
When using sudo with GUI applications:
```bash
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 [gstreamer-command]
```

## 🔧 Hardware vs Software Decoding Comparison

| **Aspect** | **Hardware (VPU)** | **Software (CPU)** |
|------------|-------------------|-------------------|
| **Decoder** | `mppvideodec` | `avdec_h264` |
| **Resource** | VPU chip | CPU cores |
| **CPU Usage** | 5-15% | 70-90% |
| **Performance** | High efficiency | Lower efficiency |
| **Permission** | Requires sudo | Normal user |
| **Power** | Lower consumption | Higher consumption |

## 🚀 GStreamer Pipeline Components

### **RTSP Source Elements**
```bash
rtspsrc location=rtsp://IP:PORT/stream protocols=tcp
```
- **rtspsrc**: RTSP client source
- **protocols=tcp**: Force TCP transport (more reliable than UDP)

### **RTP Processing**
```bash
rtph264depay ! h264parse
```
- **rtph264depay**: Extracts H.264 from RTP packets
- **h264parse**: Parses H.264 stream, adds stream info

### **Hardware Decoder (VPU)**
```bash
mppvideodec
```
- **mppvideodec**: Rockchip MPP (Media Process Platform) hardware decoder
- Supports: H.264, H.265/HEVC, VP8, VP9
- Outputs: DRM format (requires conversion)

### **Video Processing**
```bash
videoconvert ! format ! videoscale
```
- **videoconvert**: Format conversion (DRM → RGB/YUV)
- **videoscale**: Resize video frames
- **videoflip**: Rotate/flip video

### **Video Sinks (Display Options)**
```bash
# Options (try in order of preference):
ximagesink          # X11 image sink (most compatible)
xvimagesink         # X11 video sink (hardware overlay)
kmssink             # Direct DRM/KMS output
autovideosink       # Automatic selection
```

## 📝 Working Examples

### **YOLO Vehicle Detection with VPU (Recommended)**
```bash
# Multi-stream vehicle and license plate detection
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 /home/firefly/Documents/YOLO/DEMO-RTSP/V1/vehicle_cascade_multi_stream_vpu.py \
--rtsp rtsp://172.21.10.118:8554/dual \
--mode vehicle-plate \
--target-fps 8

# Visual vehicle detection with GUI
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 /home/firefly/Documents/YOLO/DEMO-RTSP/V1/vehicle_detection_rtsp_visual.py \
--rtsp rtsp://172.21.10.118:8554/dual

# Multiple camera streams
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 /home/firefly/Documents/YOLO/DEMO-RTSP/V1/vehicle_cascade_multi_stream_vpu.py \
--rtsp rtsp://172.21.10.118:8554/camera1 rtsp://172.21.10.118:8554/dual \
--mode vehicle-plate \
--target-fps 5
```

### **Basic RTSP Decode with VPU**
```bash
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
gst-launch-1.0 \
rtspsrc location=rtsp://172.21.10.118:8554/camera1 protocols=tcp ! \
rtph264depay ! \
h264parse ! \
mppvideodec ! \
videoconvert ! \
ximagesink
```
```bash
$ sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 python3 /home/firefly/Documents/YOLO/DEMO-RTSP/V1/vehicle_detection_rtsp_visual.py --rtsp rtsp://172.21.10.118:8554/dual
```

### **Save Frames to Files**
```bash
sudo gst-launch-1.0 \
rtspsrc location=rtsp://172.21.10.118:8554/camera1 protocols=tcp ! \
rtph264depay ! \
h264parse ! \
mppvideodec ! \
videoconvert ! \
jpegenc ! \
multifilesink location=frame_%05d.jpg max-files=100
```

### **Local File Decode**
```bash
sudo gst-launch-1.0 \
filesrc location=/path/to/video.mp4 ! \
qtdemux ! \
h264parse ! \
mppvideodec ! \
videoconvert ! \
ximagesink
```

### **With Audio (RTSP)**
```bash
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
gst-launch-1.0 \
rtspsrc location=rtsp://172.21.10.118:8554/camera1 protocols=tcp name=src \
src.recv_rtp_src_0 ! rtph264depay ! h264parse ! mppvideodec ! videoconvert ! ximagesink \
src.recv_rtp_src_1 ! rtpmp4adepay ! aacparse ! faad ! audioconvert ! autoaudiosink
```

## 🛠️ Available GStreamer Elements

### **Video Decoders**
```bash
# Hardware decoders (require sudo)
mppvideodec         # Rockchip MPP decoder
omxh264dec          # OpenMAX H.264 decoder
omxh265dec          # OpenMAX H.265 decoder

# Software decoders
avdec_h264          # FFmpeg H.264 decoder
avdec_h265          # FFmpeg H.265 decoder
```

### **Video Sinks**
```bash
ximagesink          # X11 software rendering
xvimagesink         # X11 hardware overlay
kmssink             # Direct kernel mode setting
glimagesink         # OpenGL rendering
gtk4paintablesink   # GTK4 widget
```

### **Video Filters**
```bash
videoconvert        # Format conversion
videoscale          # Scaling/resizing
videoflip           # Rotation/flipping
videocrop           # Cropping
gamma               # Gamma correction
```

## 🔍 Debugging and Troubleshooting

### **Check Available Decoders**
```bash
gst-inspect-1.0 | grep -i "h264\|hevc\|mpp"
```

### **Debug Pipeline**
```bash
GST_DEBUG=3 gst-launch-1.0 [your-pipeline]
```

### **Common Error Messages & Solutions**
```bash
# VPU/MPP Issues
"failed to init mpp ctx" → Use sudo for VPU access
"mpp_service: Permission denied" → Check /dev/mpp_service permissions
"No such device" → VPU driver not loaded

# Display Issues  
"Could not initialise Xv output" → Try ximagesink instead of xvimagesink
"XDG_RUNTIME_DIR not set" → Add XDG_RUNTIME_DIR=/run/user/1000
"cannot connect to X server" → Add DISPLAY=:0
"Authorization required" → Use proper sudo with env vars

# Network/RTSP Issues
"Could not connect to server" → Check RTSP URL and network connectivity
"Connection timed out" → Try protocols=tcp in rtspsrc
"Stream not found" → Verify stream name (camera1, dual, etc.)
"Network unreachable" → Check IP address and routing

# Python/Application Issues
"ModuleNotFoundError" → Activate virtual environment first
"CUDA out of memory" → Reduce batch size or use CPU mode
"Model not found" → Check .rknn model file paths
"Exit code 1" → Check application logs for specific errors
"Exit code 130" → Application interrupted (Ctrl+C)
```

## 📊 Performance Monitoring

### **Real-time System Monitoring**
```bash
# Monitor CPU/Memory usage
top -b -n1 | head -20

# Check running VPU processes
ps aux | grep -E "(python3|mpp|vpu)" | grep -v grep

# Monitor system load
watch -n 1 "uptime && free -h"
```

### **VPU Hardware Status**
```bash
# Check VPU device permissions
ls -la /dev/mpp_service

# Verify MPP service availability
sudo cat /proc/interrupts | grep -i mpp || echo "MPP interrupts not visible"

# Check recent VPU messages
sudo dmesg | tail -20 | grep -i -E "(mpp|vpu|rga|gpu)"
```

### **Performance Comparison Test**
```bash
# Test VPU performance (should show ~30-50% CPU)
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
timeout 30s python3 /path/to/vpu_app.py &
sleep 5 && top -b -n3 -p $(pgrep python3) | tail -1

# Compare with software decode (would show 70-90% CPU)
timeout 30s python3 /path/to/software_app.py &
sleep 5 && top -b -n3 -p $(pgrep python3) | tail -1
```

### **Network Stream Quality**
```bash
# Test RTSP connection
ffprobe rtsp://172.21.10.118:8554/dual

# Monitor network usage during streaming
iftop -i wlan0  # or eth0
```

### **Pipeline Statistics**
```bash
# Add to pipeline for FPS display
fpsdisplaysink video-sink=ximagesink

# Debug pipeline performance
GST_DEBUG=3 gst-launch-1.0 [your-pipeline] 2>&1 | grep -i fps
```

## 🔐 Permanent Permission Fix

### **Option 1: Add User to Video Group** (Recommended)
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

### **Option 2: Change Device Permissions**
```bash
sudo chmod 666 /dev/mpp_service
# Note: This resets after reboot
```

### **Option 3: Create udev Rule**
```bash
sudo nano /etc/udev/rules.d/99-mpp.rules
# Add: KERNEL=="mpp_service", MODE="0666"
sudo udevadm control --reload-rules
```

## 🎯 Available YOLO Applications

### **Vehicle Detection Applications**
```bash
# 1. Multi-stream VPU optimized (Best performance)
vehicle_cascade_multi_stream_vpu.py
- Supports multiple RTSP streams
- VPU hardware acceleration
- Vehicle + License plate detection
- Configurable FPS target

# 2. Visual detection with GUI
vehicle_detection_rtsp_visual.py  
- Single stream with visual output
- Real-time bounding boxes
- Performance statistics display

# 3. Standard multi-stream (Software decode)
vehicle_cascade_multi_stream.py
- CPU-based decoding
- Higher resource usage
- Fallback option

# 4. Basic cascade detection
vehicle_cascade_detection.py
- Simple vehicle detection
- Single stream processing
```

### **Command Line Examples**
```bash
# Single camera with license plate detection
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 DEMO-RTSP/V1/vehicle_cascade_multi_stream_vpu.py \
--rtsp rtsp://172.21.10.118:8554/camera1 \
--mode vehicle-plate --target-fps 10

# Dual camera setup
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 DEMO-RTSP/V1/vehicle_cascade_multi_stream_vpu.py \
--rtsp rtsp://172.21.10.118:8554/camera1 rtsp://172.21.10.118:8554/dual \
--mode vehicle-plate --target-fps 5

# Vehicle only detection (faster)
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 DEMO-RTSP/V1/vehicle_cascade_multi_stream_vpu.py \
--rtsp rtsp://172.21.10.118:8554/dual \
--target-fps 15

# Visual output with GUI
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 DEMO-RTSP/V1/vehicle_detection_rtsp_visual.py \
--rtsp rtsp://172.21.10.118:8554/dual
```

## 📚 Additional Resources

### **Supported Codecs**
- **H.264**: Baseline, Main, High profiles
- **H.265/HEVC**: Main, Main10 profiles  
- **VP8**: All profiles
- **VP9**: Profile 0, Profile 2

### **Maximum Specifications**
- **Resolution**: Up to 4K (4096x2304)
- **Frame Rate**: Up to 60fps
- **Bit Depth**: 8-bit, 10-bit (H.265)

### **Related Files**
- VPU test scripts: `/usr/local/bin/h264dec.sh`, `/usr/local/bin/h264enc.sh`
- Device file: `/dev/mpp_service`
- GStreamer plugins: `/usr/lib/aarch64-linux-gnu/gstreamer-1.0/`

## 🚀 Quick Start Guide

### **Step 1: Activate Python Environment**
```bash
cd /home/firefly/Documents/YOLO
source .venv/bin/activate
```

### **Step 2: Test VPU Access**
```bash
# Check VPU device
ls -la /dev/mpp_service
# Should show: crw-rw---- 1 root video 241, 0
```

### **Step 3: Run Application**
```bash
# Replace with your RTSP URL
sudo XDG_RUNTIME_DIR=/run/user/1000 DISPLAY=:0 \
python3 DEMO-RTSP/V1/vehicle_detection_rtsp_visual.py \
--rtsp rtsp://172.21.10.118:8554/dual
```

### **Step 4: Monitor Performance**
```bash
# In another terminal
top -b -n1 | head -10
# CPU should be 30-50% (not 70-90%)
```

## 🎯 Best Practices & Performance Tips

### **VPU Optimization**
1. **Always use sudo** for VPU hardware access
2. **Set proper environment variables** (XDG_RUNTIME_DIR, DISPLAY)
3. **Monitor CPU usage** to verify hardware acceleration (should be <50%)
4. **Use appropriate target-fps** (5-15 fps for multi-stream, up to 30 for single)

### **Network Optimization**
5. **Use TCP protocol** for RTSP streams (more reliable than UDP)
6. **Test network connectivity** before running applications
7. **Use wired connection** when possible for stable streaming
8. **Check bandwidth** - ensure sufficient network capacity

### **Display & GUI**
9. **Choose ximagesink** for display compatibility
10. **Test display access** with simple applications first
11. **Use SSH X11 forwarding** for remote access if needed

### **Application Tuning**
12. **Start with single stream** before multi-stream
13. **Adjust FPS target** based on hardware capabilities
14. **Monitor memory usage** for long-running applications
15. **Use vehicle-only mode** for better performance if license plates not needed

### **Debugging**
16. **Test with local files** first before RTSP streams
17. **Check logs** for specific error messages
18. **Use timeout commands** for testing
19. **Verify model files** (.rknn) are present and accessible

## 📈 Expected Performance

| **Configuration** | **CPU Usage** | **Memory** | **Max FPS** |
|-------------------|---------------|------------|--------------|
| Single stream + Vehicle only | 25-35% | 400-600MB | 25-30 |
| Single stream + Vehicle+Plate | 35-45% | 500-700MB | 15-20 |
| Dual stream + Vehicle+Plate | 45-65% | 600-800MB | 8-12 |
| Without VPU (Software) | 70-90% | 800MB+ | 5-10 |

---

**Note**: This guide is optimized for Rockchip RK3588/RK3399 platforms with MPP (Media Process Platform) support. Performance may vary based on stream resolution, network conditions, and system load.