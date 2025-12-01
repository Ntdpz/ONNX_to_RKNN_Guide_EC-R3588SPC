# 🔥 VPU Specification & Usage Guide

## 📋 Table of Contents
1. [VPU Overview](#vpu-overview)
2. [Hardware Specifications](#hardware-specifications)
3. [Supported Formats](#supported-formats)
4. [How VPU Works](#how-vpu-works)
5. [Usage Methods](#usage-methods)
6. [Limitations & Requirements](#limitations--requirements)
7. [Performance Comparison](#performance-comparison)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 VPU Overview

**VPU (Video Processing Unit)** คือ hardware accelerator สำหรับ encode/decode วิดีโอบน RK3588

### Key Features:
- ✅ **Hardware-accelerated** video decode/encode
- ✅ **Low CPU usage** (< 5% vs 40-60% software)
- ✅ **Low power consumption**
- ✅ **High throughput** (4K @ 60fps)
- ❌ **Limited format support** (ไม่รองรับทุก format)

---

## 🖥️ Hardware Specifications

### RK3588 VPU Capabilities:

#### **Decode Support:**
| Codec | Max Resolution | Max FPS | Bitrate |
|-------|---------------|---------|---------|
| H.264 (AVC) | 8192x8192 | 60fps | Up to 200Mbps |
| H.265 (HEVC) | 8192x8192 | 60fps | Up to 200Mbps |
| VP9 | 8192x8192 | 60fps | Up to 200Mbps |
| VP8 | 1920x1080 | 60fps | - |
| AV1 | 8192x8192 | 60fps | - |

#### **Encode Support:**
| Codec | Max Resolution | Max FPS |
|-------|---------------|---------|
| H.264 (AVC) | 8192x8192 | 60fps |
| H.265 (HEVC) | 8192x8192 | 60fps |

#### **Hardware Components:**
- **MPP (Media Process Platform)**: Rockchip's media framework
- **RGA (Raster Graphic Acceleration)**: Hardware for format conversion, scaling
- **DRM (Direct Rendering Manager)**: Zero-copy memory management

---

## 📐 Supported Formats

### ✅ **Video Formats VPU รองรับ:**

#### **Pixel Formats (Input):**
```
✅ yuv420p       - Standard range YUV 4:2:0 (16-235)
✅ nv12          - Semi-planar YUV (VPU native output)
✅ nv16          - Semi-planar YUV 4:2:2
✅ yuyv          - Packed YUV 4:2:2
❌ yuvj420p      - JPEG full range (0-255) ← ไม่รองรับ!
❌ rgb24         - RGB format ← ต้อง convert ก่อน
```

#### **H.264 Profiles:**
```
✅ Baseline Profile
✅ Main Profile  
✅ High Profile
❌ High 10 Profile (10-bit)
❌ High 422 Profile
```

#### **Color Range:**
```
✅ tv/limited range   - Y: 16-235, UV: 16-240 (Standard)
❌ pc/full range      - Y: 0-255, UV: 0-255 (JPEG style)
```

#### **Stream Format:**
```
✅ byte-stream        - Annex-B (00 00 00 01 start codes)
✅ avc                - MP4/MKV container format
⚠️ Requires SPS/PPS  - In-band or out-of-band
```

---

## ⚙️ How VPU Works

### **Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Application (Python/C++)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  GStreamer / FFmpeg                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  rtspsrc     │ → │  h264parse   │ → │ mppvideodec  │   │
│  └──────────────┘   └──────────────┘   └──────┬───────┘   │
└───────────────────────────────────────────────┼─────────────┘
                                                 │
┌────────────────────────────────────────────────▼─────────────┐
│                         MPP Library                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ H.264 Parser │ → │  VPU Driver  │ → │ DMA Buffer   │    │
│  └──────────────┘   └──────────────┘   └──────┬───────┘    │
└───────────────────────────────────────────────┼──────────────┘
                                                 │
┌────────────────────────────────────────────────▼─────────────┐
│                    Hardware VPU (RK3588)                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │   Entropy    │ → │   Inverse    │ → │     Loop     │    │
│  │   Decoder    │   │   Transform  │   │   Filter     │    │
│  └──────────────┘   └──────────────┘   └──────┬───────┘    │
└───────────────────────────────────────────────┼──────────────┘
                                                 │
                                                 ▼
                                          NV12 Frame Data
                                          (DMA Memory)
```

### **Processing Flow:**

#### **1. Input Stage:**
```
RTSP Stream → Network → rtspsrc → RTP packets
```

#### **2. Demux/Parse:**
```
RTP packets → rtph264depay → H.264 NAL units
H.264 NAL → h264parse → Add SPS/PPS, alignment
```

#### **3. VPU Decode:**
```
H.264 stream → mppvideodec → VPU hardware
VPU → NV12 frames → DMA buffer (GPU memory)
```

#### **4. Color Conversion:**
```
NV12 (DMA) → RGA hardware → BGR/RGB
DMA → CPU memory → NumPy array
```

#### **5. Display/Process:**
```
BGR array → OpenCV → Display
BGR array → RKNN NPU → Inference
```

---

## 🛠️ Usage Methods

### **Method 1: GStreamer (Recommended)**

#### **Basic Pipeline:**
```bash
gst-launch-1.0 \
  rtspsrc location="rtsp://IP:PORT/stream" protocols=tcp ! \
  rtph264depay ! \
  h264parse config-interval=-1 ! \
  video/x-h264,stream-format=byte-stream,alignment=au ! \
  mppvideodec ! \
  videoconvert ! \
  autovideosink
```

#### **Python GStreamer:**
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

pipeline_str = (
    "rtspsrc location=rtsp://IP:PORT/stream protocols=tcp ! "
    "rtph264depay ! "
    "h264parse config-interval=-1 ! "
    "video/x-h264,stream-format=byte-stream,alignment=au ! "
    "mppvideodec ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true"
)

pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name('sink')
pipeline.set_state(Gst.State.PLAYING)

# Get frames
sample = appsink.emit('pull-sample')
buffer = sample.get_buffer()
# ... process frame
```

---

### **Method 2: FFmpeg with rkmpp**

#### **Check rkmpp support:**
```bash
ffmpeg -decoders | grep rkmpp
```

Output:
```
V..... h264_rkmpp           Rockchip MPP H.264 decoder
V..... hevc_rkmpp           Rockchip MPP HEVC decoder
```

#### **Decode with FFmpeg:**
```bash
ffmpeg -c:v h264_rkmpp \
  -i rtsp://IP:PORT/stream \
  -vf hwdownload,format=nv12,format=yuv420p \
  -f rawvideo output.yuv
```

⚠️ **Note**: FFmpeg h264_rkmpp outputs `AV_PIX_FMT_DRM_PRIME` (DRM memory)
- Must use `hwdownload` filter to copy to CPU memory
- Format conversion: DRM → NV12 → YUV420P/BGR

---

### **Method 3: Direct MPP API (C/C++)**

```c
#include <rockchip/rk_mpi.h>

// Initialize MPP
MppCtx ctx;
MppApi *mpi;
mpp_create(&ctx, &mpi);
mpi->control(ctx, MPP_SET_INPUT_BLOCK, MPP_POLL_BLOCK);

// Decode setup
MppParam param;
param.type = MPP_VIDEO_CodingAVC; // H.264
mpp_init(ctx, MPP_CTX_DEC, param.type);

// Feed H.264 packets
MppPacket packet;
mpp_packet_init(&packet, data, size);
mpi->decode_put_packet(ctx, packet);

// Get decoded frames
MppFrame frame;
mpi->decode_get_frame(ctx, &frame);
// ... process NV12 frame
```

---

## ⚠️ Limitations & Requirements

### **🚫 VPU ไม่รองรับ:**

#### **1. JPEG Color Range (yuvj420p)**
```
❌ Problem: RTSP stream uses yuvj420p (full range 0-255)
✅ Solution: 
   - Option 1: Use software decode (avdec_h264)
   - Option 2: Change encoder to yuv420p (limited range 16-235)
   - Option 3: SW decode → VPU re-encode (not recommended)
```

#### **2. 10-bit Video**
```
❌ H.264 High 10 Profile
❌ H.265 Main 10 Profile
✅ Solution: Use software decode or convert to 8-bit
```

#### **3. RGB Direct Input**
```
❌ VPU only accepts YUV formats
✅ Solution: Convert RGB → YUV with RGA or videoconvert
```

#### **4. Direct Buffer Access**
```
❌ VPU outputs to DMA/DRM memory (cannot map directly in Python)
✅ Solution: Use videoconvert to copy to system memory
```

### **✅ Requirements:**

#### **For RTSP Decode:**
1. **Stream Requirements:**
   - H.264 Baseline/Main/High Profile
   - Standard color range (yuv420p, NOT yuvj420p)
   - Valid SPS/PPS in stream
   - Byte-stream format or AVC format

2. **GStreamer Elements:**
   ```bash
   # Check if installed
   gst-inspect-1.0 mppvideodec
   gst-inspect-1.0 mppvideoenc
   gst-inspect-1.0 videoconvert
   ```

3. **Kernel Modules:**
   ```bash
   lsmod | grep rockchip
   # Should see: rockchip_vpu, rockchip_rga
   ```

4. **Permissions:**
   ```bash
   ls -l /dev/dri/renderD*
   ls -l /dev/mpp_service
   # User should have access
   ```

---

## 📊 Performance Comparison

### **CPU Usage:**

| Method | CPU Usage (1080p@30fps) | Power |
|--------|------------------------|-------|
| 🔥 VPU (mppvideodec) | 5-10% | Low |
| 🖥️ Software (avdec_h264) | 40-60% | High |
| 🖥️ Software (OpenCV) | 80-100% | Very High |

### **Throughput:**

| Resolution | VPU FPS | Software FPS |
|-----------|---------|--------------|
| 1080p | 60+ | 30-40 |
| 4K | 60 | 10-15 |
| 8K | 30 | 2-5 |

### **Multi-Stream:**

| Streams | VPU (1080p each) | Software |
|---------|------------------|----------|
| 1 stream | 5% CPU | 40% CPU |
| 2 streams | 8% CPU | 80% CPU |
| 4 streams | 15% CPU | 200%+ (impossible) |

**⚡ VPU is essential for multi-stream processing!**

---

## 🔧 Troubleshooting

### **Problem 1: Black frames / Mean = 0.00**

**Possible Causes:**
- ❌ Stream is yuvj420p (JPEG color range)
- ❌ Missing SPS/PPS
- ❌ Stream format not byte-stream
- ❌ Camera sending black frames

**Debug:**
```bash
# Check stream format
ffprobe -v error -show_entries stream=pix_fmt,profile -rtsp_transport tcp rtsp://IP:PORT/stream

# If output shows yuvj420p:
pix_fmt=yuvj420p  ← Problem!

# Solution: Use software decode or fix encoder
```

**Fix Options:**
```python
# Option A: Software decode
pipeline = "... avdec_h264 ! videoconvert ! ..."

# Option B: Fix at encoder (if you control RTSP server)
ffmpeg -i input -pix_fmt yuv420p -color_range tv -f rtsp rtsp://...
```

---

### **Problem 2: Failed to map buffer**

**Cause:** VPU outputs to DMA memory, cannot map directly

**Solution:** Already in pipeline - `videoconvert` copies to system memory
```python
pipeline = "... mppvideodec ! videoconvert ! video/x-raw,format=BGR ! ..."
```

---

### **Problem 3: Pipeline fails to start**

**Check:**
```bash
# 1. VPU module loaded?
lsmod | grep rockchip

# 2. GStreamer plugin installed?
gst-inspect-1.0 mppvideodec

# 3. Permissions?
groups  # Should include 'video' or 'render'

# 4. Test simple pipeline
gst-launch-1.0 videotestsrc ! mppvideoenc ! mppvideodec ! autovideosink
```

---

### **Problem 4: Low FPS / Stuttering**

**Optimization:**
```python
# Use these parameters:
pipeline_str = (
    "rtspsrc location=... latency=100 protocols=tcp ! "  # Low latency
    "rtph264depay ! "
    "h264parse ! "
    "mppvideodec ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink emit-signals=true "
    "max-buffers=1 "      # Only keep latest frame
    "drop=true "          # Drop old frames
    "sync=false"          # Don't sync to clock
)
```

---

### **Problem 5: Memory leak**

**Solution:** Always unmap buffers
```python
success, map_info = buffer.map(Gst.MapFlags.READ)
frame = np.ndarray(..., buffer=map_info.data).copy()  # Must .copy()!
buffer.unmap(map_info)  # Always unmap
```

---

## 📚 References

### **Official Documentation:**
- Rockchip MPP: https://github.com/rockchip-linux/mpp
- GStreamer mppvideodec: https://gstreamer.freedesktop.org/
- T-Firefly Wiki: https://wiki.t-firefly.com/

### **Key Files on System:**
```
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstmpp.so
/usr/lib/aarch64-linux-gnu/librockchip_mpp.so
/usr/lib/aarch64-linux-gnu/librockchip_vpu.so
/dev/mpp_service
/dev/dri/renderD128
```

### **Useful Commands:**
```bash
# List VPU capabilities
cat /sys/kernel/debug/mpp_service/session

# Monitor VPU usage
watch -n 1 'cat /sys/kernel/debug/mpp_service/session'

# GStreamer debug
GST_DEBUG=mpp*:5 gst-launch-1.0 ...

# Check hardware decode
ffmpeg -hwaccels
```

---

## ✅ Quick Reference

### **When to use VPU:**
✅ Multiple streams (2+ cameras)  
✅ High resolution (4K)  
✅ Low power requirement  
✅ Stream format compatible (yuv420p)  

### **When to use Software:**
✅ Stream is yuvj420p (JPEG color range)  
✅ Unusual codecs (not H.264/H.265)  
✅ Need maximum compatibility  
✅ Single stream, low resolution  

### **Recommended Setup:**
```python
# Check stream first
ffprobe -show_entries stream=pix_fmt rtsp://...

# If yuv420p → Use VPU
pipeline = "... mppvideodec ! ..."

# If yuvj420p → Use Software
pipeline = "... avdec_h264 ! ..."
```

---

## 📝 Summary

| Feature | VPU | Software |
|---------|-----|----------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **CPU Usage** | ⭐⭐⭐⭐⭐ | ⭐ |
| **Compatibility** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Power Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**🎯 Best Practice:**
- Use VPU when possible (multi-stream, 4K)
- Fallback to software when needed (yuvj420p, compatibility)
- Always check stream format first with ffprobe
- Test both methods and measure performance

---

**Last Updated:** November 4, 2025  
**Platform:** Firefly RK3588, Ubuntu 20.04  
**GStreamer:** 1.16.2  
**MPP Version:** 2.3.0
