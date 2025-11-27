# 📋 Configuration Field Categories

## 🎯 Purpose

เอกสารนี้จำแนก Configuration Fields ทั้งหมดตามประเภท เพื่อให้รู้ว่า field ไหน:
- **ห้ามเปลี่ยน** (Must match training)
- **เปลี่ยนได้** (User configurable)
- **ต้องการเสมอ** (Always required)
- **ต้องเป็นค่านี้เท่านั้น** (Fixed per model type)

---

## 🔒 CRITICAL FIELDS (ห้ามเปลี่ยน!)

### Fields ที่ต้องตรงกับตอน Training ทุก Phase

| Field | Description | Example | Why Critical |
|-------|-------------|---------|--------------|
| `input_size` | Input image size (H, W) | `[640, 640]` | Model architecture fixed at this size |
| `input_format` | Color format | `"RGB"` or `"BGR"` | Model trained with specific format |
| `channels` | Number of channels | `3` (RGB) or `1` (Gray) | Model architecture fixed |
| `resize_method` | How to resize input | `"letterbox"` or `"direct"` | Affects feature extraction |
| `padding_color` | Color for padding | `[114, 114, 114]` | Model learned with this background |
| `normalize.mean` | Normalization mean | `[0, 0, 0]` | Model weights scaled accordingly |
| `normalize.std` | Normalization std | `[255, 255, 255]` | Model weights scaled accordingly |

### ⚠️ หากเปลี่ยนค่าเหล่านี้:

```
❌ Model จะให้ผลลัพธ์ผิดพลาด
❌ Accuracy ลดลงอย่างมาก (>20%)
❌ ต้อง Retrain Model ใหม่ทั้งหมด
```

### ✅ ตัวอย่างค่าที่ถูกต้อง:

```yaml
# YOLOv8 Standard
input_size: [640, 640]
input_format: "RGB"
resize_method: "letterbox"
padding_color: [114, 114, 114]
normalize:
  mean: [0, 0, 0]
  std: [255, 255, 255]

# ResNet ImageNet
input_size: [224, 224]
input_format: "RGB"
resize_method: "center_crop"
normalize:
  mean: [123.675, 116.28, 103.53]
  std: [58.395, 57.12, 57.375]

# CRNN Text Recognition
input_size: [32, 128]
input_format: "Grayscale"
channels: 1
resize_method: "direct"
normalize:
  mean: [0.5]
  std: [0.5]
```

---

## ⚠️ FIXED FIELDS (ต้องเป็นค่านี้เท่านั้น)

### Fields ที่ขึ้นกับ Model Type/Architecture

| Field | Depends On | Examples |
|-------|------------|----------|
| `channels` | Input format | RGB=3, Grayscale=1, RGBA=4 |
| `layout` | Framework | PyTorch=NCHW, TensorFlow=NHWC |
| `dtype` | Training precision | `"float32"`, `"uint8"` |
| `output_format` | Model head | YOLO=xywh, Faster-RCNN=xyxy |

### ตัวอย่าง:

```yaml
# If input_format = "RGB"
channels: 3  # ⚠️ Must be 3

# If input_format = "Grayscale"
channels: 1  # ⚠️ Must be 1

# PyTorch models
layout: "NCHW"  # ⚠️ (Batch, Channels, Height, Width)

# TensorFlow models
layout: "NHWC"  # ⚠️ (Batch, Height, Width, Channels)
```

---

## ✅ CONFIGURABLE FIELDS (เปลี่ยนได้)

### Fields ที่ปรับแต่งได้ตามความต้องการ

#### Platform Settings

| Field | Options | Default | Notes |
|-------|---------|---------|-------|
| `platform.target` | rk3588, rk3576, rk3562, rv1109, rv1126, rk1808 | `rk3588` | Match your hardware |
| `platform.sub_platform` | Chip variants | `null` | Usually not needed |

#### Quantization Settings

| Field | Options | Default | Notes |
|-------|---------|---------|-------|
| `quantization.type` | FP16, INT8, UINT8 | `FP16` | FP16=accuracy, INT8=speed |
| `quantization.algorithm` | normal, mmse, kl_divergence | `normal` | mmse=better accuracy |
| `quantization.method` | channel, layer | `channel` | channel=better accuracy |
| `optimization_level` | 0, 1, 2, 3 | `3` | 3=most optimized |

#### Inference Settings

| Field | Range | Default | Notes |
|-------|-------|---------|-------|
| `conf_threshold` | 0.0 - 1.0 | `0.25` | Higher=fewer detections |
| `iou_threshold` | 0.0 - 1.0 | `0.7` | Higher=less NMS filtering |
| `max_detections` | 1 - 1000 | `300` | Maximum output boxes |
| `min_box_size` | 0 - 100 | `10` | Minimum box size (pixels) |

#### Dataset Settings (for INT8)

| Field | Description | Recommended |
|-------|-------------|-------------|
| `dataset.path` | Path to calibration images | `dataset.txt` |
| `dataset.size` | Number of images | 500-1000 |
| `dataset.source` | Source of images | train or val (NOT test) |

### ตัวอย่างการปรับแต่ง:

```yaml
# High accuracy (development)
quantization:
  type: "FP16"
  optimization_level: 3

# High performance (production)
quantization:
  type: "INT8"
  algorithm: "mmse"
  method: "channel"
  optimization_level: 3
  dataset:
    path: "dataset.txt"
    size: 1000

# Strict detection (fewer false positives)
postprocessing:
  conf_threshold: 0.5    # Higher
  iou_threshold: 0.45    # Lower (more aggressive NMS)

# Loose detection (catch all)
postprocessing:
  conf_threshold: 0.15   # Lower
  iou_threshold: 0.85    # Higher (less NMS)
```

---

## 📊 REQUIRED FIELDS (ต้องการเสมอ)

### Fields ที่ต้องมีทุก Model

#### Model Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `model.name` | string | Model name | `"bun_detection"` |
| `model.type` | string | Architecture type | `"YOLOv8"`, `"ResNet"`, `"CRNN"` |
| `model.task` | string | Task type | `"detect"`, `"classify"`, `"segment"` |
| `model.version` | string | Model version | `"1.0.0"` |

#### Class Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `classes` | list | Class names | `["person", "car", "dog"]` |
| `num_classes` | int | Number of classes | `3` |

#### Input Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `input_size` | [int, int] | Input dimensions | `[640, 640]` |
| `input_format` | string | Color format | `"RGB"` |
| `channels` | int | Channel count | `3` |

---

## 📝 OPTIONAL FIELDS (ไม่จำเป็น)

### Fields ที่เก็บไว้เพื่ออ้างอิง

| Field | Description | Use Case |
|-------|-------------|----------|
| `model.author` | Model creator | Documentation |
| `model.description` | Model description | Documentation |
| `model.created_date` | Creation date | Version tracking |
| `training.epochs` | Training epochs | Reference |
| `training.batch_size` | Batch size used | Reference |
| `training.optimizer` | Optimizer type | Reference |
| `dataset.train_images` | Training image count | Statistics |
| `dataset.val_images` | Validation image count | Statistics |

---

## 🎯 Field Usage by Phase

### Phase 1: Training (PT)

**Required:**
- ✅ All model info fields
- ✅ All input fields
- ✅ All preprocessing fields
- ✅ Class information

**Optional:**
- Training parameters (epochs, batch_size, etc.)
- Dataset statistics

**Forbidden:**
- ONNX-specific fields
- RKNN-specific fields

### Phase 2: Export (ONNX)

**Required:**
- ✅ Copy all from Phase 1 (training fields)
- ✅ ONNX opset version
- ✅ Input/output names and shapes

**Optional:**
- Export tool version
- Simplification settings

**Configurable:**
- Dynamic axes
- Opset version (11-16)

### Phase 3: Conversion (RKNN)

**Required:**
- ✅ Copy all from Phase 2 (training + ONNX fields)
- ✅ Platform target
- ✅ Quantization type

**Optional:**
- Hybrid quantization config
- Custom optimization

**Configurable:**
- ✅ Platform, quantization, optimization
- ✅ Postprocessing thresholds
- ✅ Runtime settings

---

## 🚨 Validation Rules

### Pre-Conversion Checks

```python
def validate_config(current_phase, previous_phase):
    # Critical fields must match
    assert current['input_size'] == previous['input_size']
    assert current['input_format'] == previous['input_format']
    assert current['resize_method'] == previous['resize_method']
    assert current['padding_color'] == previous['padding_color']
    assert current['normalize'] == previous['normalize']
    
    # Class info must match
    assert current['classes'] == previous['classes']
    assert current['num_classes'] == previous['num_classes']
    
    print("✅ Configuration validated!")
```

### Post-Conversion Checks

```python
def validate_performance(current, baseline):
    # Accuracy should not drop more than 5%
    accuracy_drop = baseline['mAP50'] - current['mAP50']
    assert accuracy_drop < 0.05, f"Accuracy drop too large: {accuracy_drop:.1%}"
    
    # Should have speedup on NPU
    if current['device'] == 'NPU':
        assert current['fps'] > baseline['fps'], "No speedup on NPU"
    
    print("✅ Performance validated!")
```

---

## 📚 Quick Reference

### ❌ Never Change (Training Config)
- input_size, input_format, channels
- resize_method, padding_color
- normalize.mean, normalize.std

### ⚠️ Fixed (Per Model Type)
- layout (NCHW/NHWC)
- dtype (float32/uint8)
- output_format

### ✅ Always Adjust (Per Hardware)
- platform.target
- quantization settings

### ✅ May Adjust (Per Use Case)
- conf_threshold, iou_threshold
- max_detections, min_box_size
- optimization_level

---

**📅 Last Updated:** November 27, 2025  
**🔖 Version:** 1.0.0
