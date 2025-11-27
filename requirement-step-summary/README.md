# 📚 Requirement & Configuration Templates

## 📂 Directory Structure

```
requirement-step-summary/
├── 01_OVERVIEW.md              # Workflow overview and principles
├── 02_FIELD_CATEGORIES.md      # Field classification guide
│
├── templates/                  # Universal templates for any model
│   ├── training_source.yaml
│   ├── onnx_source.yaml
│   ├── rknn_source.yaml
│   ├── performance_pt.json
│   ├── performance_onnx.json
│   └── performance_rknn.json
│
└── examples/                   # Real-world examples
    └── yolov8_bun/            # YOLOv8 Bun Detection example
        ├── training_source.yaml
        ├── onnx_source.yaml
        ├── rknn_source.yaml
        ├── performance_pt.json
        ├── performance_onnx.json
        └── performance_rknn.json
```

---

## 📋 Overview

This directory contains **complete documentation and templates** for converting AI models from PyTorch → ONNX → RKNN.

### Purpose

1. **Standardize** model conversion workflow
2. **Track** configuration across all phases
3. **Compare** performance before and after conversion
4. **Ensure** consistency in preprocessing/postprocessing
5. **Document** optimal settings for each model

---

## 📖 Documentation

### [01_OVERVIEW.md](./01_OVERVIEW.md)

**Content:**
- Complete workflow explanation
- Package structure per phase
- Configuration inheritance
- Performance expectations
- Best practices

**Read this first** to understand the overall process.

### [02_FIELD_CATEGORIES.md](./02_FIELD_CATEGORIES.md)

**Content:**
- 🔒 **Critical Fields** - Must match training (never change)
- ⚠️ **Fixed Fields** - Determined by model type
- ✅ **Configurable Fields** - Can adjust per use case
- 📊 **Required Fields** - Always needed
- 📝 **Optional Fields** - For reference only

**Use this** to know which fields you can/cannot modify.

---

## 📦 Templates

### For Phase 1: PyTorch Training

**File:** `templates/training_source.yaml`

**Contains:**
- Model information
- Class definitions
- Input settings (size, format, channels)
- Preprocessing configuration
- Training parameters (reference)
- Performance baseline

**Purpose:** Document training configuration to ensure ONNX export matches.

---

### For Phase 2: ONNX Export

**File:** `templates/onnx_source.yaml`

**Contains:**
- Inherited training configuration (🔒 do not change)
- ONNX export settings (✅ configurable)
- Model verification results
- Requirements for RKNN conversion

**Purpose:** Bridge between training and RKNN conversion.

**File:** `templates/performance_onnx.json`

**Contains:**
- Performance metrics vs PyTorch
- Inference speed on CPU
- Validation results

---

### For Phase 3: RKNN Conversion

**File:** `templates/rknn_source.yaml`

**Contains:**
- Inherited preprocessing (🔒 do not change)
- Platform settings (✅ configurable)
- Quantization settings (✅ configurable)
- Postprocessing configuration (✅ adjustable)
- Runtime settings

**Purpose:** Final configuration for NPU deployment.

**File:** `templates/performance_rknn.json`

**Contains:**
- FP16 performance metrics
- INT8 performance metrics
- Comparison with ONNX
- Tuning history
- Deployment recommendations

---

## 🎯 Example: YOLOv8 Bun Detection

Located in `examples/yolov8_bun/`

### Real-world Configuration

**Training:**
- Model: YOLOv8 (1 class: "bun")
- Input: 640x640 RGB
- Preprocessing: Letterbox + gray padding (114,114,114)
- Normalization: mean=[0,0,0], std=[255,255,255]

**Performance:**
- PyTorch: 23 detections, ~50 FPS (GPU)
- ONNX: 23 detections, 6.9 FPS (CPU)
- RKNN FP16: 23 detections, 21.3 FPS (NPU) - **100% match!**
- RKNN INT8: 22 detections, 35.7 FPS (NPU) - **95.7% match**

### Key Findings

1. **IoU threshold = 0.85** is critical
   - 0.7 → only 15/23 detections
   - 0.85 → full 23/23 detections

2. **Confidence threshold = 0.25** matches training
   - Using training threshold gives best results

3. **MMSE algorithm** better than normal for INT8
   - Accuracy: 91% vs 88%

---

## 🚀 How to Use

### For New Model Conversion

#### Step 1: Training Phase

```bash
# 1. Copy template
cp templates/training_source.yaml your_model/training_source.yaml

# 2. Fill in actual values from training
# - Model type, classes
# - Input size, format
# - Preprocessing method
# - Normalization values

# 3. Record performance
cp templates/performance_pt.json your_model/performance_pt.json
# Fill in actual mAP, FPS, etc.
```

#### Step 2: ONNX Export

```bash
# 1. Copy template
cp templates/onnx_source.yaml your_model/onnx_source.yaml

# 2. Copy training config (🔒 critical values)
# From training_source.yaml → onnx_source.yaml

# 3. Configure export settings
# - ONNX opset version
# - Simplify settings

# 4. Export and verify
python export_to_onnx.py --source your_model/onnx_source.yaml

# 5. Record performance
cp templates/performance_onnx.json your_model/performance_onnx.json
```

#### Step 3: RKNN Conversion

```bash
# 1. Copy template
cp templates/rknn_source.yaml your_model/rknn_source.yaml

# 2. Copy preprocessing config (🔒 critical)
# From onnx_source.yaml → rknn_source.yaml

# 3. Configure RKNN settings
# - Platform (rk3588/rk3576/etc.)
# - Quantization (FP16/INT8)
# - Optimization level

# 4. Convert
python convert_to_rknn.py --source your_model/rknn_source.yaml

# 5. Test and tune
# - Adjust conf_threshold, iou_threshold
# - Compare FP16 vs INT8
# - Record optimal settings

# 6. Record performance
cp templates/performance_rknn.json your_model/performance_rknn.json
```

---

## ⚠️ Critical Rules

### 1. Never Change Training Config

```yaml
# 🔒 These must stay the same across all phases:
input_size: [640, 640]
input_format: "RGB"
resize_method: "letterbox"
padding_color: [114, 114, 114]
normalize:
  mean: [0, 0, 0]
  std: [255, 255, 255]
```

**Why?** Model was trained with these specific values. Changing them breaks the model.

### 2. Validate Each Phase

```bash
# After ONNX export
python validate.py --onnx best.onnx --baseline training_source.yaml

# After RKNN conversion
python validate.py --rknn best_fp16.rknn --baseline onnx_source.yaml
```

### 3. Document Tuning

**Always record** in performance JSON:
- Thresholds tested
- Optimal values found
- Reasoning for choices

---

## 📊 Performance Tracking

### What to Track

**Accuracy:**
- mAP50, mAP50-95
- Precision, Recall, F1
- Detection count on test image

**Speed:**
- Preprocessing time
- Inference time
- Postprocessing time
- Total FPS

**Comparison:**
- Accuracy drop vs previous phase
- Confidence score changes
- Speedup factor

### Acceptance Criteria

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| Accuracy drop (ONNX vs PT) | < 1% | Check export settings |
| Accuracy drop (FP16 vs ONNX) | < 5% | Try different optimization |
| Accuracy drop (INT8 vs ONNX) | < 10% | Try MMSE, increase dataset |
| Speedup (FP16 vs CPU) | > 2x | Check NPU utilization |
| Speedup (INT8 vs CPU) | > 4x | Check quantization settings |

---

## 🔗 Related Resources

- **Conversion Tools:**
  - `universal_onnx_to_rknn.py` - Universal converter
  - `npu_inference.py` - Inference script template

- **Documentation:**
  - `Doc/UNIVERSAL_CONVERTER_GUIDE.md` - Converter usage
  - `Doc/PREPROCESSING_POSTPROCESSING_GUIDE.md` - Pre/post processing

- **Examples:**
  - `bun_stage1_detection/` - Complete working example

---

## ✅ Checklist

### Before Starting
- [ ] Read 01_OVERVIEW.md
- [ ] Read 02_FIELD_CATEGORIES.md
- [ ] Understand your model type (YOLO/ResNet/CRNN/etc.)

### Phase 1 (Training)
- [ ] Fill training_source.yaml completely
- [ ] Record actual performance metrics
- [ ] Verify preprocessing matches training logs

### Phase 2 (ONNX)
- [ ] Copy critical values from training_source.yaml
- [ ] Verify ONNX export (onnx.checker)
- [ ] Compare ONNX vs PT performance
- [ ] Ensure 100% accuracy match

### Phase 3 (RKNN)
- [ ] Copy preprocessing from onnx_source.yaml
- [ ] Test FP16 first (baseline)
- [ ] Tune thresholds for optimal performance
- [ ] Test INT8 with calibration dataset
- [ ] Document optimal settings
- [ ] Compare FP16 vs INT8 trade-offs

---

**📅 Last Updated:** November 27, 2025  
**🔖 Version:** 1.0.0  
**📝 Maintainer:** Firefly Development Team
