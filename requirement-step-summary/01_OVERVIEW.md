# 📚 Model Conversion Workflow Overview

## 🎯 Purpose

เอกสารนี้อธิบาย Workflow การแปลง Model จาก PyTorch → ONNX → RKNN พร้อม Source Configuration และ Performance Tracking สำหรับใช้งานบน Rockchip NPU

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Training (PyTorch)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Input:  Dataset + Training Config                               │
│ Action: Train model                                             │
│ Output:                                                          │
│   ├── best.pt                    (Model weights)                │
│   ├── training_source.yaml       (Training configuration)       │
│   └── performance_pt.json        (Performance baseline)         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Export (ONNX)                                          │
├─────────────────────────────────────────────────────────────────┤
│ Input:  best.pt + training_source.yaml                          │
│ Action: Read training_source.yaml → Export ONNX                 │
│ Output:                                                          │
│   ├── best.onnx                  (ONNX model)                   │
│   ├── onnx_source.yaml           (Copy training + ONNX info)    │
│   └── performance_onnx.json      (Performance vs PT)            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Conversion (RKNN) ⚠️ Linux Ubuntu 20.04 Only           │
├─────────────────────────────────────────────────────────────────┤
│ ⚠️ Requirement: Linux Ubuntu 20.04 (RKNN-Toolkit2 supports Linux only) │
│ Input:  best.onnx + onnx_source.yaml                            │
│ Action: Read onnx_source.yaml → Convert RKNN + Quantize         │
│ Output:                                                          │
│   ├── best_fp16.rknn             (RKNN FP16 model)              │
│   ├── best_int8.rknn             (RKNN INT8 model - optional)   │
│   ├── rknn_source.yaml           (Copy ONNX + RKNN info)        │
│   ├── npu_inference.py           (Auto-generated script)        │
│   └── performance_rknn.json      (Performance vs ONNX)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Package Structure per Phase

### Phase 1: PyTorch Package
```
pt_package/
├── best.pt                      # Model weights
├── training_source.yaml         # 🔑 Training configuration
└── performance_pt.json          # 📊 Performance baseline
```

### Phase 2: ONNX Package
```
onnx_package/
├── best.onnx                    # ONNX model
├── onnx_source.yaml             # 🔑 Training config + ONNX info
└── performance_onnx.json        # 📊 Performance vs PT
```

### Phase 3: RKNN Package
```
rknn_package/
├── best_fp16.rknn               # RKNN FP16 model
├── best_int8.rknn               # RKNN INT8 model (optional)
├── rknn_source.yaml             # 🔑 Full configuration
├── npu_inference.py             # 🚀 Ready-to-use inference script
└── performance_rknn.json        # 📊 Performance vs ONNX
```

---

## 🔑 Key Principles

### 1. Configuration Inheritance

```yaml
training_source.yaml (Phase 1)
    ↓ [Copy critical values]
onnx_source.yaml (Phase 2)
    ↓ [Copy critical values]
rknn_source.yaml (Phase 3)
```

**Critical values ที่ต้อง copy ทุก phase:**
- Input size, format (RGB/BGR)
- Preprocessing method, padding color
- Normalization (mean, std)
- Class names, number of classes

### 2. Performance Tracking

**แต่ละ phase ต้องเก็บ:**
- Accuracy metrics (mAP, precision, recall)
- Inference performance (FPS, latency)
- Comparison with previous phase
- Device used (CPU/GPU/NPU)

### 3. Validation

**แต่ละ phase ต้อง validate:**
- ✅ Config values match previous phase
- ✅ Output shape correct
- ✅ Performance acceptable (no significant drop)
- ✅ Model loads successfully

---

## 🎯 Use Cases

### 1. Single Model Conversion
```bash
# Phase 1: Already have best.pt
# Create training_source.yaml manually or auto-generate

# Phase 2: Export to ONNX
python export_to_onnx.py --pt best.pt --source training_source.yaml

# Phase 3: Convert to RKNN
python convert_to_rknn.py --onnx best.onnx --source onnx_source.yaml
```

### 2. Batch Conversion
```bash
# Convert multiple models
python batch_convert.py --models models.txt --target rk3588
```

### 3. Different Quantization
```bash
# FP16 for accuracy
python convert_to_rknn.py --onnx best.onnx --dtype fp16

# INT8 for speed
python convert_to_rknn.py --onnx best.onnx --dtype int8 --dataset dataset.txt
```

---

## 📊 Performance Expectations

### Typical Results

| Phase | Model Type | Device | FPS | Accuracy vs Original |
|-------|-----------|---------|-----|---------------------|
| PT | PyTorch | GPU | 30-60 | 100% (baseline) |
| ONNX | ONNX | CPU | 5-10 | 100% |
| RKNN FP16 | RKNN | NPU | 15-30 | 98-100% |
| RKNN INT8 | RKNN | NPU | 30-60 | 95-98% |

### Expected Speedup (vs ONNX on CPU)

- **RKNN FP16 on NPU:** 2-4x faster
- **RKNN INT8 on NPU:** 4-8x faster

### Expected Accuracy Drop

- **ONNX vs PT:** 0-0.5% (negligible)
- **RKNN FP16 vs ONNX:** 0-2%
- **RKNN INT8 vs ONNX:** 2-5%

---

## ⚠️ Important Notes

### Critical Rules

1. **🔒 Never Change Training Config**
   - Input size, preprocessing, normalization must stay consistent
   - Changing these requires retraining

2. **✅ Adjustable Settings**
   - Platform, quantization type, optimization level
   - Postprocessing thresholds (conf, IoU)

3. **📋 Always Document**
   - Keep source configs for every phase
   - Record performance for comparison
   - Note any issues or tuning done

### Common Issues

1. **Accuracy Drop > 5%**
   - Check preprocessing matches training
   - Try different quantization algorithm (mmse instead of normal)
   - Increase calibration dataset size

2. **Performance Lower Than Expected**
   - Increase optimization level (0→3)
   - Use INT8 instead of FP16
   - Check NPU utilization

3. **Model Conversion Fails**
   - Check ONNX operators compatibility
   - Verify input/output shapes
   - Update RKNN toolkit version

---

## 🔗 Related Documents

- [02_FIELD_CATEGORIES.md](./02_FIELD_CATEGORIES.md) - Field classification (required/optional/forbidden)
- [templates/](./templates/) - Configuration templates
- [examples/](./examples/) - Real-world examples

---

## 📅 Document Info

**Created:** November 27, 2025  
**Version:** 1.0.0  
**Compatibility:** RKNN-Toolkit2 v2.0.0+
