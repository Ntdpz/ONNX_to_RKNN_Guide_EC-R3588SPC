# ONNX to RKNN Conversion Guide for EC-R3588SPC

Complete guide and toolkit for converting ONNX models to RKNN format optimized for RK3588 NPU on EC-R3588SPC board.

## 📁 Repository Structure

```
ONNX_to_RKNN_Guide_EC-R3588SPC/
├── Data-set/                    # Dataset สำหรับ RKNN INT8 Quantization
│   └── dataset.txt              # รายการ path ของ images สำหรับ calibration
│
├── Doc/                         # เอกสารความรู้ทั้งหมดเกี่ยวกับการทำ RKNN Model
│   ├── 01_OVERVIEW.md          # ภาพรวม workflow การแปลง PT → ONNX → RKNN
│   ├── 02_FIELD_CATEGORIES.md  # คำอธิบาย field ที่เปลี่ยนได้/เปลี่ยนไม่ได้
│   └── preprocessing_theory/   # ทฤษฎี preprocessing และ postprocessing
│
├── Model-AI/                    # AI Models ทุกรูปแบบ
│   ├── pytorch/                # PyTorch models (.pt)
│   ├── onnx/                   # ONNX models (.onnx)
│   └── rknn/                   # RKNN models (.rknn)
│       ├── fp16/               # FP16 quantized models
│       └── int8/               # INT8 quantized models
│
├── onnx_to_rknn_converter/     # เครื่องมือแปลง ONNX → RKNN
│   ├── universal_onnx_to_rknn.py      # Universal converter (รองรับทุก model type)
│   ├── yolov8_onnx_to_rknn.py         # YOLOv8 specific converter
│   └── config_validator.py             # ตรวจสอบ config ก่อนแปลง
│
├── requirement-step-summary/    # Template และ Requirements สำหรับทำ RKNN
│   ├── README.md               # คู่มือการใช้ templates
│   ├── templates/              # Universal templates สำหรับทุก model
│   │   ├── training_source.yaml
│   │   ├── onnx_source.yaml
│   │   ├── rknn_source.yaml
│   │   └── performance_*.json
│   └── examples/               # ตัวอย่างจาก models จริง
│       └── yolov8_bun/         # YOLOv8 bun detection example
│
├── test/                        # Scripts และผลการทดสอบ Model
│   ├── test_onnx.py            # ทดสอบ ONNX model
│   ├── test_rknn.py            # ทดสอบ RKNN model
│   └── results/                # ผลการทดสอบและ performance logs
│
└── old_file/                    # ไฟล์เก่า (รอลบ)
```

## 🚀 Quick Start

### 1. การเตรียม Environment

```bash
# ติดตั้ง RKNN-Toolkit2
pip install rknn-toolkit2

# ติดตั้ง dependencies
pip install opencv-python numpy onnx
```

### 2. การแปลง Model: PyTorch → ONNX → RKNN

#### Step 1: Export PyTorch เป็น ONNX
```python
import torch

model = torch.load("model.pt")
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=12,
    input_names=['images'],
    output_names=['output0']
)
```

#### Step 2: แปลง ONNX เป็น RKNN
```bash
# FP16 (แม่นยำสูง, เร็วปานกลาง)
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --onnx Model-AI/onnx/model.onnx \
    --rknn Model-AI/rknn/fp16/model.rknn \
    --platform rk3588 \
    --quantization FP16

# INT8 (เร็วสุด, ต้องมี dataset)
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --onnx Model-AI/onnx/model.onnx \
    --rknn Model-AI/rknn/int8/model.rknn \
    --platform rk3588 \
    --quantization INT8 \
    --dataset Data-set/dataset.txt
```

### 3. การทดสอบ Model

```bash
# ทดสอบ ONNX
python test/test_onnx.py --model Model-AI/onnx/model.onnx --image test/images/sample.jpg

# ทดสอบ RKNN
python test/test_rknn.py --model Model-AI/rknn/fp16/model.rknn --image test/images/sample.jpg
```

## 📋 Workflow แบบเต็ม

### 1. เตรียม Configuration Files
```bash
# Copy templates จาก requirement-step-summary/templates/
cp requirement-step-summary/templates/training_source.yaml my_model/
cp requirement-step-summary/templates/onnx_source.yaml my_model/
cp requirement-step-summary/templates/rknn_source.yaml my_model/
```

แก้ไข config ตามความต้องการ:
- **training_source.yaml**: ตั้งค่า input size, preprocessing, normalization
- **onnx_source.yaml**: inherit จาก training + export settings
- **rknn_source.yaml**: inherit preprocessing + ตั้งค่า RKNN platform, quantization

### 2. เตรียม Dataset สำหรับ INT8 Quantization
```bash
# สร้าง dataset.txt (100-500 images recommended)
ls Data-set/images/*.jpg > Data-set/dataset.txt
```

### 3. Training → Export → Convert

```bash
# 1. Train PyTorch model (ใช้ config จาก training_source.yaml)
python train.py --config my_model/training_source.yaml

# 2. Export เป็น ONNX (ใช้ config จาก onnx_source.yaml)
python export_onnx.py --config my_model/onnx_source.yaml

# 3. Convert เป็น RKNN (ใช้ config จาก rknn_source.yaml)
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --config my_model/rknn_source.yaml
```

### 4. Validate Performance

ตรวจสอบว่า:
- ✅ ONNX accuracy ≈ PyTorch accuracy (±1-2%)
- ✅ RKNN FP16 accuracy ≈ ONNX accuracy (±1-2%)
- ✅ RKNN INT8 accuracy ≥ 95% ของ FP16

บันทึกผลใน `performance_*.json` files

## ⚙️ Configuration Guidelines

### 🔒 Critical Fields (ห้ามเปลี่ยน)
Fields ที่ต้อง**คงที่**ตลอด PT → ONNX → RKNN:
- `input_size`: [640, 640]
- `format`: "RGB" or "BGR"
- `resize_method`: "letterbox", "stretch", "crop"
- `padding_color`: [114, 114, 114]
- `normalization`: mean=[0,0,0], std=[255,255,255]

ดูรายละเอียดที่: `Doc/02_FIELD_CATEGORIES.md`

### ✅ Configurable Fields
Fields ที่เปลี่ยนได้ตาม use case:
- `platform`: rk3588, rk3576, rk3566, rk3568
- `quantization`: FP16, INT8, UINT8
- `optimization_level`: 0-3
- `conf_threshold`: 0.25, 0.5, etc.
- `iou_threshold`: 0.45, 0.7, 0.85

## 📊 Performance Benchmarks

### YOLOv8 Bun Detection (640x640, 1 class)

| Platform | Format | FPS | mAP@0.5 | Detection Rate |
|----------|--------|-----|---------|----------------|
| PyTorch  | FP32   | 50  | 0.95    | 23/23 (100%)   |
| ONNX     | FP32   | 6.9 | 0.95    | 23/23 (100%)   |
| RKNN     | FP16   | 21.3| 0.95    | 23/23 (100%)   |
| RKNN     | INT8   | 35.7| 0.93    | 22/23 (95.7%)  |

**Key Finding**: IoU threshold = 0.85 จำเป็นสำหรับ detection rate 100%

## 🛠️ Tools Overview

### Universal ONNX to RKNN Converter
```bash
python onnx_to_rknn_converter/universal_onnx_to_rknn.py --help
```

Features:
- ✅ Auto-detect model type (YOLOv5/v8/v10, ResNet, CRNN, etc.)
- ✅ Support all RK platforms (rk3588, rk3576, rk3566, rk3568)
- ✅ FP16 & INT8 quantization
- ✅ Smart parameter recommendations
- ✅ ONNX model analysis

### Dataset Generator
```bash
# สร้าง dataset.txt จาก folder
python Data-set/generate_dataset.py --dir Data-set/images/ --output Data-set/dataset.txt --max 500
```

### Model Validator
```bash
# ตรวจสอบ config consistency
python onnx_to_rknn_converter/config_validator.py \
    --training my_model/training_source.yaml \
    --onnx my_model/onnx_source.yaml \
    --rknn my_model/rknn_source.yaml
```

## 📚 Documentation

### หมวดหมู่เอกสาร

1. **Doc/01_OVERVIEW.md**
   - Workflow overview: PT → ONNX → RKNN
   - Package structure design
   - Performance expectations
   - Validation checklist

2. **Doc/02_FIELD_CATEGORIES.md**
   - Field classification: 🔒 Critical, ⚠️ Fixed, ✅ Configurable
   - ตัวอย่าง config ที่ถูกต้อง/ผิด
   - Common pitfalls

3. **requirement-step-summary/README.md**
   - Template usage guide
   - Config inheritance rules
   - Performance tracking system

4. **requirement-step-summary/examples/yolov8_bun/**
   - Real-world example
   - Complete config files
   - Performance results

## 🎯 Common Use Cases

### Use Case 1: Convert YOLOv8 Model
```bash
# 1. Export ONNX
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
"

# 2. Convert to RKNN FP16
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --onnx yolov8n.onnx \
    --rknn yolov8n_fp16.rknn \
    --platform rk3588 \
    --quantization FP16

# 3. Convert to RKNN INT8
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --onnx yolov8n.onnx \
    --rknn yolov8n_int8.rknn \
    --platform rk3588 \
    --quantization INT8 \
    --dataset Data-set/dataset.txt
```

### Use Case 2: Custom Model with Config Files
```bash
# 1. Create configs from templates
cp requirement-step-summary/templates/*.yaml my_model/

# 2. Edit configs (ดู examples/yolov8_bun/ เป็นตัวอย่าง)
nano my_model/training_source.yaml
nano my_model/onnx_source.yaml
nano my_model/rknn_source.yaml

# 3. Convert with config
python onnx_to_rknn_converter/universal_onnx_to_rknn.py \
    --config my_model/rknn_source.yaml
```

## 🔍 Troubleshooting

### ปัญหาที่พบบ่อย

#### 1. Detection Loss (ONNX works, RKNN fails)
```bash
# เช็ค preprocessing parameters
- ตรวจสอบว่า resize_method เหมือนกันทุก phase
- ตรวจสอบ padding_color (ควรเป็น [114,114,114] สำหรับ YOLO)
- ตรวจสอบ normalization (mean, std)
- ลอง tune iou_threshold (เพิ่มจาก 0.45 → 0.7 → 0.85)
```

#### 2. Low INT8 Accuracy
```bash
# ปรับปรุง dataset.txt
- ใช้ภาพ 100-500 images (diverse)
- ภาพต้องครอบคลุม use case จริง
- ลอง quantization_algorithm="kl_divergence" แทน "normal"
```

#### 3. ONNX Export Error
```bash
# ตรวจสอบ opset version
- YOLOv8: opset_version=12
- YOLOv10: opset_version=13
- เช็คว่า PyTorch model ใช้ operations ที่ ONNX รองรับ
```

## 🔗 Useful Links

- [RKNN-Toolkit2 Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [RK3588 NPU Specs](https://www.rock-chips.com/a/cn/product/RK35xilie/2022/0926/1660.html)
- [EC-R3588SPC Board Info](https://www.edatec.cn/en/product/detail/edatec-r3588spc.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## 📝 License

Please refer to original RKNN-Toolkit2 license and model licenses.

## 🤝 Contributing

Issues และ Pull Requests ยินดีต้อนรับ!

---

**Hardware**: EC-R3588SPC (RK3588, 6 TOPS NPU)  
**Toolkit**: RKNN-Toolkit2 v2.3.2  
**Python**: 3.8.10+  
**Last Updated**: November 27, 2025
