# YOLOv5 to RKNN Complete Conversion Guide

คู่มือแปลง YOLOv5 model ที่เทรนมาจาก Windows ให้ใช้งานบน NPU (RKNN) แบบครบวงจร

## 📋 Overview

การแปลง YOLOv5 model จาก Windows-trained format ไปเป็น RKNN format สำหรับใช้งานบน RK3588 NPU ประกอบด้วย 3 ขั้นตอนหลัก:

1. **Windows Model → PyTorch (PT)** - แก้ไข path compatibility
2. **PyTorch (PT) → ONNX** - Export เป็น standard format
3. **ONNX → RKNN** - แปลงสำหรับ NPU acceleration

## 🛠️ Prerequisites

### Environment Setup
```bash
# ตรวจสอบ Python environment
source /home/firefly/Documents/YOLO/.venv/bin/activate

# ตรวจสอบ RKNN toolkit
python -c "import rknn; print('RKNN toolkit available')"
```

### Required Files
- Windows-trained YOLOv5 model: `runs/train/[model_name]/weights/best.pt`
- YOLOv5 export script: `/home/firefly/Documents/YOLO/yolov5/export.py`
- Universal RKNN converter: `/home/firefly/Documents/YOLO/Convert_tools/universal_onnx_to_rknn.py`

## 📂 Directory Structure

```
/home/firefly/Documents/YOLO/
├── yolov5/
│   ├── export.py
│   └── runs/train/[model_name]/weights/best.pt
├── model-final-V2/
│   ├── PT/           # PyTorch models
│   ├── ONNX/         # ONNX models
│   └── RKNN/         # RKNN models
└── Convert_tools/
    └── universal_onnx_to_rknn.py
```

## 🔄 Step-by-Step Conversion Process

### Step 1: Windows Model to PyTorch (PT)

**Problem**: Windows-trained models มี path format ที่ไม่เข้ากันกับ Linux

**Solution**: ใช้ pathlib monkey patching

```bash
cd /home/firefly/Documents/YOLO/yolov5

# แปลง Vehicle Type Detection Model
PYTHONPATH=/home/firefly/Documents/YOLO:$PYTHONPATH /home/firefly/Documents/YOLO/.venv/bin/python -c "
import pathlib
import platform
if platform.system() != 'Windows':
    import posixpath
    pathlib.PurePath._flavour = pathlib._posix_flavour
    pathlib.WindowsPath = pathlib.PosixPath

import torch
checkpoint = torch.load('runs/train/vehicle_type_detection_cuda/weights/best.pt', map_location='cpu')
torch.save(checkpoint, '/home/firefly/Documents/YOLO/model-final-V2/PT/vehicle_type_detection.pt')
print('✅ Vehicle type detection model exported to PT format')
print('📁 Location: /home/firefly/Documents/YOLO/model-final-V2/PT/vehicle_type_detection.pt')
"
```

**Alternative Script Method**:
```bash
# สร้าง script สำหรับแปลงซ้ำได้
/home/firefly/Documents/YOLO/.venv/bin/python /home/firefly/Documents/YOLO/export_vehicle_alternative.py
```

**⚠️ Important PT Model Format Requirements**:

สำหรับ models ที่มี architecture ซับซ้อน (เช่น license plate detection) จำเป็นต้องแปลงเป็น **fused layers format** ก่อน export เป็น ONNX:

```bash
# 1. แปลง Windows model เป็น PT format ธรรมดา
cd /home/firefly/Documents/YOLO/yolov5
PYTHONPATH=/home/firefly/Documents/YOLO:$PYTHONPATH /home/firefly/Documents/YOLO/.venv/bin/python -c "
import pathlib
import platform
if platform.system() != 'Windows':
    import posixpath
    pathlib.PurePath._flavour = pathlib._posix_flavour
    pathlib.WindowsPath = pathlib.PosixPath

import torch
checkpoint = torch.load('runs/train/license_plate_model_87percent/weights/best.pt', map_location='cpu')
torch.save(checkpoint, '/home/firefly/Documents/YOLO/model-final-V2/PT/license_plate_model_87percent.pt')
print('✅ License plate model exported to PT format')
"

# 2. แปลง PT เป็น fused layers format
/home/firefly/Documents/YOLO/.venv/bin/python -c "
import torch
import sys
sys.path.append('.')
from models.experimental import attempt_load
from utils.torch_utils import select_device

# Load model and fuse layers
device = select_device('cpu')
model = attempt_load('/home/firefly/Documents/YOLO/model-final-V2/PT/license_plate_model_87percent.pt', device=device, inplace=True, fuse=True)
model.eval()

# Save fused model
torch.save({
    'model': model,
    'epoch': -1,
    'best_fitness': None,
    'training_results': None,
    'optimizer': None
}, '/home/firefly/Documents/YOLO/model-final-V2/PT/license_plate_model_87percent_fused.pt')

print('✅ Fused model saved successfully')
"
```

**PT Model Format Types**:
- **Standard PT**: เหมาะสำหรับ models ที่ไม่ซับซ้อน (vehicle detection, basic object detection)
- **Fused PT**: จำเป็นสำหรับ models ซับซ้อน (license plate detection, custom architectures)
- **Fused layers** ช่วยป้องกัน channel mismatch errors ระหว่าง ONNX export

### Step 2: PyTorch (PT) to ONNX

**Export เป็น ONNX format** พร้อม optimization:

```bash
cd /home/firefly/Documents/YOLO/yolov5

# Export Vehicle Type Detection to ONNX
/home/firefly/Documents/YOLO/.venv/bin/python export.py \
    --weights /home/firefly/Documents/YOLO/model-final-V2/PT/vehicle_type_detection.pt \
    --include onnx \
    --opset 12 \
    --simplify \
    --device cpu
```

**Parameters Explanation**:
- `--weights`: Path ไปยัง PT model
- `--include onnx`: Export format เป็น ONNX
- `--opset 12`: ONNX opset version (compatible กับ RKNN)
- `--simplify`: ทำ graph optimization
- `--device cpu`: ใช้ CPU สำหรับ export (stable กว่า)

### Step 3: ONNX to RKNN

**ใช้ Universal RKNN Converter** สำหรับแปลงเป็น NPU format:

#### FP32 Strategy (Highest Accuracy)
```bash
cd /home/firefly/Documents/YOLO/Convert_tools

/home/firefly/Documents/YOLO/.venv/bin/python universal_onnx_to_rknn.py \
    --input /home/firefly/Documents/YOLO/model-final-V2/ONNX/vehicle_type_detection.onnx \
    --output /home/firefly/Documents/YOLO/model-final-V2/RKNN/vehicle_type_detection_fp32.rknn \
    --strategy fp32 \
    --verbose
```

#### FP16 Strategy (Balanced)
```bash
/home/firefly/Documents/YOLO/.venv/bin/python universal_onnx_to_rknn.py \
    --input /home/firefly/Documents/YOLO/model-final-V2/ONNX/vehicle_type_detection.onnx \
    --output /home/firefly/Documents/YOLO/model-final-V2/RKNN/vehicle_type_detection_fp16.rknn \
    --strategy fp16 \
    --verbose
```

#### Auto Strategy (Try Best First)
```bash
/home/firefly/Documents/YOLO/.venv/bin/python universal_onnx_to_rknn.py \
    --input /home/firefly/Documents/YOLO/model-final-V2/ONNX/vehicle_type_detection.onnx \
    --output /home/firefly/Documents/YOLO/model-final-V2/RKNN/vehicle_type_detection_auto.rknn \
    --strategy auto \
    --verbose
```

## 📊 Model Size Comparison

| Format | Size | Accuracy | NPU Acceleration |
|--------|------|----------|------------------|
| PT     | ~13.8 MB | Original | ❌ |
| ONNX   | ~27.3 MB | Original | ❌ |
| RKNN FP32 | ~15.2 MB | High | ✅ |
| RKNN FP16 | ~15.2 MB | High | ✅ |

## 🧪 Example: Complete Vehicle Type Detection Conversion

### 1. Setup Environment
```bash
cd /home/firefly/Documents/YOLO
source .venv/bin/activate
```

### 2. Convert Windows Model to PT
```bash
cd yolov5
PYTHONPATH=/home/firefly/Documents/YOLO:$PYTHONPATH /home/firefly/Documents/YOLO/.venv/bin/python -c "
import pathlib
import platform
if platform.system() != 'Windows':
    import posixpath
    pathlib.PurePath._flavour = pathlib._posix_flavour
    pathlib.WindowsPath = pathlib.PosixPath

import torch
checkpoint = torch.load('runs/train/vehicle_type_detection_cuda/weights/best.pt', map_location='cpu')
torch.save(checkpoint, '/home/firefly/Documents/YOLO/model-final-V2/PT/vehicle_type_detection.pt')
print('✅ Model exported to PT format')
"
```

### 3. Export PT to ONNX
```bash
/home/firefly/Documents/YOLO/.venv/bin/python export.py \
    --weights /home/firefly/Documents/YOLO/model-final-V2/PT/vehicle_type_detection.pt \
    --include onnx \
    --opset 12 \
    --simplify
```

### 4. Convert ONNX to RKNN
```bash
cd /home/firefly/Documents/YOLO/Convert_tools
/home/firefly/Documents/YOLO/.venv/bin/python universal_onnx_to_rknn.py \
    --input /home/firefly/Documents/YOLO/model-final-V2/ONNX/vehicle_type_detection.onnx \
    --output /home/firefly/Documents/YOLO/model-final-V2/RKNN/vehicle_type_detection.rknn \
    --strategy fp32 \
    --verbose
```

### 5. Verify Results
```bash
ls -la /home/firefly/Documents/YOLO/model-final-V2/RKNN/
# Expected: vehicle_type_detection.rknn (~15.2 MB)
```

## 🔧 Universal RKNN Converter Usage

### Available Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `fp32` | 32-bit floating point | Highest accuracy |
| `fp16` | 16-bit floating point | Balanced accuracy/performance |
| `int8` | 8-bit integer quantization | Smallest size, fastest |
| `hybrid` | Mixed precision | Optimal balance |
| `auto` | Try strategies in order | Automatic best result |

### Command Options
```bash
python universal_onnx_to_rknn.py [OPTIONS]

Options:
  --input, -i      Input ONNX model file
  --output, -o     Output RKNN file path
  --strategy, -s   Conversion strategy [fp32/fp16/int8/hybrid/auto]
  --dataset, -d    Dataset file for quantization (INT8/hybrid only)
  --verbose, -v    Verbose output
  --log, -l        Save conversion log to file
```

### Batch Conversion
```bash
python universal_onnx_to_rknn.py \
    --batch /path/to/onnx/directory \
    --output_dir /path/to/rknn/directory \
    --strategy auto
```

## ⚠️ Common Issues & Solutions

### Issue 1: WindowsPath Error
```
AttributeError: module 'pathlib' has no attribute 'WindowsPath'
```
**Solution**: ใช้ pathlib monkey patching ตาม Step 1

### Issue 2: RKNN Quantization Error
```
quantized_dtype 'dynamic_fixed_point-i16' not supported
```
**Solution**: ใช้ universal converter ที่อัปเดตแล้วสำหรับ RKNN toolkit 2.3.0

### Issue 3: ONNX Export Error
```
RuntimeError: Exporting the operator to ONNX opset version 11 is not supported
```
**Solution**: ใช้ `--opset 12` หรือสูงกว่า

### Issue 4: Channel Mismatch Error during ONNX Export
```
RuntimeError: Given groups=1, weight of size [48, 12, 3, 3], expected input[1, 48, 320, 320] to have 12 channels, but got 48 channels instead
```
**Solution**: แปลง PT model เป็น fused layers format ก่อน export:

```bash
# สำหรับ models ที่มี architecture ซับซ้อน ต้องใช้ fused format
cd /home/firefly/Documents/YOLO/yolov5
/home/firefly/Documents/YOLO/.venv/bin/python -c "
import torch
import sys
sys.path.append('.')

# Load fused model
checkpoint = torch.load('/path/to/fused_model.pt', map_location='cpu')
model = checkpoint['model']
model.eval()

# Export using torch.onnx.export directly
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    '/path/to/output.onnx',
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output']
)
"
```

## 🚀 Performance Optimization Tips

### 1. Model Size Optimization
- ใช้ `--simplify` เมื่อ export ONNX
- เลือก strategy ที่เหมาะสมตามความต้องการ
- พิจารณา INT8 quantization สำหรับ production

### 2. NPU Utilization
- RKNN models จะใช้ NPU acceleration อัตโนมัติ
- Monitor NPU usage ด้วย `sudo cat /sys/kernel/debug/rknpu/load`

### 3. Inference Speed
- FP16 มักให้ performance ดีที่สุดบน RK3588
- INT8 เร็วที่สุดแต่อาจลด accuracy

## 📝 File Naming Convention

```
model_name_[strategy].rknn

Examples:
- vehicle_type_detection_fp32.rknn
- license_plate_detection_fp16.rknn
- object_detection_int8.rknn
```

## 🔍 Verification Commands

### Check Model Information
```bash
# PT model info
python -c "import torch; print(torch.load('model.pt', map_location='cpu').keys())"

# ONNX model info
python -c "import onnx; model = onnx.load('model.onnx'); print(f'Inputs: {len(model.graph.input)}, Outputs: {len(model.graph.output)}')"
```

### Test RKNN Model
```bash
# Basic RKNN model loading test
python -c "
from rknn.api import RKNN
rknn = RKNN()
ret = rknn.load_rknn('model.rknn')
print('✅ RKNN model loaded successfully' if ret == 0 else '❌ Failed to load RKNN model')
rknn.release()
"
```

## 📚 Additional Resources

- [YOLOv5 Official Documentation](https://github.com/ultralytics/yolov5)
- [RKNN Toolkit Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [RK3588 NPU Performance Guide](https://wiki.radxa.com/Rock5/guide/rknn)

---

**Last Updated**: October 27, 2025  
**Compatible**: RKNN Toolkit 2.3.0, YOLOv5 7.0+, RK3588 NPU