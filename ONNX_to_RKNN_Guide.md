# Universal ONNX to RKNN Conversion Guide

คู่มือกลาง (Universal Guide) สำหรับการแปลงโมเดล ONNX เป็น RKNN เพื่อใช้งานบนบอร์ด RK3588

## 🔄 กระบวนการแปลง (Conversion Workflow)

### 1. **ONNX Model** → **RKNN Toolkit2** → **RKNN Model** → **NPU**

```
ONNX Model (Float32) 
       ↓
RKNN Toolkit2 Processing:
  • Load ONNX (Opset 12 recommended)
  • Configure Target (RK3588)
  • Build & Optimize  
  • Quantization (Optional: FP16 vs INT8)
       ↓
RKNN Model (Ready for NPU)
```

## 🛠️ เครื่องมือที่มี (Tools)

### 1. `onnx_to_rknn_converter.py` (Single Model)
เครื่องมือแปลงไฟล์แบบไฟล์เดียวจบ รองรับทั้ง FP16 และ INT8

```bash
# แบบง่าย (FP16) - ไม่ต้องใช้รูปภาพ
python3 onnx_to_rknn_converter.py 
  --onnx your_model.onnx 
  --rknn output_model.rknn

# แบบเร็ว (INT8) - ต้องใช้รูปภาพสำหรับ Calibration
python3 onnx_to_rknn_converter.py 
  --onnx your_model.onnx 
  --rknn output_model.rknn 
  --quantize 
  --images ./dataset_folder/
```

### 2. `batch_onnx_converter.py` (Batch Processing)
เครื่องมือแปลงไฟล์แบบเหมาเข่ง สำหรับคนที่มีหลายโมเดล

```bash
# แปลงทุกไฟล์ในโฟลเดอร์
python3 batch_onnx_converter.py 
  --onnx-dir ./onnx_models/ 
  --output-dir ./rknn_models/
```

## ⚙️ โหมดการแปลง (Conversion Modes)

### 1. **FP16 Mode (Default)**
- **ข้อดี:** ง่าย, ไม่ต้องเตรียม Dataset, ความแม่นยำสูง (ใกล้เคียงต้นฉบับ)
- **ข้อเสีย:** ไฟล์ใหญ่กว่า, ช้ากว่า INT8 เล็กน้อย
- **เหมาะสำหรับ:** การทดสอบเบื้องต้น, โมเดลที่ต้องการความละเอียดสูง

### 2. **INT8 Mode (Quantized)**
- **ข้อดี:** ไฟล์เล็ก (ลดลง 50%), ทำงานเร็วขึ้น (Speed up 20-50%)
- **ข้อเสีย:** ต้องเตรียมรูปภาพ (Calibration Dataset), ความแม่นยำอาจลดลงเล็กน้อย
- **เหมาะสำหรับ:** การใช้งานจริง (Production), งานที่ต้องการ FPS สูงสุด

## 📋 Model Requirements (ข้อกำหนดของโมเดล)

เพื่อให้การแปลงราบรื่น โมเดล ONNX ของคุณควรมีคุณสมบัติดังนี้:

1.  **Opset Version:** 12 (แนะนำ) หรือ 11
2.  **Input Size:** 640x640 (มาตรฐาน YOLO)
3.  **Batch Size:** 1 (Static shape)
4.  **Color Format:** RGB (Mean=[0,0,0], Std=[255,255,255])

## 💡 Examples (ตัวอย่างผลการทดสอบ)

นี่คือตัวอย่างผลลัพธ์จากการแปลงโมเดลจริง (เพื่อเป็นแนวทาง):

| Model Type | Original (ONNX) | Converted (RKNN) | Status |
|------------|-----------------|------------------|--------|
| **YOLOv5s (FP16)** | ~14 MB | ~15 MB | ✅ Success |
| **YOLOv5s (INT8)** | ~14 MB | ~8 MB | ✅ Success |
| **YOLOv5m (FP16)** | ~40 MB | ~42 MB | ✅ Success |

**Note:** เวลาในการแปลงเฉลี่ยอยู่ที่ 5-15 วินาทีต่อโมเดล

## 🚨 Common Issues & Solutions

### 1. **"Unsupported operator"**
- **สาเหตุ:** ใช้ Opset version ใหม่เกินไป หรือมี Layer แปลกๆ
- **วิธีแก้:** Export ONNX ใหม่โดยใช้ `opset=12`

### 2. **"Model build failed"**
- **สาเหตุ:** Input shape ไม่คงที่ (Dynamic shape)
- **วิธีแก้:** Export ONNX โดยระบุ Input size ให้ชัดเจน (เช่น 640x640)

### 3. **Accuracy Drop (ไม่แม่นหลังแปลง)**
- **สาเหตุ:** รูปภาพที่ใช้ทำ Calibration (INT8) น้อยเกินไป หรือไม่ครอบคลุม
- **วิธีแก้:** เพิ่มจำนวนรูปภาพใน Dataset (แนะนำ 50-100 รูป) หรือเปลี่ยนกลับไปใช้ FP16

## 🔧 Configuration Options (การตั้งค่าเพิ่มเติม)

### Normalization (ค่าสี)
สคริปต์ตั้งค่ามาตรฐานไว้ที่:
```python
mean_values=[[0, 0, 0]]
std_values=[[255, 255, 255]]
```
(หมายความว่ารับค่า 0-255 แล้วหารด้วย 255 เพื่อให้เป็น 0-1)

### Target Platform
```bash
--target rk3588
```
(รองรับ rk3588, rk3566, rk3568)

## 🛠️ เครื่องมือที่สร้าง

### 1. `onnx_to_rknn_converter.py` - Single Model Converter

```bash
# Convert without quantization (FP16)
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model_fp16.rknn

# Convert with quantization (INT8)
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model_int8.rknn \
  --quantize \
  --images ./calibration_images/

# Test converted model
python3 onnx_to_rknn_converter.py \
  --rknn model.rknn \
  --test \
  --image test.jpg
```

### 2. `batch_onnx_converter.py` - Batch Converter

```bash
# Convert all ONNX models in directory (FP16)
python3 batch_onnx_converter.py \
  --onnx-dir /path/to/onnx/models/ \
  --output-dir ./converted_rknn/

# Convert with quantization (INT8)
python3 batch_onnx_converter.py \
  --onnx-dir /path/to/onnx/models/ \
  --output-dir ./converted_rknn/ \
  --quantize \
  --images-dir ./calibration_images/
```

## 📊 ผลการทดสอบการแปลง

### Models ที่แปลงสำเร็จ:

| Model | ONNX Size | RKNN Size | Conversion Time | Status |
|-------|-----------|-----------|-----------------|---------|
| **codeprovince_best** | - | 15.1 MB | 9.0s | ✅ Success |
| **codeprovince_last** | - | 15.1 MB | 8.9s | ✅ Success |
| **licenseplate_best** | - | 15.1 MB | 5.5s | ✅ Success |
| **licenseplate_last** | - | 15.1 MB | 5.5s | ✅ Success |
| **license_plate_model_87percent** | - | 46.5 MB | 16.5s | ✅ Success |
| **vehicle_type_detection_best** | - | 15.2 MB | 8.7s | ✅ Success |
| **vehicle_type_detection_last** | - | 15.2 MB | 8.9s | ✅ Success |

**Total**: 7/7 models (100% success rate)
**Total Size**: 137.4 MB  
**Average Time**: 9.3s per model

## 🎯 Performance Comparison

### Original vs Converted Models:

| Model | Type | Inference Time | FPS | Output Format |
|-------|------|----------------|-----|---------------|
| **codeprovince_best_fp32.rknn** (original) | FP32 | 68-86ms | 11-14 | (1,25200,7) |
| **codeprovince_best_fp16.rknn** (converted) | FP16 | 75.8ms | 13.2 | (1,25200,7) |
| **licenseplate_best_fp16.rknn** (converted) | FP16 | 73.1ms | 13.7 | (1,25200,6) |

**ข้อสังเกต**: Performance เทียบเท่ากัน! License plate model มี output format แตกต่าง (6 classes) 🚀

## ⚙️ การแปลงแบบละเอียด

### 1. **FP16 Mode (Non-quantized)**
```bash
# ไม่ต้องใช้ calibration dataset
python3 onnx_to_rknn_converter.py \
  --onnx codeprovince_best.onnx \
  --rknn codeprovince_best_fp16.rknn \
  --target rk3588
```

**Features:**
- ✅ ไม่ต้อง calibration images
- ✅ แปลงเร็ว (8-16 วินาที)
- ✅ คุณภาพดี (precision สูง)
- ❌ ไฟล์ใหญ่กว่า INT8

### 2. **INT8 Mode (Quantized)**
```bash
# ต้องใช้ calibration dataset
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model_int8.rknn \
  --quantize \
  --images ./calibration_images/ \
  --num-images 100
```

**Features:**
- ✅ ไฟล์เล็ก (~50% ของ FP16)
- ✅ เร็วกว่าบน NPU
- ❌ ต้อง calibration images
- ❌ อาจลด accuracy เล็กน้อย

## 📋 Model Requirements

### Input ONNX Model:
1. **Format**: ONNX v1.7+
2. **Input Shape**: [1, 3, 640, 640] (CHW format)
3. **Data Type**: Float32 (recommended)
4. **Operations**: รองรับ RK3588 ops
5. **Output**: YOLO detection format

### Output RKNN Model:
1. **Platform**: RK3588 compatible
2. **Input Format**: NHWC [1, 640, 640, 3]  
3. **Data Type**: FP16 or INT8
4. **Optimizations**: NPU-specific optimizations applied
5. **Runtime**: RKNN Runtime compatible

## 🔧 Configuration Options

### Normalization Settings:
```bash
# Default (0-255 input)
--mean 0 0 0 --std 255 255 255

# ImageNet normalization  
--mean 123.675 116.28 103.53 --std 58.395 57.12 57.375

# Custom normalization
--mean 127.5 127.5 127.5 --std 127.5 127.5 127.5
```

### Optimization Levels:
- **Level 0**: Basic conversion
- **Level 1**: Standard optimization (default)
- **Level 2**: Aggressive optimization  
- **Level 3**: Maximum optimization

## 🚨 Common Issues & Solutions

### 1. **"Unsupported operator" Error**
```bash
# Check ONNX model operators
python3 -c "
import onnx
model = onnx.load('model.onnx')
ops = set(node.op_type for node in model.graph.node)
print('Operators:', sorted(ops))
"
```

### 2. **"Model build failed" Error**
- ตรวจสอบ ONNX model validity
- ลด optimization level
- ปรับ input shape/format

### 3. **Poor Accuracy after Quantization**
- เพิ่มจำนวน calibration images
- ใช้ representative calibration data
- ลองปรับ normalization parameters

### 4. **Large Model Size**
- ใช้ quantization (INT8)
- เพิ่ม optimization level
- ลด model complexity

## 💡 Best Practices

### 1. **Model Preparation:**
```python
# Export ONNX with proper settings
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,    # Use compatible version
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None    # Fixed input size
)
```

### 2. **Calibration Dataset:**
- ใช้ representative images (มีลักษณะคล้าย training data)
- จำนวน 50-200 รูป (ขึ้นกับ model complexity)
- ความละเอียดเดียวกับ training (640x640)
- มี diversity ในแสง/มุมมอง/สี

### 3. **Performance Testing:**
```bash
# Test converted model
python3 npu_inference.py \
  --model converted_model.rknn \
  --image test.jpg

# Compare with original
python3 realtime_monitor.py \
  --model converted_model.rknn \
  --image test.jpg \
  --interval 0.1
```

## 📈 Conversion Pipeline

### Complete Pipeline Example:
```bash
# Step 1: Convert ONNX to RKNN
python3 batch_onnx_converter.py \
  --onnx-dir ./onnx_models/ \
  --output-dir ./rknn_models/

# Step 2: Test converted models  
python3 batch_npu_inference.py \
  --model ./rknn_models/model_fp16.rknn \
  --input ./test_images/

# Step 3: Performance analysis
python3 npu_monitor.py test \
  --model ./rknn_models/model_fp16.rknn \
  --image test.jpg \
  --iterations 20
```

## 🎯 Summary

### การแปลง ONNX เป็น RKNN ทำให้:
1. **NPU Acceleration**: ใช้ 6 TOPS performance ของ RK3588
2. **Memory Efficiency**: Optimized memory usage
3. **Power Efficiency**: ประหยัดพลังงานกว่า CPU/GPU
4. **Integration**: ง่ายต่อการ integrate ใน application

### Workflow:
**ONNX** → **RKNN Toolkit2** → **RKNN Model** → **NPU Runtime** → **Fast Inference** 🚀

**ผลลัพธ์**: Model ที่เร็วกว่า ใช้พลังงานน้อยกว่า และ optimized สำหรับ RK3588 NPU!