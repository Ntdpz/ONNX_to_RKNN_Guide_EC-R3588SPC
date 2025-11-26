# Universal NPU Inference & ONNX Conversion Tools for EC-R3588SPC

ชุดเครื่องมือกลาง (Universal Tools) สำหรับใช้งาน NPU บนบอร์ด EC-R3588SPC และแปลงโมเดล ONNX เป็น RKNN รองรับโมเดล YOLOv5/v8 ทุกประเภท

## 🚀 Overview

ชุดเครื่องมือนี้ถูกออกแบบมาให้เป็น **Generic Tools** ที่สามารถใช้ได้กับทุกโมเดล:
- **Universal Inference**: รองรับ YOLO models ทุกประเภท (1 class หรือ 80 classes)
- **Auto-Detection**: ตรวจจับจำนวน class และ output format อัตโนมัติ
- **Batch Processing**: รองรับการแปลงและรันหลายไฟล์พร้อมกัน
- **Standard Pipeline**: รองรับมาตรฐาน YOLO (640x640, RGB)

## 📋 Table of Contents

1. [เครื่องมือที่มี (Tools)](#tools)
2. [Quick Start](#quick-start)
3. [NPU Inference (การใช้งาน)](#npu-inference)
4. [ONNX to RKNN Conversion (การแปลงไฟล์)](#onnx-conversion)
5. [Performance Monitoring](#monitoring)
6. [Examples (ตัวอย่างการใช้งาน)](#examples)

## 🛠️ เครื่องมือที่มี {#tools}

### NPU Inference Tools (สำหรับรันโมเดล)
| Script | Description | Features |
|--------|-------------|----------|
| `npu_inference.py` | รัน inference รูปเดียว | รองรับทุก YOLO model, Auto-NMS, Visualization |
| `batch_npu_inference.py` | รัน inference หลายรูป | Parallel processing, Progress tracking |
| `npu_monitor.py` | ตรวจสอบสถานะ NPU | System info, Benchmarking |
| `realtime_monitor.py` | ดูทรัพยากรแบบ Real-time | Live CPU/Memory/NPU tracking |

### ONNX Conversion Tools (สำหรับแปลงไฟล์)
| Script | Description | Features |
|--------|-------------|----------|
| `onnx_to_rknn_converter.py` | แปลง ONNX เป็น RKNN | รองรับ FP16/INT8, Auto-Calibration |
| `batch_onnx_converter.py` | แปลงหลาย model พร้อมกัน | Batch conversion, Summary report |

## 🚀 Quick Start {#quick-start}

### 1. ตรวจสอบระบบ

```bash
# Check NPU status
python3 npu_monitor.py info

# Start rknn_server if needed
sudo systemctl start rknn_server
```

### 2. รัน NPU Inference (แบบพื้นฐาน)

```bash
# รันโมเดลอะไรก็ได้ (สคริปต์จะ detect class เอง)
python3 npu_inference.py 
  --model your_model.rknn 
  --image test_image.jpg
```

### 3. แปลง ONNX เป็น RKNN (แบบพื้นฐาน)

```bash
# แปลงเป็น FP16 (ไม่ต้องใช้รูป Calibration)
python3 onnx_to_rknn_converter.py 
  --onnx your_model.onnx 
  --rknn your_model.rknn
```

## 🎯 NPU Inference {#npu-inference}

เครื่องมือ `npu_inference.py` ถูกออกแบบมาให้ยืดหยุ่น:
- **ไม่ต้องระบุ Classes**: ถ้าไม่ระบุ `--classes` โปรแกรมจะใช้เลข Class ID (0, 1, 2...) แทน
- **Auto-Resize**: ปรับขนาดภาพให้เข้ากับโมเดล (640x640) อัตโนมัติ
- **Auto-NMS**: ตัดกล่องซ้ำให้อัตโนมัติ

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | required | Path ไฟล์ RKNN model |
| `--image` | str | required | Path รูปภาพ input |
| `--classes` | list | auto | รายชื่อ class names (ถ้าไม่ใส่จะโชว์เป็น ID) |
| `--conf` | float | 0.5 | Confidence threshold (กรองความมั่นใจ) |
| `--iou` | float | 0.45 | IoU threshold (สำหรับตัดกล่องซ้ำ) |

## 🔄 ONNX to RKNN Conversion {#onnx-conversion}

เครื่องมือ `onnx_to_rknn_converter.py` รองรับการแปลง 2 โหมดหลัก:

### 1. FP16 Mode (เร็วและง่าย)
เหมาะสำหรับทดสอบโมเดลใหม่ ไม่ต้องเตรียมรูปภาพ
```bash
python3 onnx_to_rknn_converter.py 
  --onnx model.onnx 
  --rknn model_fp16.rknn 
  --target rk3588
```

### 2. INT8 Mode (เร็วสูงสุด)
เหมาะสำหรับใช้งานจริง (Production) ต้องใช้รูปภาพตัวอย่างเพื่อทำ Quantization
```bash
python3 onnx_to_rknn_converter.py 
  --onnx model.onnx 
  --rknn model_int8.rknn 
  --quantize 
  --images ./dataset_folder/
```

## 📊 Performance Monitoring {#monitoring}

### System Information
```bash
python3 npu_monitor.py info
```

### Benchmark (วัดความเร็ว)
```bash
python3 npu_monitor.py test 
  --model your_model.rknn 
  --image test.jpg 
  --iterations 100
```

## 💡 Examples (ตัวอย่างการใช้งานจริง) {#examples}

### Example 1: License Plate Recognition (เฉพาะทาง)
```bash
# ระบุชื่อ Class เองเพื่อให้แสดงผลถูกต้อง
python3 npu_inference.py 
  --model license_plate.rknn 
  --image car.jpg 
  --classes license_plate 
  --conf 0.7
```

### Example 2: Vehicle Classification (หลาย Class)
```bash
# ระบุหลาย Class ตามลำดับที่เทรนมา
python3 npu_inference.py 
  --model vehicle_type.rknn 
  --image traffic.jpg 
  --classes car truck bus motorcycle 
  --conf 0.4
```

### Example 3: Batch Processing (ทำทีละเยอะๆ)
```bash
# แปลงทุกไฟล์ในโฟลเดอร์
python3 batch_onnx_converter.py 
  --onnx-dir ./my_onnx_models/ 
  --output-dir ./my_rknn_models/

# รันทุกรูปในโฟลเดอร์
python3 batch_npu_inference.py 
  --model my_model.rknn 
  --input ./test_images/ 
  --output ./results/
```

## 🚨 Troubleshooting

### Common Issues
1. **"Unsupported operator"**: ONNX Opset เก่า/ใหม่ไป -> แนะนำให้ใช้ **Opset 12** ตอน export จาก PyTorch
2. **"Input size mismatch"**: โมเดลรับ 640x640 แต่ส่งขนาดอื่น -> สคริปต์นี้มี Auto-resize ช่วยจัดการให้
3. **"NPU timeout"**: โมเดลใหญ่เกินไป หรือความร้อนสูง -> เช็ค `npu_monitor.py`

---
**Note:** เครื่องมือนี้ทดสอบบนบอร์ด **EC-R3588SPC** (RK3588) รองรับ YOLOv5 และ YOLOv8


## 🛠️ เครื่องมือที่มี {#tools}

### NPU Inference Tools
| Script | Description | Features |
|--------|-------------|----------|
| `npu_inference.py` | รัน inference รูปเดียว | NPU acceleration, NMS, visualization |
| `batch_npu_inference.py` | รัน inference หลายรูป | Parallel processing, progress tracking |
| `npu_monitor.py` | NPU performance monitoring | System info, benchmarking |
| `realtime_monitor.py` | Real-time resource monitor | Live CPU/Memory/NPU tracking |

### ONNX Conversion Tools
| Script | Description | Features |
|--------|-------------|----------|
| `onnx_to_rknn_converter.py` | แปลง ONNX เป็น RKNN | FP16/INT8, quantization, testing |
| `batch_onnx_converter.py` | แปลงหลาย model พร้อมกัน | Batch conversion, summary report |

### Documentation
| File | Description |
|------|-------------|
| `NPU_TOOLS_GUIDE.md` | คู่มือใช้งาน NPU tools |
| `ONNX_to_RKNN_Guide.md` | คู่มือแปลง ONNX เป็น RKNN |
| `README_NPU_Inference.md` | รายละเอียด NPU inference |

## 🚀 Quick Start {#quick-start}

### 1. ตรวจสอบระบบ

```bash
# Check NPU status
python3 npu_monitor.py info

# Start rknn_server if needed
sudo systemctl start rknn_server
```

### 2. รัน NPU Inference

```bash
# Basic inference
python3 npu_inference.py \
  --model model.rknn \
  --image image.jpg \
  --classes code province

# Batch processing
python3 batch_npu_inference.py \
  --model model.rknn \
  --input ./images/ \
  --classes code province
```

### 3. แปลง ONNX เป็น RKNN

```bash
# Single model conversion
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model.rknn

# Batch conversion
python3 batch_onnx_converter.py \
  --onnx-dir ./onnx_models/ \
  --output-dir ./rknn_models/
```

## 🎯 NPU Inference {#npu-inference}

### Basic Usage

```bash
# CodeProvince Detection
python3 npu_inference.py \
  --model codeprovince_best_fp32.rknn \
  --image test.jpg \
  --classes code province \
  --conf 0.5

# Vehicle Type Detection  
python3 npu_inference.py \
  --model vehicle_detection_best_fp32.rknn \
  --image car.jpg \
  --classes car truck bus motorcycle \
  --conf 0.4
```

### Batch Processing

```bash
# Process all images in directory
python3 batch_npu_inference.py \
  --model codeprovince_best_fp32.rknn \
  --input ./test_images/ \
  --output ./results/ \
  --classes code province \
  --workers 4
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | required | Path ไฟล์ RKNN model |
| `--image` | str | required | Path รูปภาพ input |
| `--classes` | list | auto | รายชื่อ class names |
| `--conf` | float | 0.5 | Confidence threshold |
| `--iou` | float | 0.45 | IoU threshold สำหรับ NMS |
| `--output` | str | auto | Output directory |

## 🔄 ONNX to RKNN Conversion {#onnx-conversion}

### วิธีการแปลง

```bash
# FP16 Conversion (No quantization)
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model_fp16.rknn \
  --target rk3588

# INT8 Conversion (With quantization)
python3 onnx_to_rknn_converter.py \
  --onnx model.onnx \
  --rknn model_int8.rknn \
  --quantize \
  --images ./calibration_images/
```

### Batch Conversion

```bash
# Convert all ONNX models
python3 batch_onnx_converter.py \
  --onnx-dir /path/to/onnx/models/ \
  --output-dir ./converted_models/ \
  --test-image test.jpg
```

### Conversion Flow

```
ONNX Model (Float32/16) 
       ↓
RKNN Toolkit2 Processing:
  • Load ONNX
  • Configure (RK3588)
  • Build & Optimize  
  • Quantization (optional)
       ↓
RKNN Model (FP16/INT8)
       ↓
NPU Runtime (6 TOPS)
```

## 📊 Performance Monitoring {#monitoring}

### System Information

```bash
# Show NPU and system info
python3 npu_monitor.py info
```

### Performance Benchmarking

```bash
# Benchmark model performance
python3 npu_monitor.py test \
  --model model.rknn \
  --image test.jpg \
  --iterations 20 \
  --output results.json
```

### Real-time Resource Monitoring

```bash
# Monitor resources during inference
python3 realtime_monitor.py \
  --model model.rknn \
  --image test.jpg \
  --interval 0.1 \
  --output detailed_usage.json
```

## 📈 ผลการทดสอบ {#results}

### NPU Performance

| Model | Type | Inference Time | FPS | NPU Usage | Memory Impact |
|-------|------|----------------|-----|-----------|---------------|
| CodeProvince (FP32) | Original | 68-86ms | 11-14 | 1.8% | +1.2% |
| CodeProvince (FP16) | Converted | 75.8ms | 13.2 | 1.9% | +1.3% |
| Vehicle Detection (FP32) | Original | 76ms | 13.1 | 1.9% | +1.3% |

### System Resources

| Resource | Baseline | During Inference | Peak | Status |
|----------|----------|------------------|------|--------|
| **CPU** | 47-54% | 51-61% (avg) | 100% (spike) | 🟢 Normal |
| **Memory** | 56.7-57.3% | +1.2-1.3% | 58.5% | 🟢 Normal |
| **NPU Freq** | 1000 MHz | 1000 MHz | 1000 MHz | 🟢 Max Speed |
| **Temperature** | 44-46°C | +2.8-4.6°C | 48-52°C | 🟢 Normal |

### ONNX Conversion Results

| Model | ONNX→RKNN Size | Time | Status | Performance |
|-------|----------------|------|---------|-------------|
| codeprovince_best | →15.1MB | 9.0s | ✅ Success | Same as original |
| license_plate_model | →46.5MB | 16.5s | ✅ Success | Optimized |
| vehicle_detection_best | →15.2MB | 8.7s | ✅ Success | Same as original |

**Conversion Success Rate**: 5/5 (100%)

## 🔧 Configuration Examples

### Model-Specific Configurations

```bash
# CodeProvince Detection (2 classes)
python3 npu_inference.py \
  --model codeprovince_best_fp32.rknn \
  --image license_plate.jpg \
  --classes code province \
  --conf 0.6 \
  --iou 0.4

# License Plate Detection  
python3 npu_inference.py \
  --model license_plate_model.rknn \
  --image car_image.jpg \
  --classes license_plate \
  --conf 0.7

# Vehicle Type Classification (7 classes)
python3 npu_inference.py \
  --model vehicle_detection_best_fp32.rknn \
  --image traffic.jpg \
  --classes car truck bus motorcycle bicycle motorbike van \
  --conf 0.4
```

### Normalization Settings

```bash
# Default (0-255 input)
--mean 0 0 0 --std 255 255 255

# ImageNet normalization
--mean 123.675 116.28 103.53 --std 58.395 57.12 57.375
```

## 🚨 Troubleshooting {#troubleshooting}

### NPU Issues

```bash
# RKNN server not running
sudo systemctl start rknn_server
sudo systemctl enable rknn_server

# Check NPU devices
ls -la /dev/rknpu*

# Check NPU frequency
cat /sys/class/devfreq/fdab0000.npu/cur_freq
```

### Memory Issues

```bash
# Check system memory
free -h
python3 npu_monitor.py info

# Monitor during inference
python3 realtime_monitor.py --model model.rknn --image test.jpg
```

### Performance Issues

```bash
# Set NPU to performance mode
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor

# Check thermal throttling
python3 npu_monitor.py info
```

### Model Conversion Issues

```bash
# Check ONNX model validity
python3 -c "import onnx; onnx.checker.check_model(onnx.load('model.onnx'))"

# Test converted model
python3 onnx_to_rknn_converter.py --rknn model.rknn --test --image test.jpg
```

## 💡 Best Practices

### Performance Optimization

1. **NPU Governor**: ใช้ `performance` mode สำหรับ consistent performance
2. **Batch Processing**: ใช้ parallel workers สำหรับหลายรูป  
3. **Model Selection**: FP16 สำหรับ accuracy, INT8 สำหรับ speed
4. **Input Size**: 640x640 optimal สำหรับ YOLO models

### Model Preparation

```python
# Export ONNX with optimal settings
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None  # Fixed input size
)
```

### Calibration Dataset

- ใช้ representative images (คล้าย training data)
- จำนวน 50-200 รูป
- ความละเอียด 640x640
- หลากหลายในแสง/มุมมอง/สี

## 🎯 Use Cases

### 1. License Plate Recognition Pipeline

```bash
# Step 1: Detect license plates
python3 npu_inference.py \
  --model license_plate_model.rknn \
  --image car.jpg \
  --classes license_plate \
  --conf 0.7

# Step 2: Extract code/province
python3 npu_inference.py \
  --model codeprovince_model.rknn \
  --image cropped_plate.jpg \
  --classes code province \
  --conf 0.6
```

### 2. Traffic Monitoring

```bash
# Batch process traffic camera footage
python3 batch_npu_inference.py \
  --model vehicle_detection_best.rknn \
  --input ./traffic_images/ \
  --output ./traffic_results/ \
  --classes car truck bus motorcycle \
  --workers 4 \
  --conf 0.4
```

### 3. Model Development Workflow

```bash
# Step 1: Convert ONNX to RKNN
python3 onnx_to_rknn_converter.py \
  --onnx new_model.onnx \
  --rknn new_model.rknn

# Step 2: Test performance  
python3 npu_monitor.py test \
  --model new_model.rknn \
  --image test.jpg \
  --iterations 50

# Step 3: Validate accuracy
python3 batch_npu_inference.py \
  --model new_model.rknn \
  --input ./validation_set/ \
  --output ./validation_results/
```

## 📚 Additional Resources

### Documentation Files
- `NPU_TOOLS_GUIDE.md` - Detailed usage guide
- `ONNX_to_RKNN_Guide.md` - Conversion guide
- `README_NPU_Inference.md` - Inference details

### Example Outputs
- `./npu_results/` - Inference results with bounding boxes
- `./batch_results/` - Batch processing summaries  
- `./converted_models/` - Converted RKNN models
- `conversion_summary.txt` - Conversion statistics
- `batch_summary.txt` - Batch processing statistics

### Hardware Requirements
- **Platform**: EC-R3588SPC with RK3588 SoC
- **NPU**: 6 TOPS @ 1000 MHz
- **Memory**: 8+ GB RAM recommended  
- **Storage**: SSD recommended for model loading
- **OS**: Ubuntu 20.04+ with RKNN Runtime

### Software Dependencies
- **RKNN Toolkit2**: v2.3.0+
- **Python**: 3.8+
- **OpenCV**: 4.4+
- **NumPy**: 1.17+
- **rknn_server**: Running and enabled

## 🎉 Summary

### เครื่องมือครบชุดสำหรับ NPU:
- ✅ **NPU Inference**: Fast, efficient object detection
- ✅ **ONNX Conversion**: Seamless model deployment  
- ✅ **Performance Monitoring**: Real-time resource tracking
- ✅ **Batch Processing**: Scale to multiple images/models
- ✅ **Documentation**: Complete guides and examples

### Key Benefits:
- 🚀 **6 TOPS Performance** on RK3588 NPU
- ⚡ **13+ FPS** inference speed  
- 💾 **Low Memory Usage** (+1-2% during inference)
- 🌡️ **Thermal Efficient** (+2-4°C temperature rise)
- 📊 **100% Success Rate** in ONNX→RKNN conversion

**Ready for Production**: Deploy your YOLO models on NPU with confidence! 🎯

---

## 📞 Support

For issues or questions:
1. Check `Troubleshooting` section above
2. Review log outputs from monitoring tools
3. Validate model format and requirements  
4. Test with provided example models first

**Happy NPU Computing!** 🚀