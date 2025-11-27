# 🚀 Universal ONNX to RKNN Converter - คู่มือการใช้งาน

## 📋 Overview

**Universal ONNX to RKNN Converter** เป็นเครื่องมือที่รองรับการแปลง ONNX Model ทุกประเภทเป็น RKNN Format พร้อมระบบ Auto-detection และ Configuration ครบทุกอย่าง

### ✨ Features

- ✅ **Auto-detect Model Type** - วิเคราะห์ YOLOv5/v8/v10 อัตโนมัติ
- ✅ **Full Configuration** - ปรับแต่งได้ทุก parameter
- ✅ **Multi-Platform Support** - รองรับ RK3588, RK3576, RK3562, RV1109/1126, RK1808, RK3399Pro
- ✅ **Flexible Quantization** - FP16, INT8, UINT8 + Multiple algorithms
- ✅ **Smart Recommendations** - แนะนำ preprocessing/postprocessing ตาม model type
- ✅ **Verification System** - ตรวจสอบ model หลัง convert

---

## 📦 Requirements

### ซอฟต์แวร์ที่ต้องมี

```bash
# Python
Python >= 3.8

# Libraries
rknn-toolkit2 >= 2.0.0
onnx >= 1.12.0
numpy >= 1.19.0
```

### การติดตั้ง Dependencies

```bash
# ติดตั้ง RKNN-Toolkit2
pip3 install rknn-toolkit2

# ติดตั้ง ONNX
pip3 install onnx

# ติดตั้ง NumPy
pip3 install numpy
```

### ตรวจสอบการติดตั้ง

```bash
python3 -c "from rknn.api import RKNN; print('RKNN OK')"
python3 -c "import onnx; print('ONNX OK')"
```

---

## 🚀 Quick Start

### 1. Basic FP16 Conversion (แนะนำสำหรับเริ่มต้น)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_fp16.rknn
```

**ผลลัพธ์:**
- Model RKNN ที่ใช้ FP16 quantization
- Accuracy ใกล้เคียง ONNX มากที่สุด
- ขนาดไฟล์ประมาณครึ่งหนึ่งของ FP32

### 2. INT8 Quantization (สำหรับ Performance)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_int8.rknn \
    --quantize \
    --dataset dataset.txt
```

**ผลลัพธ์:**
- Model RKNN ที่ใช้ INT8 quantization
- ความเร็วเพิ่มขึ้น 2-4 เท่า
- ขนาดไฟล์เล็กลง 60-70%
- Accuracy ลดลงเล็กน้อย (1-5%)

### 3. With Verification

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model.rknn \
    --verify
```

**ตรวจสอบ:**
- Model โหลดได้ถูกต้อง
- Runtime initialization (บน target platform)

---

## 📖 ตัวอย่างการใช้งาน

### Example 1: YOLOv8 Object Detection (Basic)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx yolov8n.onnx \
    --rknn yolov8n_fp16.rknn \
    --platform rk3588
```

**Output:**
```
🔍 Analyzing ONNX model...
   📝 Graph Name: torch_jit
   📐 Input Shape: [1, 3, 640, 640]
   📊 Output Shapes:
      [0] output0: [1, 84, 8400]
   🎯 Detected Type: YOLOv8

📋 Conversion Settings:
   📁 Input:  yolov8n.onnx
   💾 Output: yolov8n_fp16.rknn
   🎯 Platform: rk3588
   🔧 Quantization: FP16
   ⚡ Optimization Level: 3

✅ Conversion completed successfully!

💡 Recommendations for YOLOv8:
   Preprocessing:
   - Use Letterbox resize (maintain aspect ratio)
   - Padding with gray color (114, 114, 114)
   - Convert BGR → RGB
   - Normalize: mean=[0,0,0], std=[255,255,255]
```

### Example 2: YOLOv8 with INT8 Quantization

```bash
python3 universal_onnx_to_rknn.py \
    --onnx yolov8n.onnx \
    --rknn yolov8n_int8.rknn \
    --quantize \
    --dataset coco_calibration.txt \
    --algorithm mmse \
    --verify
```

**คำอธิบาย:**
- `--quantize`: เปิดใช้งาน INT8 quantization
- `--dataset`: ไฟล์รายชื่อรูปสำหรับ calibration (500-1000 รูป)
- `--algorithm mmse`: ใช้ MMSE algorithm (accuracy ดีกว่า normal)
- `--verify`: ตรวจสอบ model หลัง convert

### Example 3: Custom Normalization

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model.rknn \
    --mean 123.675 116.28 103.53 \
    --std 58.395 57.12 57.375
```

**Use case:**
- Model ที่ train ด้วย ImageNet normalization
- Custom preprocessing pipeline

### Example 4: Different Platform (RK3576)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_rk3576.rknn \
    --platform rk3576 \
    --quantize \
    --dataset dataset.txt
```

### Example 5: Advanced Configuration

```bash
python3 universal_onnx_to_rknn.py \
    --onnx yolov8s.onnx \
    --rknn yolov8s_hybrid.rknn \
    --platform rk3588 \
    --quantize \
    --dtype INT8 \
    --algorithm kl_divergence \
    --method channel \
    --dataset dataset.txt \
    --optimization 3 \
    --mean 0 0 0 \
    --std 255 255 255 \
    --hybrid-quant \
    --hybrid-quant-file hybrid_config.txt \
    --custom-string "v1.0.0-production" \
    --verify \
    --verbose
```

**คำอธิบาย:**
- `--dtype INT8`: ระบุ data type สำหรับ quantization
- `--algorithm kl_divergence`: ใช้ KL Divergence algorithm
- `--method channel`: Quantize แบบ per-channel (accuracy ดีกว่า per-layer)
- `--optimization 3`: Optimization สูงสุด
- `--hybrid-quant`: Mix FP16 และ INT8 (layers ที่สำคัญใช้ FP16)
- `--custom-string`: เพิ่ม version tag
- `--verbose`: แสดงข้อมูล debug ละเอียด

---

## ⚙️ Parameters Reference

### Required Parameters

| Parameter | Short | Type | Description |
|-----------|-------|------|-------------|
| `--onnx` | `-i` | string | Path to input ONNX model file |
| `--rknn` | `-o` | string | Path to output RKNN model file |

### Platform Settings

| Parameter | Short | Default | Choices | Description |
|-----------|-------|---------|---------|-------------|
| `--platform` | `-p` | `rk3588` | rk3588, rk3576, rk3562, rv1109, rv1126, rk1808, rk3399pro | Target platform |
| `--sub-platform` | - | `None` | - | Sub-platform for specific chip variants |

### Quantization Settings

| Parameter | Short | Default | Choices | Description |
|-----------|-------|---------|---------|-------------|
| `--quantize` | `-q` | `False` | - | Enable quantization (INT8) |
| `--dtype` | - | `INT8` | INT8, FP16, UINT8 | Quantization data type |
| `--algorithm` | - | `normal` | normal, mmse, kl_divergence | Quantization algorithm |
| `--method` | - | `channel` | channel, layer | Quantization method |
| `--dataset` | `-d` | `None` | - | Path to dataset.txt for calibration |

### Optimization Settings

| Parameter | Short | Default | Range | Description |
|-----------|-------|---------|-------|-------------|
| `--optimization` | - | `3` | 0-3 | Optimization level (higher = more optimized) |

### Model Settings

| Parameter | Short | Format | Example | Description |
|-----------|-------|--------|---------|-------------|
| `--mean` | - | R G B | `--mean 0 0 0` | Mean values for normalization |
| `--std` | - | R G B | `--std 255 255 255` | Std values for normalization |
| `--input-size` | - | C H W | `--input-size 3 640 640` | Input size (auto-detected if omitted) |
| `--outputs` | - | list | `--outputs output0 output1` | Output layer names (auto-detected if omitted) |

### Advanced Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hybrid-quant` | `False` | Enable hybrid quantization (mix FP16+INT8) |
| `--hybrid-quant-file` | `None` | Path to hybrid quantization config file |
| `--custom-string` | `None` | Custom string for model version tracking |

### Other Options

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--verify` | `-v` | `False` | Verify model after conversion |
| `--verbose` | - | `False` | Enable verbose output |

---

## 📊 Quantization Algorithm Comparison

### Normal (แนะนำสำหรับทั่วไป)

```bash
--algorithm normal
```

**ข้อดี:**
- ✅ เร็วที่สุด
- ✅ ใช้งานง่าย
- ✅ Accuracy ดีสำหรับ model ส่วนใหญ่

**ข้อเสีย:**
- ⚠️ Accuracy อาจต่ำกว่า algorithm อื่น

**Use case:**
- Prototyping
- Model ทั่วไป
- ต้องการความเร็วในการ convert

### MMSE (แนะนำสำหรับ Accuracy)

```bash
--algorithm mmse
```

**ข้อดี:**
- ✅ Accuracy สูงกว่า normal
- ✅ เหมาะกับ complex model
- ✅ ลด quantization error ได้ดี

**ข้อเสีย:**
- ⚠️ ช้ากว่า normal (2-3 เท่า)
- ⚠️ ใช้ memory มากกว่า

**Use case:**
- Production deployment
- Model ที่ต้องการ accuracy สูง
- YOLOv8, Complex architectures

### KL Divergence (แนะนำสำหรับ specific cases)

```bash
--algorithm kl_divergence
```

**ข้อดี:**
- ✅ ดีกับ model ที่มี activation แปลกๆ
- ✅ Handle outliers ได้ดี

**ข้อเสีย:**
- ⚠️ ช้ามาก
- ⚠️ อาจไม่ดีกว่า MMSE ในบาง case

**Use case:**
- Model ที่มี extreme values
- Classification tasks
- Research purposes

---

## 📁 Dataset Preparation (สำหรับ INT8)

### ข้อกำหนด Dataset

1. **จำนวนรูป:** 500-1000 รูป (แนะนำ)
2. **รูปแบบ:** JPG, PNG, BMP
3. **ขนาด:** ไม่จำกัด (จะ resize อัตโนมัติ)
4. **ความหลากหลาย:** ครอบคลุมทุก use case

### วิธีสร้าง dataset.txt

#### วิธีที่ 1: Manual

```bash
# สร้างไฟล์ dataset.txt
nano dataset.txt
```

```
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
...
```

#### วิธีที่ 2: Script (แนะนำ)

```bash
# ใช้ create_dataset_txt.py
python3 create_dataset_txt.py \
    -i /path/to/images \
    -o dataset.txt \
    -n 1000
```

#### วิธีที่ 3: Command Line

```bash
# Linux/Mac
find /path/to/images -name "*.jpg" | head -1000 > dataset.txt

# ใช้ absolute path
find "$(pwd)/images" -name "*.jpg" | head -1000 > dataset.txt
```

### ตัวอย่าง dataset.txt ที่ถูกต้อง

```
/home/user/dataset/train/img_001.jpg
/home/user/dataset/train/img_002.jpg
/home/user/dataset/train/img_003.jpg
/home/user/dataset/val/img_001.jpg
/home/user/dataset/val/img_002.jpg
```

### Best Practices

- ✅ ใช้ **absolute path** (ไม่ใช่ relative path)
- ✅ เลือกรูป**หลากหลาย** จาก training set
- ✅ จำนวน **500-1000 รูป** (ไม่ต้องมากเกินไป)
- ✅ ตรวจสอบ path **ถูกต้อง** (ไม่มี typo)
- ❌ อย่าใช้ test set (ป้องกัน data leakage)

---

## 🎯 Platform-Specific Guides

### RK3588 (EC-R3588SPC, Orange Pi 5, etc.)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model.rknn \
    --platform rk3588 \
    --quantize \
    --dataset dataset.txt
```

**Hardware:**
- NPU: 6 TOPS @ 1GHz
- Cores: 3x NPU cores
- Memory: Shared with system

**Recommendations:**
- Optimization level: **3** (maximum)
- Quantization: **INT8** for best performance
- Algorithm: **mmse** for accuracy

### RK3576

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model.rknn \
    --platform rk3576
```

**Hardware:**
- NPU: 6 TOPS
- Latest generation NPU

### RV1109/RV1126 (Embedded Vision)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model.rknn \
    --platform rv1126 \
    --quantize \
    --dtype INT8
```

**Hardware:**
- NPU: 2 TOPS
- Limited memory

**Recommendations:**
- **ต้อง**ใช้ INT8 (FP16 จะช้า)
- เลือก model เล็กๆ (YOLOv8n, YOLOv5s)
- Optimization level: **3**

---

## 🔍 Auto-Detection Examples

### YOLOv8 Detection

**ONNX Info:**
- Input: `(1, 3, 640, 640)`
- Output: `(1, 84, 8400)`

**Auto-detected Settings:**
```
✅ Detected: YOLOv8
   
💡 Recommendations:
   - Preprocessing: Letterbox + RGB
   - Confidence: 0.25
   - IoU: 0.7-0.85
   - NMS: Required
```

### YOLOv5 Detection

**ONNX Info:**
- Input: `(1, 3, 640, 640)`
- Output: `(1, 25200, 85)`

**Auto-detected Settings:**
```
✅ Detected: YOLOv5
   
💡 Recommendations:
   - Preprocessing: Letterbox + RGB
   - Confidence: 0.25-0.5
   - IoU: 0.45-0.7
   - NMS: Required
```

### YOLOv10 Detection

**ONNX Info:**
- Input: `(1, 3, 640, 640)`
- Output: `(1, 300, 6)`

**Auto-detected Settings:**
```
✅ Detected: YOLOv10
   
💡 Recommendations:
   - Preprocessing: Letterbox + RGB
   - NMS: Not required (built-in)
```

---

## 🐛 Troubleshooting

### ❌ Error: "Failed to load ONNX model"

**สาเหตุ:**
- ไฟล์ ONNX เสีย
- ONNX version ไม่รองรับ
- Operators ไม่รองรับ

**วิธีแก้:**
```bash
# ตรวจสอบ ONNX
python3 -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"

# Export ONNX ใหม่ (PyTorch)
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=11)
```

### ❌ Error: "Build failed"

**สาเหตุ:**
- Dataset path ผิด
- Memory ไม่พอ
- Operators ไม่รองรับ

**วิธีแก้:**
```bash
# ตรวจสอบ dataset
head -5 dataset.txt
ls -l $(head -1 dataset.txt)  # ตรวจสอบไฟล์แรก

# ลด optimization level
--optimization 2  # จาก 3

# ใช้ FP16 แทน INT8
# ลบ --quantize flag
```

### ⚠️ Warning: "Runtime initialization failed"

**สาเหตุ:**
- รัน verify บน x86 platform (ปกติ)

**วิธีแก้:**
- ไม่ต้องแก้! เป็นเรื่องปกติบน x86
- Model จะทำงานได้บน RK3588 hardware

### ❌ Error: "Dataset file not found"

**วิธีแก้:**
```bash
# ใช้ absolute path
pwd  # ดู current directory
# แก้ dataset.txt ให้ใช้ full path
```

### 📊 Model Size ใหญ่เกินไป

**ปัญหา:** RKNN model ใหญ่กว่า ONNX

**วิธีแก้:**
```bash
# ใช้ INT8 quantization
--quantize --dataset dataset.txt

# ตรวจสอบขนาด
ls -lh model.onnx model.rknn
```

**Expected sizes:**
- FP32 ONNX: 12 MB → FP16 RKNN: 7.6 MB (60%)
- FP32 ONNX: 12 MB → INT8 RKNN: 4.7 MB (40%)

---

## 📈 Performance Optimization

### 1. Quantization Strategy

**Development Phase:**
```bash
# ใช้ FP16 - Accuracy สูงสุด
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_fp16.rknn
```

**Production Phase:**
```bash
# ใช้ INT8 - Performance สูงสุด
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_int8.rknn \
    --quantize \
    --algorithm mmse \
    --dataset dataset.txt
```

### 2. Optimization Level

**Testing:**
```bash
--optimization 1  # รวดเร็ว แต่ performance ต่ำ
```

**Production:**
```bash
--optimization 3  # ช้ากว่า แต่ performance สูงสุด
```

### 3. Hybrid Quantization

**Use case:** Layers บางตัวสำคัญมาก (e.g., detection head)

```bash
# สร้าง hybrid_config.txt
echo "output_layer FP16" > hybrid_config.txt
echo "bbox_head FP16" >> hybrid_config.txt

python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_hybrid.rknn \
    --quantize \
    --hybrid-quant \
    --hybrid-quant-file hybrid_config.txt
```

**ผลลัพธ์:**
- Accuracy ใกล้เคียง FP16
- Performance ใกล้เคียง INT8
- ขนาดไฟล์ระหว่าง FP16-INT8

---

## 🎓 Best Practices

### ✅ DO

1. **ใช้ FP16 สำหรับ Development**
   ```bash
   # Quick testing
   python3 universal_onnx_to_rknn.py --onnx model.onnx --rknn model.rknn
   ```

2. **สร้าง Dataset ที่ดี**
   - 500-1000 รูป
   - หลากหลายครอบคลุม use cases
   - ใช้รูปจาก training set

3. **Verify Model**
   ```bash
   --verify  # เสมอ!
   ```

4. **เปรียบเทียบ Accuracy**
   - Test ONNX vs RKNN
   - ตรวจสอบ mAP, F1-score

5. **ทดสอบบน Hardware จริง**
   - Benchmark บน RK3588
   - วัด FPS, Latency

### ❌ DON'T

1. **อย่าใช้ Test Set ใน Dataset.txt**
   - Data leakage!

2. **อย่าใช้ Optimization 0**
   - Performance แย่

3. **อย่าลืม Dataset สำหรับ INT8**
   - Accuracy จะต่ำมาก

4. **อย่าใช้ Relative Path**
   - จะหา file ไม่เจอ

5. **อย่าใช้ Dataset น้อยเกินไป**
   - < 100 รูป → Accuracy แย่

---

## 📚 Workflow Example

### Complete Workflow: YOLOv8 Custom Model

#### Step 1: Export ONNX

```python
# PyTorch
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', simplify=True)
```

#### Step 2: Prepare Dataset

```bash
python3 create_dataset_txt.py \
    -i dataset/train/images \
    -o dataset.txt \
    -n 1000
```

#### Step 3: Convert to FP16 (Development)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx best.onnx \
    --rknn best_fp16.rknn \
    --verify
```

#### Step 4: Test on Hardware

```bash
# Transfer to RK3588
scp best_fp16.rknn firefly@192.168.1.100:~/models/

# Run inference
python3 npu_inference.py \
    --model best_fp16.rknn \
    --image test.jpg \
    --conf 0.25 \
    --iou 0.85
```

#### Step 5: Convert to INT8 (Production)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx best.onnx \
    --rknn best_int8.rknn \
    --quantize \
    --algorithm mmse \
    --dataset dataset.txt \
    --verify
```

#### Step 6: Benchmark

```bash
# Compare FP16 vs INT8
python3 benchmark.py --model best_fp16.rknn --runs 100
python3 benchmark.py --model best_int8.rknn --runs 100
```

#### Step 7: Deploy

```bash
# Final deployment
cp best_int8.rknn /opt/models/production/
```

---

## 🔗 Related Documentation

- [PREPROCESSING_POSTPROCESSING_GUIDE.md](./PREPROCESSING_POSTPROCESSING_GUIDE.md) - คู่มือ Pre/Post processing
- [COMPLETE_BEGINNER_GUIDE.md](./COMPLETE_BEGINNER_GUIDE.md) - คู่มือเริ่มต้น
- [ONNX_to_RKNN_Guide.md](./ONNX_to_RKNN_Guide.md) - คู่มือ ONNX to RKNN พื้นฐาน

---

## ❓ FAQ

### Q: FP16 กับ INT8 ต่างกันยังไง?

**A:**
| | FP16 | INT8 |
|---|------|------|
| **Accuracy** | ✅ สูงสุด (99%) | ⚠️ ลดลงเล็กน้อย (95-98%) |
| **Speed** | ⚡ Baseline | ⚡⚡⚡ เร็วกว่า 2-4x |
| **Size** | 📦 Baseline | 📦 เล็กกว่า 50% |
| **Use case** | Development, High accuracy | Production, Real-time |

### Q: Dataset ต้องเป็นรูปจาก Training Set หรือ?

**A:** ✅ ใช่! ต้องเป็นรูปจาก training set (ไม่ใช่ test set)
- ✅ Training set: OK
- ✅ Validation set: OK  
- ❌ Test set: NOT OK (data leakage)

### Q: จำนวนรูปใน Dataset ควรเท่าไหร่?

**A:**
- ✅ **500-1000 รูป**: แนะนำ
- ⚠️ 100-500 รูป: พอใช้
- ❌ < 100 รูป: น้อยเกินไป

### Q: MMSE ช้ากว่า Normal มากไหม?

**A:** ช้ากว่า 2-3 เท่า แต่ accuracy ดีกว่า
- Normal: 30s
- MMSE: 60-90s
- KL Divergence: 120-180s

### Q: Hybrid Quantization คืออะไร?

**A:** Mix FP16 + INT8
- Layers สำคัญ → FP16 (accuracy)
- Layers ทั่วไป → INT8 (speed)
- ได้ทั้ง accuracy และ performance

### Q: แปลงบน x86 แล้วใช้บน ARM ได้ไหม?

**A:** ✅ ได้! แปลงบน x86/x64 Linux → ใช้บน RK3588 ARM

### Q: รองรับ Model อะไรบ้าง?

**A:**
- ✅ YOLOv5, YOLOv8, YOLOv10, YOLO-NAS
- ✅ ResNet, MobileNet, EfficientNet
- ✅ Custom models (ถ้า operators รองรับ)

---

## 📞 Support & Resources

### Documentation
- [RKNN-Toolkit2 Official Docs](https://github.com/rockchip-linux/rknn-toolkit2)
- [ONNX Documentation](https://onnx.ai/onnx/)

### Community
- [Rockchip NPU Forum](https://forum.radxa.com/)
- GitHub Issues

### Tools
- `create_dataset_txt.py` - Dataset creation
- `npu_inference.py` - Inference testing
- `benchmark.py` - Performance testing

---

**📅 Last Updated:** November 27, 2025  
**✍️ Author:** Firefly EC-R3588SPC Development Team  
**🔖 Version:** 1.0.0
