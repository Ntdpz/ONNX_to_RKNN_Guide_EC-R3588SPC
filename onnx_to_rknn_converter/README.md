# Universal ONNX to RKNN Converter - คู่มือการใช้งาน

เครื่องมือสำหรับแปลง ONNX Model เป็น RKNN Model สำหรับ Rockchip NPU (RK3588, RK3576, etc.)

---

## 📋 ภาพรวม

`universal_onnx_to_rknn.py` เป็น Universal Converter ที่รองรับการแปลง ONNX Model ทุกประเภทเป็น RKNN Model พร้อมฟีเจอร์:
- ✅ Auto-detection (YOLOv5, YOLOv8, YOLOv10, Classification)
- ✅ FP16 และ INT8 Quantization
- ✅ Configurable ทุกพารามิเตอร์
- ✅ รองรับหลาย Platform (RK3588, RK3576, RK3562, etc.)
- ✅ Output อัตโนมัติไปยัง `Model-AI/<model-name>/`

---

## 🔧 Environment & Requirements

### ระบบที่รองรับ
- **OS:** Ubuntu 20.04+ / WSL2
- **Python:** 3.8 - 3.10
- **Architecture:** x86_64 (สำหรับ development)

### ติดตั้ง RKNN-Toolkit2

```bash
# 1. Clone RKNN-Toolkit2
cd /path/to/your/workspace
git clone https://github.com/rockchip-linux/rknn-toolkit2.git

# 2. ติดตั้ง Dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libxslt1-dev zlib1g-dev libglib2.0-dev
sudo apt-get install -y libsm6 libgl1-mesa-glx libprotobuf-dev gcc

# 3. ติดตั้ง RKNN-Toolkit2 (Python 3.10)
cd rknn-toolkit2/rknn-toolkit2/packages
pip3 install rknn_toolkit2-2.3.0-cp310-cp310-linux_x86_64.whl

# หรือสำหรับ Python 3.8
# pip3 install rknn_toolkit2-2.3.0-cp38-cp38-linux_x86_64.whl

# 4. ติดตั้ง Dependencies อื่นๆ
pip3 install onnx
pip3 install numpy
```

### ตรวจสอบการติดตั้ง

```bash
python3 -c "from rknn.api import RKNN; print('RKNN Toolkit2 installed successfully')"
```

---

## 📂 โครงสร้างโฟลเดอร์

```
onnx_to_rknn_converter/
├── README.md                        # เอกสารนี้
└── universal_onnx_to_rknn.py       # Converter script
```

**Output จะไปที่:**
```
Model-AI/
└── <model-name>/
    └── <output-name>.rknn
```

---

## 🚀 วิธีการใช้งาน

### Syntax พื้นฐาน

```bash
python3 universal_onnx_to_rknn.py \
    --onnx <ไฟล์ ONNX> \
    --model-name <ชื่อโมเดล> \
    --output-name <ชื่อไฟล์ .rknn>
```

---

## 📖 Step-by-Step Guide

### Step 1: เตรียม ONNX Model

วาง ONNX model ไว้ในตำแหน่งที่ต้องการ เช่น:
```
Model-AI/bun_stage1_detection/best.onnx
```

### Step 2: (Optional) เตรียม Dataset สำหรับ INT8 Quantization

ถ้าต้องการแปลงเป็น INT8 ต้องมีไฟล์ dataset.txt:
```bash
cd ../Data-set
python3 create_dataset_txt.py \
    -i ./bun_stage1_detection/train/images \
    -d bun_train \
    -n 500
```

### Step 3: เปิด Terminal ที่โฟลเดอร์ onnx_to_rknn_converter

```bash
cd /home/nz/firefly/ONNX_to_RKNN_Guide_EC-R3588SPC/onnx_to_rknn_converter
```

### Step 4: รัน Converter

**สำหรับ FP16:**
```bash
python3 universal_onnx_to_rknn.py \
    --onnx ../Model-AI/bun_stage1_detection/best.onnx \
    --model-name bun_stage1_detection \
    --output-name best_fp16.rknn
```

**สำหรับ INT8:**
```bash
python3 universal_onnx_to_rknn.py \
    --onnx ../Model-AI/bun_stage1_detection/best.onnx \
    --model-name bun_stage1_detection \
    --output-name best_int8.rknn \
    --quantize \
    --dataset ../Data-set/output/bun_train_dataset.txt
```

### Step 5: ตรวจสอบผลลัพธ์

```bash
ls -lh ../Model-AI/bun_stage1_detection/
```

---

## ⚙️ Parameters ทั้งหมด

### Required Parameters (บังคับ)

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--onnx` | `-i` | ไฟล์ ONNX input | `--onnx model.onnx` |
| `--model-name` | `-m` | ชื่อโมเดล (ชื่อโฟลเดอร์ใน Model-AI) | `--model-name bun_detection` |
| `--output-name` | `-o` | ชื่อไฟล์ .rknn output | `--output-name model_fp16.rknn` |

### Platform Settings

| Parameter | Default | Choices | Description |
|-----------|---------|---------|-------------|
| `--platform` `-p` | `rk3588` | `rk3588`, `rk3576`, `rk3562`, `rv1109`, `rv1126`, `rk1808`, `rk3399pro` | Target platform |
| `--sub-platform` | `None` | - | Sub-platform สำหรับ chip variants |

### Quantization Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--quantize` `-q` | `False` | เปิดใช้งาน quantization (INT8) |
| `--dtype` | `INT8` | ประเภท data type: `INT8`, `FP16`, `UINT8` |
| `--algorithm` | `normal` | Algorithm: `normal`, `mmse`, `kl_divergence` |
| `--method` | `channel` | Method: `channel`, `layer` |
| `--dataset` `-d` | `None` | Path ไปยังไฟล์ dataset.txt (จำเป็นสำหรับ INT8) |

### Optimization Settings

| Parameter | Default | Choices | Description |
|-----------|---------|---------|-------------|
| `--optimization` | `3` | `0`, `1`, `2`, `3` | Optimization level (3 = สูงสุด) |

### Model Settings

| Parameter | Format | Description | Example |
|-----------|--------|-------------|---------|
| `--mean` | 3 floats | Mean values สำหรับ normalization | `--mean 0 0 0` |
| `--std` | 3 floats | Std values สำหรับ normalization | `--std 255 255 255` |
| `--input-size` | 3 ints | Input size [C, H, W] | `--input-size 3 640 640` |
| `--outputs` | list | Output layer names | `--outputs output0 output1` |

### Advanced Settings

| Parameter | Description |
|-----------|-------------|
| `--hybrid-quant` | เปิดใช้ hybrid quantization (FP16 + INT8) |
| `--hybrid-quant-file` | Path ไปยังไฟล์ config สำหรับ hybrid quant |
| `--custom-string` | Custom string สำหรับ version tracking |
| `--verify` `-v` | ตรวจสอบโมเดลหลังแปลง |
| `--verbose` | แสดง output แบบละเอียด |

---

## 💡 ตัวอย่างการใช้งาน

### ตัวอย่างที่ 1: Basic FP16 Conversion

```bash
python3 universal_onnx_to_rknn.py \
    --onnx ../Model-AI/bun_stage1_detection/best.onnx \
    --model-name bun_stage1_detection \
    --output-name best_fp16.rknn
```

**Output:** `Model-AI/bun_stage1_detection/best_fp16.rknn`

---

### ตัวอย่างที่ 2: INT8 Quantization

```bash
python3 universal_onnx_to_rknn.py \
    --onnx ../Model-AI/bun_stage1_detection/best.onnx \
    --model-name bun_stage1_detection \
    --output-name best_int8.rknn \
    --quantize \
    --dataset ../Data-set/output/bun_train_dataset.txt
```

**Output:** `Model-AI/bun_stage1_detection/best_int8.rknn`

---

### ตัวอย่างที่ 3: Full Configuration

```bash
python3 universal_onnx_to_rknn.py \
    --onnx ../Model-AI/bun_stage1_detection/best.onnx \
    --model-name bun_stage1_detection \
    --output-name best_int8_optimized.rknn \
    --platform rk3588 \
    --quantize \
    --dtype INT8 \
    --algorithm mmse \
    --method channel \
    --dataset ../Data-set/output/bun_train_dataset.txt \
    --optimization 3 \
    --mean 0 0 0 \
    --std 255 255 255 \
    --verify
```

---

### ตัวอย่างที่ 4: YOLOv8 Custom Size

```bash
python3 universal_onnx_to_rknn.py \
    --onnx yolov8n.onnx \
    --model-name yolov8_custom \
    --output-name yolov8n_640_fp16.rknn \
    --input-size 3 640 640 \
    --mean 0 0 0 \
    --std 255 255 255
```

---

### ตัวอย่างที่ 5: Different Platform (RK3576)

```bash
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --model-name my_model \
    --output-name model_rk3576_int8.rknn \
    --platform rk3576 \
    --quantize \
    --dataset dataset.txt
```

---

## 🔄 กระบวนการแปลง ONNX → RKNN

### ภาพรวมการทำงาน

```
ONNX Model
    ↓
[1] Analyze Model (Auto-detect type, input/output)
    ↓
[2] Config RKNN (Platform, optimization, quantization)
    ↓
[3] Load ONNX (Import to RKNN)
    ↓
[4] Build Model (Optimize & Quantize)
    ↓
[5] Export RKNN (Save to Model-AI/<model-name>/)
    ↓
RKNN Model (พร้อมใช้งานบน RK3588)
```

### รายละเอียดแต่ละขั้นตอน

#### Step 1: Analyze Model
```python
# อ่านไฟล์ ONNX และวิเคราะห์
- ชื่อ Graph
- Input shape (e.g., [1, 3, 640, 640])
- Output shapes
- Auto-detect model type (YOLOv5/v8/v10, Classification)
```

**ตัวอย่าง Output:**
```
🔍 Analyzing ONNX model...
   📝 Graph Name: torch_jit
   📐 Input Shape: [1, 3, 640, 640]
   📊 Output Shapes:
      [0] output0: [1, 84, 8400]
   🎯 Detected Type: YOLOv8
```

---

#### Step 2: Config RKNN
```python
# กำหนด Configuration
rknn.config(
    target_platform='rk3588',           # Platform เป้าหมาย
    optimization_level=3,                # ระดับ optimization (0-3)
    quantized_dtype='INT8',             # ประเภท quantization
    quantized_algorithm='normal',       # Algorithm: normal/mmse/kl_divergence
    quantized_method='channel',         # Method: channel/layer
    mean_values=[[0, 0, 0]],           # Mean สำหรับ normalization
    std_values=[[255, 255, 255]]       # Std สำหรับ normalization
)
```

**พารามิเตอร์สำคัญ:**
- **target_platform**: ระบุ NPU ที่จะใช้งาน (RK3588, RK3576, etc.)
- **optimization_level**: 
  - `0` = ไม่ optimize
  - `3` = optimize สูงสุด (แนะนำ)
- **quantized_dtype**: 
  - `FP16` = Precision สูง, ช้ากว่า
  - `INT8` = เร็วกว่า, accuracy ลดลงเล็กน้อย
- **mean/std values**: ค่าที่ใช้ normalize input image

---

#### Step 3: Load ONNX
```python
# Import ONNX model เข้า RKNN
rknn.load_onnx(
    model='model.onnx',
    inputs=['images'],              # Input layer name
    outputs=['output0']             # Output layer names
)
```

**การทำงาน:**
- อ่าน ONNX graph
- แปลง operators เป็น RKNN operators
- ตรวจสอบความเข้ากันได้

---

#### Step 4: Build Model
```python
# Build และ Quantize model
rknn.build(
    do_quantization=True,                    # เปิด quantization
    dataset='dataset.txt',                   # ไฟล์ calibration dataset
    rknn_batch_size=1                        # Batch size
)
```

**การทำงาน:**
1. **Graph Optimization:**
   - Fuse operators (Convolution + BatchNorm + ReLU → Single op)
   - Remove redundant nodes
   - Optimize memory layout

2. **Quantization (ถ้าเปิดใช้งาน):**
   - โหลดรูปจาก `dataset.txt` (50-1000 รูป)
   - คำนวณ calibration data
   - แปลง weights จาก FP32 → INT8
   - เก็บ scale/zero_point สำหรับแต่ละ layer

3. **NPU Mapping:**
   - แมป operators ไปยัง NPU hardware
   - จัดสรร memory
   - สร้าง execution plan

**Algorithm การ Quantize:**
- **normal**: เร็ว, accuracy ดี
- **mmse**: ช้ากว่า, accuracy ดีกว่า (Minimize Mean Square Error)
- **kl_divergence**: ใช้ KL Divergence เพื่อหา optimal threshold

---

#### Step 5: Export RKNN
```python
# Export เป็นไฟล์ .rknn
rknn.export_rknn('output.rknn')
```

**การทำงาน:**
- บันทึก optimized graph
- บันทึก quantized weights
- บันทึก metadata (input/output shapes, normalization params)
- Compress และบันทึกเป็นไฟล์ .rknn

**ไฟล์ .rknn ประกอบด้วย:**
- Model architecture
- Quantized weights
- NPU execution instructions
- Preprocessing parameters

---

## 🎯 Quantization Algorithms

### 1. Normal Quantization (แนะนำสำหรับส่วนใหญ่)

```bash
--quantize --algorithm normal
```

**วิธีการทำงาน:**
- ใช้ min-max range จาก calibration dataset
- คำนวณ scale และ zero_point
- แปลง FP32 → INT8

**ข้อดี:**
- เร็ว
- Accuracy ดีในกรณีส่วนใหญ่
- ใช้งานง่าย

**ข้อเสีย:**
- อาจมี outliers ที่ทำให้ accuracy ลดลง

---

### 2. MMSE (Minimum Mean Square Error)

```bash
--quantize --algorithm mmse
```

**วิธีการทำงาน:**
- หา threshold ที่ minimize MSE ระหว่าง FP32 และ INT8
- Iterative search สำหรับ optimal clipping range

**ข้อดี:**
- Accuracy สูงกว่า normal
- จัดการ outliers ได้ดีกว่า

**ข้อเสีย:**
- ช้ากว่า (ใช้เวลา calibration นานกว่า)

---

### 3. KL Divergence

```bash
--quantize --algorithm kl_divergence
```

**วิธีการทำงาน:**
- ใช้ KL Divergence เพื่อหา optimal threshold
- พยายาม preserve information distribution

**ข้อดี:**
- ดีสำหรับบาง model architectures
- Preserve distribution ได้ดี

**ข้อเสีย:**
- ช้าที่สุด
- อาจไม่เหมาะกับทุก model

---

## 🔍 Quantization Method

### Channel-wise Quantization (แนะนำ)

```bash
--method channel
```

**วิธีการทำงาน:**
- Quantize แยกตาม channel
- แต่ละ channel มี scale/zero_point ของตัวเอง

**ข้อดี:**
- Accuracy สูงกว่า layer-wise
- Flexible กว่า

**ข้อเสีย:**
- ใช้ memory เยอะกว่าเล็กน้อย

---

### Layer-wise Quantization

```bash
--method layer
```

**วิธีการทำงาน:**
- Quantize ทั้ง layer ด้วย scale/zero_point เดียว

**ข้อดี:**
- ใช้ memory น้อยกว่า
- เร็วกว่า

**ข้อเสีย:**
- Accuracy ต่ำกว่า channel-wise

---

## 📊 Normalization Parameters

### Mean และ Std Values

สำหรับ normalize input image ก่อนเข้า model

```bash
--mean 0 0 0 --std 255 255 255
```

**ความหมาย:**
```python
normalized_pixel = (pixel - mean) / std

# ตัวอย่าง: pixel = [128, 64, 200]
# mean = [0, 0, 0], std = [255, 255, 255]
normalized = ([128, 64, 200] - [0, 0, 0]) / [255, 255, 255]
           = [0.502, 0.251, 0.784]
```

### ค่า Mean/Std ที่ใช้บ่อย

**1. ImageNet Normalization:**
```bash
--mean 123.675 116.28 103.53 \
--std 58.395 57.12 57.375
```

**2. 0-1 Range:**
```bash
--mean 0 0 0 \
--std 255 255 255
```

**3. -1 to 1 Range:**
```bash
--mean 127.5 127.5 127.5 \
--std 127.5 127.5 127.5
```

**หมายเหตุ:** ต้องใช้ค่าเดียวกับตอน Training

---

## 🛠️ Troubleshooting

### ❌ Error: "Import RKNN-Toolkit2 failed"

**วิธีแก้:**
```bash
# ตรวจสอบว่าติดตั้ง RKNN-Toolkit2 แล้ว
pip3 list | grep rknn

# ถ้ายังไม่มี ให้ติดตั้ง
pip3 install /path/to/rknn_toolkit2-2.3.0-cp310-cp310-linux_x86_64.whl
```

---

### ❌ Error: "Dataset file not found"

**วิธีแก้:**
```bash
# ตรวจสอบว่าไฟล์ dataset.txt มีจริง
ls -la ../Data-set/output/

# ถ้ายังไม่มี ให้สร้าง
cd ../Data-set
python3 create_dataset_txt.py -i ./train/images -d mydataset -n 500
```

---

### ⚠️ Warning: "Quantization accuracy drop"

**สาเหตุ:** 
- Dataset calibration ไม่เพียงพอ
- Algorithm ไม่เหมาะสม

**วิธีแก้:**
```bash
# 1. เพิ่มจำนวนรูปใน dataset (500-1000 รูป)
# 2. ลอง algorithm อื่น
--algorithm mmse

# 3. ลอง hybrid quantization
--hybrid-quant
```

---

### 🐌 ช้ามาก (Quantization นาน)

**วิธีแก้:**
```bash
# 1. ลดจำนวนรูปใน dataset
-n 200

# 2. ใช้ algorithm normal แทน mmse
--algorithm normal

# 3. ใช้ optimization level ต่ำกว่า
--optimization 1
```

---

## 📁 Output Location

ไฟล์ทั้งหมดจะถูกบันทึกที่:
```
/home/nz/firefly/ONNX_to_RKNN_Guide_EC-R3588SPC/Model-AI/<model-name>/
```

**ตัวอย่าง:**
```
Model-AI/
└── bun_stage1_detection/
    ├── best.onnx                    # Original ONNX
    ├── best_fp16.rknn              # FP16 RKNN
    ├── best_int8.rknn              # INT8 RKNN
    └── model_config.yaml            # Config file
```

---

## 🎯 Best Practices

### 1. เลือก Quantization Type

| Use Case | Recommendation |
|----------|---------------|
| ต้องการ accuracy สูงสุด | FP16 (ไม่ใส่ `--quantize`) |
| ต้องการความเร็ว | INT8 (`--quantize`) |
| Balance | INT8 with MMSE (`--quantize --algorithm mmse`) |

---

### 2. Dataset สำหรับ Calibration

**แนะนำ:**
- ใช้รูปจาก **training set**
- จำนวน **200-1000 รูป**
- ครอบคลุมทุก class
- รูปต้องมีลักษณะใกล้เคียงกับข้อมูลจริงที่จะใช้

```bash
# สร้าง dataset calibration
python3 ../Data-set/create_dataset_txt.py \
    -i ./train/images \
    -d model_calibration \
    -n 500
```

---

### 3. Optimization Level

| Level | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| 0 | ช้าที่สุด | สูงสุด | Debug |
| 1 | ช้า | สูง | Development |
| 2 | กลาง | กลาง | Testing |
| 3 | เร็วสุด | ดี | Production (แนะนำ) |

---

### 4. Workflow แนะนำ

```bash
# 1. ทดสอบด้วย FP16 ก่อน
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --model-name test_model \
    --output-name model_fp16.rknn \
    --verify

# 2. ถ้า FP16 ทำงานได้ ลอง INT8
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --model-name test_model \
    --output-name model_int8.rknn \
    --quantize \
    --dataset dataset.txt \
    --verify

# 3. ถ้า accuracy ลดลงมาก ลอง MMSE
python3 universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --model-name test_model \
    --output-name model_int8_mmse.rknn \
    --quantize \
    --algorithm mmse \
    --dataset dataset.txt \
    --verify
```

---

## 🔗 Related Documents

| Document | Location | Description |
|----------|----------|-------------|
| **Dataset Creator** | `../Data-set/README.md` | วิธีสร้างไฟล์ dataset.txt |
| **RKNN Toolkit Guide** | `../Doc/Custom_Model_to_RKNN_Guide_v2.3.2.md` | คู่มือ RKNN Toolkit2 ฉบับเต็ม |
| **Model Storage** | `../Model-AI/README.md` | โครงสร้างการเก็บโมเดล |
| **Universal Converter Guide** | `../Doc/UNIVERSAL_CONVERTER_GUIDE.md` | คู่มือเฉพาะ Converter |

---

## 📚 References

### Official Documentation
- **RKNN-Toolkit2 GitHub:** https://github.com/rockchip-linux/rknn-toolkit2
- **Rockchip NPU Docs:** https://github.com/airockchip/rknn-toolkit2/tree/master/doc

### Model Export Guides
- **YOLOv5 to ONNX:** https://github.com/ultralytics/yolov5
- **YOLOv8 to ONNX:** https://github.com/ultralytics/ultralytics
- **PyTorch to ONNX:** https://pytorch.org/docs/stable/onnx.html

---

## 🆘 Support & Custom

### ถ้าต้องการ Customize

**ไฟล์ที่ต้องแก้:**
- `universal_onnx_to_rknn.py` - Main converter script

**ส่วนที่อาจต้อง Custom:**
1. **Model Detection** (line 60-90): เพิ่ม logic สำหรับ detect model ใหม่
2. **Default Parameters** (line 200-250): ปรับค่า default config
3. **Output Path** (line 540-560): เปลี่ยนตำแหน่ง output
4. **Preprocessing** (line 170-200): ปรับ mean/std values

### ติดต่อ / Issues

ถ้ามีปัญหาหรือต้องการปรับปรุง:
1. เปิด Issue ใน GitHub repository
2. ดูเอกสารใน `../Doc/` folder
3. ตรวจสอบ `../requirement-step-summary/` สำหรับ performance metrics

---

**Last Updated:** November 27, 2025  
**Version:** 2.0  
**Compatible with:** RKNN-Toolkit2 v2.3.0+  
**Supported Platforms:** RK3588, RK3576, RK3562, RV1109, RV1126
