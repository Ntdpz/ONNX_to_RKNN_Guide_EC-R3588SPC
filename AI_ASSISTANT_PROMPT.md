# 🤖 AI Assistant Context - ONNX to RKNN Conversion for Rockchip NPU

> **วิธีใช้:** Copy เนื้อหาทั้งหมดในไฟล์นี้ไปวางใน System Prompt หรือ Context ของ AI เพื่อให้ AI เข้าใจโปรเจกต์และช่วยงานได้อย่างมีประสิทธิภาพ

---

## 📌 โปรเจกต์นี้คืออะไร

เป็น Toolkit สำหรับแปลง AI Model (ONNX) ไปเป็น RKNN Format เพื่อรันบน Rockchip NPU (RK3588) โดยเฉพาะบอร์ด EC-R3588SPC

## 🎯 วัตถุประสงค์หลัก

- แปลง ONNX Model → RKNN Model (FP16/INT8)
- รองรับ YOLOv5, YOLOv8, YOLOv10, และ Custom Models
- ทำ Quantization (INT8) สำหรับเพิ่มประสิทธิภาพ
- รักษาความถูกต้องของ Preprocessing และ Postprocessing

---

## 🔧 เครื่องมือหลัก

| Component | Version/Detail |
|-----------|----------------|
| **SDK** | RKNN-Toolkit2 v2.3.2 |
| **Platform** | RK3588 (6 TOPS NPU), รองรับ RK3576, RK3566, RK3568 |
| **Python** | 3.8+ |
| **ONNX Opset** | รองรับ 12-19 (แนะนำ 12) |

---

## 📁 โครงสร้าง Repository

```
ONNX_to_RKNN_Guide_EC-R3588SPC/
├── Data-set/                    # Dataset สำหรับ INT8 Quantization
│   └── create_dataset_txt.py    # สร้าง dataset.txt
│
├── Doc/                         # เอกสารความรู้ทั้งหมด
│   ├── Custom_Model_to_RKNN_Guide_v2.3.0.md   # คู่มือ Custom Model (v2.3.0)
│   ├── Custom_Model_to_RKNN_Guide_v2.3.2.md   # คู่มือ Custom Model (v2.3.2)
│   ├── PREPROCESSING_POSTPROCESSING_GUIDE.md  # คู่มือ Pre/Post processing
│   └── UNIVERSAL_CONVERTER_GUIDE.md           # คู่มือ Converter
│
├── Model-AI/                    # เก็บ Model (ONNX, RKNN)
│   └── <model_name>/
│       ├── best.onnx
│       ├── best_fp16.rknn
│       ├── best_int8.rknn
│       └── model_config.yaml
│
├── onnx_to_rknn_converter/      # เครื่องมือแปลง ONNX → RKNN
│   └── universal_onnx_to_rknn.py
│
├── requirement-step-summary/    # Templates และ Config
│   ├── 01_OVERVIEW.md           # Workflow overview
│   ├── 02_FIELD_CATEGORIES.md   # จำแนก fields ที่เปลี่ยนได้/ไม่ได้
│   ├── templates/               # YAML templates
│   └── examples/                # ตัวอย่างจริง (yolov8_bun)
│
└── test/                        # Scripts ทดสอบ
    ├── file/                    # ไฟล์ทดสอบ
    └── tools/                   # เครื่องมือทดสอบ
```

---

## ⚡ กฎเหล็ก 4 ข้อ สำหรับ Custom Model

### 1️⃣ ต้อง "ตัดหัว" (Remove Post-processing)

```
❌ ห้ามใส่ใน Model:
   - NMS (Non-Maximum Suppression)
   - Decode Box (คำนวณพิกัด x, y, w, h)
   - Confidence Thresholding

✅ ต้องทำ:
   - ส่ง Feature Map (Raw Output) ออกมา
   - ทำ Post-processing ด้วย Python/C++ ภายนอก
```

**ตัวอย่างจาก Official SDK:**
```python
# NPU คำนวณแค่ Raw outputs
outputs = rknn.inference(inputs=[img])

# Post-process ภายนอก NPU ด้วย Python
boxes, classes, scores = yolov5_post_process(outputs)
```

### 2️⃣ ต้อง Static Shape (ขนาดคงที่)

```python
# ❌ ผิด - Dynamic shape
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}}  # ห้าม!
)

# ✅ ถูก - Static shape
torch.onnx.export(
    model, 
    torch.randn(1, 3, 640, 640),  # กำหนดขนาดตายตัว
    "model.onnx",
    opset_version=12
    # ไม่มี dynamic_axes
)
```

**เหตุผล:** NPU ต้องจอง Memory ล่วงหน้าแบบ Fixed size

### 3️⃣ ใช้ ONNX Opset 12-19 (แนะนำ 12)

```python
# ✅ แนะนำ - Opset 12 (เสถียรสูงสุด)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=12)

# ✅ ใช้ได้ - Opset 13-19
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=19)
```

### 4️⃣ ระวัง 5D Tensor และ Reshape/Transpose

```
❌ หลีกเลี่ยง:
   - Reshape เป็น 5 มิติในโมเดล
   - Permute มิติซับซ้อนในโมเดล

✅ ควรทำ:
   - ปล่อยให้ NPU ส่ง output ในรูปแบบของมัน
   - ทำ Reshape/Transpose ภายนอกด้วย Python
```

---

## 📊 Workflow การแปลง Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Training (PyTorch)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Output: best.pt, training_source.yaml, performance_pt.json      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Export (ONNX)                                          │
├─────────────────────────────────────────────────────────────────┤
│ Output: best.onnx, onnx_source.yaml, performance_onnx.json      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Conversion (RKNN)                                      │
├─────────────────────────────────────────────────────────────────┤
│ Output: best_fp16.rknn, best_int8.rknn, rknn_source.yaml        │
└─────────────────────────────────────────────────────────────────┘
```

### คำสั่งแปลง FP16 (แนะนำเริ่มต้น)

```bash
python universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_fp16.rknn \
    --platform rk3588
```

### คำสั่งแปลง INT8 (Production)

```bash
python universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_int8.rknn \
    --platform rk3588 \
    --quantize \
    --algorithm mmse \
    --dataset dataset.txt
```

---

## 🔒 Critical Fields (ห้ามเปลี่ยน!)

ค่าเหล่านี้ **ต้องตรงกับตอน Training** ทุก Phase:

| Field | ตัวอย่าง (YOLOv8) | คำอธิบาย |
|-------|-------------------|----------|
| `input_size` | `[640, 640]` | ขนาด Input ที่ Model ต้องการ |
| `input_format` | `"RGB"` | Color format (RGB/BGR) |
| `channels` | `3` | จำนวน channels |
| `resize_method` | `"letterbox"` | วิธี Resize (letterbox/direct) |
| `padding_color` | `[114, 114, 114]` | สี Padding (เทาสำหรับ YOLO) |
| `normalize.mean` | `[0, 0, 0]` | Mean normalization |
| `normalize.std` | `[255, 255, 255]` | Std normalization |

### ⚠️ หากเปลี่ยนค่าเหล่านี้:

```
❌ Model จะให้ผลลัพธ์ผิดพลาด
❌ Accuracy ลดลงอย่างมาก (>20%)
❌ ต้อง Retrain Model ใหม่ทั้งหมด
```

---

## ✅ Configurable Fields (เปลี่ยนได้)

### Platform Settings

| Field | Options | Default | Notes |
|-------|---------|---------|-------|
| `platform` | rk3588, rk3576, rk3562, rv1109, rv1126, rk1808 | `rk3588` | ตาม hardware |

### Quantization Settings

| Field | Options | Default | Notes |
|-------|---------|---------|-------|
| `quantization.type` | FP16, INT8, UINT8 | `FP16` | FP16=accuracy, INT8=speed |
| `quantization.algorithm` | normal, mmse, kl_divergence | `normal` | mmse=accuracy ดีกว่า |
| `quantization.method` | channel, layer | `channel` | channel=accuracy ดีกว่า |
| `optimization_level` | 0, 1, 2, 3 | `3` | 3=optimized สูงสุด |

### Inference Settings

| Field | Range | Default | Notes |
|-------|-------|---------|-------|
| `conf_threshold` | 0.0 - 1.0 | `0.25` | สูง=detections น้อย |
| `iou_threshold` | 0.0 - 1.0 | `0.85` | สูง=NMS กรองน้อย |
| `max_detections` | 1 - 1000 | `300` | Maximum output boxes |

---

## 📝 Preprocessing Pipeline (YOLOv8)

```python
def preprocess_image(image_path, target_size=640):
    """
    Preprocessing pipeline สำหรับ YOLOv8
    ต้องทำเหมือนกับตอน Train Model ไม่งั้น Model จะงง!
    """
    
    # 1. โหลดรูปภาพ
    img = cv2.imread(image_path)  # Shape: (H, W, 3) - BGR
    h, w = img.shape[:2]
    
    # 2. Letterbox Resize (รักษาอัตราส่วน)
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Padding ด้วยสีเทา (114, 114, 114) - มาตรฐาน YOLO
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
    
    # 4. Color Conversion (BGR → RGB)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # 5. Add Batch Dimension
    img_array = np.expand_dims(img_rgb, axis=0)  # (640,640,3) → (1,640,640,3)
    
    # เก็บข้อมูลสำหรับ Postprocessing
    return img_array, (h, w), scale, (pad_w, pad_h)
```

### ⚠️ สิ่งที่ต้องระวัง

| ❌ ผิด | ✅ ถูก | เหตุผล |
|--------|---------|---------|
| Direct Resize | Letterbox + Padding | รักษาอัตราส่วนรูป ไม่บิด |
| Padding สีดำ (0,0,0) | Padding สีเทา (114,114,114) | ตามมาตรฐาน YOLO Training |
| BGR Color Space | RGB Color Space | Model Train ด้วย RGB |

---

## 📝 Postprocessing Pipeline (YOLOv8)

```python
def postprocess_yolo(outputs, original_shape, scale, padding, 
                     conf_threshold=0.25, iou_threshold=0.85):
    """
    Postprocessing pipeline สำหรับ YOLOv8
    """
    
    # 1. Extract predictions
    predictions = outputs[0][0]  # (5, 8400) หรือ (8400, 5)
    
    # 2. Transpose ถ้าจำเป็น
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T  # (5, 8400) → (8400, 5)
    
    # 3. Decode predictions
    boxes = predictions[:, :4]        # x, y, w, h (center format)
    confidences = predictions[:, 4]   # confidence scores
    
    # 4. Filter by Confidence Threshold
    valid_mask = confidences > conf_threshold
    boxes = boxes[valid_mask]
    scores = confidences[valid_mask]
    
    # 5. Convert to Corner Format (x1, y1, x2, y2)
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # 6. Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores.tolist(),
        conf_threshold,
        iou_threshold  # ✅ 0.85 = กรองน้อย, 0.45 = กรองเยอะ
    )
    
    # 7. Scale back to Original Coordinates
    if len(indices) > 0:
        indices = indices.flatten()
        boxes_xyxy = boxes_xyxy[indices]
        scores = scores[indices]
        
        # ลบ Padding
        pad_w, pad_h = padding
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        
        # Scale กลับขนาดจริง
        boxes_xyxy /= scale
        
        # Clip ให้อยู่ในขอบเขตรูป
        h, w = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)
        
        return boxes_xyxy, scores
    
    return [], []
```

---

## ⚠️ ปัญหาที่พบบ่อยและวิธีแก้

### 1. 🔴 Detection Loss (ONNX ได้ 23, RKNN ได้ 6)

**สาเหตุ:**
- Preprocessing ไม่ตรง (Direct resize แทน Letterbox)
- IoU threshold ต่ำเกิน (0.45 → NMS กรองเยอะ)
- Confidence threshold สูงเกิน

**แก้ไข:**
```python
# ✅ ใช้ Letterbox + Gray Padding
img_padded = np.full((640, 640, 3), 114, dtype=np.uint8)

# ✅ เพิ่ม IoU Threshold
iou_threshold = 0.85  # จาก 0.45

# ✅ ลด Confidence Threshold
conf_threshold = 0.25  # จาก 0.5
```

### 2. 🔴 Accuracy ต่ำหลัง INT8 Quantization

**สาเหตุ:**
- Dataset สำหรับ calibration น้อยเกิน
- ใช้ test set ใน dataset.txt (data leakage)
- Algorithm ไม่เหมาะสม

**แก้ไข:**
```bash
# ✅ ใช้ dataset 500-1000 รูป (หลากหลาย)
python create_dataset_txt.py -i ./train/images -d my_model -n 1000

# ✅ ใช้ algorithm = "mmse"
python universal_onnx_to_rknn.py --algorithm mmse ...

# ✅ อย่าใช้ test set ใน dataset.txt
```

### 3. 🔴 Coordinates ผิดตำแหน่ง

**สาเหตุ:**
- ลืมลบ Padding
- ลืม Scale กลับขนาดจริง
- ลำดับการคำนวณผิด

**แก้ไข:**
```python
# ✅ ลำดับที่ถูกต้อง
boxes[:, [0, 2]] -= pad_w      # 1. ลบ Padding ก่อน
boxes[:, [1, 3]] -= pad_h
boxes /= scale                  # 2. Scale กลับทีหลัง
```

### 4. 🔴 ONNX Export Error

**แก้ไข:**
```python
# ตรวจสอบ opset version
# YOLOv8: opset_version=12
# YOLOv10: opset_version=13

torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=12,
    do_constant_folding=True
)
```

### 5. 🔴 Output Shape ผิด

**สาเหตุ:**
- RKNN output (5, 8400) แต่โค้ดคาดหวัง (8400, 5)

**แก้ไข:**
```python
# ✅ Auto-transpose
if predictions.shape[0] < predictions.shape[1]:
    predictions = predictions.T
```

---

## 📊 Performance Benchmarks

### YOLOv8 Bun Detection (640x640, 1 class)

| Platform | Format | FPS | mAP@0.5 | Detection Rate |
|----------|--------|-----|---------|----------------|
| PyTorch | FP32 | 50 | 0.95 | 23/23 (100%) |
| ONNX | FP32 | 6.9 | 0.95 | 23/23 (100%) |
| RKNN | FP16 | 21.3 | 0.95 | 23/23 (100%) |
| RKNN | INT8 | 35.7 | 0.93 | 22/23 (95.7%) |

### Quantization Algorithm Comparison

| Algorithm | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| `normal` | ⚡⚡⚡ เร็วสุด | ⭐⭐ | Prototyping |
| `mmse` | ⚡⚡ ปานกลาง | ⭐⭐⭐ | Production |
| `kl_divergence` | ⚡ ช้า | ⭐⭐⭐ | Special cases |

### FP16 vs INT8

| | FP16 | INT8 |
|---|------|------|
| **Accuracy** | ✅ สูงสุด (99%) | ⚠️ ลดลงเล็กน้อย (95-98%) |
| **Speed** | ⚡ Baseline | ⚡⚡⚡ เร็วกว่า 2-4x |
| **Size** | 📦 Baseline | 📦 เล็กกว่า 50% |
| **Use case** | Development | Production |

---

## 🚀 คำสั่งที่ใช้บ่อย

### สร้าง Dataset

```bash
# สร้าง dataset.txt (500-1000 รูป)
python create_dataset_txt.py -i ./images -d my_model -n 1000
```

### แปลง ONNX → RKNN

```bash
# FP16 (Development - Accuracy สูง)
python universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_fp16.rknn \
    --platform rk3588

# INT8 (Production - Speed สูง)
python universal_onnx_to_rknn.py \
    --onnx model.onnx \
    --rknn model_int8.rknn \
    --platform rk3588 \
    --quantize \
    --algorithm mmse \
    --dataset dataset.txt \
    --verify
```

### Export PyTorch → ONNX

```python
import torch

model = torch.load("best.pt")
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=12,
    input_names=['images'],
    output_names=['output0'],
    do_constant_folding=True
)
```

### ทดสอบบน RK3588

```bash
python Bun-detech.py --model model.rknn --image test.jpg
```

---

## 📖 เอกสารอ้างอิงในโปรเจกต์

| ไฟล์ | เนื้อหา |
|------|---------|
| `Doc/Custom_Model_to_RKNN_Guide_v2.3.2.md` | คู่มือ Custom Model ฉบับสมบูรณ์ |
| `Doc/PREPROCESSING_POSTPROCESSING_GUIDE.md` | คู่มือ Pre/Post processing |
| `Doc/UNIVERSAL_CONVERTER_GUIDE.md` | คู่มือ Converter ครบทุกคำสั่ง |
| `requirement-step-summary/01_OVERVIEW.md` | Workflow overview |
| `requirement-step-summary/02_FIELD_CATEGORIES.md` | จำแนก fields ที่เปลี่ยนได้/ไม่ได้ |

---

## 💡 Quick Reference - เมื่อผู้ใช้ถาม

| คำถาม | ตอบ/แนะนำ |
|-------|-----------|
| การแปลง Model | ใช้ `universal_onnx_to_rknn.py` |
| การเตรียม Dataset | ใช้ `create_dataset_txt.py` (500-1000 รูป) |
| Preprocessing/Postprocessing | ดู `PREPROCESSING_POSTPROCESSING_GUIDE.md` |
| Custom Model | ดู `Custom_Model_to_RKNN_Guide_v2.3.2.md` |
| Config ที่ห้ามเปลี่ยน | ดู `02_FIELD_CATEGORIES.md` |
| Detection Loss | เพิ่ม `iou_threshold` เป็น 0.85 |
| INT8 Accuracy ต่ำ | ใช้ `--algorithm mmse` + dataset 1000 รูป |
| ONNX Operators ที่รองรับ | Opset 12-19, ดู SDK documentation |

---

## 🔗 External Links

- [RKNN-Toolkit2 GitHub](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Documentation](https://onnx.ai/onnx/)

---

**Hardware:** EC-R3588SPC (RK3588, 6 TOPS NPU)  
**Toolkit:** RKNN-Toolkit2 v2.3.2  
**Python:** 3.8+  
**Last Updated:** December 1, 2025
