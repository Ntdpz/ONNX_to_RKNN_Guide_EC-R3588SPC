# 📚 คู่มือ Preprocessing และ Postprocessing สำหรับ RKNN Model

## 🎯 Overview

การทำงานของ AI Model แบ่งออกเป็น 3 ส่วนหลัก:

```
Input Image → [Preprocessing] → [Model/NPU] → [Postprocessing] → Final Results
   (จริง)         (เราเขียน)      (คำนวณ)        (เราเขียน)         (ใช้งาน)
```

---

## 📊 Pipeline สมบูรณ์

### ภาพรวมการทำงาน

| ขั้นตอน | Input | Output | ผู้รับผิดชอบ |
|---------|-------|--------|--------------|
| **Preprocessing** | รูปภาพจริง (เช่น 1108x1477) | Tensor (1,640,640,3) | 👨‍💻 Developer |
| **Model Inference** | Tensor (1,640,640,3) | Raw Output (1,5,8400) | 🤖 NPU/Model |
| **Postprocessing** | Raw Output (1,5,8400) | Bounding Boxes + Labels | 👨‍💻 Developer |

---

## 1️⃣ Preprocessing (เตรียมข้อมูลก่อนเข้า Model)

### 🎯 หน้าที่
แปลงรูปภาพจริงให้เป็น **Input Format** ที่ Model คาดหวัง

### 📝 ขั้นตอนสำหรับ YOLOv8

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
| Direct Resize `cv2.resize(img, (640,640))` | Letterbox + Padding | รักษาอัตราส่วนรูป ไม่บิด |
| Padding สีดำ (0,0,0) | Padding สีเทา (114,114,114) | ตามมาตรฐาน YOLO Training |
| BGR Color Space | RGB Color Space | Model Train ด้วย RGB |
| ไม่ Normalize | Normalize (mean/std หรือ ÷255) | ขึ้นกับ Model Config |

### 🔍 ตัวอย่างการแปลง

```
Input:  1108x1477 pixels (สีน้ำเงิน-เหลือง-แดง)
         ↓
Step 1: Scale = min(640/1477, 640/1108) = 0.433
        → Resize เป็น 480x640
         ↓
Step 2: Padding 80 pixels ซ้าย-ขวา (สีเทา 114)
        → ได้ 640x640 pixels
         ↓
Step 3: BGR → RGB
         ↓
Step 4: Add batch dimension
        → Shape: (1, 640, 640, 3)
         ↓
Ready for Model! ✅
```

---

## 2️⃣ Model Inference (NPU คำนวณ)

### 🎯 หน้าที่
รับ Input Tensor → คำนวณด้วย Neural Network → ส่ง Output Tensor

### 🤖 การทำงานภายใน

```python
# Model เป็น Black Box ที่ทำงานอัตโนมัติ
outputs = rknn.inference(inputs=[img_array])

# Input:  (1, 640, 640, 3)  ← รูปที่ประมวลผลแล้ว
# Output: (1, 5, 8400)       ← ผลลัพธ์ดิบ (Raw predictions)
```

### 📊 ความหมายของ Output

```
Shape: (1, 5, 8400)
       │  │   └─── 8400 predictions (grid cells)
       │  └─────── 5 values per prediction:
       │            [x, y, w, h, confidence]
       └────────── Batch size = 1
```

### 💡 สิ่งที่ Model ไม่รู้

- ❌ ขนาดรูปภาพต้นฉบับ (1108x1477)
- ❌ มี Padding เท่าไหร่ (80 pixels)
- ❌ Scale factor (0.433)
- ❌ Class name ("bun")
- ❌ จำนวน bbox ที่ต้องการ

**👉 Model แค่คำนวณตัวเลขตาม Weight ที่ Train ไว้!**

---

## 3️⃣ Postprocessing (แปลงผลลัพธ์ให้ใช้งานได้)

### 🎯 หน้าที่
แปลง **Raw Output Tensor** ให้เป็น **Bounding Boxes** ที่ใช้งานได้จริง

### 📝 ขั้นตอนสำหรับ YOLOv8

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
    # จาก 8400 predictions → เหลือ ~218 predictions
    
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
    # จาก 218 boxes → เหลือ 23 boxes (bbox ที่ดีที่สุด)
    
    # 7. Scale back to Original Coordinates
    if len(indices) > 0:
        indices = indices.flatten()
        boxes_xyxy = boxes_xyxy[indices]
        scores = scores[indices]
        
        # ลบ Padding
        pad_w, pad_h = padding
        boxes_xyxy[:, [0, 2]] -= pad_w  # Remove horizontal padding
        boxes_xyxy[:, [1, 3]] -= pad_h  # Remove vertical padding
        
        # Scale กลับขนาดจริง
        boxes_xyxy /= scale
        
        # Clip ให้อยู่ในขอบเขตรูป
        h, w = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)
        
        return boxes_xyxy, scores
    
    return [], []
```

### 🔍 ขั้นตอนละเอียด

#### **Step 1-2: Reshape Output**
```
(1, 5, 8400) → (5, 8400) → (8400, 5)
                Remove      Transpose
                batch dim   if needed
```

#### **Step 3: Decode Predictions**
```
(8400, 5) แต่ละแถว:
[x_center, y_center, width, height, confidence]
[320.5,    240.3,    150.2, 180.7,  0.978     ]
```

#### **Step 4: Filter by Confidence**
```
Before: 8400 predictions
Filter: confidence > 0.25
After:  218 predictions ✅
```

#### **Step 5: Convert Format**
```
Center Format (x, y, w, h):
[320, 240, 150, 180]

Corner Format (x1, y1, x2, y2):
[245, 150, 395, 330]
     ↑    ↑    ↑    ↑
    x1   y1   x2   y2
```

#### **Step 6: Non-Maximum Suppression**
```
Input: 218 overlapping boxes

NMS with IoU = 0.85:
- ถ้า 2 boxes ซ้อนกัน > 85% → เอาแค่อันที่ confidence สูงกว่า
- ถ้า 2 boxes ซ้อนกัน < 85% → เก็บทั้ง 2 อัน

Output: 23 best boxes ✅
```

#### **Step 7: Coordinate Transformation**
```
Model Coordinates (640x640):
[x=320, y=240, w=150, h=180]
    ↓ ลบ padding (80, 0)
[x=240, y=240, w=150, h=180]
    ↓ หาร scale (0.433)
[x=554, y=554, w=346, h=415]
    ↓ Clip to image (0-1108, 0-1477)
Original Coordinates (1108x1477) ✅
```

---

## 🎯 ตัวอย่างการทำงานทั้งหมด

### Input
```
รูปภาพ: bun.jpg (1108x1477 pixels)
Model: best_yolov8_fp16.rknn
Classes: ["bun"]
```

### Preprocessing
```python
img_array, original_shape, scale, padding = preprocess_image("bun.jpg")

# ผลลัพธ์:
# - img_array: (1, 640, 640, 3) - Ready for Model
# - original_shape: (1477, 1108)
# - scale: 0.433
# - padding: (80, 0)
```

### Model Inference
```python
outputs = rknn.inference(inputs=[img_array])

# ผลลัพธ์:
# - outputs: [(1, 5, 8400)] - Raw predictions
# - Inference time: 46.97 ms
# - Throughput: 21.3 FPS
```

### Postprocessing
```python
boxes, scores = postprocess_yolo(
    outputs, 
    original_shape, 
    scale, 
    padding,
    conf_threshold=0.25,
    iou_threshold=0.85
)

# ผลลัพธ์:
# - boxes: 23 bounding boxes
# - scores: [0.978, 0.974, 0.970, ..., 0.900]
# - Detections: 23/23 ✅ (100% recovery)
```

---

## 📊 เปรียบเทียบ Pre/Post Processing

| แง่มุม | Preprocessing | Postprocessing |
|--------|---------------|----------------|
| **Input** | รูปภาพจริง (ขนาดไม่แน่นอน) | Raw tensor (ขนาดคงที่) |
| **Output** | Tensor ขนาดคงที่ | Bounding boxes ใช้งานได้ |
| **การแปลง** | รูป → Tensor | Tensor → Boxes |
| **ต้องรู้** | Input size Model ต้องการ | Output format ของ Model |
| **ต้องเก็บ** | Scale, Padding | - |
| **ความยาก** | ⭐⭐ ต้องระวังบิดรูป | ⭐⭐⭐⭐ Logic ซับซ้อน |

---

## ⚙️ Parameters สำคัญ

### Preprocessing Parameters

| Parameter | ค่าแนะนำ | คำอธิบาย |
|-----------|----------|----------|
| `target_size` | 640 | ขนาด Input ที่ Model ต้องการ |
| `padding_color` | (114, 114, 114) | สีเทา - มาตรฐาน YOLO |
| `interpolation` | INTER_LINEAR | วิธี Resize รูป |
| `color_format` | RGB | Format ที่ Model Train |

### Postprocessing Parameters

| Parameter | ค่าแนะนำ | ผลกระทบ |
|-----------|----------|----------|
| `conf_threshold` | 0.25 | ต่ำ = bbox เยอะ, สูง = bbox น้อย |
| `iou_threshold` | 0.85 | สูง = กรอง NMS น้อย, ต่ำ = กรองเยอะ |

### 🔍 ผลกระทบของ IoU Threshold

```
IoU = 0.45 (เข้มงวด):
218 boxes → NMS → 15 boxes ❌ (กรองเยอะไป)

IoU = 0.85 (ผ่อนปรน):
218 boxes → NMS → 23 boxes ✅ (ได้ครบ!)
```

---

## 🚨 ปัญหาที่พบบ่อย

### 1. Bbox Detection Loss

**อาการ:** ONNX detect ได้ 23 bbox แต่ RKNN detect ได้แค่ 6 bbox

**สาเหตุ:**
- ❌ Preprocessing ไม่ตรง (Direct resize แทน Letterbox)
- ❌ IoU threshold ต่ำเกิน (0.45 → NMS กรองเยอะ)
- ❌ Confidence threshold สูงเกิน

**วิธีแก้:**
```python
# ✅ ใช้ Letterbox + Gray Padding
img_padded = np.full((640, 640, 3), 114, dtype=np.uint8)

# ✅ เพิ่ม IoU Threshold
iou_threshold = 0.85  # จาก 0.45

# ✅ ลด Confidence Threshold
conf_threshold = 0.25  # จาก 0.5
```

### 2. Coordinates ไม่ตรงกับรูปจริง

**อาการ:** Bbox วาดผิดตำแหน่ง

**สาเหตุ:**
- ❌ ลืมลบ Padding
- ❌ ลืม Scale กลับขนาดจริง
- ❌ ใช้ Scale factor ผิด

**วิธีแก้:**
```python
# ✅ ลำดับที่ถูกต้อง
boxes[:, [0, 2]] -= pad_w      # 1. ลบ Padding ก่อน
boxes[:, [1, 3]] -= pad_h
boxes /= scale                  # 2. Scale กลับทีหลัง
```

### 3. Output Shape ผิด

**อาการ:** Error shape mismatch

**สาเหตุ:**
- ❌ RKNN output (5, 8400) แต่โค้ดคาดหวัง (8400, 5)

**วิธีแก้:**
```python
# ✅ Auto-transpose
if predictions.shape[0] < predictions.shape[1]:
    predictions = predictions.T
```

---

## ✅ Checklist การ Debug

### Preprocessing
- [ ] รูปต้นฉบับโหลดได้ถูกต้อง
- [ ] Letterbox resize (ไม่บิดรูป)
- [ ] Padding สีเทา (114, 114, 114)
- [ ] Color conversion BGR → RGB
- [ ] Shape = (1, 640, 640, 3)
- [ ] บันทึก scale และ padding

### Model Inference
- [ ] Model โหลดสำเร็จ
- [ ] Runtime initialize ได้
- [ ] Output shape ถูกต้อง (1, 5, 8400)
- [ ] Inference time สมเหตุสมผล (<100ms)

### Postprocessing
- [ ] Transpose shape ถ้าจำเป็น
- [ ] Confidence threshold ตรงกับ Training
- [ ] IoU threshold สูงพอ (0.85)
- [ ] NMS ทำงาน (boxes ลดลง)
- [ ] ลบ Padding ถูกต้อง
- [ ] Scale coordinates กลับ
- [ ] Clip ให้อยู่ในรูป

---

## 📚 สรุป

### 🎯 หลักการสำคัญ

1. **Model = เครื่องคำนวณ**
   - Input: Tensor → Output: Tensor
   - ไม่รู้ความหมาย ไม่รู้ Class ไม่รู้ขนาดรูปจริง

2. **Pre/Post Processing = งานของ Developer**
   - เตรียมข้อมูล + แปลงผลลัพธ์
   - ต้องเขียนเองให้ถูกต้อง

3. **ต้องตรงกับ Training**
   - Preprocessing ผิด → Model งง → ผลลัพธ์แย่
   - Postprocessing ผิด → Coordinates ผิด → Bbox วาดผิดที่

### 🚀 Best Practices

```python
# ✅ DO
- ใช้ Letterbox resize (รักษาอัตราส่วน)
- Padding สีเทา (114, 114, 114)
- Convert BGR → RGB
- IoU threshold สูง (0.85)
- Confidence threshold ตรงกับ Training (0.25)

# ❌ DON'T
- Direct resize (บิดรูป)
- Padding สีดำ (0, 0, 0)
- ลืม Convert color space
- IoU threshold ต่ำเกิน (0.45)
- Confidence threshold สูงเกิน (0.5)
```

### 📊 Performance Target

```
Preprocessing:  < 10 ms
Inference:      ~ 47 ms  (21.3 FPS)
Postprocessing: < 5 ms
─────────────────────────
Total:          ~ 62 ms  (16 FPS)
```

### 🎉 Success Criteria

- ✅ RKNN detections = ONNX detections (23/23 = 100%)
- ✅ Confidence scores สมเหตุสมผล (0.9+)
- ✅ Coordinates ถูกต้อง (วาด bbox ตรงของจริง)
- ✅ Performance ดี (> 20 FPS)

---

## 🔗 เอกสารอ้างอิง

- [RKNN Toolkit2 Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**📅 Updated:** November 27, 2025  
**📝 Author:** Firefly EC-R3588SPC Development Team  
**🔖 Version:** 1.0
