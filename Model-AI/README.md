# Model-AI - Model Storage

โฟลเดอร์สำหรับเก็บโมเดล AI ทั้งหมด (ONNX และ RKNN)

---

## 📋 ภาพรวม

โฟลเดอร์นี้ใช้สำหรับ**เก็บโมเดล AI** ที่ผ่านการ Training และแปลงเป็นรูปแบบต่างๆ แล้ว ไม่มีสคริปต์หรือเครื่องมือในโฟลเดอร์นี้ เป็นเพียง **Storage** สำหรับโมเดลเท่านั้น

---

## 📂 โครงสร้างโฟลเดอร์

```
Model-AI/
└── <ชื่อ Model>/
    ├── best.onnx                    # โมเดล ONNX ต้นฉบับ
    ├── best_fp16.rknn              # โมเดล RKNN แบบ FP16 (precision สูง)
    ├── best_int8.rknn              # โมเดล RKNN แบบ INT8 (quantized)
    ├── best_yolov8_fp16.rknn       # โมเดล YOLOv8 RKNN แบบ FP16
    └── model_config.yaml            # ไฟล์ configuration ของโมเดล
```

### ตัวอย่าง

```
Model-AI/
├── bun_stage1_detection/
│   ├── best.onnx
│   ├── best_fp16.rknn
│   ├── best_int8.rknn
│   ├── best_yolov8_fp16.rknn
│   └── model_config.yaml
├── bun_stage2_classification/
│   ├── model.onnx
│   ├── model_fp16.rknn
│   └── model_config.yaml
└── ...
```

---

## 📝 ประเภทไฟล์

### 1. ONNX Model (`.onnx`)
- โมเดลต้นฉบับที่แปลงจาก PyTorch/TensorFlow
- ใช้สำหรับการแปลงเป็น RKNN
- Format: Open Neural Network Exchange

### 2. RKNN FP16 Model (`.rknn`)
- โมเดลที่แปลงสำหรับ Rockchip NPU
- Precision: FP16 (16-bit floating point)
- Accuracy: สูง
- Speed: ปานกลาง

### 3. RKNN INT8 Model (`.rknn`)
- โมเดลที่ผ่าน Quantization
- Precision: INT8 (8-bit integer)
- Accuracy: ลดลงเล็กน้อย
- Speed: เร็วกว่า FP16

### 4. Model Config (`.yaml`)
- ไฟล์ Configuration ของโมเดล
- เก็บข้อมูล metadata, input/output shape, preprocessing parameters
- รายละเอียดดูใน `../requirement-step-summary/`

---

## 🎯 วัตถุประสงค์

โฟลเดอร์นี้ใช้เพื่อ:
- ✅ เก็บโมเดลที่พร้อมใช้งาน
- ✅ จัดเก็บโมเดลแยกตามโปรเจค
- ✅ เก็บทั้ง ONNX และ RKNN ไว้ในที่เดียวกัน
- ✅ เก็บ Configuration ของแต่ละโมเดล

---

## 📌 หมายเหตุสำคัญ

- โฟลเดอร์นี้**ไม่มีสคริปต์**สำหรับ Training หรือ Conversion
- สคริปต์สำหรับแปลง ONNX → RKNN อยู่ที่ `../onnx_to_rknn_converter/`
- วิธีการ Training และรายละเอียดโมเดลอยู่ที่ `../requirement-step-summary/`
- Dataset อยู่ที่ `../Data-set/`
- Documentation อยู่ที่ `../Doc/`

---

## 🔗 Related Folders

| Folder | Description |
|--------|-------------|
| `../Data-set/` | Raw datasets และ dataset list files |
| `../onnx_to_rknn_converter/` | สคริปต์แปลง ONNX → RKNN |
| `../requirement-step-summary/` | รายละเอียด training, performance, config |
| `../Doc/` | คู่มือการใช้งาน RKNN Toolkit |

---

## 📚 Reference

สำหรับรายละเอียดเพิ่มเติมเกี่ยวกับการใช้งานโมเดล:
- **การแปลง ONNX → RKNN:** ดูใน `../Doc/Custom_Model_to_RKNN_Guide_v2.3.2.md`
- **Model Configuration:** ดูใน `../requirement-step-summary/02_FIELD_CATEGORIES.md`
- **Performance Metrics:** ดูใน `../requirement-step-summary/examples/`

---

**Last Updated:** November 27, 2025  
**Purpose:** Model Storage Only  
**Compatible with:** RKNN-Toolkit2 v2.3.0+
