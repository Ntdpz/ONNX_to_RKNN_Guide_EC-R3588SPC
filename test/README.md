# Test - Testing & Validation Tools

โฟลเดอร์สำหรับทดสอบและ validate โมเดล RKNN หลังจากแปลงแล้ว

---

## 📋 ภาพรวม

โฟลเดอร์นี้ใช้สำหรับ:
- ✅ ทดสอบโมเดล RKNN บน RK3588
- ✅ Validate accuracy และ performance
- ✅ Debug และตรวจสอบ output
- ✅ เก็บไฟล์ทดสอบและเครื่องมือ

---

## 📂 โครงสร้างโฟลเดอร์

```
test/
├── README.md              # เอกสารนี้
├── file/                  # ไฟล์สำหรับทดสอบ
│   └── test.jpg          # รูปภาพทดสอบ
└── tools/                 # เครื่องมือทดสอบ
    └── Bun-detech.py     # สคริปต์ทดสอบ detection
```

---

## 🎯 วัตถุประสงค์

### 1. file/ - Test Files
เก็บไฟล์ที่ใช้สำหรับทดสอบโมเดล:
- รูปภาพทดสอบ (`.jpg`, `.png`)
- วิดีโอทดสอบ (`.mp4`, `.avi`)
- ข้อมูลทดสอบอื่นๆ

**การใช้งาน:**
```bash
# วางไฟล์ทดสอบไว้ที่นี่
test/file/
├── test.jpg
├── test_image_2.jpg
├── test_video.mp4
└── ...
```

---

### 2. tools/ - Testing Tools
เก็บสคริปต์และเครื่องมือสำหรับทดสอบ:
- Detection scripts
- Inference scripts
- Benchmark tools
- Validation scripts

---

## 🚀 วิธีการใช้งาน

### ทดสอบโมเดล Detection

```bash
cd /home/nz/firefly/ONNX_to_RKNN_Guide_EC-R3588SPC/test/tools

# รันสคริปต์ทดสอบ
python3 Bun-detech.py \
    --model ../../Model-AI/bun_stage1_detection/best_int8.rknn \
    --image ../file/test.jpg \
    --output result.jpg
```

**หมายเหตุ:** สคริปต์จะต้องรันบน **RK3588 hardware** เท่านั้น (ไม่สามารถรันบน x86 ได้)

---

## 📝 การเพิ่มไฟล์ทดสอบ

### เพิ่มรูปภาพทดสอบ

```bash
# Copy รูปภาพไปยังโฟลเดอร์ file/
cp /path/to/your/image.jpg test/file/

# หรือดาวน์โหลดจาก URL
wget -O test/file/test_new.jpg https://example.com/image.jpg
```

---

### เพิ่มสคริปต์ทดสอบใหม่

```bash
# สร้างสคริปต์ใหม่ในโฟลเดอร์ tools/
cd test/tools
nano my_test_script.py
```

**โครงสร้างสคริปต์แนะนำ:**
```python
#!/usr/bin/env python3
"""
Test Script Template
"""

from rknn.api import RKNN
import cv2
import numpy as np

def run_inference(model_path, image_path):
    # Load RKNN model
    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target='rk3588')
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    # ... preprocessing ...
    
    # Run inference
    outputs = rknn.inference(inputs=[img])
    
    # Process results
    # ... postprocessing ...
    
    rknn.release()
    return results

if __name__ == '__main__':
    # Your test code here
    pass
```

---

## 🔍 ตัวอย่างการทดสอบ

### 1. ทดสอบ Single Image

```bash
cd test/tools
python3 Bun-detech.py \
    --model ../../Model-AI/bun_stage1_detection/best_int8.rknn \
    --image ../file/test.jpg
```

---

### 2. ทดสอบ Multiple Images

```bash
# สร้างสคริปต์ batch test
for img in ../file/*.jpg; do
    python3 Bun-detech.py \
        --model ../../Model-AI/bun_stage1_detection/best_int8.rknn \
        --image "$img" \
        --output "result_$(basename $img)"
done
```

---

### 3. Benchmark Performance

```bash
# วัด FPS และ latency
python3 benchmark.py \
    --model ../../Model-AI/bun_stage1_detection/best_int8.rknn \
    --iterations 100
```

---

## 🛠️ เครื่องมือที่แนะนำ

### สำหรับการทดสอบบน RK3588

**1. RKNN Runtime:**
```bash
# ติดตั้งบน RK3588 board
pip3 install rknnlite2
```

**2. OpenCV:**
```bash
sudo apt-get install python3-opencv
```

**3. Visualization Tools:**
```bash
pip3 install matplotlib
pip3 install pillow
```

---

## 📊 การ Validate Accuracy

### วิธี Validate Model

```python
# ตัวอย่างโค้ด validation
import numpy as np

def calculate_accuracy(predictions, ground_truth):
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy

# Run validation
# accuracy = calculate_accuracy(model_outputs, true_labels)
```

---

## 🎯 Checklist การทดสอบ

เมื่อได้โมเดล RKNN ใหม่ ควรทดสอบ:

- [ ] โหลดโมเดลได้สำเร็จ
- [ ] Input preprocessing ถูกต้อง
- [ ] Output shape ถูกต้อง
- [ ] Detection/Classification ทำงานได้
- [ ] Accuracy ใกล้เคียงกับ ONNX model
- [ ] Performance (FPS) เป็นไปตามที่คาดหวัง
- [ ] ทดสอบกับรูปหลายๆ แบบ
- [ ] Edge cases ทำงานได้

---

## 🐛 Troubleshooting

### ❌ Error: "Model file not found"

**วิธีแก้:**
```bash
# ตรวจสอบ path ของโมเดล
ls -la ../../Model-AI/<model-name>/

# ใช้ absolute path
python3 script.py --model /absolute/path/to/model.rknn
```

---

### ❌ Error: "Init runtime failed"

**สาเหตุ:** รันบน x86 แทน RK3588

**วิธีแก้:**
- ต้องรันบน RK3588 hardware เท่านั้น
- หรือใช้ Simulator mode (accuracy อาจไม่ตรง)

```python
# Simulator mode (สำหรับทดสอบบน x86)
rknn.init_runtime(target='rk3588', target_sub_class='RKNN3588')
```

---

### ⚠️ Warning: "Output mismatch"

**สาเหตุ:** Preprocessing ไม่ตรงกับตอน training

**วิธีแก้:**
```python
# ตรวจสอบ mean/std values
# ต้องตรงกับที่ใช้ตอน convert ONNX → RKNN

# ตัวอย่าง
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = (img - [0, 0, 0]) / [255, 255, 255]  # ต้องตรงกับ --mean --std
```

---

## 📁 ตัวอย่างโครงสร้างไฟล์

```
test/
├── README.md
├── file/                          # Test files
│   ├── test.jpg
│   ├── test_bun_1.jpg
│   ├── test_bun_2.jpg
│   └── test_video.mp4
├── tools/                         # Testing tools
│   ├── Bun-detech.py             # Detection script
│   ├── benchmark.py               # Performance benchmark
│   ├── validate_accuracy.py       # Accuracy validation
│   └── inference_video.py         # Video inference
└── results/                       # Test results (optional)
    ├── result_1.jpg
    ├── result_2.jpg
    └── metrics.json
```

---

## 🔗 Related Documents

| Document | Location | Description |
|----------|----------|-------------|
| **ONNX Converter** | `../onnx_to_rknn_converter/README.md` | วิธีแปลง ONNX → RKNN |
| **Model Storage** | `../Model-AI/README.md` | โมเดลที่พร้อมทดสอบ |
| **RKNN Guide** | `../Doc/Custom_Model_to_RKNN_Guide_v2.3.2.md` | คู่มือ RKNN Toolkit2 |

---

## 📚 References

### Testing & Deployment
- **RKNN Runtime API:** https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit-lite2
- **Python API Examples:** https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit2/examples

### Performance Optimization
- **NPU Performance Guide:** ดูใน `../Doc/` folder
- **Benchmark Tools:** https://github.com/airockchip/rknn_model_zoo

---

## 📝 หมายเหตุ

- โฟลเดอร์นี้เป็น **Testing Environment** เท่านั้น
- ไม่ใช่สำหรับ Production deployment
- สำหรับ Production ให้ดูใน `../Doc/` สำหรับ deployment guide
- Test scripts ควรรันบน **RK3588 hardware** เพื่อความแม่นยำ

---

**Last Updated:** November 27, 2025  
**Purpose:** Testing & Validation  
**Platform:** RK3588 (EC-R3588SPC)
