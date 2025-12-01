# 📋 RKNN NPU Complete Guide - EC-R3588SPC
**สำหรับการใช้งาน NPU บน RK3588 SoC**

---

## 🎯 Overview
การใช้งาน NPU บน EC-R3588SPC ต้องใช้ **RKNN format** เท่านั้น โดยต้องผ่าน **RKNN Server** เพื่อเข้าถึง Hardware NPU

### 🏗️ NPU Architecture
```
YOLOv5 Model (.pt/.onnx) → RKNN Converter → .rknn Model → RKNN Server → NPU Hardware
```

---

## 🔄 วิธีแปลง ONNX YOLOv5 FP32 เป็น RKNN

### 📋 Requirements
```bash
- RKNN Toolkit2 v2.3.0+
- Python 3.8+
- ONNX Model (YOLOv5 format)
- Calibration Images (สำหรับ INT8 quantization)
```

### 🛠️ Step 1: เตรียม Environment
```bash
# Install RKNN Toolkit2
pip install rknn-toolkit2

# Verify installation
python3 -c "from rknn import RKNN; print('RKNN OK')"
```

### 🔧 Step 2: แปลง ONNX เป็น RKNN
```python
#!/usr/bin/env python3
# onnx_to_rknn_converter.py

from rknn import RKNN
import numpy as np
import cv2
import os

def convert_yolov5_to_rknn(onnx_path, rknn_path, quantize=True):
    """
    แปลง YOLOv5 ONNX เป็น RKNN format
    
    Args:
        onnx_path: path ไฟล์ ONNX
        rknn_path: path output RKNN
        quantize: ใช้ INT8 quantization หรือไม่
    """
    
    # 1. สร้าง RKNN object
    rknn = RKNN(verbose=True)
    
    try:
        # 2. Config สำหรับ RK3588
        print("🔧 Configuring for RK3588...")
        ret = rknn.config(
            mean_values=[[0, 0, 0]],           # YOLOv5 normalization
            std_values=[[255, 255, 255]],      # Scale to 0-1
            target_platform='rk3588'           # Target hardware
        )
        if ret != 0:
            raise Exception("Config failed!")
        
        # 3. Load ONNX model
        print(f"📥 Loading ONNX: {onnx_path}")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise Exception("Load ONNX failed!")
        
        # 4. Build RKNN model
        print("🏗️ Building RKNN model...")
        if quantize:
            # สำหรับ INT8 quantization (ประหยัดพื้นที่, เร็วกว่า)
            dataset_path = create_calibration_dataset()
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            # สำหรับ FP32 (แม่นยำกว่า)
            ret = rknn.build(do_quantization=False)
        
        if ret != 0:
            raise Exception("Build failed!")
        
        # 5. Export RKNN file
        print(f"💾 Exporting to: {rknn_path}")
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise Exception("Export failed!")
        
        print("✅ Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Error: {e}")
        return False
    
    finally:
        rknn.release()

def create_calibration_dataset(image_folder="./calibration_images", 
                              dataset_path="./dataset.txt"):
    """
    สร้าง calibration dataset สำหรับ INT8 quantization
    """
    if not os.path.exists(image_folder):
        print(f"⚠️ Creating sample calibration images...")
        os.makedirs(image_folder, exist_ok=True)
        
        # สร้างรูปตัวอย่าง (ควรใช้รูปจริงจาก training set)
        for i in range(10):
            sample_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(f"{image_folder}/sample_{i}.jpg", sample_img)
    
    # สร้าง dataset.txt
    with open(dataset_path, 'w') as f:
        for img_file in os.listdir(image_folder):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                f.write(f"{image_folder}/{img_file}\n")
    
    return dataset_path

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # Configuration
    ONNX_MODEL = "yolov5s.onnx"          # Input ONNX file
    RKNN_MODEL = "yolov5s_fp32.rknn"     # Output RKNN file
    
    # Convert with FP32 precision
    success = convert_yolov5_to_rknn(
        onnx_path=ONNX_MODEL,
        rknn_path=RKNN_MODEL,
        quantize=False  # FP32 mode
    )
    
    if success:
        print(f"🎉 Model converted: {RKNN_MODEL}")
        print(f"📊 File size: {os.path.getsize(RKNN_MODEL)/1024/1024:.1f} MB")
    else:
        print("💥 Conversion failed!")
```

### 🚀 Step 3: การใช้งาน Script
```bash
# แปลงโดยตรง
python3 onnx_to_rknn_converter.py

# หรือใช้ผ่าน function
python3 -c "
from onnx_to_rknn_converter import convert_yolov5_to_rknn
convert_yolov5_to_rknn('model.onnx', 'model.rknn', quantize=False)
"
```

---

## 🎯 วิธีใช้งาน RKNN Model

### 📋 Requirements
```bash
- RKNN Runtime (มีแล้วใน EC-R3588SPC)
- RKNN Server (รันอัตโนมัติ)
- Python 3.8+ with rknn-toolkit2
```

### 🔧 RKNN Inference Script
```python
#!/usr/bin/env python3
# npu_inference.py

import cv2
import numpy as np
from rknn import RKNN
import time

class NPUInference:
    def __init__(self, model_path, class_names=None):
        """
        Initialize NPU inference
        
        Args:
            model_path: path to .rknn model
            class_names: list of class names for detection
        """
        self.model_path = model_path
        self.class_names = class_names or []
        self.rknn = RKNN()
        self.input_size = 640  # YOLOv5 default
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load RKNN model to NPU"""
        try:
            print(f"📥 Loading model: {self.model_path}")
            
            # Load RKNN model
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise Exception("Failed to load RKNN model")
            
            # Initialize runtime (ใช้ NPU)
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise Exception("Failed to initialize NPU runtime")
            
            print("✅ NPU model loaded successfully")
            
        except Exception as e:
            print(f"💥 Error loading model: {e}")
            raise e
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for YOLOv5 inference
        
        Args:
            image_path: path to input image
            
        Returns:
            processed_image, original_image, scale_factor, padding
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Calculate scale and padding for letterbox
        scale = min(self.input_size/w, self.input_size/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Create letterbox (pad to square)
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_h, self.input_size-new_h-pad_h, 
            pad_w, self.input_size-new_w-pad_w, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        img_input = np.expand_dims(img_rgb, axis=0)
        
        return img_input, original_img, scale, (pad_w, pad_h)
    
    def inference(self, image_path):
        """
        Run inference on NPU
        
        Args:
            image_path: path to input image
            
        Returns:
            detection results, inference time
        """
        try:
            # Preprocess
            img_input, original_img, scale, padding = self.preprocess_image(image_path)
            
            # Run inference
            print("⚡ Running inference on NPU...")
            start_time = time.time()
            
            outputs = self.rknn.inference(inputs=[img_input])
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            print(f"✅ Inference completed in {inference_time:.2f} ms")
            print(f"🎯 Throughput: {1000/inference_time:.1f} FPS")
            
            # Post-process results
            results = self.postprocess_yolo(outputs[0], original_img.shape, scale, padding)
            
            return results, inference_time
            
        except Exception as e:
            print(f"💥 Inference error: {e}")
            return None, 0
    
    def postprocess_yolo(self, output, img_shape, scale, padding):
        """
        Post-process YOLOv5 output
        
        Args:
            output: raw model output
            img_shape: original image shape (h, w, c)
            scale: scale factor from preprocessing
            padding: padding values (pad_w, pad_h)
            
        Returns:
            list of detections [x1, y1, x2, y2, confidence, class_id]
        """
        pad_w, pad_h = padding
        h, w = img_shape[:2]
        
        # Reshape output (batch, num_anchors, 5+num_classes)
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Filter by confidence threshold
        conf_threshold = 0.5
        mask = output[:, 4] > conf_threshold
        output = output[mask]
        
        if len(output) == 0:
            return []
        
        # Extract box coordinates and class scores
        boxes = output[:, :4]
        scores = output[:, 4]
        class_scores = output[:, 5:] if output.shape[1] > 5 else np.ones((len(output), 1))
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        confidences = scores * np.max(class_scores, axis=1)
        
        # Convert boxes from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale boxes back to original image coordinates
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip boxes to image boundaries
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        # Apply NMS (Non-Maximum Suppression)
        boxes_for_nms = np.column_stack([x1, y1, x2, y2])
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(), 
            confidences.tolist(), 
            conf_threshold, 
            0.4  # NMS threshold
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append([
                    int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
                    float(confidences[i]), int(class_ids[i])
                ])
        
        return results
    
    def draw_results(self, image_path, results, output_path=None):
        """
        Draw detection results on image
        
        Args:
            image_path: input image path
            results: detection results
            output_path: output image path
        """
        img = cv2.imread(image_path)
        
        for result in results:
            x1, y1, x2, y2, confidence, class_id = result
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1-25), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"💾 Result saved: {output_path}")
        
        return img
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'rknn'):
            self.rknn.release()

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "yolov5s.rknn"
    IMAGE_PATH = "test_image.jpg"
    OUTPUT_PATH = "result.jpg"
    CLASS_NAMES = ["person", "bicycle", "car"]  # YOLOv5 classes
    
    # Initialize NPU inference
    npu = NPUInference(MODEL_PATH, CLASS_NAMES)
    
    # Run inference
    results, inference_time = npu.inference(IMAGE_PATH)
    
    if results:
        print(f"🎯 Found {len(results)} objects")
        
        # Draw and save results
        npu.draw_results(IMAGE_PATH, results, OUTPUT_PATH)
        
        # Print detection details
        for i, result in enumerate(results):
            x1, y1, x2, y2, conf, class_id = result
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            print(f"  [{i+1}] {class_name}: {conf:.3f} at ({x1},{y1},{x2},{y2})")
    else:
        print("⚠️ No objects detected")
```

### 🚀 การใช้งาน Script
```bash
# รัน inference
python3 npu_inference.py

# หรือใช้ผ่าน command line
python3 -c "
from npu_inference import NPUInference
npu = NPUInference('model.rknn', ['class1', 'class2'])
results, time = npu.inference('test.jpg')
print(f'Found {len(results)} objects in {time:.1f}ms')
"
```

---

## ⚠️ ข้อจำกัดตอนใช้งาน

### 🚫 Hardware Limitations

#### 1. **Single NPU Core Processing**
```bash
❌ ไม่สามารถประมวลผล multiple models พร้อมกันได้
❌ ทำงานแบบ sequential queue เท่านั้น
✅ ใช้ทรัพยากร 100% เมื่อทำงาน (binary mode: 0% หรือ 100%)
```

#### 2. **Memory Constraints**
```bash
⚠️ NPU Memory: จำกัดตามขนาด model
⚠️ Model Size: ยิ่งใหญ่ ยิ่งใช้เวลา load นาน
⚠️ Batch Size: แนะนำ batch size = 1 สำหรับ real-time
```

#### 3. **Model Format Restrictions**
```bash
✅ รองรับ: .rknn format เท่านั้น
❌ ไม่รองรับ: .pt, .onnx, .tflite โดยตรง
❌ Dynamic shapes: ไม่รองรับ dynamic input shapes
```

### 🔧 Software Limitations

#### 1. **RKNN Server Dependencies**
```bash
⚠️ ต้องมี rknn_server running เสมอ
⚠️ หาก server crash = NPU ใช้ไม่ได้
⚠️ Restart service: sudo systemctl restart rknn_server
```

#### 2. **Model Conversion Issues**
```bash
❌ ไม่ใช่ทุก ONNX model แปลงได้
❌ Custom operators อาจไม่รองรับ
❌ บาง activation functions ไม่รองรับ
⚠️ ต้องเทสต์ model หลังแปลงเสมอ
```

#### 3. **Performance Limitations**
```bash
⚠️ Model switching: ใช้เวลา reload model (~1-2 วินาที)
⚠️ Concurrent requests: ทำให้ช้าลงเพราะต้องรอคิว
⚠️ Large models: อาจใช้เวลา inference นานกว่า GPU
```

### 🔍 Debugging Common Issues

#### 1. **NPU ไม่ทำงาน**
```bash
# ตรวจสอบ RKNN Server
systemctl status rknn_server

# Restart service
sudo systemctl restart rknn_server

# ตรวจสอบ NPU frequency
cat /sys/class/devfreq/fdab0000.npu/cur_freq
```

#### 2. **Model Load ไม่ได้**
```bash
# ตรวจสอบไฟล์ .rknn
ls -la model.rknn

# ทดสอบ model
python3 -c "
from rknn import RKNN
rknn = RKNN()
ret = rknn.load_rknn('model.rknn')
print('Load result:', ret)
rknn.release()
"
```

#### 3. **Performance ช้า**
```bash
# ตรวจสอบ NPU governor
cat /sys/class/devfreq/fdab0000.npu/governor

# เปลี่ยนเป็น performance mode
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

### 💡 Best Practices

#### 1. **Model Optimization**
```bash
✅ ใช้ FP16 quantization สำหรับ balance ระหว่าง speed/accuracy
✅ ใช้ INT8 quantization สำหรับ maximum speed
✅ เก็บ model files ใน fast storage (SSD)
```

#### 2. **Application Design**
```bash
✅ Load model 1 ครั้ง, ใช้หลายครั้ง
✅ ใช้ batch processing แทน concurrent requests
✅ Implement proper error handling สำหรับ NPU failures
```

#### 3. **Performance Monitoring**
```bash
# ติดตาม NPU usage
watch -n 1 'cat /sys/class/devfreq/fdab0000.npu/cur_freq'

# ติดตาม memory usage
htop

# ติดตาม inference time
time python3 npu_inference.py
```

---

## 📊 Performance Benchmarks

### 🎯 Typical Performance (RK3588 NPU)

| Model Type | Input Size | FP32 | FP16 | INT8 |
|------------|------------|------|------|------|
| **YOLOv5s** | 640x640 | 75ms | 65ms | 45ms |
| **YOLOv5m** | 640x640 | 120ms | 95ms | 70ms |
| **YOLOv5l** | 640x640 | 180ms | 140ms | 100ms |

### 📈 Throughput Comparison

```bash
NPU (RK3588):     13-22 FPS (depending on model size)
CPU (A76 cores):  2-5 FPS (same models)
GPU (Mali-G610):  8-12 FPS (same models)

🏆 NPU Winner: 2-4x faster than CPU, 1.5-2x faster than GPU
```

---

## 🎯 สรุปการใช้งาน RKNN NPU

### ✅ จุดแข็ง
- **Performance**: เร็วกว่า CPU/GPU อย่างชัดเจน
- **Power Efficiency**: ประหยัดไฟกว่า GPU
- **Offline Operation**: ทำงานได้ 100% แบบ offline
- **Edge Computing**: เหมาะสำหรับ embedded applications

### ⚠️ ข้อควรระวัง
- **Single Task**: ประมวลผลทีละ model เท่านั้น
- **Format Dependency**: ต้องใช้ .rknn format เท่านั้น
- **Server Dependency**: พึ่ง rknn_server service
- **Limited Flexibility**: น้อยกว่า GPU ในเรื่อง customization

### 🚀 แนะนำการใช้งาน
1. **ใช้สำหรับ Production Inference**: เร็วและประหยัดไฟ
2. **ไม่ใช้สำหรับ Training**: NPU สำหรับ inference เท่านั้น
3. **เหมาะสำหรับ Edge Devices**: ที่ต้องการ real-time processing
4. **ทดสอบ Model ก่อนใช้งานจริง**: เพื่อตรวจสอบ accuracy หลังแปลง

---

## 📚 Additional Resources

### 🔗 Documentation Links
- [Firefly NPU Usage Guide](https://wiki.t-firefly.com/en/EC-R3588SPC/usage_npu.html)
- [RKNN Toolkit2 Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [RK3588 NPU Performance Guide](https://wiki.t-firefly.com/en/EC-R3588SPC/)

### 🛠️ Tools และ Scripts
- `onnx_to_rknn_converter.py`: ONNX → RKNN converter
- `npu_inference.py`: NPU inference tool
- `npu_monitor.py`: Performance monitoring tool
- `batch_npu_inference.py`: Batch processing tool

### 📞 Support
สำหรับปัญหาการใช้งาน สามารถดู logs ได้ที่:
```bash
# RKNN Server logs
journalctl -u rknn_server -f

# System logs
dmesg | grep -i npu

# Performance logs
cat /sys/class/devfreq/fdab0000.npu/load
```

---

**📝 Document Version**: 1.0  
**📅 Last Updated**: October 29, 2025  
**👨‍💻 Author**: YOLO NPU Implementation Team  
**🏢 Hardware**: EC-R3588SPC (RK3588 SoC, 6 TOPS NPU)
