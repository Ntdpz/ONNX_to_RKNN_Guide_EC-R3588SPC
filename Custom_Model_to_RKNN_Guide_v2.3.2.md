# RKNN-Toolkit2 v2.3.2 - คู่มือฉบับสมบูรณ์

> **เวอร์ชัน:** 2.3.2  
> **วันที่อัปเดท:** November 26, 2025  
> **ผู้จัดทำ:** สรุปจากเอกสารและโค้ดตัวอย่าง Official SDK

---

## 📑 สารบัญ

1. [ภาพรวม RKNN Software Stack](#ภาพรวม-rknn-software-stack)
2. [ฟีเจอร์ใหม่ใน v2.3.2](#ฟีเจอร์ใหม่ใน-v232)
3. [แพลตฟอร์มที่รองรับ](#แพลตฟอร์มที่รองรับ)
4. [การติดตั้ง](#การติดตั้ง)
5. [Operator Support (ONNX/PyTorch/Caffe/TensorFlow)](#operator-support)
6. [คู่มือการใช้งาน - Custom Model](#คู่มือการใช้งาน---custom-model)
7. [ตัวอย่างโค้ด](#ตัวอย่างโค้ด)
8. [ฟีเจอร์พิเศษ](#ฟีเจอร์พิเศษ)
9. [เอกสารอ้างอิง](#เอกสารอ้างอิง)

---

## ภาพรวม RKNN Software Stack

RKNN (Rockchip Neural Network) เป็น SDK สำหรับแปลงและรัน AI Model บนชิป NPU ของ Rockchip โดยมีส่วนประกอบหลัก 3 ส่วน:

### 🔧 RKNN-Toolkit2 (PC/Server)
- **หน้าที่:** แปลงโมเดล (ONNX/PyTorch/TensorFlow/Caffe) → RKNN
- **การใช้งาน:** Model conversion, Quantization, Inference simulation
- **ระบบปฏิบัติการ:** Linux x86_64, ARM64

### 📱 RKNN-Toolkit-Lite2 (Edge Device)
- **หน้าที่:** รันโมเดล RKNN บนบอร์ด (Python API)
- **การติดตั้ง:** ผ่าน pip
- **ระบบปฏิบัติการ:** Linux ARM64

### ⚡ RKNN Runtime (Edge Device)
- **หน้าที่:** รันโมเดล RKNN บนบอร์ด (C/C++ API)
- **ประสิทธิภาพ:** สูงสุด เหมาะกับ Production
- **ระบบปฏิบัติการ:** Linux ARM64

### 🔩 RKNPU Kernel Driver
- **หน้าที่:** Interface ระหว่าง Software ↔ NPU Hardware
- **แหล่งที่มา:** Open source ใน Rockchip kernel

---

## ฟีเจอร์ใหม่ใน v2.3.2

### 🆕 สิ่งที่เพิ่มมา
| ฟีเจอร์ | รายละเอียด |
|---------|-----------|
| **RV1126B Support** | รองรับแพลตฟอร์มใหม่ RV1126B |
| **Improved Einsum & Norm** | ปรับปรุง operator สำหรับ Transformer |
| **Automatic Mixed Precision** | ใช้ INT8 + FP16 ผสมกันอัตโนมัติเพื่อความแม่นยำ |
| **Graph Optimization** | ปรับปรุงการ optimize graph ก่อน convert |

### 📊 เปรียบเทียบกับเวอร์ชันก่อนหน้า

| เวอร์ชัน | จุดเด่น |
|---------|---------|
| **v2.3.2** | RV1126B, Auto Mixed Precision |
| v2.3.0 | ARM64 support, W4A16 quantization (RK3576) |
| v2.2.0 | Pip installation, Python 3.12 |
| v2.1.0 | Flash Attention (RK3562/RK3576) |
| v1.6.0 | **ONNX Opset 12-19 Support** ⭐ |

---

## แพลตฟอร์มที่รองรับ

### ✅ รองรับเต็มรูปแบบ
- **RK3588 Series** - High-end (6 TOPS)
- **RK3576 Series** - Mid-to-High (6 TOPS)
- **RK3566/RK3568 Series** - Mid-range (1 TOPS)
- **RK3562 Series** - Entry-level
- **RV1103/RV1106** - Vision processors
- **RV1103B/RV1106B** - Vision processors (Updated)
- **RV1126B** - 🆕 New in v2.3.2
- **RK2118** - Audio processors

### ⚠️ แพลตฟอร์มเก่า (ใช้ Toolkit v1)
สำหรับ **RK1808/RV1109/RV1126/RK3399Pro** กรุณาใช้:
- https://github.com/airockchip/rknn-toolkit
- https://github.com/airockchip/rknpu

---

## การติดตั้ง

### 🐍 Python Version Support
```
Python 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
```

### 📦 ติดตั้งผ่าน Pip (แนะนำ)

```bash
# สำหรับ PC (x86_64)
pip install rknn-toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# สำหรับ PC (ARM64)
pip install rknn-toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# สำหรับบอร์ด (RKNN-Toolkit-Lite2)
pip install rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

**หมายเหตุ:** เปลี่ยน `cp310` ตาม Python version ของคุณ (cp36, cp37, cp38, cp39, cp311, cp312)

### 📥 ติดตั้งจาก Local File

```bash
cd rknn-toolkit2-2.3.2/rknn-toolkit2/packages/x86_64/
pip install rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 🔧 ติดตั้ง Dependencies

```bash
# ติดตั้งตาม Python version
pip install -r requirements_cp310-2.3.2.txt
```

---

## Operator Support

### 🔹 ONNX Operators (Opset 12-19)

**จำนวน Operators ที่รองรับ:** 100+ operators

#### ✅ Operators ที่รองรับเต็มรูปแบบ

| Category | Operators |
|----------|-----------|
| **Activation** | Relu, Sigmoid, Tanh, LeakyRelu, PRelu, Elu, HardSigmoid, HardSwish, Softmax, Softplus, Mish |
| **Convolution** | Conv, ConvTranspose, DepthToSpace, SpaceToDepth |
| **Pooling** | AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool, MaxRoiPool, MaxUnpool |
| **Normalization** | BatchNormalization, InstanceNormalization, LayerNormalization, LRN, LpNormalization, MeanVarianceNormalization |
| **Arithmetic** | Add, Sub, Mul, Div, Pow, Mod |
| **Reduction** | ReduceMean, ReduceMax, ReduceMin, ReduceSum |
| **Shape** | Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Concat, Split, Slice |
| **Math** | Exp, Log, Sqrt, Sin, Cos, Floor, Clip, Erf |
| **Logical** | And, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Where |
| **RNN** | LSTM, GRU (batchsize: 1) |
| **Other** | Pad, Resize (nearest/bilinear), Gather, GatherElements, ScatterND, Cast, Constant, ConstantOfShape, Dropout, Expand, Gemm, MatMul, RoiAlign, Shape, Size, ReverseSequence |

#### ❌ Operators ที่ไม่รองรับ

```
Abs, Acos, Asin, Atan, Ceil, Einsum, NonMaxSuppression, 
TopK, Loop, Scan, Range, OneHot, และอื่นๆ
```

**ดูรายการเต็ม:** `doc/RKNNToolKit2_OP_Support-2.3.2.md`

---

### 🔹 PyTorch Operators

**รองรับโมเดลจาก:** PyTorch >= 1.6.0

#### ✅ Operators ที่รองรับ (ตัวอย่าง)

```python
aten::_convolution      # Conv layers
aten::adaptive_avg_pool2d
aten::add, aten::mul, aten::div
aten::batch_norm
aten::relu, aten::sigmoid, aten::tanh
aten::hardswish, aten::mish, aten::silu
aten::matmul, aten::bmm
aten::cat, aten::split
aten::reshape, aten::flatten
aten::max_pool2d, aten::avg_pool2d
aten::lstm, aten::gru
aten::layer_norm
aten::softmax
aten::transpose, aten::permute
```

**หมายเหตุ:** PyTorch model ควร export เป็น TorchScript (.pt) หรือ ONNX ก่อน

---

### 🔹 Caffe Operators

**Protocol Version:** Berkeley Caffe (commit 21d0608) + Custom extensions

#### ✅ Operators ที่รองรับ

```
BatchNorm, bn (BatchNorm+Scale), Convolution, ConvolutionDepthwise
Deconvolution, Pooling, InnerProduct, Concat, Eltwise
Relu, Relu6, PRelu, Sigmoid, TanH, Softmax
LRN, Dropout, Flatten, Reshape, Permute, Slice
Normalize, Scale, Power, Crop, Reorg
Lstm, Proposal, ROIPooling, Resize, Upsample
```

---

### 🔹 TensorFlow Operators

**รองรับเวอร์ชัน:**
- TensorFlow 1.x: v1.12 - v1.15
- TensorFlow 2.x: v2.3 - v2.5

#### ✅ Operators ที่รองรับ

```
Add, AvgPool, MaxPool, Conv2D, DepthwiseConv2d
Div, LeakyRelu, Relu, Sigmoid, Softmax, Tanh
MatMul, Concat, Reshape, Transpose, Squeeze
Pad, Slice, Split, StridedSlice
ResizeBilinear, ResizeNearestNeighbor
DepthToSpace, SpaceToDepth
Mean, LRN, Softplus, Dropout, Flatten
```

---

### 🔹 Darknet Operators

```
add, batchnormalize, concat
convolutional, depthwise_convolutional
fullconnect, leakyrelu, mish
pooling (Average/Max/Global)
route, shortcut, softmax, upsampling
```

---

## คู่มือการใช้งาน - Custom Model

### 🎯 กฎเหล็ก 4 ข้อ สำหรับ Custom Model

#### 1️⃣ ต้อง "ตัดหัว" (Remove Post-processing)

**ห้ามใส่ในโมเดล:**
- Decode Box (คำนวณพิกัด x, y, w, h)
- NMS (Non-Maximum Suppression)
- Confidence Thresholding
- Class filtering

**ต้องทำ:** ให้โมเดลส่งค่า Feature Map (Raw output) ออกมา แล้วทำ Post-processing ภายนอกด้วย Python/C++

**ตัวอย่างจาก Official Code:**
```python
# rknn-toolkit2-2.3.2/examples/onnx/yolov5/test.py (line 284-315)

# 1. Inference (ได้ Raw outputs)
outputs = rknn.inference(inputs=[img2], data_format=['nhwc'])

# 2. Post-process ภายนอก NPU
input0_data = outputs[0]  # Feature map 80x80
input1_data = outputs[1]  # Feature map 40x40
input2_data = outputs[2]  # Feature map 20x20

# 3. คำนวณ boxes, classes, scores ด้วย Python
boxes, classes, scores = yolov5_post_process(input_data)
```

---

#### 2️⃣ ต้อง "Static Shape" (ขนาดคงที่)

**ห้าม:** ใช้ `dynamic_axes` ใน ONNX export

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

---

#### 3️⃣ ใช้ ONNX Opset 12-19 (แนะนำ 12)

```python
# ✅ แนะนำ - Opset 12 (เสถียรสูงสุด)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=12)

# ✅ ใช้ได้ - Opset 13-19
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=19)
```

**อ้างอิง:** CHANGELOG v1.6.0 - "Support ONNX model of OPSET 12~19"

---

#### 4️⃣ ระวัง "Tensor Reshape/Transpose" ในโมเดล

**หลีกเลี่ยง:** การ Reshape เป็น 5D หรือ permute มิติซับซ้อนในโมเดล

**ควรทำ:** ปล่อยให้ NPU ส่ง output ในรูปแบบของมัน แล้วทำ Reshape/Transpose ภายนอก

**ตัวอย่าง:**
```python
# NPU ส่ง output มาในรูปแบบ Hardware-specific
input0_data = outputs[0]

# Reshape และ Transpose ภายนอก
input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
input0_data = np.transpose(input0_data, (2, 3, 0, 1))
```

---

### 📝 Export Script สำหรับ Custom Model

```python
import torch
import torch.nn as nn

# ========================================
# สมมติ Custom Model ของคุณ
# ========================================
class MyCustomYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ...  # ResNet, EfficientNet, etc.
        self.neck = ...      # FPN, PANet, etc.
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 85, 1)  # 85 = (x,y,w,h,conf) + 80 classes
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        
        # ✅ จุดสำคัญ: ส่ง Feature Map ออกไปเลย
        # ❌ อย่าทำ: decode_boxes(), NMS() ในนี้
        return x

# ========================================
# Export สำหรับ RKNN (The Golden Script)
# ========================================
def export_for_rknn():
    model = MyCustomYOLO()
    model.eval()
    
    # ✅ Static Shape (1, 3, 640, 640)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    torch.onnx.export(
        model,
        dummy_input,
        "custom_yolo_rknn.onnx",
        
        # ✅ Opset 12 (เสถียร)
        opset_version=12,
        
        # ✅ ตั้งชื่อ Input/Output ชัดเจน
        input_names=['images'],
        output_names=['output'],
        
        # ✅ ห้ามใช้ dynamic_axes
        # dynamic_axes={'images': {0: 'batch'}}  # <-- ลบทิ้ง!
        
        # ช่วย Optimize constant folding
        do_constant_folding=True
    )
    
    print("✅ Export สำเร็จ! พร้อมแปลงเป็น RKNN")

if __name__ == "__main__":
    export_for_rknn()
```

---

## ตัวอย่างโค้ด

### 🔹 ตัวอย่างที่ 1: ONNX YOLOv5 (แบบ Python Script)

```python
from rknn.api import RKNN
import cv2
import numpy as np

# 1. สร้าง RKNN object
rknn = RKNN(verbose=True)

# 2. Config preprocessing
rknn.config(
    mean_values=[[0, 0, 0]], 
    std_values=[[255, 255, 255]], 
    target_platform='rk3588'
)

# 3. Load ONNX model
rknn.load_onnx(model='yolov5s_relu.onnx')

# 4. Build (ทำ Quantization)
rknn.build(do_quantization=True, dataset='./dataset.txt')

# 5. Export RKNN model
rknn.export_rknn('./yolov5s_relu.rknn')

# 6. Init runtime (ถ้าจะ inference บน PC)
rknn.init_runtime()

# 7. Prepare input
img = cv2.imread('bus.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = np.expand_dims(img, 0)

# 8. Inference
outputs = rknn.inference(inputs=[img], data_format=['nhwc'])

# 9. Post-processing (ภายนอก NPU)
boxes, classes, scores = yolov5_post_process(outputs)

# 10. Release
rknn.release()
```

---

### 🔹 ตัวอย่างที่ 2: PyTorch ResNet18

```python
from rknn.api import RKNN
import torch
import torchvision.models as models

# 1. Export PyTorch to TorchScript
model = models.resnet18(pretrained=True)
model.eval()
trace_model = torch.jit.trace(model, torch.Tensor(1, 3, 224, 224))
trace_model.save('./resnet18.pt')

# 2. Convert to RKNN
rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[123.675, 116.28, 103.53], 
    std_values=[58.395, 58.395, 58.395], 
    target_platform='rk3588'
)
rknn.load_pytorch(model='./resnet18.pt', input_size_list=[[1, 3, 224, 224]])
rknn.build(do_quantization=True, dataset='./dataset.txt')
rknn.export_rknn('./resnet18.rknn')
```

---

### 🔹 ตัวอย่างที่ 3: Model Config YAML (rknn_convert)

สร้างไฟล์ `model_config.yml`:

```yaml
models:
    name: yolov5s_relu           # ชื่อโมเดล output
    platform: onnx               # onnx, pytorch, tensorflow, caffe
    model_file_path: ./yolov5s_relu.onnx
    quantize: true               # เปิด Quantization
    dataset: ./dataset.txt       # Path to calibration dataset
    configs:
      quantized_dtype: asymmetric_quantized-8  # INT8 quantization
      mean_values: [0, 0, 0]
      std_values: [255, 255, 255]
      quant_img_RGB2BGR: false
      quantized_algorithm: normal  # normal, mmse
      quantized_method: channel    # channel, layer
```

**รันคำสั่ง:**
```bash
python3 -m rknn.api.rknn_convert -t rk3588 -i ./model_config.yml -o ./
```

---

## ฟีเจอร์พิเศษ

### 🔹 1. Dynamic Shape (กำหนดชุด Shape ล่วงหน้า)

```python
# กำหนดชุด Input shapes ที่อนุญาต
dynamic_input = [
    [[1,3,256,256]],    # Set 1
    [[1,3,160,160]],    # Set 2
    [[1,3,224,224]],    # Set 3
]

rknn.config(
    mean_values=[103.94, 116.78, 123.68],
    std_values=[58.82, 58.82, 58.82],
    target_platform='rk3588',
    dynamic_input=dynamic_input  # เปิดใช้ dynamic shape
)

# Inference ด้วย shape ต่างๆ
img1 = cv2.resize(img, (224,224))
outputs1 = rknn.inference(inputs=[img1], data_format=['nhwc'])

img2 = cv2.resize(img, (160,160))
outputs2 = rknn.inference(inputs=[img2], data_format=['nhwc'])
```

**หมายเหตุ:** ไม่ใช่ Dynamic แบบไม่จำกัด แต่เป็นการเลือกจากชุดที่กำหนดไว้

---

### 🔹 2. Hybrid Quantization (INT8 + FP16)

```python
rknn.config(
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',  # INT8
    hybrid_quantization_step='fp16',           # ใช้ FP16 สำหรับ layer ที่ sensitive
)
```

**ประโยชน์:** รักษาความแม่นยำในบางส่วน + ความเร็วจาก INT8

---

### 🔹 3. Weight Compression (ลดขนาดโมเดล)

```python
rknn.config(
    target_platform='rk3588',
    optimization_level=3,        # 0=none, 1=low, 2=medium, 3=high
    weight_sharing=True,         # Share duplicate weights
    weight_compression=True,     # Compress weights (RK3588/RV1106)
)
```

**ผลลัพธ์:** ลดขนาดไฟล์ .rknn และ Memory usage

---

### 🔹 4. Multi-Core Mode (RK3588 only)

```python
rknn.init_runtime(
    core_mask=RKNN.NPU_CORE_0_1_2  # ใช้ทั้ง 3 cores
)
```

**Options:**
- `RKNN.NPU_CORE_0` - Core 0 เท่านั้น
- `RKNN.NPU_CORE_1` - Core 1 เท่านั้น
- `RKNN.NPU_CORE_2` - Core 2 เท่านั้น
- `RKNN.NPU_CORE_0_1_2` - ใช้ทั้ง 3 cores

---

### 🔹 5. Accuracy Analysis (วิเคราะห์ความแม่นยำ)

```python
rknn.accuracy_analysis(
    inputs=['./test_data/input.npy'],
    output_dir='./accuracy_analysis',
    target='rk3588'
)
```

**ผลลัพธ์:** รายงานความแตกต่างระหว่าง Original model vs RKNN model แต่ละ layer

---

### 🔹 6. Custom Operator Support

สำหรับ Operator ที่ NPU ไม่รองรับ สามารถใช้ CPU/GPU fallback:

```python
rknn.config(
    target_platform='rk3588',
    custom_op={
        'op_name': 'MyCustomOp',
        'op_type': 'CPU',  # CPU or GPU
        'op_lib': './libcustom_op.so'
    }
)
```

---

## เอกสารอ้างอิง

### 📄 เอกสารใน SDK

| ไฟล์ | เนื้อหา |
|------|---------|
| `CHANGELOG.md` | ประวัติการอัปเดททุกเวอร์ชัน |
| `README.md` | ภาพรวม SDK และ Platform support |
| `doc/RKNNToolKit2_OP_Support-2.3.2.md` | รายการ Operators ทั้งหมด (ONNX/PyTorch/Caffe/TF/Darknet) |
| `doc/rknn_server_proxy.md` | วิธีใช้ rknn_server สำหรับ Remote debugging |
| `doc/Using RKNN-ToolKit2 in WSL.md` | ใช้ RKNN-Toolkit2 บน WSL (Windows) |

### 📁 โฟลเดอร์ตัวอย่าง

```
rknn-toolkit2/examples/
├── onnx/
│   ├── resnet50v2/
│   └── yolov5/              # ⭐ ตัวอย่างสำคัญ
├── pytorch/
│   ├── resnet18/
│   ├── resnet18_qat/
│   └── yolov5/
├── tensorflow/
│   ├── ssd_mobilenet_v1/
│   └── inception_v3_qat/
├── caffe/
│   ├── mobilenet_v2/
│   └── vgg-ssd/
├── tflite/
│   ├── mobilenet_v1/
│   └── mobilenet_v1_qat/
├── darknet/
│   └── yolov3_416x416/
└── functions/
    ├── accuracy_analysis/   # วิเคราะห์ความแม่นยำ
    ├── dynamic_shape/       # ⭐ Dynamic input shape
    ├── hybrid_quant/        # INT8+FP16 mixed quantization
    ├── multi_batch/         # Batch processing
    ├── custom_op/           # Custom operator
    ├── model_pruning/       # Model pruning
    ├── codegen/             # Generate C++ deployment code
    └── onnx_edit/           # Edit ONNX graph
```

### 🌐 แหล่งข้อมูลออนไลน์

- **Official GitHub:** https://github.com/airockchip/rknn-toolkit2
- **RKNN Model Zoo:** https://github.com/airockchip/rknn_model_zoo
- **RKNPU2 SDK Download:** https://console.zbox.filez.com/l/I00fc3 (รหัส: rknn)
- **RKNN-LLM (Large Language Model):** https://github.com/airockchip/rknn-llm
- **Redmine (Official Support):** https://redmine.rock-chips.com

### 💬 Community Support

- **QQ Group 1:** 1025468710 (เต็ม)
- **QQ Group 2:** 547021958 (เต็ม)
- **QQ Group 3:** 469385426 (เต็ม)
- **QQ Group 4:** 958083853 ✅

---

## 🎯 สรุป Checklist สำหรับ Custom Model

### ก่อน Export ONNX
- [ ] โมเดลใช้เฉพาะ Operators ที่ RKNN รองรับ (ตรวจสอบจาก OP Support List)
- [ ] ตัด Post-processing ออก (NMS, Decode, Thresholding)
- [ ] กำหนดขนาด Input แบบ Static (เช่น 640x640)
- [ ] ใช้ Opset 12-19 (แนะนำ 12)

### Export ONNX
```python
torch.onnx.export(
    model,
    torch.randn(1, 3, 640, 640),  # ✅ Static shape
    "model.onnx",
    opset_version=12,              # ✅ Opset 12
    input_names=['images'],
    output_names=['output'],
    do_constant_folding=True
    # ❌ ไม่มี dynamic_axes
)
```

### Convert to RKNN
```python
rknn = RKNN(verbose=True)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
rknn.load_onnx(model='model.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('model.rknn')
```

### หลัง Convert
- [ ] ทดสอบ Inference บน PC ด้วย `rknn.init_runtime()`
- [ ] ตรวจสอบความแม่นยำด้วย `rknn.accuracy_analysis()`
- [ ] ทำ Post-processing ภายนอก NPU
- [ ] Deploy บนบอร์ดด้วย RKNN Runtime (C++) หรือ Toolkit-Lite2 (Python)

---

## 📌 ข้อควรระวัง

1. **ONNX Opset:** รองรับ 12-19 แต่แนะนำ 12 เพื่อความเสถียร
2. **Post-processing:** ต้องทำภายนอก NPU เสมอ (NMS, Decode Box)
3. **Dynamic Shape:** ไม่ใช่แบบไม่จำกัด ต้องกำหนดชุด shapes ล่วงหน้า
4. **Quantization Dataset:** ต้องมีภาพตัวอย่าง 100-500 ภาพเพื่อ calibrate INT8
5. **Platform Compatibility:** RKNN-Toolkit2 ไม่เข้ากันกับ RKNN-Toolkit v1

---

**เอกสารนี้จัดทำจาก:** RKNN-Toolkit2 v2.3.2 Official SDK  
**อัปเดทล่าสุด:** November 26, 2025  
**License:** ตามเงื่อนไข Rockchip RKNN SDK
