# 🚀 NPU Parallel Inference Guide (RKNN)

## 📋 Table of Contents
1. [NPU Architecture](#npu-architecture)
2. [Multi-Core Support](#multi-core-support)
3. [Core Mask Configuration](#core-mask-configuration)
4. [Parallel Inference Methods](#parallel-inference-methods)
5. [Python Implementation](#python-implementation)
6. [C++ Implementation](#c-implementation)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Benchmarking](#benchmarking)

---

## 🖥️ NPU Architecture

### RK3588 NPU Specifications:

```
┌─────────────────────────────────────────────┐
│         RK3588 Neural Process Unit          │
├─────────────────────────────────────────────┤
│  Total Performance: 6 TOPS (INT8)           │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ NPU Core │  │ NPU Core │  │ NPU Core │  │
│  │    0     │  │    1     │  │    2     │  │
│  │ 2 TOPS   │  │ 2 TOPS   │  │ 2 TOPS   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│       ↓              ↓              ↓       │
│  ┌──────────────────────────────────────┐  │
│  │      Shared Memory & Cache            │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Core Capabilities:
| Feature | Value |
|---------|-------|
| **Total Cores** | 3 |
| **Per Core Performance** | 2 TOPS (INT8) |
| **Total Performance** | 6 TOPS (INT8) |
| **Supported Precision** | INT8, INT16, FP16 (partial) |
| **Max Batch Size** | Platform dependent |
| **Concurrent Contexts** | Multiple (limited by memory) |

---

## 🔢 Multi-Core Support

### Core Modes:

RK3588 NPU has **3 independent cores** that can be used in different configurations:

#### **1. Single Core Mode**
- Use one specific core
- Lower power consumption
- Simple workload

#### **2. Dual Core Mode**
- Use two cores simultaneously
- Better for medium workloads
- Load balancing

#### **3. Triple Core Mode (All Cores)**
- Maximum performance
- Best for heavy workloads
- Highest power consumption

#### **4. Auto Mode**
- System decides which core(s) to use
- Dynamic allocation
- Recommended for most cases

---

## 🎯 Core Mask Configuration

### Core Mask Values:

```c
typedef enum _rknn_core_mask {
    RKNN_NPU_CORE_AUTO = 0,      // Auto: System chooses (Default)
    RKNN_NPU_CORE_0 = 1,          // 0b001: Use Core 0 only
    RKNN_NPU_CORE_1 = 2,          // 0b010: Use Core 1 only
    RKNN_NPU_CORE_2 = 4,          // 0b100: Use Core 2 only
    RKNN_NPU_CORE_0_1 = 3,        // 0b011: Use Core 0 + Core 1
    RKNN_NPU_CORE_0_1_2 = 7,      // 0b111: Use All 3 Cores
    RKNN_NPU_CORE_ALL = 0xffff,   // Auto: Use all available cores
} rknn_core_mask;
```

### Binary Representation:

```
Core 2  Core 1  Core 0
  ↓       ↓       ↓
  0       0       1    = 1  (RKNN_NPU_CORE_0)
  0       1       0    = 2  (RKNN_NPU_CORE_1)
  1       0       0    = 4  (RKNN_NPU_CORE_2)
  0       1       1    = 3  (RKNN_NPU_CORE_0_1)
  1       1       1    = 7  (RKNN_NPU_CORE_0_1_2)
```

---

## 🚀 Parallel Inference Methods

### **Method 1: Single Model, Multiple Instances (Recommended)**

**Use Case:** Multiple video streams, same model

```
Stream 1 → Model Instance (Core 0) → Detection Result 1
Stream 2 → Model Instance (Core 1) → Detection Result 2
Stream 3 → Model Instance (Core 2) → Detection Result 3
```

**Advantages:**
- ✅ Maximum throughput
- ✅ Independent processing
- ✅ No core contention

---

### **Method 2: Multi-Core Single Instance**

**Use Case:** Single large workload, need max speed

```
Single Large Image → Model (All 3 Cores) → Fast Result
```

**Advantages:**
- ✅ Lowest latency for single inference
- ✅ Best for high-resolution images
- ✅ Simple implementation

---

### **Method 3: Pipeline Parallel**

**Use Case:** Multi-stage detection (cascade)

```
Core 0: Vehicle Detection
   ↓
Core 1: License Plate Detection
   ↓
Core 2: Character Recognition
```

**Advantages:**
- ✅ Different models on different cores
- ✅ Pipeline efficiency
- ✅ No waiting between stages

---

### **Method 4: Data Parallel**

**Use Case:** Batch processing

```
Batch Images [0-9]   → Core 0
Batch Images [10-19] → Core 1
Batch Images [20-29] → Core 2
```

**Advantages:**
- ✅ Process multiple images simultaneously
- ✅ High throughput
- ✅ Good for offline processing

---

## 🐍 Python Implementation

### **Method 1: Multiple Contexts (Multi-Stream)**

```python
from rknnlite.api import RKNNLite
import threading
import queue

class MultiStreamNPU:
    """Multi-stream inference with different NPU cores"""
    
    def __init__(self, model_path, num_streams=3):
        self.contexts = []
        self.threads = []
        self.input_queues = []
        self.output_queues = []
        
        # Create separate context for each stream
        for stream_id in range(num_streams):
            # Initialize RKNN context
            rknn = RKNNLite()
            
            # Load model
            ret = rknn.load_rknn(model_path)
            if ret != 0:
                raise Exception(f"Load model failed for stream {stream_id}")
            
            # Set core mask (pin to specific core)
            core_mask = 1 << stream_id  # 1, 2, 4 for core 0, 1, 2
            
            # Init runtime with specific core
            ret = rknn.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise Exception(f"Init runtime failed for stream {stream_id}")
            
            self.contexts.append(rknn)
            self.input_queues.append(queue.Queue(maxsize=10))
            self.output_queues.append(queue.Queue(maxsize=10))
            
            # Start worker thread for this core
            thread = threading.Thread(
                target=self._worker,
                args=(stream_id,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
    
    def _worker(self, stream_id):
        """Worker thread for specific NPU core"""
        rknn = self.contexts[stream_id]
        input_q = self.input_queues[stream_id]
        output_q = self.output_queues[stream_id]
        
        while True:
            # Get input from queue
            frame = input_q.get()
            if frame is None:
                break
            
            # Run inference
            outputs = rknn.inference(inputs=[frame])
            
            # Put result to output queue
            output_q.put(outputs)
    
    def infer(self, stream_id, frame):
        """Submit frame for inference"""
        self.input_queues[stream_id].put(frame)
    
    def get_result(self, stream_id, timeout=1.0):
        """Get inference result"""
        try:
            return self.output_queues[stream_id].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self):
        """Release all contexts"""
        # Stop all workers
        for q in self.input_queues:
            q.put(None)
        
        # Wait for threads
        for thread in self.threads:
            thread.join()
        
        # Release contexts
        for rknn in self.contexts:
            rknn.release()


# Usage Example
if __name__ == '__main__':
    import cv2
    import numpy as np
    
    # Create multi-stream NPU
    npu = MultiStreamNPU('/path/to/model.rknn', num_streams=3)
    
    # Process 3 streams in parallel
    caps = [
        cv2.VideoCapture('rtsp://camera1'),
        cv2.VideoCapture('rtsp://camera2'),
        cv2.VideoCapture('rtsp://camera3')
    ]
    
    while True:
        for stream_id, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                # Preprocess
                input_data = preprocess(frame)
                
                # Submit to NPU
                npu.infer(stream_id, input_data)
        
        # Get results
        for stream_id in range(3):
            result = npu.get_result(stream_id, timeout=0.1)
            if result is not None:
                # Post-process and display
                process_result(result, stream_id)
    
    npu.release()
```

---

### **Method 2: Single Context, All Cores (Max Performance)**

```python
from rknnlite.api import RKNNLite

class SingleModelAllCores:
    """Use all NPU cores for single model"""
    
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        
        # Load model
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise Exception("Load model failed")
        
        # Init runtime with ALL cores
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise Exception("Init runtime failed")
    
    def infer(self, input_data):
        """Run inference using all 3 cores"""
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs
    
    def release(self):
        self.rknn.release()


# Usage
model = SingleModelAllCores('/path/to/model.rknn')

# Single inference using all cores (fastest per-inference)
result = model.infer(input_data)

model.release()
```

---

### **Method 3: Pipeline Parallel (Cascade Detection)**

```python
from rknnlite.api import RKNNLite
import queue
import threading

class CascadePipeline:
    """Multi-stage pipeline on different cores"""
    
    def __init__(self, model_paths):
        """
        model_paths: list of model paths for each stage
        Example: ['vehicle.rknn', 'plate.rknn', 'char.rknn']
        """
        self.stages = []
        self.queues = []
        self.threads = []
        
        # Create stage for each model
        for stage_id, model_path in enumerate(model_paths):
            rknn = RKNNLite()
            rknn.load_rknn(model_path)
            
            # Pin each stage to different core
            core_mask = 1 << stage_id
            rknn.init_runtime(core_mask=core_mask)
            
            self.stages.append(rknn)
            self.queues.append(queue.Queue(maxsize=10))
            
            # Start worker for this stage
            thread = threading.Thread(
                target=self._stage_worker,
                args=(stage_id,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Output queue
        self.output_queue = queue.Queue(maxsize=10)
    
    def _stage_worker(self, stage_id):
        """Worker for specific pipeline stage"""
        rknn = self.stages[stage_id]
        input_q = self.queues[stage_id]
        
        # Get next queue (or output queue for last stage)
        if stage_id < len(self.stages) - 1:
            output_q = self.queues[stage_id + 1]
        else:
            output_q = self.output_queue
        
        while True:
            data = input_q.get()
            if data is None:
                break
            
            # Run inference for this stage
            result = rknn.inference(inputs=[data])
            
            # Pass to next stage
            output_q.put(result)
    
    def process(self, input_data):
        """Submit data to first stage"""
        self.queues[0].put(input_data)
    
    def get_final_result(self, timeout=1.0):
        """Get result from last stage"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self):
        for q in self.queues:
            q.put(None)
        for thread in self.threads:
            thread.join()
        for rknn in self.stages:
            rknn.release()


# Usage: 3-stage cascade detection
pipeline = CascadePipeline([
    'vehicle_detection.rknn',   # Core 0
    'plate_detection.rknn',     # Core 1
    'character_recognition.rknn' # Core 2
])

# Process frame through pipeline
pipeline.process(frame)

# Get final result (after all 3 stages)
final_result = pipeline.get_final_result(timeout=0.5)

pipeline.release()
```

---

## 💻 C++ Implementation

### **Multi-Core Inference (C++ RKNN API)**

```cpp
#include "rknn_api.h"
#include <vector>
#include <thread>
#include <mutex>
#include <queue>

class MultiCoreInference {
private:
    std::vector<rknn_context> contexts;
    std::vector<std::thread> threads;
    std::vector<std::queue<cv::Mat>> input_queues;
    std::vector<std::queue<std::vector<float>>> output_queues;
    std::vector<std::mutex> mutexes;
    bool running = true;

public:
    MultiCoreInference(const char* model_path, int num_cores = 3) {
        contexts.resize(num_cores);
        input_queues.resize(num_cores);
        output_queues.resize(num_cores);
        mutexes.resize(num_cores);
        threads.resize(num_cores);
        
        // Initialize each core
        for (int i = 0; i < num_cores; i++) {
            // Load model
            int ret = rknn_init(&contexts[i], (void*)model_path, 0, 0, NULL);
            if (ret != RKNN_SUCC) {
                printf("rknn_init failed for core %d\n", i);
                continue;
            }
            
            // Set core mask (pin to specific core)
            rknn_core_mask core_mask = (rknn_core_mask)(1 << i);
            rknn_set_core_mask(contexts[i], core_mask);
            
            // Start worker thread
            threads[i] = std::thread(&MultiCoreInference::worker, this, i);
        }
    }
    
    void worker(int core_id) {
        rknn_context ctx = contexts[core_id];
        
        while (running) {
            // Check if input available
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(mutexes[core_id]);
                if (input_queues[core_id].empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                frame = input_queues[core_id].front();
                input_queues[core_id].pop();
            }
            
            // Prepare input
            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].size = frame.cols * frame.rows * frame.channels();
            inputs[0].fmt = RKNN_TENSOR_NHWC;
            inputs[0].buf = frame.data;
            
            // Set inputs
            int ret = rknn_inputs_set(ctx, 1, inputs);
            
            // Run inference
            ret = rknn_run(ctx, NULL);
            
            // Get outputs
            rknn_output outputs[1];
            memset(outputs, 0, sizeof(outputs));
            outputs[0].want_float = 1;
            
            ret = rknn_outputs_get(ctx, 1, outputs, NULL);
            
            // Process output
            std::vector<float> result((float*)outputs[0].buf, 
                                     (float*)outputs[0].buf + outputs[0].size/sizeof(float));
            
            // Release output
            rknn_outputs_release(ctx, 1, outputs);
            
            // Store result
            {
                std::lock_guard<std::mutex> lock(mutexes[core_id]);
                output_queues[core_id].push(result);
            }
        }
    }
    
    void submit(int core_id, const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutexes[core_id]);
        input_queues[core_id].push(frame);
    }
    
    bool get_result(int core_id, std::vector<float>& result) {
        std::lock_guard<std::mutex> lock(mutexes[core_id]);
        if (output_queues[core_id].empty()) {
            return false;
        }
        result = output_queues[core_id].front();
        output_queues[core_id].pop();
        return true;
    }
    
    ~MultiCoreInference() {
        running = false;
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        for (auto& ctx : contexts) {
            rknn_destroy(ctx);
        }
    }
};

// Usage
int main() {
    MultiCoreInference npu("model.rknn", 3);
    
    // Submit frames to different cores
    cv::Mat frame1 = cv::imread("image1.jpg");
    cv::Mat frame2 = cv::imread("image2.jpg");
    cv::Mat frame3 = cv::imread("image3.jpg");
    
    npu.submit(0, frame1);
    npu.submit(1, frame2);
    npu.submit(2, frame3);
    
    // Get results
    std::vector<float> result1, result2, result3;
    npu.get_result(0, result1);
    npu.get_result(1, result2);
    npu.get_result(2, result3);
    
    return 0;
}
```

---

## 📊 Performance Optimization

### **1. Core Assignment Strategy**

| Workload | Strategy | Core Mask | Reason |
|----------|----------|-----------|--------|
| Single stream | All cores | `0x7` (111) | Max speed per inference |
| 2 streams | 1 core each | `0x1`, `0x2` | No contention |
| 3 streams | 1 core each | `0x1`, `0x2`, `0x4` | Perfect balance |
| 4+ streams | Round-robin | Rotate | Share cores |

### **2. Memory Management**

```python
# ✅ Good: Reuse buffers
input_buffer = np.zeros((640, 640, 3), dtype=np.uint8)

while True:
    # Reuse same buffer
    input_buffer[:] = preprocess(frame)
    result = rknn.inference(inputs=[input_buffer])

# ❌ Bad: Create new buffer every time
while True:
    input_buffer = np.zeros((640, 640, 3), dtype=np.uint8)  # Memory allocation!
    result = rknn.inference(inputs=[input_buffer])
```

### **3. Queue Sizing**

```python
# Small queue (1-2): Lower latency, may drop frames
input_queue = queue.Queue(maxsize=2)

# Large queue (10+): No drops, higher latency
input_queue = queue.Queue(maxsize=10)

# Recommended: 2-5 frames
input_queue = queue.Queue(maxsize=3)
```

---

## ✅ Best Practices

### **1. When to Use Multi-Core**

✅ **Use Multi-Core When:**
- Processing multiple streams (2+ cameras)
- Need high throughput
- Real-time requirements
- Models can fit in memory multiple times

❌ **Don't Use Multi-Core When:**
- Single stream, single model
- Memory constrained (large models)
- Low power requirement
- Models need > 2GB RAM each

### **2. Core Mask Selection**

```python
# Scenario 1: Single 4K stream, need max speed
core_mask = RKNNLite.NPU_CORE_0_1_2  # Use all 3 cores

# Scenario 2: 3 x 1080p streams
# Each stream gets dedicated core
stream1_mask = RKNNLite.NPU_CORE_0
stream2_mask = RKNNLite.NPU_CORE_1
stream3_mask = RKNNLite.NPU_CORE_2

# Scenario 3: Auto (let system decide)
core_mask = RKNNLite.NPU_CORE_AUTO  # Recommended for most cases
```

### **3. Thread Management**

```python
# ✅ Good: One thread per core
num_threads = num_cores  # 3 threads for 3 cores

# ❌ Bad: More threads than cores
num_threads = 10  # Causes context switching overhead

# ✅ Good: Use daemon threads for workers
thread = threading.Thread(target=worker, daemon=True)
```

---

## 🎯 Benchmarking

### **Run RKNN Benchmark Tool**

```bash
cd /path/to/rknn-toolkit2/rknpu2/examples/rknn_benchmark

# Build
./build-linux.sh

# Run with different core masks
cd install/rknn_benchmark_Linux

# Core 0 only
./rknn_benchmark model.rknn input.jpg 100 1

# Core 1 only
./rknn_benchmark model.rknn input.jpg 100 2

# Core 2 only
./rknn_benchmark model.rknn input.jpg 100 4

# Core 0 + 1
./rknn_benchmark model.rknn input.jpg 100 3

# All cores
./rknn_benchmark model.rknn input.jpg 100 7
```

### **Expected Performance (YOLOv5s 640x640)**

| Configuration | FPS (Single Stream) | FPS (3 Streams) | Latency |
|---------------|--------------------:|----------------:|--------:|
| 1 Core | 25-30 FPS | 8-10 FPS each | 33ms |
| 2 Cores | 40-50 FPS | 20-25 FPS each | 25ms |
| 3 Cores | 50-60 FPS | 25-30 FPS each | 20ms |
| Auto | 45-55 FPS | 22-28 FPS each | 22ms |

### **Python Benchmark Script**

```python
import time
import numpy as np
from rknnlite.api import RKNNLite

def benchmark_core_mask(model_path, core_mask, iterations=100):
    """Benchmark specific core mask"""
    rknn = RKNNLite()
    rknn.load_rknn(model_path)
    rknn.init_runtime(core_mask=core_mask)
    
    # Dummy input
    input_data = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        rknn.inference(inputs=[input_data])
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        rknn.inference(inputs=[input_data])
    elapsed = time.time() - start
    
    fps = iterations / elapsed
    latency = elapsed / iterations * 1000  # ms
    
    rknn.release()
    
    return fps, latency

# Test all configurations
core_configs = {
    'Core 0': RKNNLite.NPU_CORE_0,
    'Core 1': RKNNLite.NPU_CORE_1,
    'Core 2': RKNNLite.NPU_CORE_2,
    'Core 0+1': RKNNLite.NPU_CORE_0_1,
    'All Cores': RKNNLite.NPU_CORE_0_1_2,
    'Auto': RKNNLite.NPU_CORE_AUTO
}

print(f"{'Configuration':<15} {'FPS':>10} {'Latency (ms)':>15}")
print("-" * 45)

for name, mask in core_configs.items():
    fps, latency = benchmark_core_mask('model.rknn', mask)
    print(f"{name:<15} {fps:>10.2f} {latency:>15.2f}")
```

---

## 📝 Summary

### **Quick Reference**

| Scenario | Core Mask | Python API | C++ API |
|----------|-----------|------------|---------|
| Single stream, max speed | `0x7` (all) | `core_mask=RKNNLite.NPU_CORE_0_1_2` | `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2)` |
| 3 streams parallel | `0x1`, `0x2`, `0x4` | `core_mask=1`, `2`, `4` | `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)` |
| Auto (recommended) | `0x0` or `0xffff` | `core_mask=RKNNLite.NPU_CORE_AUTO` | `rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO)` |

### **Performance Tips**

1. ✅ **Use separate contexts** for multiple streams (best throughput)
2. ✅ **Pin each context to specific core** (avoid contention)
3. ✅ **Use all cores** for single large inference (lowest latency)
4. ✅ **Reuse buffers** to avoid memory allocation
5. ✅ **Keep queues small** (2-5 frames) for low latency

### **Common Mistakes**

1. ❌ Sharing single context between threads (thread-unsafe)
2. ❌ Creating/destroying contexts frequently (slow)
3. ❌ Not pinning cores (random core assignment, unpredictable performance)
4. ❌ Large queues (high latency)
5. ❌ Not checking return values (silent failures)

---

## 📚 References

### **Official Documentation:**
- RKNN SDK: `/RKNN-tools/rknn-toolkit2-v2.3.0-2024-11-08/doc/`
- API Reference: `02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.0_EN.pdf`
- Runtime API: `04_Rockchip_RKNPU_API_Reference_RKNNRT_V2.3.0_EN.pdf`

### **Example Code:**
- Benchmark: `/RKNN-tools/rknn-toolkit2/rknpu2/examples/rknn_benchmark/`
- YOLOv5 Demo: `/RKNN-tools/rknn-toolkit2/rknpu2/examples/rknn_yolov5_demo/`
- Model Zoo: `/RKNN-tools/rknn_model_zoo-v2.3.0-2024-11-08/`

### **Useful Commands:**

```bash
# Check NPU status
cat /sys/kernel/debug/rknpu/version

# Monitor NPU usage
watch -n 1 'cat /sys/kernel/debug/rknpu/load'

# Check available cores
cat /sys/kernel/debug/rknpu/info
```

---

**Last Updated:** November 4, 2025  
**Platform:** Firefly RK3588, RKNN SDK v2.3.0  
**Supported Models:** RKNN format (INT8/INT16)

