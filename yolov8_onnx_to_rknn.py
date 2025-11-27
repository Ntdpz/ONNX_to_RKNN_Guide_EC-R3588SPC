#!/usr/bin/env python3
"""
YOLOv8 ONNX to RKNN Converter
แก้ปัญหาโมเดลเพี้ยนหลังจากแปลง ONNX -> RKNN
รองรับ YOLOv8 Object Detection โดยเฉพาะ
"""

import os
import sys
import argparse
import time
import numpy as np
from rknn.api import RKNN

class YOLOv8ToRKNN:
    def __init__(self):
        self.rknn = None
        
    def convert(self, 
                onnx_path, 
                rknn_path,
                target_platform='rk3588',
                quantize=False,
                dataset_path=None,
                input_size=640):
        """
        แปลง YOLOv8 ONNX เป็น RKNN
        
        Args:
            onnx_path: path ไฟล์ ONNX
            rknn_path: path ไฟล์ RKNN output
            target_platform: platform (rk3588, rk3576, etc.)
            quantize: ใช้ INT8 quantization หรือไม่
            dataset_path: path ไฟล์ dataset.txt สำหรับ quantization
            input_size: ขนาด input (default: 640)
        """
        
        print("=" * 70)
        print("🚀 YOLOv8 ONNX to RKNN Converter")
        print("=" * 70)
        print(f"📁 ONNX Model: {os.path.basename(onnx_path)}")
        print(f"💾 RKNN Output: {os.path.basename(rknn_path)}")
        print(f"🎯 Target: {target_platform}")
        print(f"📊 Input Size: {input_size}x{input_size}")
        print(f"🔧 Quantization: {'✅ INT8' if quantize else '❌ FP16'}")
        if quantize and dataset_path:
            print(f"📄 Dataset: {os.path.basename(dataset_path)}")
        print("=" * 70)
        
        # Validate files
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX file not found: {onnx_path}")
            return False
            
        if quantize and dataset_path and not os.path.exists(dataset_path):
            print(f"❌ Dataset file not found: {dataset_path}")
            return False
        
        try:
            # Initialize RKNN
            print("\n🔧 Step 1: Initializing RKNN...")
            self.rknn = RKNN(verbose=False)
            print("✅ RKNN initialized")
            
            # Configure model - IMPORTANT FOR YOLOv8
            print("\n⚙️  Step 2: Configuring model...")
            print("   📌 Mean values: [0, 0, 0]")
            print("   📌 Std values: [255, 255, 255]")
            print("   📌 Optimization level: 3 (highest)")
            
            ret = self.rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=target_platform,
                optimization_level=3,  # Maximum optimization
                quantized_algorithm='normal',
                quantized_method='channel'
            )
            
            if ret != 0:
                print(f"❌ Configuration failed: {ret}")
                return False
            print("✅ Model configured")
            
            # Load ONNX
            print("\n📥 Step 3: Loading ONNX model...")
            ret = self.rknn.load_onnx(model=onnx_path)
            if ret != 0:
                print(f"❌ Failed to load ONNX: {ret}")
                return False
            print("✅ ONNX loaded successfully")
            
            # Build model
            print("\n🏗️  Step 4: Building RKNN model...")
            build_start = time.time()
            
            if quantize and dataset_path:
                print("   🔢 Building with INT8 quantization...")
                ret = self.rknn.build(
                    do_quantization=True,
                    dataset=dataset_path,
                    rknn_batch_size=1
                )
            else:
                print("   🔢 Building with FP16 (no quantization)...")
                ret = self.rknn.build(do_quantization=False)
            
            if ret != 0:
                print(f"❌ Build failed: {ret}")
                return False
            
            build_time = time.time() - build_start
            print(f"✅ Build completed in {build_time:.1f}s")
            
            # Export RKNN
            print("\n💾 Step 5: Exporting RKNN model...")
            os.makedirs(os.path.dirname(rknn_path) or '.', exist_ok=True)
            
            ret = self.rknn.export_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Export failed: {ret}")
                return False
            
            # Verify output
            if os.path.exists(rknn_path):
                file_size = os.path.getsize(rknn_path) / (1024*1024)
                print(f"✅ RKNN exported successfully")
                print(f"   📁 File: {rknn_path}")
                print(f"   📊 Size: {file_size:.2f} MB")
            else:
                print(f"❌ Output file not created")
                return False
            
            print("\n" + "=" * 70)
            print("🎉 Conversion completed successfully!")
            print("=" * 70)
            return True
            
        except Exception as e:
            print(f"\n❌ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.rknn:
                self.rknn.release()
    
    def verify_model(self, rknn_path, input_size=640):
        """
        ตรวจสอบโมเดล RKNN ว่าสามารถโหลดได้หรือไม่
        """
        print("\n" + "=" * 70)
        print("🔍 Verifying RKNN model...")
        print("=" * 70)
        
        try:
            rknn = RKNN(verbose=False)
            
            # Load model
            print("📥 Loading RKNN model...")
            ret = rknn.load_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Failed to load model: {ret}")
                rknn.release()
                return False
            print("✅ Model loaded successfully")
            
            # Get file size
            file_size = os.path.getsize(rknn_path) / (1024*1024)
            print(f"   📊 File size: {file_size:.2f} MB")
            
            # Note about runtime testing
            print("\n📋 Model Information:")
            print("   ℹ️  Model can be loaded successfully")
            print("   ℹ️  Runtime testing requires actual RK3588 device")
            print("   ℹ️  Verification on x86 is limited (expected)")
            
            rknn.release()
            
            print("\n✅ Model verification passed!")
            print("   ✅ File exists and can be loaded")
            print("   ✅ Ready to deploy on RK3588 device")
            print("=" * 70)
            return True
            
        except Exception as e:
            print(f"❌ Verification error: {str(e)[:100]}")
            print("⚠️  This may be normal on x86 platform")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 ONNX to RKNN Converter (แก้ปัญหาโมเดลเพี้ยน)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📖 ตัวอย่างการใช้งาน:

1. แปลงแบบ FP16 (ไม่ใช้ quantization):
   python3 yolov8_onnx_to_rknn.py \\
       --onnx model.onnx \\
       --rknn model_fp16.rknn

2. แปลงแบบ INT8 (ใช้ quantization):
   python3 yolov8_onnx_to_rknn.py \\
       --onnx model.onnx \\
       --rknn model_int8.rknn \\
       --quantize \\
       --dataset dataset.txt

3. ระบุขนาด input และ platform:
   python3 yolov8_onnx_to_rknn.py \\
       --onnx model.onnx \\
       --rknn model.rknn \\
       --input-size 640 \\
       --platform rk3588

4. แปลงและตรวจสอบโมเดล:
   python3 yolov8_onnx_to_rknn.py \\
       --onnx model.onnx \\
       --rknn model.rknn \\
       --verify

⚠️  สิ่งที่ต้องระวัง:
- ONNX ต้อง export ด้วย Opset 12
- ขนาด input ต้องตรงกับที่ใช้เทรน
- ถ้าใช้ quantization ต้องมีไฟล์ dataset.txt
        """
    )
    
    parser.add_argument('--onnx', required=True,
                        help='Path to YOLOv8 ONNX model')
    parser.add_argument('--rknn', required=True,
                        help='Output RKNN model path')
    parser.add_argument('--platform', default='rk3588',
                        choices=['rk3566', 'rk3568', 'rk3576', 'rk3588'],
                        help='Target platform (default: rk3588)')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--dataset',
                        help='Dataset file for quantization (dataset.txt)')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input size (default: 640)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify model after conversion')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.quantize and not args.dataset:
        print("❌ Error: --dataset required when using --quantize")
        sys.exit(1)
    
    if args.input_size not in [320, 416, 480, 640, 1280]:
        print(f"⚠️  Warning: Input size {args.input_size} is unusual for YOLOv8")
    
    # Convert model
    converter = YOLOv8ToRKNN()
    success = converter.convert(
        onnx_path=args.onnx,
        rknn_path=args.rknn,
        target_platform=args.platform,
        quantize=args.quantize,
        dataset_path=args.dataset,
        input_size=args.input_size
    )
    
    if not success:
        print("\n💥 Conversion failed!")
        sys.exit(1)
    
    # Verify if requested
    if args.verify:
        success = converter.verify_model(args.rknn, args.input_size)
        if not success:
            print("\n⚠️  Verification failed (but model may still work on device)")
    
    print("\n✅ All done! You can now use the RKNN model on your device.")
    print(f"📁 Output: {args.rknn}")


if __name__ == "__main__":
    main()
