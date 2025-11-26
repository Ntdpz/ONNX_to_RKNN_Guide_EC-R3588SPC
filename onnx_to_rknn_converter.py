#!/usr/bin/env python3
"""
ONNX to RKNN Converter for NPU
แปลง ONNX model เป็น RKNN model สำหรับรันบน RK3588 NPU
ตาม documentation: https://wiki.t-firefly.com/en/EC-R3588SPC/usage_npu.html
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
from rknn.api import RKNN

class ONNXToRKNNConverter:
    def __init__(self):
        self.rknn = None
        
    def create_calibration_dataset(self, images_dir, dataset_file, num_images=50):
        """สร้าง calibration dataset สำหรับ quantization"""
        
        print(f"🔧 Creating calibration dataset...")
        print(f"   📂 Images directory: {images_dir}")
        print(f"   📄 Dataset file: {dataset_file}")
        print(f"   🔢 Number of images: {num_images}")
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            import glob
            pattern = os.path.join(images_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(images_dir, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            print(f"❌ No images found in: {images_dir}")
            return False
        
        # Limit to requested number
        if len(image_files) > num_images:
            image_files = image_files[:num_images]
        
        print(f"   📊 Found {len(image_files)} images")
        
        # Create dataset file
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            for img_path in image_files:
                # Convert to absolute path
                abs_path = os.path.abspath(img_path)
                f.write(abs_path + '\\n')
        
        print(f"✅ Dataset created: {dataset_file}")
        return True
    
    def convert_onnx_to_rknn(self, onnx_path, rknn_path, 
                            target_platform='rk3588',
                            do_quantization=True,
                            dataset_file=None,
                            mean_values=[0, 0, 0],
                            std_values=[255, 255, 255],
                            optimization_level=1):
        """แปลง ONNX เป็น RKNN model"""
        
        print("🚀 ONNX to RKNN Converter")
        print("=" * 50)
        print(f"📁 ONNX Model: {os.path.basename(onnx_path)}")
        print(f"💾 RKNN Output: {os.path.basename(rknn_path)}")
        print(f"🎯 Target Platform: {target_platform}")
        print(f"📊 Quantization: {'✅ Enabled' if do_quantization else '❌ Disabled'}")
        if dataset_file:
            print(f"📄 Dataset: {os.path.basename(dataset_file)}")
        
        # Validate inputs
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX model not found: {onnx_path}")
            return False
        
        if do_quantization and dataset_file and not os.path.exists(dataset_file):
            print(f"❌ Dataset file not found: {dataset_file}")
            return False
        
        # Create output directory
        os.makedirs(os.path.dirname(rknn_path), exist_ok=True)
        
        try:
            # Initialize RKNN
            print("\\n🔧 Initializing RKNN...")
            self.rknn = RKNN(verbose=False)
            
            # Configure model
            print("⚙️  Configuring model...")
            ret = self.rknn.config(
                mean_values=[mean_values], 
                std_values=[std_values], 
                target_platform=target_platform,
                optimization_level=optimization_level
            )
            if ret != 0:
                print(f"❌ Model configuration failed: {ret}")
                return False
            print("✅ Model configured")
            
            # Load ONNX model
            print("📥 Loading ONNX model...")
            ret = self.rknn.load_onnx(model=onnx_path)
            if ret != 0:
                print(f"❌ Failed to load ONNX model: {ret}")
                return False
            print("✅ ONNX model loaded")
            
            # Build model
            print("🏗️  Building RKNN model...")
            build_start = time.time()
            
            if do_quantization and dataset_file:
                ret = self.rknn.build(do_quantization=True, dataset=dataset_file)
            else:
                ret = self.rknn.build(do_quantization=False)
                
            if ret != 0:
                print(f"❌ Model build failed: {ret}")
                return False
            
            build_time = time.time() - build_start
            print(f"✅ Model built successfully in {build_time:.1f}s")
            
            # Export RKNN model
            print("💾 Exporting RKNN model...")
            ret = self.rknn.export_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Failed to export RKNN model: {ret}")
                return False
            
            # Check output file
            if os.path.exists(rknn_path):
                file_size = os.path.getsize(rknn_path) / (1024*1024)  # MB
                print(f"✅ RKNN model exported successfully")
                print(f"   📁 Output: {rknn_path}")
                print(f"   📊 Size: {file_size:.1f} MB")
            else:
                print(f"❌ Output file not created: {rknn_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.rknn:
                self.rknn.release()
    
    def test_converted_model(self, rknn_path, test_image_path, target_platform='rk3588'):
        """ทดสอบ RKNN model ที่แปลงแล้ว"""
        
        print(f"\\n🧪 Testing converted RKNN model...")
        print(f"📁 Model: {os.path.basename(rknn_path)}")
        print(f"🖼️  Test image: {os.path.basename(test_image_path)}")
        
        if not os.path.exists(rknn_path):
            print(f"❌ RKNN model not found: {rknn_path}")
            return False
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return False
        
        try:
            # Initialize RKNN for testing
            rknn = RKNN(verbose=False)
            
            # Load model
            ret = rknn.load_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Failed to load RKNN model: {ret}")
                return False
            print("✅ RKNN model loaded for testing")
            
            # Initialize runtime
            ret = rknn.init_runtime(target=target_platform)
            if ret != 0:
                print(f"❌ Failed to initialize runtime: {ret}")
                return False
            print("✅ Runtime initialized")
            
            # Load and preprocess test image
            img = cv2.imread(test_image_path)
            if img is None:
                print(f"❌ Failed to load test image")
                return False
            
            # Resize to 640x640 (standard YOLO input)
            img_resized = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_rgb, axis=0)
            
            print(f"📐 Input shape: {img_array.shape}")
            
            # Run inference
            print("⚡ Running test inference...")
            start_time = time.time()
            outputs = rknn.inference(inputs=[img_array])
            inference_time = time.time() - start_time
            
            print(f"✅ Test inference completed")
            print(f"   ⏱️  Time: {inference_time*1000:.1f}ms")
            print(f"   📊 Output shapes: {[out.shape for out in outputs]}")
            print(f"   🎯 FPS: {1.0/inference_time:.1f}")
            
            # Basic output validation
            if outputs and len(outputs) > 0:
                output = outputs[0]
                print(f"   🔍 Output range: [{np.min(output):.3f}, {np.max(output):.3f}]")
                print(f"   📈 Output mean: {np.mean(output):.3f}")
                
                # Check for reasonable YOLO output format
                if len(output.shape) >= 2:
                    last_dim = output.shape[-1]
                    if last_dim >= 5:
                        print(f"   ✅ Output format looks like YOLO (last dim: {last_dim})")
                    else:
                        print(f"   ⚠️  Unusual output format (last dim: {last_dim})")
            
            rknn.release()
            return True
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(
        description='ONNX to RKNN Converter for NPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert without quantization (FP16)
  python3 onnx_to_rknn_converter.py \\
    --onnx model.onnx \\
    --rknn model.rknn
  
  # Convert with quantization (INT8) 
  python3 onnx_to_rknn_converter.py \\
    --onnx model.onnx \\
    --rknn model.rknn \\
    --quantize \\
    --images ./images/ \\
    --dataset ./calibration.txt
  
  # Test converted model
  python3 onnx_to_rknn_converter.py \\
    --rknn model.rknn \\
    --test \\
    --image test.jpg
        """
    )
    
    # Input/Output
    parser.add_argument('--onnx', help='Input ONNX model path')
    parser.add_argument('--rknn', required=True, help='Output RKNN model path')
    
    # Conversion options
    parser.add_argument('--target', default='rk3588', 
                        choices=['rk3566', 'rk3568', 'rk3576', 'rk3588'],
                        help='Target platform (default: rk3588)')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--images', help='Directory containing calibration images')
    parser.add_argument('--dataset', help='Calibration dataset file path')
    parser.add_argument('--num-images', type=int, default=50,
                        help='Number of calibration images (default: 50)')
    
    # Model configuration  
    parser.add_argument('--mean', nargs=3, type=float, default=[0, 0, 0],
                        help='Mean values for normalization (default: 0 0 0)')
    parser.add_argument('--std', nargs=3, type=float, default=[255, 255, 255],
                        help='Std values for normalization (default: 255 255 255)')
    parser.add_argument('--optimization', type=int, default=1, choices=[0, 1, 2, 3],
                        help='Optimization level (default: 1)')
    
    # Testing
    parser.add_argument('--test', action='store_true',
                        help='Test the converted/existing RKNN model')
    parser.add_argument('--image', help='Test image for inference')
    
    args = parser.parse_args()
    
    converter = ONNXToRKNNConverter()
    
    # Test mode
    if args.test:
        if not args.image:
            print("❌ --image required for testing")
            return
        
        success = converter.test_converted_model(args.rknn, args.image, args.target)
        if success:
            print("\\n🎉 Model test completed successfully!")
        else:
            print("\\n💥 Model test failed!")
        return
    
    # Conversion mode
    if not args.onnx:
        print("❌ --onnx required for conversion")
        parser.print_help()
        return
    
    # Handle dataset creation
    dataset_file = args.dataset
    if args.quantize:
        if not args.images and not args.dataset:
            print("❌ --images or --dataset required for quantization")
            return
        
        if args.images and not args.dataset:
            # Auto-create dataset file
            dataset_file = os.path.splitext(args.rknn)[0] + '_calibration.txt'
            success = converter.create_calibration_dataset(
                args.images, dataset_file, args.num_images
            )
            if not success:
                return
    
    # Convert model
    success = converter.convert_onnx_to_rknn(
        onnx_path=args.onnx,
        rknn_path=args.rknn,
        target_platform=args.target,
        do_quantization=args.quantize,
        dataset_file=dataset_file,
        mean_values=args.mean,
        std_values=args.std,
        optimization_level=args.optimization
    )
    
    if success:
        print("\\n🎉 ONNX to RKNN conversion completed successfully!")
        
        # Auto-test if image provided
        if args.image and os.path.exists(args.image):
            print("\\n🧪 Running automatic test...")
            test_success = converter.test_converted_model(args.rknn, args.image, args.target)
            if test_success:
                print("✅ Automatic test passed!")
            else:
                print("⚠️  Automatic test failed (but conversion succeeded)")
    else:
        print("\\n💥 ONNX to RKNN conversion failed!")

if __name__ == "__main__":
    main()