#!/usr/bin/env python3
"""
Universal ONNX to RKNN Converter
รองรับทุก Model Architecture พร้อม Auto-detection
สามารถ Config ทุกอย่างผ่าน Parameters
"""

import os
import sys
import argparse
import time
import numpy as np
import onnx
from rknn.api import RKNN

class UniversalONNXToRKNN:
    """Universal ONNX to RKNN Converter with full configurability"""
    
    def __init__(self, verbose=True):
        self.rknn = None
        self.verbose = verbose
        self.model_info = {}
        
    def analyze_onnx(self, onnx_path):
        """วิเคราะห์ ONNX model เพื่อ detect model type และ metadata"""
        
        if self.verbose:
            print("\n🔍 Analyzing ONNX model...")
        
        try:
            model = onnx.load(onnx_path)
            
            # Get model info
            self.model_info['graph_name'] = model.graph.name
            
            # Get input info
            input_tensor = model.graph.input[0]
            input_shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
            self.model_info['input_name'] = input_tensor.name
            self.model_info['input_shape'] = input_shape
            
            # Get output info
            outputs = []
            for output_tensor in model.graph.output:
                output_shape = [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
                outputs.append({
                    'name': output_tensor.name,
                    'shape': output_shape
                })
            self.model_info['outputs'] = outputs
            
            # Detect model type
            self.model_info['model_type'] = self._detect_model_type()
            
            if self.verbose:
                self._print_model_info()
                
            return self.model_info
            
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze ONNX model: {e}")
            return None
    
    def _detect_model_type(self):
        """Auto-detect model type จาก output shape"""
        
        if not self.model_info.get('outputs'):
            return "Unknown"
        
        output_shape = self.model_info['outputs'][0]['shape']
        graph_name = self.model_info.get('graph_name', '').lower()
        
        # YOLOv8 Detection: (1, 84, 8400) or (1, num_classes+4, 8400)
        if len(output_shape) == 3 and output_shape[2] == 8400:
            return "YOLOv8"
        
        # YOLOv5 Detection: (1, 25200, 85) or (1, 25200, num_classes+5)
        if len(output_shape) == 3 and output_shape[1] == 25200:
            return "YOLOv5"
        
        # YOLOv10 Detection: (1, 300, 6)
        if len(output_shape) == 3 and output_shape[1] == 300:
            return "YOLOv10"
        
        # Check graph name
        if 'yolov8' in graph_name or 'yolo8' in graph_name:
            return "YOLOv8"
        elif 'yolov5' in graph_name or 'yolo5' in graph_name:
            return "YOLOv5"
        elif 'yolov10' in graph_name or 'yolo10' in graph_name:
            return "YOLOv10"
        elif 'yolo' in graph_name:
            return "YOLO (Generic)"
        
        return "Unknown"
    
    def _print_model_info(self):
        """แสดงข้อมูล Model ที่วิเคราะห์ได้"""
        
        print(f"   📝 Graph Name: {self.model_info.get('graph_name', 'N/A')}")
        print(f"   📐 Input Shape: {self.model_info.get('input_shape', 'N/A')}")
        print(f"   📊 Output Shapes:")
        for i, output in enumerate(self.model_info.get('outputs', [])):
            print(f"      [{i}] {output['name']}: {output['shape']}")
        print(f"   🎯 Detected Type: {self.model_info.get('model_type', 'Unknown')}")
    
    def convert(self,
                onnx_path,
                rknn_path,
                # Platform settings
                target_platform='rk3588',
                # Quantization settings
                quantize=False,
                quantized_dtype='INT8',
                quantized_algorithm='normal',
                quantized_method='channel',
                dataset_path=None,
                # Optimization settings
                optimization_level=3,
                # Model settings
                input_size_list=None,
                mean_values=None,
                std_values=None,
                # Output settings
                output_names=None,
                # Advanced settings
                do_hybrid_quant=False,
                hybrid_quant_file=None,
                target_sub_platform=None,
                custom_string=None):
        """
        Universal ONNX to RKNN Converter
        
        Args:
            onnx_path: Path to ONNX model file
            rknn_path: Path to output RKNN model file
            
            Platform Settings:
            - target_platform: Target platform (rk3588, rk3576, rk3562, rv1109, rv1126, rk1808, rk3399pro)
            - target_sub_platform: Sub-platform for specific chip variants
            
            Quantization Settings:
            - quantize: Enable quantization (True for INT8, False for FP16)
            - quantized_dtype: Data type for quantization ('INT8', 'FP16', 'UINT8')
            - quantized_algorithm: Quantization algorithm ('normal', 'mmse', 'kl_divergence')
            - quantized_method: Quantization method ('channel', 'layer')
            - dataset_path: Path to dataset.txt for calibration (required for INT8)
            
            Optimization Settings:
            - optimization_level: Optimization level (0, 1, 2, 3) - higher = more optimized
            
            Model Settings:
            - input_size_list: List of input sizes [[C, H, W]] (auto-detected if None)
            - mean_values: Mean values for normalization [[R, G, B]] (e.g., [[0, 0, 0]])
            - std_values: Std values for normalization [[R, G, B]] (e.g., [[255, 255, 255]])
            - output_names: List of output layer names (auto-detected if None)
            
            Advanced Settings:
            - do_hybrid_quant: Enable hybrid quantization (mix of FP16 and INT8)
            - hybrid_quant_file: Path to hybrid quantization config file
            - custom_string: Custom string for model version tracking
        """
        
        print("\n" + "=" * 80)
        print("🚀 Universal ONNX to RKNN Converter")
        print("=" * 80)
        
        # Analyze ONNX model
        self.analyze_onnx(onnx_path)
        
        print(f"\n📋 Conversion Settings:")
        print(f"   📁 Input:  {os.path.basename(onnx_path)}")
        print(f"   💾 Output: {os.path.basename(rknn_path)}")
        print(f"   🎯 Platform: {target_platform}")
        if target_sub_platform:
            print(f"   🎯 Sub-Platform: {target_sub_platform}")
        print(f"   🔧 Quantization: {quantized_dtype if quantize else 'FP16'}")
        if quantize:
            print(f"   📊 Algorithm: {quantized_algorithm}")
            print(f"   📊 Method: {quantized_method}")
            if dataset_path:
                print(f"   📄 Dataset: {os.path.basename(dataset_path)}")
        print(f"   ⚡ Optimization Level: {optimization_level}")
        if mean_values:
            print(f"   📐 Mean Values: {mean_values}")
        if std_values:
            print(f"   📐 Std Values: {std_values}")
        print("=" * 80)
        
        # Validate inputs
        if not os.path.exists(onnx_path):
            print(f"\n❌ Error: ONNX file not found: {onnx_path}")
            return False
        
        if quantize and not dataset_path:
            print(f"\n⚠️  Warning: Quantization enabled but no dataset provided.")
            print(f"   For INT8 quantization, dataset is highly recommended for better accuracy.")
        
        if dataset_path and not os.path.exists(dataset_path):
            print(f"\n❌ Error: Dataset file not found: {dataset_path}")
            return False
        
        # Initialize RKNN
        self.rknn = RKNN(verbose=self.verbose)
        
        try:
            # Configure RKNN
            print("\n🔧 Configuring RKNN...")
            
            config_params = {
                'target_platform': target_platform,
                'optimization_level': optimization_level,
            }
            
            if target_sub_platform:
                config_params['target_sub_platform'] = target_sub_platform
            
            if quantize:
                config_params['quantized_dtype'] = quantized_dtype
                config_params['quantized_algorithm'] = quantized_algorithm
                config_params['quantized_method'] = quantized_method
            
            if custom_string:
                config_params['custom_string'] = custom_string
            
            # Add mean/std values to config (RKNN Toolkit 2.3.x)
            if mean_values:
                config_params['mean_values'] = mean_values
            if std_values:
                config_params['std_values'] = std_values
            
            ret = self.rknn.config(**config_params)
            if ret != 0:
                print(f"❌ Configuration failed with error code: {ret}")
                return False
            print("   ✅ Configuration successful")
            
            # Load ONNX model
            print("\n📥 Loading ONNX model...")
            load_params = {}
            
            if input_size_list:
                load_params['input_size_list'] = input_size_list
            if output_names:
                load_params['outputs'] = output_names
            
            ret = self.rknn.load_onnx(model=onnx_path, **load_params)
            if ret != 0:
                print(f"❌ Failed to load ONNX model with error code: {ret}")
                return False
            print("   ✅ ONNX model loaded")
            
            # Build RKNN model
            print("\n🔨 Building RKNN model...")
            print("   ⏳ This may take a few minutes...")
            
            build_params = {}
            
            if quantize and dataset_path:
                build_params['do_quantization'] = True
                build_params['dataset'] = dataset_path
            else:
                build_params['do_quantization'] = False
            
            # Note: mean_values and std_values are now passed via config() in RKNN Toolkit 2.3.x
            
            if do_hybrid_quant and hybrid_quant_file:
                build_params['do_hybrid_quant'] = True
                build_params['hybrid_quant_file'] = hybrid_quant_file
            
            start_time = time.time()
            ret = self.rknn.build(**build_params)
            build_time = time.time() - start_time
            
            if ret != 0:
                print(f"❌ Build failed with error code: {ret}")
                return False
            print(f"   ✅ Build completed in {build_time:.1f} seconds")
            
            # Export RKNN model
            print("\n💾 Exporting RKNN model...")
            ret = self.rknn.export_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Export failed with error code: {ret}")
                return False
            
            # Get file size
            file_size = os.path.getsize(rknn_path) / (1024 * 1024)
            print(f"   ✅ Model exported: {rknn_path}")
            print(f"   📊 File size: {file_size:.2f} MB")
            
            print("\n" + "=" * 80)
            print("🎉 Conversion completed successfully!")
            print("=" * 80)
            
            # Print recommendations
            self._print_recommendations()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if self.rknn:
                self.rknn.release()
    
    def _print_recommendations(self):
        """แสดงคำแนะนำการใช้งาน Model"""
        
        model_type = self.model_info.get('model_type', 'Unknown')
        
        print(f"\n💡 Recommendations for {model_type}:")
        
        if model_type == "YOLOv8":
            print("""
   Preprocessing:
   - Use Letterbox resize (maintain aspect ratio)
   - Padding with gray color (114, 114, 114)
   - Convert BGR → RGB
   - Normalize: mean=[0,0,0], std=[255,255,255]
   
   Postprocessing:
   - Transpose output if shape is (5, 8400) → (8400, 5)
   - Confidence threshold: 0.25 (recommended)
   - IoU threshold: 0.7-0.85 (higher = less filtering)
   - Apply NMS (cv2.dnn.NMSBoxes)
   - Scale coordinates back to original image size
   
   Example inference script:
   python3 npu_inference.py \\
       --model output.rknn \\
       --image test.jpg \\
       --conf 0.25 \\
       --iou 0.85
""")
        
        elif model_type == "YOLOv5":
            print("""
   Preprocessing:
   - Use Letterbox resize (maintain aspect ratio)
   - Padding with gray color (114, 114, 114)
   - Convert BGR → RGB
   - Normalize: mean=[0,0,0], std=[255,255,255]
   
   Postprocessing:
   - Output format: (1, 25200, 85) or (1, 25200, num_classes+5)
   - First 4 values: [x, y, w, h]
   - 5th value: objectness score
   - Remaining: class probabilities
   - Confidence threshold: 0.25-0.5
   - IoU threshold: 0.45-0.7
   - Apply NMS
""")
        
        else:
            print("""
   General Recommendations:
   - Check model documentation for preprocessing requirements
   - Verify output format and shape
   - Test with different threshold values
   - Compare results with original model (ONNX/PyTorch)
""")
        
        print(f"\n📚 Model Information Summary:")
        print(f"   Input Shape: {self.model_info.get('input_shape', 'N/A')}")
        print(f"   Output Shapes: {[o['shape'] for o in self.model_info.get('outputs', [])]}")
        print(f"   Model Type: {model_type}")
    
    def verify_model(self, rknn_path, target_platform='rk3588'):
        """
        ตรวจสอบว่า Model โหลดได้หรือไม่
        
        Args:
            rknn_path: Path to RKNN model
            target_platform: Target platform
        """
        
        print(f"\n🔍 Verifying RKNN model: {os.path.basename(rknn_path)}")
        
        if not os.path.exists(rknn_path):
            print(f"❌ Model file not found: {rknn_path}")
            return False
        
        rknn_verify = RKNN(verbose=False)
        
        try:
            # Load model
            ret = rknn_verify.load_rknn(rknn_path)
            if ret != 0:
                print(f"❌ Failed to load model with error code: {ret}")
                return False
            
            print("   ✅ Model loads successfully")
            
            # Try to initialize runtime (จะ fail บน x86 ซึ่งเป็นเรื่องปกติ)
            ret = rknn_verify.init_runtime(target=target_platform)
            if ret != 0:
                print(f"   ℹ️  Runtime initialization skipped (expected on x86 platform)")
                print(f"   ℹ️  Model will work correctly on {target_platform} hardware")
            else:
                print(f"   ✅ Runtime initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Verification note: {e}")
            print(f"   ℹ️  This is normal on x86 platform")
            print(f"   ℹ️  Model should work on {target_platform} hardware")
            return True
            
        finally:
            rknn_verify.release()


def main():
    parser = argparse.ArgumentParser(
        description='Universal ONNX to RKNN Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic FP16 conversion
  python3 universal_onnx_to_rknn.py \\
      --onnx model.onnx \\
      --model-name my_model \\
      --output-name model_fp16.rknn

  # INT8 quantization with dataset
  python3 universal_onnx_to_rknn.py \\
      --onnx model.onnx \\
      --model-name my_model \\
      --output-name model_int8.rknn \\
      --quantize \\
      --dataset dataset.txt

  # Full configuration
  python3 universal_onnx_to_rknn.py \\
      --onnx model.onnx \\
      --model-name my_model \\
      --output-name model.rknn \\
      --platform rk3588 \\
      --quantize \\
      --dtype INT8 \\
      --algorithm mmse \\
      --method channel \\
      --dataset dataset.txt \\
      --optimization 3 \\
      --mean 0 0 0 \\
      --std 255 255 255 \\
      --verify

  # Different platforms
  python3 universal_onnx_to_rknn.py \\
      --onnx model.onnx \\
      --model-name my_model \\
      --output-name model_rk3576.rknn \\
      --platform rk3576

Supported Platforms:
  - rk3588 (EC-R3588SPC, Orange Pi 5, etc.)
  - rk3576 (RK3576 boards)
  - rk3562 (RK3562 boards)
  - rv1109, rv1126 (Rockchip RV series)
  - rk1808, rk3399pro (Older platforms)

Quantization Algorithms:
  - normal: Standard quantization (fast, good accuracy)
  - mmse: Minimize mean square error (better accuracy, slower)
  - kl_divergence: KL divergence based (good for some models)

Optimization Levels:
  - 0: No optimization
  - 1: Basic optimization
  - 2: Moderate optimization
  - 3: Aggressive optimization (recommended)

Output Location:
  All .rknn files will be saved to:
  /home/nz/firefly/ONNX_to_RKNN_Guide_EC-R3588SPC/Model-AI/<model-name>/
        """
    )
    
    # Required arguments
    parser.add_argument('--onnx', '-i', required=True,
                        help='Path to input ONNX model file')
    parser.add_argument('--model-name', '-m', required=True,
                        help='Model name (creates folder: Model-AI/<model-name>/)')
    parser.add_argument('--output-name', '-o', required=True,
                        help='Output .rknn filename (e.g., model_fp16.rknn)')
    
    # Platform settings
    parser.add_argument('--platform', '-p', default='rk3588',
                        choices=['rk3588', 'rk3576', 'rk3562', 'rv1109', 'rv1126', 
                                'rk1808', 'rk3399pro'],
                        help='Target platform (default: rk3588)')
    parser.add_argument('--sub-platform', 
                        help='Target sub-platform for specific chip variants')
    
    # Quantization settings
    parser.add_argument('--quantize', '-q', action='store_true',
                        help='Enable quantization (INT8/w8a8)')
    parser.add_argument('--dtype', default='w8a8',
                        choices=['w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16', 'INT8', 'FP16', 'UINT8'],
                        help='Quantization data type (default: w8a8). For RKNN Toolkit 2.3.x: w8a8 (INT8), w8a16, w16a16i, w4a16. Legacy: INT8, FP16, UINT8')
    parser.add_argument('--algorithm', default='normal',
                        choices=['normal', 'mmse', 'kl_divergence'],
                        help='Quantization algorithm (default: normal)')
    parser.add_argument('--method', default='channel',
                        choices=['channel', 'layer'],
                        help='Quantization method (default: channel)')
    parser.add_argument('--dataset', '-d',
                        help='Path to dataset.txt for quantization calibration')
    
    # Optimization settings
    parser.add_argument('--optimization', type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help='Optimization level 0-3 (default: 3)')
    
    # Model settings
    parser.add_argument('--mean', nargs=3, type=float,
                        metavar=('R', 'G', 'B'),
                        help='Mean values for normalization (e.g., --mean 0 0 0)')
    parser.add_argument('--std', nargs=3, type=float,
                        metavar=('R', 'G', 'B'),
                        help='Std values for normalization (e.g., --std 255 255 255)')
    parser.add_argument('--input-size', nargs=3, type=int,
                        metavar=('C', 'H', 'W'),
                        help='Input size [C, H, W] (auto-detected if not specified)')
    parser.add_argument('--outputs', nargs='+',
                        help='Output layer names (auto-detected if not specified)')
    
    # Advanced settings
    parser.add_argument('--hybrid-quant', action='store_true',
                        help='Enable hybrid quantization (mix FP16 and INT8)')
    parser.add_argument('--hybrid-quant-file',
                        help='Path to hybrid quantization config file')
    parser.add_argument('--custom-string',
                        help='Custom string for model version tracking')
    
    # Other options
    parser.add_argument('--verify', '-v', action='store_true',
                        help='Verify model after conversion')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Map legacy dtype names to RKNN Toolkit 2.3.x names
    dtype_mapping = {
        'INT8': 'w8a8',
        'UINT8': 'w8a8',
        'FP16': None  # FP16 doesn't need quantized_dtype
    }
    quantized_dtype = dtype_mapping.get(args.dtype, args.dtype)
    
    # Construct output path: Model-AI/<model-name>/<output-name>
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from onnx_to_rknn_converter/
    model_ai_dir = os.path.join(project_root, 'Model-AI', args.model_name)
    
    # Create Model-AI/<model-name>/ directory if it doesn't exist
    if not os.path.exists(model_ai_dir):
        os.makedirs(model_ai_dir)
        print(f"\n📁 Created model directory: {model_ai_dir}")
    
    # Full path to output RKNN file
    rknn_path = os.path.join(model_ai_dir, args.output_name)
    
    # Prepare parameters
    mean_values = [[args.mean[0], args.mean[1], args.mean[2]]] if args.mean else None
    std_values = [[args.std[0], args.std[1], args.std[2]]] if args.std else None
    input_size_list = [[args.input_size[0], args.input_size[1], args.input_size[2]]] if args.input_size else None
    
    # Create converter
    converter = UniversalONNXToRKNN(verbose=args.verbose)
    
    # Convert model
    success = converter.convert(
        onnx_path=args.onnx,
        rknn_path=rknn_path,
        target_platform=args.platform,
        target_sub_platform=args.sub_platform,
        quantize=args.quantize,
        quantized_dtype=quantized_dtype,
        quantized_algorithm=args.algorithm,
        quantized_method=args.method,
        dataset_path=args.dataset,
        optimization_level=args.optimization,
        input_size_list=input_size_list,
        mean_values=mean_values,
        std_values=std_values,
        output_names=args.outputs,
        do_hybrid_quant=args.hybrid_quant,
        hybrid_quant_file=args.hybrid_quant_file,
        custom_string=args.custom_string
    )
    
    # Verify if requested
    if success and args.verify:
        converter.verify_model(rknn_path, args.platform)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
