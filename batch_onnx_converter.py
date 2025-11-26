#!/usr/bin/env python3
"""
Batch ONNX to RKNN Converter
แปลง ONNX models หลายตัวเป็น RKNN พร้อมกัน
"""

import os
import subprocess
import argparse
import time
from datetime import datetime

def convert_model(onnx_path, output_dir, quantize=False, images_dir=None, test_image=None):
    """Convert single ONNX model to RKNN"""
    
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    # Output paths
    if quantize:
        rknn_path = os.path.join(output_dir, f"{model_name}_int8.rknn")
    else:
        rknn_path = os.path.join(output_dir, f"{model_name}_fp16.rknn")
    
    print(f"\\n{'='*60}")
    print(f"🔄 Converting: {model_name}")
    print(f"📁 ONNX: {onnx_path}")
    print(f"💾 RKNN: {rknn_path}")
    print(f"⚙️  Mode: {'INT8 Quantized' if quantize else 'FP16 Non-quantized'}")
    
    # Build command
    cmd = [
        'python3', 'onnx_to_rknn_converter.py',
        '--onnx', onnx_path,
        '--rknn', rknn_path,
        '--target', 'rk3588'
    ]
    
    if quantize and images_dir:
        cmd.extend(['--quantize', '--images', images_dir])
    
    if test_image:
        cmd.extend(['--image', test_image])
    
    start_time = time.time()
    
    try:
        # Run conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        conversion_time = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            # Check if file was created
            if os.path.exists(rknn_path):
                file_size = os.path.getsize(rknn_path) / (1024*1024)
                print(f"✅ Success! Created {rknn_path}")
                print(f"   ⏱️  Time: {conversion_time:.1f}s")
                print(f"   📊 Size: {file_size:.1f} MB")
            else:
                success = False
                print(f"❌ Conversion reported success but file not found")
        else:
            print(f"❌ Conversion failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
        
        return {
            'model': model_name,
            'onnx_path': onnx_path,
            'rknn_path': rknn_path,
            'success': success,
            'time': conversion_time,
            'quantized': quantize,
            'file_size_mb': os.path.getsize(rknn_path) / (1024*1024) if success and os.path.exists(rknn_path) else 0,
            'stderr': result.stderr if not success else ''
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timed out after 300 seconds")
        return {
            'model': model_name,
            'onnx_path': onnx_path,
            'rknn_path': rknn_path,
            'success': False,
            'time': 300,
            'quantized': quantize,
            'file_size_mb': 0,
            'stderr': 'Timeout after 300 seconds'
        }
    
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return {
            'model': model_name,
            'onnx_path': onnx_path,
            'rknn_path': rknn_path,
            'success': False,
            'time': time.time() - start_time,
            'quantized': quantize,
            'file_size_mb': 0,
            'stderr': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Batch ONNX to RKNN Converter')
    
    parser.add_argument('--onnx-dir', required=True,
                        help='Directory containing ONNX models')
    parser.add_argument('--output-dir', default='./converted_rknn',
                        help='Output directory for RKNN models')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--images-dir', 
                        help='Directory with calibration images for quantization')
    parser.add_argument('--test-image',
                        help='Test image for validation')
    parser.add_argument('--models', nargs='+',
                        help='Specific model names to convert (without .onnx)')
    
    args = parser.parse_args()
    
    print("🚀 Batch ONNX to RKNN Converter")
    print("=" * 50)
    print(f"📂 ONNX Directory: {args.onnx_dir}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"⚙️  Quantization: {'✅ Enabled' if args.quantize else '❌ Disabled'}")
    if args.images_dir:
        print(f"🖼️  Calibration Images: {args.images_dir}")
    if args.test_image:
        print(f"🧪 Test Image: {args.test_image}")
    
    # Validate directories
    if not os.path.exists(args.onnx_dir):
        print(f"❌ ONNX directory not found: {args.onnx_dir}")
        return
    
    if args.quantize and args.images_dir and not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find ONNX models
    onnx_files = []
    for file in os.listdir(args.onnx_dir):
        if file.endswith('.onnx'):
            model_name = os.path.splitext(file)[0]
            if args.models is None or model_name in args.models:
                onnx_files.append(os.path.join(args.onnx_dir, file))
    
    if not onnx_files:
        print(f"❌ No ONNX models found in: {args.onnx_dir}")
        if args.models:
            print(f"   Looking for: {', '.join(args.models)}")
        return
    
    print(f"\\n📊 Found {len(onnx_files)} ONNX models to convert")
    for onnx_file in onnx_files:
        print(f"   • {os.path.basename(onnx_file)}")
    
    # Convert models
    results = []
    total_start_time = time.time()
    
    for i, onnx_file in enumerate(onnx_files, 1):
        print(f"\\n[{i}/{len(onnx_files)}] Converting {os.path.basename(onnx_file)}...")
        
        result = convert_model(
            onnx_file, 
            args.output_dir, 
            args.quantize, 
            args.images_dir, 
            args.test_image
        )
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\\n{'='*60}")
    print(f"📊 Conversion Summary")
    print(f"{'='*60}")
    print(f"📁 Total Models: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📈 Success Rate: {successful/len(results)*100:.1f}%")
    
    if successful > 0:
        avg_time = sum(r['time'] for r in results if r['success']) / successful
        total_size = sum(r['file_size_mb'] for r in results if r['success'])
        print(f"⚡ Average Time: {avg_time:.1f}s per model")
        print(f"💾 Total Size: {total_size:.1f} MB")
    
    # Detailed results
    print(f"\\n📋 Detailed Results:")
    print("-" * 80)
    print(f"{'Model':<25} {'Status':<10} {'Time':<8} {'Size':<8} {'Type'}")
    print("-" * 80)
    
    for result in results:
        status = "✅ OK" if result['success'] else "❌ FAIL"
        time_str = f"{result['time']:.1f}s"
        size_str = f"{result['file_size_mb']:.1f}MB" if result['success'] else "N/A"
        type_str = "INT8" if result['quantized'] else "FP16"
        
        print(f"{result['model']:<25} {status:<10} {time_str:<8} {size_str:<8} {type_str}")
    
    # Failed models
    if failed > 0:
        print(f"\\n❌ Failed Conversions:")
        for result in results:
            if not result['success']:
                print(f"   • {result['model']}: {result['stderr']}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'conversion_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"ONNX to RKNN Conversion Summary\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write(f"Input Directory: {args.onnx_dir}\\n")
        f.write(f"Output Directory: {args.output_dir}\\n")
        f.write(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}\\n")
        f.write(f"Total Models: {len(results)}\\n")
        f.write(f"Successful: {successful}\\n")
        f.write(f"Failed: {failed}\\n")
        f.write(f"Total Time: {total_time:.1f}s\\n")
        f.write(f"Success Rate: {successful/len(results)*100:.1f}%\\n\\n")
        
        f.write("Detailed Results:\\n")
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            f.write(f"{result['model']:<25} {status:<10} {result['time']:.1f}s\\n")
            if not result['success']:
                f.write(f"    Error: {result['stderr']}\\n")
    
    print(f"\\n💾 Summary saved: {summary_file}")
    
    if successful == len(results):
        print("\\n🎉 All conversions completed successfully!")
    else:
        print(f"\\n⚠️  {failed} conversions failed. Check errors above.")

if __name__ == "__main__":
    main()