#!/usr/bin/env python3
"""
NPU Inference Tool using RKNN Toolkit2
สำหรับใช้งาน model RKNN บน EC-R3588SPC NPU
ไม่ต้องใช้ dataset สำหรับ quantization
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from rknn.api import RKNN

def check_npu_status():
    """ตรวจสอบสถานะ NPU และ rknn_server"""
    import subprocess
    
    try:
        # Check rknn_server
        result = subprocess.run(['pgrep', 'rknn_server'], 
                               capture_output=True, text=True, timeout=1)
        server_running = result.returncode == 0
        
        # Check NPU devices
        npu_devices = []
        for dev in ['/dev/rknpu', '/dev/rknpu0', '/dev/rknpu1']:
            if os.path.exists(dev):
                npu_devices.append(dev)
        
        return server_running, npu_devices
    except:
        return False, []

def preprocess_image(image_path, target_size=640):
    """Preprocess image สำหรับ YOLO model"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    original_shape = img.shape[:2]  # (height, width)
    print(f"📐 Original image size: {original_shape[1]}x{original_shape[0]}")
    
    # Letterbox resize (maintain aspect ratio)
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (gray padding)
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Center the image
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
    
    # Convert to RGB (RKNN expects RGB input)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # Add batch dimension
    img_array = np.expand_dims(img_rgb, axis=0)
    
    return img_array, original_shape, scale, (pad_w, pad_h)

def postprocess_yolo(outputs, original_shape, scale, padding, conf_threshold=0.5, iou_threshold=0.45):
    """Post-process YOLO outputs"""
    
    # Handle different output formats
    if len(outputs) == 1:
        # Single output (newer YOLO format)
        predictions = outputs[0]
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        print(f"🔍 Output shape: {predictions.shape}")
        
        if predictions.shape[-1] >= 5:
            # Standard format: [x, y, w, h, conf, class_probs...]
            boxes = predictions[:, :4]
            obj_conf = predictions[:, 4]
            
            if predictions.shape[-1] > 5:
                class_probs = predictions[:, 5:]
                num_classes = class_probs.shape[-1]
                class_conf = np.max(class_probs, axis=1)
                class_ids = np.argmax(class_probs, axis=1)
            else:
                # Single class or binary classification
                class_conf = np.ones_like(obj_conf)
                class_ids = np.zeros_like(obj_conf, dtype=int)
                num_classes = 1
            
            # Final confidence
            final_conf = obj_conf * class_conf
            
            # Filter by confidence
            valid_mask = final_conf > conf_threshold
            if not np.any(valid_mask):
                return [], [], [], num_classes
            
            boxes = boxes[valid_mask]
            scores = final_conf[valid_mask]
            class_ids = class_ids[valid_mask]
            
            # Convert from center format (xywh) to corner format (xyxy)
            boxes_xyxy = np.copy(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(), 
                scores.tolist(), 
                conf_threshold, 
                iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                boxes_xyxy = boxes_xyxy[indices]
                scores = scores[indices]
                class_ids = class_ids[indices]
                
                # Scale back to original image coordinates
                pad_w, pad_h = padding
                boxes_xyxy[:, [0, 2]] -= pad_w  # Remove horizontal padding
                boxes_xyxy[:, [1, 3]] -= pad_h  # Remove vertical padding
                boxes_xyxy /= scale                # Scale back to original size
                
                # Clip to image boundaries
                h, w = original_shape
                boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
                boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)
                
                return boxes_xyxy, scores, class_ids, num_classes
    
    return [], [], [], 0

def draw_results(image, boxes, scores, class_ids, class_names=None):
    """Draw detection results on image"""
    
    # Default colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    result_img = image.copy()
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Select color
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_names and class_id < len(class_names):
            label = f'{class_names[class_id]}: {score:.2f}'
        else:
            label = f'Class {class_id}: {score:.2f}'
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(result_img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_img

def run_npu_inference(model_path, image_path, output_dir="./npu_results", 
                     conf_threshold=0.5, iou_threshold=0.45, class_names=None):
    """Run inference using RKNN model on NPU"""
    
    print("🚀 RKNN NPU Inference Tool")
    print("=" * 50)
    
    # Check parameters
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    print(f"📁 Model: {os.path.basename(model_path)}")
    print(f"🖼️  Image: {os.path.basename(image_path)}")
    
    # Check NPU status
    server_running, npu_devices = check_npu_status()
    print(f"🔧 RKNN Server: {'✅ Running' if server_running else '❌ Not Running'}")
    print(f"💾 NPU Devices: {', '.join(npu_devices) if npu_devices else 'None found'}")
    
    if not server_running:
        print("⚠️  Warning: RKNN Server not running. Try: sudo systemctl start rknn_server")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize RKNN
    print("\n🔧 Initializing RKNN...")
    rknn = RKNN(verbose=False)
    
    try:
        # Load RKNN model
        print("📥 Loading RKNN model...")
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f"❌ Failed to load model, error code: {ret}")
            return False
        print("✅ Model loaded successfully")
        
        # Initialize runtime for RK3588
        print("🚀 Initializing NPU runtime...")
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            print(f"❌ Failed to initialize runtime, error code: {ret}")
            return False
        print("✅ NPU runtime initialized")
        
        # Preprocess image
        print("\n🖼️  Preprocessing image...")
        img_array, original_shape, scale, padding = preprocess_image(image_path)
        print(f"📐 Input tensor shape: {img_array.shape}")
        print(f"📊 Scale factor: {scale:.3f}")
        print(f"📦 Padding: {padding}")
        
        # Run inference
        print("\n⚡ Running inference on NPU...")
        start_time = time.time()
        outputs = rknn.inference(inputs=[img_array])
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time*1000:.2f} ms")
        print(f"🎯 Throughput: {1.0/inference_time:.1f} FPS")
        print(f"📊 Output shapes: {[out.shape for out in outputs]}")
        
        # Post-process results
        print("\n🔍 Post-processing results...")
        boxes, scores, class_ids, num_classes = postprocess_yolo(
            outputs, original_shape, scale, padding, conf_threshold, iou_threshold
        )
        
        print(f"🎯 Found {len(boxes)} detections")
        print(f"📋 Model has {num_classes} classes")
        
        if len(boxes) > 0:
            # Load original image for visualization
            original_img = cv2.imread(image_path)
            result_img = draw_results(original_img, boxes, scores, class_ids, class_names)
            
            # Save result
            timestamp = int(time.time())
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_path = os.path.join(output_dir, f'{model_name}_result_{timestamp}.jpg')
            cv2.imwrite(output_path, result_img)
            
            print(f"💾 Result saved: {output_path}")
            
            # Print detection details
            print("\n📊 Detection Details:")
            print("-" * 60)
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                
                if class_names and class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                print(f"  [{i+1:2d}] {class_name:15s} | Conf: {score:.3f} | "
                      f"Box: ({x1:3d},{y1:3d},{w:3d}x{h:3d})")
            
            print(f"\n🎉 Success! Found {len(boxes)} objects")
            print(f"💾 Output: {output_path}")
            return True
        else:
            print("⚠️  No objects detected")
            return False
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        rknn.release()

def main():
    parser = argparse.ArgumentParser(
        description='NPU Inference using RKNN Toolkit2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 npu_inference.py --model codeprovince_best_fp32.rknn --image test.jpg
  
  # With custom classes
  python3 npu_inference.py --model vehicle_model.rknn --image car.jpg --classes car truck bus
  
  # Adjust thresholds
  python3 npu_inference.py --model model.rknn --image image.jpg --conf 0.3 --iou 0.5
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                        help='Path to RKNN model file (.rknn)')
    parser.add_argument('--image', '-i', required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o', default='./npu_results',
                        help='Output directory (default: ./npu_results)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--classes', nargs='+',
                        help='Class names (e.g., --classes person car truck)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model.endswith('.rknn'):
        print("⚠️  Warning: Model file should have .rknn extension")
    
    if args.conf < 0 or args.conf > 1:
        print("❌ Confidence threshold must be between 0 and 1")
        return
    
    if args.iou < 0 or args.iou > 1:
        print("❌ IoU threshold must be between 0 and 1")
        return
    
    # Run inference
    success = run_npu_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        class_names=args.classes
    )
    
    if success:
        print("\n🎉 Inference completed successfully!")
    else:
        print("\n💥 Inference failed!")

if __name__ == "__main__":
    main()