#!/usr/bin/env python3
"""
Dataset List Creator - Universal Tool
สร้างไฟล์ dataset.txt สำหรับใช้กับ RKNN Quantization
รองรับหลาย dataset และใช้งานผ่าน command line arguments
"""

import os
import argparse
import sys


def create_image_list_file(img_dir, output_file, limit=None, recursive=True):
    """
    สแกนโฟลเดอร์, สร้างรายการพาธเต็มของไฟล์รูปภาพ, และบันทึกลงในไฟล์ .txt
    
    Args:
        img_dir (str): โฟลเดอร์ที่เก็บรูปภาพ
        output_file (str): ชื่อไฟล์ output .txt
        limit (int, optional): จำนวนไฟล์สูงสุด (None = ทั้งหมด)
        recursive (bool): สแกนโฟลเดอร์ย่อยด้วยหรือไม่ (default: True)
    
    Returns:
        bool: True ถ้าสำเร็จ, False ถ้าล้มเหลว
    """
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่
    if not os.path.isdir(img_dir):
        print(f"❌ Error: ไม่พบโฟลเดอร์ '{img_dir}'")
        return False

    print(f"🔍 กำลังสแกนรูปภาพจาก: {img_dir}")
    if recursive:
        print(f"   📂 โหมด: Recursive (รวมโฟลเดอร์ย่อย)")
    else:
        print(f"   📂 โหมด: Single directory เท่านั้น")

    image_files = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    # ถ้าต้องการสแกนแบบ recursive (ลงไปในโฟลเดอร์ย่อย)
    if recursive:
        for root, dirs, files in os.walk(img_dir):
            for filename in files:
                # ตรวจสอบว่าเป็นไฟล์รูปภาพหรือไม่
                if filename.lower().endswith(image_extensions):
                    absolute_path = os.path.abspath(os.path.join(root, filename))
                    image_files.append(absolute_path)
    else:
        # วนลูปไฟล์ทั้งหมดในโฟลเดอร์เดียว
        for filename in os.listdir(img_dir):
            file_path = os.path.join(img_dir, filename)
            # ตรวจสอบว่าเป็นไฟล์รูปภาพหรือไม่
            if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
                absolute_path = os.path.abspath(file_path)
                image_files.append(absolute_path)

    if not image_files:
        print("⚠️  ไม่พบไฟล์รูปภาพในโฟลเดอร์ที่ระบุ")
        return False

    # เรียงลำดับไฟล์
    image_files.sort()
    
    print(f"   📊 พบรูปภาพทั้งหมด: {len(image_files)} ไฟล์")

    # จำกัดจำนวนไฟล์ถ้ามีการระบุ limit
    if limit and len(image_files) > limit:
        image_files_to_write = image_files[:limit]
        print(f"   ✂️  จำกัดจำนวน: {limit} ไฟล์")
    else:
        image_files_to_write = image_files

    # เขียนรายการไฟล์ลงใน output_filename
    try:
        # สร้างโฟลเดอร์ output ถ้ายังไม่มี
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w') as f:
            for path in image_files_to_write:
                f.write(path + '\n')
        
        print("\n" + "=" * 60)
        print(f"✅ สร้างไฟล์ '{output_file}' สำเร็จแล้ว!")
        print(f"   📝 รายการรูปภาพ: {len(image_files_to_write)} ไฟล์")
        print(f"   📁 ตำแหน่งไฟล์: {os.path.abspath(output_file)}")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดขณะเขียนไฟล์: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Dataset List Creator - สร้างไฟล์ dataset.txt สำหรับ RKNN Quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน:
  # สร้าง dataset.txt จากโฟลเดอร์ (รวมโฟลเดอร์ย่อย)
  python3 create_dataset_txt.py \\
    --images ./dataset/train \\
    --output dataset.txt
  
  # จำกัดจำนวนรูปภาพ
  python3 create_dataset_txt.py \\
    --images ./dataset/train \\
    --output dataset.txt \\
    --max-files 500
  
  # สแกนเฉพาะโฟลเดอร์เดียว (ไม่รวมโฟลเดอร์ย่อย)
  python3 create_dataset_txt.py \\
    --images ./images \\
    --output dataset.txt \\
    --no-recursive
  
  # ใช้กับ dataset หลายประเภท
  python3 create_dataset_txt.py \\
    -i /path/to/yolov5/train/images \\
    -o yolov5_dataset.txt \\
    -n 1000
        """
    )
    
    parser.add_argument('-i', '--images', required=True,
                        help='โฟลเดอร์ที่เก็บรูปภาพ (รองรับ absolute หรือ relative path)')
    parser.add_argument('-o', '--output', default='dataset.txt',
                        help='ชื่อไฟล์ output (default: dataset.txt)')
    parser.add_argument('-n', '--max-files', type=int, default=None,
                        help='จำนวนไฟล์สูงสุด (default: ไม่จำกัด)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='ไม่สแกนโฟลเดอร์ย่อย (default: สแกนทุกโฟลเดอร์)')
    
    args = parser.parse_args()
    
    # แปลง path เป็น absolute path
    img_dir = os.path.abspath(args.images)
    output_file = args.output
    
    print("\n🚀 Dataset List Creator")
    print("=" * 60)
    print(f"📂 Input directory: {img_dir}")
    print(f"💾 Output file: {output_file}")
    if args.max_files:
        print(f"🔢 Max files: {args.max_files}")
    print("=" * 60 + "\n")
    
    # สร้างไฟล์ dataset
    success = create_image_list_file(
        img_dir=img_dir,
        output_file=output_file,
        limit=args.max_files,
        recursive=not args.no_recursive
    )
    
    if success:
        print("\n🎉 ทำงานเสร็จสมบูรณ์!")
        sys.exit(0)
    else:
        print("\n💥 เกิดข้อผิดพลาด!")
        sys.exit(1)


if __name__ == '__main__':
    main()
