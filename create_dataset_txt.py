import os

# --- 1. ตั้งค่า ---
# ระบุพาธไปยังโฟลเดอร์ที่เก็บรูปภาพ train ของคุณ
# คุณสามารถเปลี่ยนพาธนี้เป็นโฟลเดอร์อื่นได้ตามต้องการ
# ตัวอย่างที่ 1: CodeProvince
image_directory = r'/home/firefly/Documents/YOLO/Data-set/CodeProvince.v11i.yolov5pytorch/train/images'

# ตัวอย่างที่ 2: LicensePlate (หากต้องการใช้ ให้ลบ # ข้างหน้าบรรทัดถัดไป และใส่ # หน้าบรรทัด CodeProvince)
# image_directory = r'D:\train_tolov5\Data-set\LicensePlate.v6-final_2025-07-18-6-44pm.yolov5pytorch\train\images'

# ชื่อไฟล์ output ที่จะสร้าง
output_filename = 'dataset.txt'

# จำนวนไฟล์สูงสุดที่ต้องการใส่ใน list (ใส่ None ถ้าต้องการทั้งหมด)
max_files = 1000 
# -----------------


def create_image_list_file(img_dir, output_file, limit=None):
    """
    สแกนโฟลเดอร์, สร้างรายการพาธเต็มของไฟล์รูปภาพ, และบันทึกลงในไฟล์ .txt
    """
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่
    if not os.path.isdir(img_dir):
        print(f"Error: ไม่พบโฟลเดอร์ '{img_dir}'")
        return

    print(f"กำลังสแกนรูปภาพจาก: {img_dir}")

    image_files = []
    # วนลูปไฟล์ทั้งหมดในโฟลเดอร์
    for filename in os.listdir(img_dir):
        # ตรวจสอบว่าเป็นไฟล์รูปภาพหรือไม่ (เช็คจากนามสกุลไฟล์)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # สร้าง Absolute Path (พาธเต็ม) ของไฟล์
            absolute_path = os.path.join(img_dir, filename)
            image_files.append(absolute_path)

    if not image_files:
        print("ไม่พบไฟล์รูปภาพในโฟลเดอร์ที่ระบุ")
        return

    # จำกัดจำนวนไฟล์ถ้ามีการระบุ limit
    if limit and len(image_files) > limit:
        image_files_to_write = image_files[:limit]
    else:
        image_files_to_write = image_files

    # เขียนรายการไฟล์ลงใน output_filename
    try:
        with open(output_file, 'w') as f:
            for path in image_files_to_write:
                f.write(path + '\n')
        
        print("-" * 30)
        print(f"สร้างไฟล์ '{output_file}' สำเร็จแล้ว!")
        print(f"มีรายการรูปภาพทั้งหมด {len(image_files_to_write)} ไฟล์")
        print(f"ตำแหน่งไฟล์: {os.path.abspath(output_file)}")
        print("-" * 30)

    except Exception as e:
        print(f"เกิดข้อผิดพลาดขณะเขียนไฟล์: {e}")


if __name__ == '__main__':
    create_image_list_file(image_directory, output_filename, max_files)
