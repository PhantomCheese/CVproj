import os
import cv2
from deepface import DeepFace
import pandas as pd

# 定义数据集路径
data_dir = 'VGG-Face2/data/train'  # 请修改为你的VGGFace2数据集路径

# 定义人脸数据库路径
database_dir = 'my_face_database/'

# 创建数据库目录
if not os.path.exists(database_dir):
    os.makedirs(database_dir)

subject_folders = sorted(os.listdir(data_dir))[:5]

# 遍历数据集目录，每个文件夹对应一个人
for subject in subject_folders:
    subject_path = os.path.join(data_dir, subject)
    if os.path.isdir(subject_path):
        # 提取前十张图片
        for i, image_name in enumerate(os.listdir(subject_path)[:2]):
            image_path = os.path.join(subject_path, image_name)
            dest_path = os.path.join(database_dir, f"{subject}_{i}.jpg")
            img = cv2.imread(image_path)
            cv2.imwrite(dest_path, img)

# 定义要识别的图像路径
query_image_path = '0016_01.jpg'  # 请修改为你要识别的图片路径

# 在数据库中查找最匹配的脸
results = DeepFace.find(img_path=query_image_path, db_path=database_dir, model_name="VGG-Face")

# 检查结果类型
print(type(results))
print(results)

# 提取 DataFrame
if isinstance(results, list) and len(results) > 0:
    results_df = results[0]  # 提取列表中的第一个 DataFrame
    
    # 输出匹配结果
    for index, result in results_df.iterrows():
        matched_image_path = result['identity']
        matched_person = os.path.basename(matched_image_path).split('_')[0]
        print(f"Matched image: {matched_image_path}, Person: {matched_person}")

        # 显示匹配到的图片
        img = cv2.imread(matched_image_path)
        if img is not None:
            cv2.imshow(f"Matched image - {matched_person}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not read image at {matched_image_path}")
else:
    print("No results found or unexpected results format.")