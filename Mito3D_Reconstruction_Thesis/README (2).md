import zipfile
import os

# 定义需要打包的文件
files_to_pack = [
    'generate.py', 
    'train.py', 
    'README.md', 
    'final_mitochondria.obj', 
    'Thesis_Final_Result.png',
    'checkpoints/model_final.pth'
]

output_zip = 'Mito3D_Graduation_Project.zip'

with zipfile.ZipFile(output_zip, 'w') as zipf:
    for file in files_to_pack:
        if os.path.exists(file):
            zipf.write(file)
            print(f"已加入包: {file}")
        else:
            print(f"⚠️ 缺失文件: {file}")

print(f"\n✅ 打包完成！请在文件夹中查看: {output_zip}")