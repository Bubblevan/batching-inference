import os
import glob

# 指定路径
path = '.'  # 当前目录，你可以根据需要修改这个路径

# 构造一个glob模式来匹配所有.png文件
png_files = glob.glob(os.path.join(path, '*.png'))

# 遍历匹配到的文件列表并删除每个文件
for png_file in png_files:
    try:
        os.remove(png_file)
        print(f"Deleted file: {png_file}")
    except OSError as e:
        print(f"Error: {e.strerror} - {png_file}")

print("Done. All PNG files have been deleted.")