import chardet

file_path = '/Users/chuanhanyuan/Desktop/code/testdata/train/1.csv'

with open(file_path, 'rb') as f:
    raw_data = f.read()
non_ascii_bytes = [byte for byte in raw_data if byte >= 128]
if non_ascii_bytes:
    print(f"Non-ASCII bytes found: {non_ascii_bytes}")
else:
    print("All bytes are ASCII.")

# 读取文件内容
try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()
    print("File read successfully as UTF-8.")
except UnicodeDecodeError as e:
    print(f"Error: {e}")
    print("Retrying with errors ignored...")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.readlines()

# 输出内容
print(content[:10])