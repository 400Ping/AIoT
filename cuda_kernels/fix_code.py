import os
import glob

def clean_file(filepath):
    print(f"Processing {filepath}...")
    try:
        # 嘗試用 UTF-8 讀取，如果不行為則用預設編碼
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='mbcs') as f: # Windows 預設編碼
                content = f.read()

        # 移除所有非 ASCII 字元 (包含中文註解)
        # 並確保檔頭有 NOMINMAX 防止巨集衝突
        new_lines = []
        if "#define NOMINMAX" not in content:
            new_lines.append("#define NOMINMAX\n")
            
        for line in content.splitlines():
            # 簡單過濾：只保留 ASCII 範圍內的字元 (0-127)
            clean_line = "".join(c for c in line if ord(c) < 128)
            new_lines.append(clean_line)

        # 寫回檔案 (強制 UTF-8)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(new_lines))
        print(f" -> Fixed: {filepath}")

    except Exception as e:
        print(f" -> Error processing {filepath}: {e}")

# 針對 src 資料夾下的 .cu 檔案進行清洗
target_files = glob.glob(os.path.join("src", "*.cu"))
for f in target_files:
    clean_file(f)

print("\n[完成] 所有 .cu 檔案已清洗為純英文且無亂碼。")