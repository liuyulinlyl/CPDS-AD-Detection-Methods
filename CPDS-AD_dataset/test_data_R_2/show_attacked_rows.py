import os
import pandas as pd

def main():
    # 获取当前文件夹路径
    current_dir = os.getcwd()
    
    # 获取所有 Excel 文件
    excel_files = [f for f in os.listdir(current_dir) 
                   if f.endswith('.xlsx') or f.endswith('.xls')]
    
    if not excel_files:
        print("当前目录下没有找到 Excel 文件")
        return
    
    print(f"共找到 {len(excel_files)} 个 Excel 文件\n")
    
    for file in excel_files:
        file_path = os.path.join(current_dir, file)
        
        try:
            # 读取 Excel 文件
            df = pd.read_excel(file_path)
            
            # 检查是否包含 labels 列
            if 'labels' not in df.columns:
                print(f"文件 {file} 中不存在 'labels' 列\n")
                continue
            
            # 筛选 labels == 1 的行
            abnormal_rows = df[df['labels'] == 1]
            
            if abnormal_rows.empty:
                print(f"文件 {file} 中没有 labels == 1 的行\n")
            else:
                print("=" * 80)
                print(f"文件名: {file}")
                print(f"labels == 1 的行数: {len(abnormal_rows)}")
                print(abnormal_rows)
                print("\n")
                
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}\n")

if __name__ == "__main__":
    main()