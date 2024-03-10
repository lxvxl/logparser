def extract_lines(input_file, output_file, num_lines):
    with open(input_file, 'r', encoding='utf-8') as in_file:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for i in range(num_lines):
                line = in_file.readline()
                if not line:
                    break
                out_file.write(line)

def process_file(input_file_path, output_file_path):
    # 打开原始文件
    with open(input_file_path, 'r') as f:
        lines = f.readlines()

    # 提取每行中第一个冒号之后的内容
    new_lines = [line.split(':', 1)[1].strip() for line in lines]

    # 将提取出的内容写入新文件
    with open(output_file_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# 调用函数并传入文件路径

input_file_path = r'BigData\HDFS\HDFS_full.log'  # 替换为实际的输入文件路径
output_file_path = r'BigData\HDFS\HDFS_simple.log'  # 替换为实际的输出文件路径

process_file(input_file_path, output_file_path)


#extract_lines(input_file_path, output_file_path, num_lines_to_extract)