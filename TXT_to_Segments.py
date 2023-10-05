def txt_to_segments(txt_file_path, segments_file_path):
    # 读取.txt文件的内容
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    # 写入到.segments文件
    with open(segments_file_path, 'w') as seg_file:
        for line in lines:
            seg_file.write(line) 
            

