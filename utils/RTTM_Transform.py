def RTTM_transform(input_file,output_file):
    #打开输入文件以读取标签数据
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    #打开输出文件以写入 RTTM 数据
    with open(output_file, "w") as outfile:
        for line in lines:
            #按照标签文件的格式解析每一行
            parts = line.strip().split("\t")
            if len(parts) == 3:
                start_time, end_time, label = parts
                start_time = float(start_time)
                end_time = float(end_time)

                #将标签信息写入 RTTM 文件
                rttm_line = f"SPEAKER 1 {start_time:.2f} {end_time - start_time:.2f} {label}\n"
                outfile.write(rttm_line)

    outfile.close()
