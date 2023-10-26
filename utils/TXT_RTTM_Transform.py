def txt_to_rttm(input_file,output_file,file_name):
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
                name = file_name

                #将标签信息写入 RTTM 文件
                rttm_line = f"SPEAKER {name} 1 {start_time:.2f} {end_time - start_time:.2f} <NA> <NA> {label} <NA> <NA>\n"
                outfile.write(rttm_line)

    outfile.close()

def rttm_to_txt(input_file, output_file):
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    with open(output_file, "w") as outfile:
        for line in lines:
            parts = line.strip().split()
            
            if len(parts) == 10 and parts[0] == "SPEAKER":
                start_time = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                end_time = start_time + duration

                txt_line = f"{start_time:.2f}\t{end_time:.2f}\t{label}\n"
                outfile.write(txt_line)