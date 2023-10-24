import tkinter as tk
import keyboard  # 需要安装这个库

def draw_line():
    # 创建主窗口
    root = tk.Tk()
    root.attributes("-topmost", True)  # 确保窗口置于最前端
    root.overrideredirect(1)  # 移除窗口装饰
    root.attributes('-transparentcolor', root['bg'])  # 设置背景透明

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 设置直线的位置
    first_line_height = 210  # 第一条直线距离屏幕顶部200像素
    second_line_height = 220  # 第二条直线距离屏幕顶部400像素

    # 创建画布
    canvas = tk.Canvas(root, height=screen_height, width=screen_width, highlightthickness=0, bg='systemTransparent')
    canvas.pack()

    # 绘制第一条直线
    canvas.create_line(0, first_line_height, screen_width, first_line_height, fill="red")  # 直线的坐标和颜色

    # 绘制第二条直线
    canvas.create_line(0, second_line_height, screen_width, second_line_height, fill="blue")  # 直线的坐标和颜色

    # 检测'ESC'按键
    def check_esc():
        if keyboard.is_pressed('esc'):  # 如果按下'ESC'键
            root.quit()  # 退出tkinter主循环
            root.destroy()  # 销毁窗口

    # 每100ms检查一次是否按下'ESC'键
    def check_key():
        check_esc()
        root.after(100, check_key)

    root.after(100, check_key)
    root.mainloop()

if __name__ == "__main__":
    draw_line()
