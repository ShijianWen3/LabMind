import matplotlib.pyplot as plt
import csv

def read_csv_points(file_path):
    x = []
    y = []
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue  # 跳过异常行
            try:
                xi = float(row[0])
                yi = float(row[1])
                x.append(xi)
                y.append(yi)
            except ValueError:
                continue  # 跳过无法转换的行
    
    return x, y


def plot_line(x, y):
    n = len(x)

    # 🎯 根据点数动态调整画布大小
    # 点越多，横向越宽
    width = max(6, min(20, n * 0.1))   # 限制范围避免过大或过小
    height = 4 + min(6, n * 0.02)

    # plt.figure(figsize=(width, height))
    plt.figure()

    # 绘制折线图
    plt.plot(x, y, marker='o', linestyle='-')

    # 优化显示
    plt.title("Line Plot from CSV")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # # 🎯 点太多时，减少刻度密度
    # if n > 20:
    #     step = n // 10
    #     plt.xticks(x[::step])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "./dataset/2026-3-18/合格/1-1.CSV"  # 改成你的文件路径
    x, y = read_csv_points(file_path)
    
    if x and y:
        plot_line(x, y)
    else:
        print("没有读取到有效数据")
