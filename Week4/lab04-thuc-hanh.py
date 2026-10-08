"""
BÀI THỰC HÀNH LAB 04: MÔ HÌNH TUYẾN TÍNH & HỒI QUY TUYẾN TÍNH
Môn học: Xác suất Thống kê (UET.MAT1052)
Môi trường: Python 3 (pandas, statsmodels, plotnine)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_smooth,
    geom_hline,
    geom_line,
    labs,
    theme_minimal,
    scale_color_manual,
    scale_shape_manual,
)

print("=================================================================")
print("  UET.MAT1052 - XÁC SUẤT THỐNG KÊ - BÀI THỰC HÀNH LAB 04")
print("  Chủ đề: Mô hình tuyến tính & Hồi quy tuyến tính (Linear Regression)")
print("=================================================================\n")

# Đường dẫn dữ liệu
base_dir = os.path.dirname(os.path.abspath(__file__))
poverty_path = os.path.join(base_dir, "du-lieu", "poverty_grad.csv")
books_path = os.path.join(base_dir, "du-lieu", "books.csv")

# ---------------------------------------------------------------
# BÀI 1: KHÁM PHÁ DỮ LIỆU VÀ TƯƠNG QUAN TRÊN 51 BANG HOA KỲ
# ---------------------------------------------------------------
print("--- BÀI 1: TƯƠNG QUAN & SCATTERPLOT TRÊN 51 BANG HOA KỲ ---")
df_poverty = pd.read_csv(poverty_path)
df_poverty.rename(columns={"graduates": "Graduates", "poverty": "Poverty"}, inplace=True)
print(f"Số lượng quan sát: {len(df_poverty)} bang.")
print(df_poverty.head(6))

# Thống kê mẫu
mx = df_poverty["Graduates"].mean()
sx = df_poverty["Graduates"].std()
my = df_poverty["Poverty"].mean()
sy = df_poverty["Poverty"].std()
r = df_poverty["Graduates"].corr(df_poverty["Poverty"])

print(f"\nGraduates (x): Trung bình = {mx:.2f}%, Độ lệch chuẩn sx = {sx:.2f}%")
print(f"Poverty   (y): Trung bình = {my:.2f}%, Độ lệch chuẩn sy = {sy:.2f}%")
print(f"Hệ số tương quan Pearson r = {r:.4f}")

# Tính tay hệ số OLS giải tích
b1_manual = r * (sy / sx)
b0_manual = my - b1_manual * mx
print(f"\n[Tính tay giải tích OLS]:")
print(f"  Độ dốc b1 = r * (sy / sx) = {b1_manual:.4f}")
print(f"  Hệ số chặn b0 = y_bar - b1 * x_bar = {b0_manual:.4f}")
print(f"  Phương trình: Poverty_hat = {b0_manual:.2f} + ({b1_manual:.2f}) * Graduates")

# Trực quan hóa Bài 1 bằng plotnine
p1 = (
    ggplot(df_poverty, aes(x="Graduates", y="Poverty"))
    + geom_point(color="#0284c7", size=3, alpha=0.75)
    + geom_smooth(method="lm", se=True, color="#ea580c", fill="#ea580c", size=1.2)
    + labs(
        title="Biểu đồ phân tán & Đường xu hướng hồi quy (51 Bang Hoa Kỳ)",
        x="Tỷ lệ tốt nghiệp THPT (%)",
        y="Tỷ lệ sống dưới ngưỡng nghèo (%)",
    )
    + theme_minimal()
)
p1.save(os.path.join(base_dir, "hinh_01_poverty_scatter.png"), width=8, height=5, dpi=150)
print("-> Đã lưu đồ thị Bài 1: hinh_01_poverty_scatter.png")

# ---------------------------------------------------------------
# BÀI 2: KHỚP MÔ HÌNH HỒI QUY BẰNG STATSMODELS & ĐỒ THỊ PHẦN DƯ
# ---------------------------------------------------------------
print("\n--- BÀI 2: KHỚP MÔ HÌNH HỒI QUY ĐƠN BIẾN (OLS) ---")
model_poverty = smf.ols("Poverty ~ Graduates", data=df_poverty).fit()
print("Khớp mô hình thành công với statsmodels!")
print(model_poverty.summary().tables[1])
print(f"\nR-squared = {model_poverty.rsquared:.4f} ({model_poverty.rsquared*100:.1f}%)")

# Dự báo tại Graduates = 85.0%
pred_85 = model_poverty.predict(pd.DataFrame({"Graduates": [85.0]}))[0]
print(f"Dự báo tỷ lệ nghèo tại Graduates = 85.0%: {pred_85:.2f}%")

# Tính phần dư cho RI và DC
df_poverty["fitted"] = model_poverty.fittedvalues
df_poverty["residual"] = model_poverty.resid

ri = df_poverty[df_poverty["state"] == "Rhode Island"].iloc[0]
dc = df_poverty[df_poverty["state"] == "District of Columbia"].iloc[0]
print(f"Rhode Island: Thực tế = {ri['Poverty']}%, Khớp = {ri['fitted']:.2f}%, Phần dư = {ri['residual']:.2f}%")
print(f"Washington DC: Thực tế = {dc['Poverty']}%, Khớp = {dc['fitted']:.2f}%, Phần dư = {dc['residual']:.2f}%")

# Trực quan hóa Bài 2: Đồ thị phần dư bằng plotnine
p_resid = (
    ggplot(df_poverty, aes(x="fitted", y="residual"))
    + geom_point(color="#0284c7", size=3, alpha=0.8)
    + geom_hline(yintercept=0, color="#ea580c", linetype="dashed", size=1)
    + labs(
        title="Đồ thị phần dư (Residuals vs Fitted Values)",
        x="Giá trị khớp (ŷ)",
        y="Phần dư (e)",
    )
    + theme_minimal()
)
p_resid.save(os.path.join(base_dir, "hinh_02_residuals_plot.png"), width=8, height=5, dpi=150)
print("-> Đã lưu đồ thị Bài 2: hinh_02_residuals_plot.png")

# ---------------------------------------------------------------
# BÀI 3: MÔ HÌNH HỒI QUY ĐA BIẾN VỚI BIẾN PHÂN LOẠI (BOOKS)
# ---------------------------------------------------------------
print("\n--- BÀI 3: HỒI QUY ĐA BIẾN SONG SONG (DỮ LIỆU BOOKS) ---")
df_books = pd.read_csv(books_path)
print(f"Số lượng sách khảo sát: {len(df_books)} cuốn.")
print(df_books.head(6))

model_books = smf.ols("weight ~ volume + C(cover)", data=df_books).fit()
print("\nKết quả hồi quy mô hình song song (Parallel Slopes):")
print(model_books.summary().tables[1])

b0_b = model_books.params["Intercept"]
b1_b = model_books.params["volume"]
b2_b = model_books.params["C(cover)[T.paperback]"]

print(f"\nPhương trình tổng quát:")
print(f"  weight_hat = {b0_b:.2f} + {b1_b:.2f} * volume + ({b2_b:.2f}) * D_paperback")
print(f"\n1. Sách bìa cứng (Hardcover, D=0): weight_hat = {b0_b:.2f} + {b1_b:.2f} * volume")
print(f"2. Sách bìa mềm (Paperback, D=1):  weight_hat = {b0_b + b2_b:.2f} + {b1_b:.2f} * volume")

# Dự báo sách bìa mềm thể tích 600 cm3
pred_book = model_books.predict(pd.DataFrame({"volume": [600], "cover": ["paperback"]}))[0]
print(f"\nDự báo sách bìa mềm thể tích 600 cm3: {pred_book:.2f} gram")

# Trực quan hóa Bài 3: Mô hình hai đường song song bằng plotnine
df_books["fitted_weight"] = model_books.fittedvalues

p_books = (
    ggplot(df_books, aes(x="volume", y="weight", color="cover", shape="cover"))
    + geom_point(size=3.5, alpha=0.85)
    + geom_line(aes(y="fitted_weight", group="cover"), size=1.2)
    + scale_color_manual(values={"hardcover": "#0284c7", "paperback": "#ea580c"})
    + labs(
        title="Mô hình Hai Đường Song Song: Khối lượng ~ Thể tích + Loại bìa",
        x="Thể tích (cm³)",
        y="Khối lượng (g)",
        color="Loại bìa",
        shape="Loại bìa",
    )
    + theme_minimal()
)
p_books.save(os.path.join(base_dir, "hinh_03_parallel_slopes.png"), width=9, height=5.5, dpi=150)
print("-> Đã lưu đồ thị Bài 3: hinh_03_parallel_slopes.png")

print("\n=================================================================")
print("  HOÀN THÀNH TOÀN BỘ SCRIPT THỰC HÀNH LAB 04!")
print("=================================================================")
