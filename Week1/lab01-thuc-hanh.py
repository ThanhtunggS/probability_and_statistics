#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Học phần: UET.MAT1052 - Xác suất thống kê (VNU-UET)
Bài thực hành Lab 01: Khám Phá Cấu Trúc Dữ Liệu Với Python/pandas
Tác giả: Biên soạn theo chuẩn bài giảng UET (Tuần 01)
"""

import os
import pandas as pd
import numpy as np

def in_tieu_de(tieu_de):
    print("\n" + "=" * 70)
    print(f"  {tieu_de}")
    print("=" * 70)

def main():
    # Định vị thư mục dữ liệu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "du-lieu")

    in_tieu_de("PHẦN 1: KHẢO SÁT BỘ DỮ LIỆU KHOẢN VAY (loan50.csv)")
    loan_path = os.path.join(data_dir, "loan50.csv")
    if os.path.exists(loan_path):
        df_loan = pd.read_csv(loan_path)
        print(f"1. Kích thước bảng (shape): {df_loan.shape[0]} hàng x {df_loan.shape[1]} cột")
        print("\n2. Danh sách 10 cột đầu tiên:")
        print(df_loan.columns[:10].tolist())

        print("\n3. Lát cắt 4 biến đại diện (term, grade, state, interest_rate):")
        sub_loan = df_loan[['term', 'grade', 'state', 'interest_rate']].head(5)
        print(sub_loan)

        print("\n4. Kiểu dữ liệu lưu trữ máy tính (dtypes):")
        print(sub_loan.dtypes)

        print("\n5. Nhận xét phân loại thống kê:")
        print(" - term (dtype: int64): Biến số rời rạc (kỳ hạn vay theo tháng).")
        print(" - grade (dtype: object): Biến phân loại thứ bậc (mức xếp hạng tín dụng A > B > C...).")
        print(" - state (dtype: object): Biến phân loại danh nghĩa (mã bang là nhãn định danh).")
        print(" - interest_rate (dtype: float64): Biến số liên tục (tỷ lệ lãi suất %/năm).")
    else:
        print(f"Không tìm thấy tệp {loan_path}!")

    in_tieu_de("PHẦN 2: KHẢO SÁT BỘ DỮ LIỆU THƯ ĐIỆN TỬ (email50.csv)")
    email_path = os.path.join(data_dir, "email50.csv")
    if os.path.exists(email_path):
        df_email = pd.read_csv(email_path)
        print(f"1. Kích thước bảng (shape): {df_email.shape[0]} hàng x {df_email.shape[1]} cột")
        
        print("\n2. Kiểm tra các biến đặc trưng (spam, cc, image, number):")
        cols = ['spam', 'cc', 'image', 'number']
        print(df_email[cols].head(5))

        print("\n3. Đơn vị quan sát và phân tích bản chất:")
        print(" - Đơn vị quan sát: 1 bức thư điện tử (email).")
        print(" - spam: Biến phân loại danh nghĩa (nhị phân 0/1). Số 1/0 là nhãn, không có ý nghĩa tính trung bình cộng.")
        print(" - cc: Biến số rời rạc (số người nhận bản sao).")
        print(" - image: Biến số rời rạc (số ảnh đính kèm).")
        print(" - number: Biến phân loại thứ bậc (nhận các giá trị 'none', 'small', 'big').")
        print("   => CẢNH BÁO: Tên biến 'number' không đồng nghĩa đây là biến số!")
    else:
        print(f"Không tìm thấy tệp {email_path}!")

    in_tieu_de("PHẦN 3: KHẢO SÁT BỘ DỮ LIỆU CHIM CÁNH CỤT (penguins.csv)")
    penguins_path = os.path.join(data_dir, "penguins.csv")
    if os.path.exists(penguins_path):
        df_penguins = pd.read_csv(penguins_path)
        print(f"1. Kích thước bảng (shape): {df_penguins.shape[0]} hàng x {df_penguins.shape[1]} cột")
        print("\n2. Cấu trúc tổng thể với df.info():")
        df_penguins.info()

        print("\n3. Tóm tắt nhanh số lượng theo loài (species):")
        print(df_penguins['species'].value_counts())

        print("\n4. Phân loại biến:")
        print(" - 4 biến số liên tục: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g.")
        print(" - 3 biến phân loại danh nghĩa: species, island, sex.")
        print(" - Đơn vị quan sát: 1 cá thể chim cánh cụt.")
    else:
        print(f"Không tìm thấy tệp {penguins_path}!")

    in_tieu_de("PHẦN 4: THỬ THÁCH PHẢN BIỆN (W01-PY01)")
    print("Mô phỏng bảng sinh viên UET kích thước (1200, 6):")
    mock_data = pd.DataFrame({
        'ma_sinh_vien': [24020001, 24020002, 24020003, 24020004],
        'muc_danh_gia': [1, 2, 5, 4],
        'nganh_hoc': ['CNTT', 'KTPM', 'KHMT', 'DTVT'],
        'gpa': [3.45, 2.80, 3.85, 3.10]
    })
    print(mock_data)
    print("\nKiểu lưu trữ dtypes:")
    print(mock_data.dtypes)
    print("\nKết luận cốt lõi:")
    print(" 1. ma_sinh_vien có dtype=int64 nhưng là BIẾN PHÂN LOẠI DANH NGHĨA (mã định danh).")
    print(" 2. muc_danh_gia có dtype=int64 nhưng là BIẾN PHÂN LOẠI THỨ BẬC (thang đo mức độ).")
    print(" 3. gpa có dtype=float64 và là BIẾN SỐ LIÊN TỤC (độ lớn có ý nghĩa định lượng).")
    print("\n=> Luôn nhìn ngữ nghĩa đo lường thực tế trước khi áp dụng công thức thống kê!\n")

if __name__ == "__main__":
    main()
