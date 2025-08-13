import pandas as pd

# Đọc file .dta
df = pd.read_stata("./data/2010.dta")  # đổi đường dẫn cho đúng

# Hiển thị 10 hàng đầu
print(df.head(10))

# Nếu muốn xem thông tin tổng quan
print(df.info())