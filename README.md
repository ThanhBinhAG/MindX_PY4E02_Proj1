# Mini Game Analytics Dashboard

Dashboard phân tích dữ liệu game Steam với giao diện trực quan, hỗ trợ xem dữ liệu thông thường và chế độ doanh thu.

## Tính năng

- 📊 **Dashboard trực quan**: Hiển thị các biểu đồ về game Steam
- 🎮 **Chế độ xem thông thường**: Xem dữ liệu game theo popularity, thể loại, đánh giá
- 💰 **Chế độ doanh thu**: Xem dữ liệu doanh thu, bản bán, phân loại theo publisher
- 🔍 **Tìm kiếm và lọc**: Lọc theo thể loại, năm, giá, tìm kiếm game
- 📈 **Nhiều loại biểu đồ**: Bar chart, Line chart, Pie chart, Scatter chart

## Cài đặt

### Yêu cầu
- Python 3.7 trở lên
- Các thư viện Python (xem `requirements.txt`)

### Các bước cài đặt

1. Clone hoặc tải project về máy

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Chuẩn bị dữ liệu:
   - Đặt file `steam.csv` vào thư mục `data/`
   - Đặt file `Steam_2024_bestRevenue_1500.csv` vào thư mục `data/` (cho chế độ doanh thu)

4. Chạy ứng dụng:
```bash
python mini_game_api.py
```

5. Mở trình duyệt và truy cập:
```
http://localhost:5000
```

## Cấu trúc thư mục

```
mini-game-dashboard/
├── data/                          # Thư mục chứa dữ liệu CSV
│   ├── steam.csv                  # Dữ liệu game Steam chính
│   └── Steam_2024_bestRevenue_1500.csv  # Dữ liệu doanh thu
├── frontend/                      # Giao diện web
│   └── index.html                 # File HTML chính
├── mini_game_api.py               # Backend API Flask
├── requirements.txt               # Danh sách thư viện Python
└── README.md                      # File này
```

## Sử dụng

### Chế độ xem thông thường
- Xem top 10 game phổ biến
- Biểu đồ số lượng game theo năm (2013-2019)
- Phân bố theo thể loại
- Đánh giá vs lượt đánh giá

### Chế độ doanh thu
- Bật toggle "Chế độ doanh thu" ở góc trên bên phải
- Xem top 10 game theo doanh thu
- Phân bố doanh thu theo loại publisher
- Tổng doanh thu và tổng bản bán

### Lọc dữ liệu
- **Thể loại**: Chọn thể loại game
- **Năm**: Lọc theo năm phát hành
- **Tìm kiếm**: Tìm game theo tên
- **Giá**: Lọc theo khoảng giá (tối thiểu - tối đa)

## API Endpoints

- `GET /api/stats/summary` - Tổng quan chỉ số (KPI)
- `GET /api/top` - Top N game theo chỉ số
- `GET /api/series` - Dữ liệu time-series theo năm
- `GET /api/aggregate` - Phân bố theo genre/publisher/region
- `GET /api/revenue/analytics` - Analytics cho dữ liệu doanh thu
- `GET /api/reviews` - Tổng hợp đánh giá
- `GET /api/suggest` - Gợi ý tên game
- `GET /api/game/<appid>` - Chi tiết một game
- `GET /api/export` - Xuất dữ liệu ra CSV

## Ghi chú

- Dữ liệu được xử lý nhẹ bằng pandas, không cần database
- Có thể thay đổi đường dẫn file CSV qua biến môi trường `STEAM_CSV` và `REVENUE_CSV`
- Port mặc định là 5000, có thể thay đổi qua biến môi trường `PORT`

## Tác giả
by Light.
MindX Project - Py4E02
