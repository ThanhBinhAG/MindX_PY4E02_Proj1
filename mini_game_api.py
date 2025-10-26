#!/usr/bin/env python3
"""
Mini Game Analytics Dashboard API
---------------------------------
Backend Flask đọc dữ liệu Steam CSV, xử lý nhẹ bằng pandas,
cung cấp các endpoint phân tích và trực quan hóa, đồng thời
serve file index.html trong thư mục frontend/ để demo.
"""

from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import os
import re  # <-- THÊM DÒNG NÀY
from datetime import datetime

# ===================== CONFIG =====================
CSV_PATH = os.environ.get("STEAM_CSV", "mini-game-dashboard/data/steam.csv")
PARQUET_PATH = None  # không dùng parquet
DEFAULT_DATE_COL = "release_date"
PORT = int(os.environ.get("PORT", 5000))
# ==================================================

app = Flask(__name__)
CORS(app)

# ----------------- Utilities -----------------
def load_df():
    """Load CSV (hoặc parquet nếu có)."""
    if PARQUET_PATH and os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        return df
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy dataset tại {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    return light_clean(df)

def light_clean(df):
    """Xử lý nhẹ dữ liệu: ngày phát hành, giá, owners, tỉ lệ đánh giá,..."""
    # Ngày phát hành
    if DEFAULT_DATE_COL in df.columns:
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL], errors='coerce')
    else:
        for c in df.columns:
            if 'release' in c.lower():
                df[DEFAULT_DATE_COL] = pd.to_datetime(df[c], errors='coerce')
                break
        if DEFAULT_DATE_COL not in df.columns:
            df[DEFAULT_DATE_COL] = pd.NaT
    df["release_year"] = df[DEFAULT_DATE_COL].dt.year.fillna(0).astype(int)

    # Giá
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        df["price"] = 0.0

    # Positive / negative
    pos = next((c for c in df.columns if "positive" in c.lower()), None)
    neg = next((c for c in df.columns if "negative" in c.lower()), None)
    df["positive"] = pd.to_numeric(df[pos], errors="coerce").fillna(0) if pos else 0
    df["negative"] = pd.to_numeric(df[neg], errors="coerce").fillna(0) if neg else 0
    df["positive_rate"] = df.apply(
        lambda r: r["positive"] / (r["positive"] + r["negative"])
        if (r["positive"] + r["negative"]) > 0
        else 0,
        axis=1,
    )

    # Owners [SỬ DỤNG HÀM MỚI]
    owner_col = next((c for c in df.columns if "owner" in c.lower()), None)
    if owner_col:
        # Áp dụng hàm parse_owner_range mới
        df["owners"] = df[owner_col].apply(parse_owner_range).fillna(0).astype(int)
    else:
        df["owners"] = 0

    # Popularity
    df["popularity"] = np.log1p(df["owners"]) * (df["positive_rate"] + 0.01)

    # Genres
    if "genres" in df.columns:
        df["genres"] = df["genres"].fillna("").astype(str)
    else:
        df["genres"] = ""

    # Region fallback
    if "region" not in df.columns:
        df["region"] = "Global"

    # Avg playtime
    play_col = next((c for c in df.columns if "playtime" in c.lower()), None)
    if play_col:
        df["avg_playtime"] = pd.to_numeric(df[play_col], errors="coerce").fillna(0)
    else:
        df["avg_playtime"] = 0

    # Ensure name & appid
    if "name" not in df.columns:
        df["name"] = df.index.astype(str)
    if "appid" not in df.columns:
        df["appid"] = range(1, len(df) + 1)

    return df


def apply_filters(df, params):
    """Lọc dữ liệu theo tham số URL."""
    d = df
    start = params.get("start") or params.get("start_date")
    end = params.get("end") or params.get("end_date")
    genre = params.get("genre")
    region = params.get("region")
    publisher = params.get("publisher")
    q = params.get("q")
    min_price = params.get("min_price")
    max_price = params.get("max_price")

    if start:
        d = d[d[DEFAULT_DATE_COL] >= pd.to_datetime(start, errors="coerce")]
    if end:
        d = d[d[DEFAULT_DATE_COL] <= pd.to_datetime(end, errors="coerce")]
    if genre:
        d = d[d["genres"].str.contains(genre, case=False, na=False)]
    if region:
        d = d[d["region"].str.contains(region, case=False, na=False)]
    if publisher and "publisher" in d.columns:
        d = d[d["publisher"].astype(str).str.contains(publisher, case=False, na=False)]
    if q:
        d = d[d["name"].astype(str).str.contains(q, case=False, na=False)]
    if min_price:
        d = d[d["price"] >= float(min_price)]
    if max_price:
        d = d[d["price"] <= float(max_price)]

    return d


# [MÃ ĐÃ SỬA] Hàm trợ giúp để xử lý khoảng giá trị "owners"
def parse_owner_range(owner_str):
    """
    Chuyển đổi chuỗi "100,000 - 200,000" hoặc "100,000" thành số.
    Sử dụng regex để tìm tất cả các số trong chuỗi, bất kể định dạng.
    """
    try:
        # 1. Làm sạch chuỗi: loại bỏ dấu phẩy
        cleaned_str = str(owner_str).replace(',', '')
        
        # 2. Tìm tất cả các chuỗi số (ví dụ: "100000", "200000")
        numbers = re.findall(r'\d+', cleaned_str)
        
        if len(numbers) == 0:
            # Không tìm thấy số (ví dụ: "N/A", "nan")
            return 0
        elif len(numbers) == 1:
            # Chỉ có 1 số (ví dụ: "20000")
            return int(numbers[0])
        elif len(numbers) >= 2:
            # Có 2 số trở lên (ví dụ: "100000 - 200000")
            # Chúng ta chỉ lấy 2 số đầu tiên
            low = int(numbers[0])
            high = int(numbers[1])
            # Lấy trung bình cộng
            return (low + high) / 2
    except Exception:
        # Bất kỳ lỗi nào khác
        return 0

# ----------------- API Endpoints -----------------
@app.route("/api/stats/summary")
def summary():
    """Tổng quan chỉ số (KPI)."""
    df = load_df()
    df = apply_filters(df, request.args)
    total_games = len(df)
    avg_price = round(df["price"].mean(), 2)
    avg_playtime = round(df["avg_playtime"].mean(), 2)
    total_owners = int(df["owners"].sum())
    top_genre = (
        df["genres"].str.split(";").explode().str.strip().value_counts().idxmax()
        if not df.empty and not df["genres"].str.strip().eq("").all()
        else "N/A" # Thêm kiểm tra để tránh lỗi nếu không có genre
    )
    return jsonify(
        {
            "total_games": total_games,
            "avg_price": avg_price,
            "avg_playtime": avg_playtime,
            "total_owners": total_owners,
            "top_genre": top_genre,
        }
    )


@app.route("/api/top")
def top_games():
    """Top N theo chỉ số."""
    df = load_df()
    df = apply_filters(df, request.args)
    metric = request.args.get("metric", "popularity")
    n = int(request.args.get("n", 10))
    if metric not in df.columns:
        return jsonify({"error": f"Metric '{metric}' không tồn tại."}), 400
    top = df.sort_values(metric, ascending=False).head(n)
    return jsonify(top[["appid", "name", metric, "price", "release_year"]].to_dict(orient="records"))


@app.route("/api/series")
def series():
    """Dữ liệu time-series theo năm."""
    df = load_df()
    df = apply_filters(df, request.args)
    
    # Lọc bỏ những năm không hợp lệ (ví dụ: năm 0)
    df_valid_year = df[df["release_year"] > 1990] # Giả sử chỉ lấy game sau 1990
    
    metric = request.args.get("metric", "count")
    group = df_valid_year.groupby("release_year") # Nhóm theo 'release_year' đã xử lý
    
    if metric == "avg_price":
        res = group["price"].mean()
    else:
        res = group.size()
        
    out = {str(int(k)): float(v) for k, v in res.sort_index().items()}
    return jsonify(out)


@app.route("/api/aggregate")
def aggregate():
    """Phân bố theo genre / region / publisher."""
    df = load_df()
    df = apply_filters(df, request.args)
    by = request.args.get("by", "genre")
    if by == "region":
        s = df["region"].fillna("Unknown").value_counts().to_dict()
    elif by == "publisher" and "publisher" in df.columns:
        s = df["publisher"].fillna("Unknown").value_counts().to_dict()
    else:
        # Tách 'genres', loại bỏ khoảng trắng, loại bỏ giá trị rỗng và đếm
        s = df["genres"].str.split(";").explode().str.strip()
        s = s[s != ''].value_counts().to_dict()
    return jsonify(s)


@app.route("/api/export")
def export_csv():
    """Xuất dữ liệu lọc hiện tại ra CSV."""
    df = load_df()
    df = apply_filters(df, request.args)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="export_filtered.csv",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


# ----------------- Serve frontend -----------------
@app.route("/")
def serve_frontend():
    """Trả về file index.html nếu có, hoặc hướng dẫn API."""
    base = os.path.join(os.path.dirname(__file__), "frontend")
    index_path = os.path.join(base, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(base, "index.html")
    return jsonify(
        {
            "message": "Mini Game API đang chạy!",
            "hint": "Thêm file frontend/index.html để hiển thị dashboard.",
            "endpoints": ["/api/stats/summary", "/api/top", "/api/series", "/api/aggregate"],
        }
    )


# ----------------- Run -----------------
if __name__ == "__main__":
    csv_status = "✅ found" if os.path.exists(CSV_PATH) else "❌ missing"
    print("=" * 60)
    print("🚀 Starting Mini Game Analytics API")
    print(f"📦 Dataset: {CSV_PATH} → {csv_status}")
    print(f"🌐 Running on: http://127.0.0.1:{PORT}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=True)