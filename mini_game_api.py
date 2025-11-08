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
CSV_PATH = os.environ.get("STEAM_CSV", "data/steam.csv")
REVENUE_CSV_PATH = os.environ.get("REVENUE_CSV", "data/Steam_2024_bestRevenue_1500.csv")
PARQUET_PATH = None  # không dùng parquet
DEFAULT_DATE_COL = "release_date"
PORT = int(os.environ.get("PORT", 5000))
# ==================================================

app = Flask(__name__)
CORS(app)

# ----------------- Utilities -----------------
def load_df(use_revenue=False):
    """Load CSV (hoặc parquet nếu có). Nếu use_revenue=True, load dữ liệu revenue."""
    if use_revenue:
        if not os.path.exists(REVENUE_CSV_PATH):
            raise FileNotFoundError(f"Không tìm thấy dataset revenue tại {REVENUE_CSV_PATH}")
        df = pd.read_csv(REVENUE_CSV_PATH, low_memory=False)
        return clean_revenue_df(df)
    
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
    df["total_reviews"] = (pd.to_numeric(df["positive"], errors="coerce").fillna(0) +
                            pd.to_numeric(df["negative"], errors="coerce").fillna(0)).astype(int)

    # Owners [SỬ DỤNG HÀM MỚI]
    owner_col = next((c for c in df.columns if "owner" in c.lower()), None)
    if owner_col:
        # Áp dụng hàm parse_owner_range mới
        df["owners"] = df[owner_col].apply(parse_owner_range).fillna(0).astype(int)
    else:
        df["owners"] = 0

    # Popularity
    df["popularity"] = np.log1p(df["owners"]) * (df["positive_rate"] + 0.01)

    # Business-friendly features
    # Revenue proxy: owners * price (chỉ là xấp xỉ để tham khảo)
    df["revenue_proxy"] = (df["owners"].astype(float) * df["price"].astype(float)).fillna(0.0)

    # Price band (vectorized)
    price_bins = [-float('inf'), 0, 5, 15, 30, float('inf')]
    price_labels = ["Free", "<$5", "$5-$15", "$15-$30", ">$30"]
    df["price_band"] = pd.cut(df["price"], bins=price_bins, labels=price_labels, right=False)

    # Owners tier (vectorized)
    owners_bins = [-float('inf'), 50000, 200000, 1000000, float('inf')]
    owners_labels = ["Indie (<50k)", "Mid (50k-200k)", "Hit (200k-1M)", "Blockbuster (>=1M)"]
    df["owners_tier"] = pd.cut(df["owners"], bins=owners_bins, labels=owners_labels, right=False)

    # Review band (vectorized, descending for right cut)
    review_bins = [-float('inf'), 0.4, 0.6, 0.8, 0.9, float('inf')]
    review_labels = [
        "Negative (<40%)",
        "Mixed (40-60%)",
        "Mostly Positive (60-80%)",
        "Very Positive (80-90%)",
        "Overwhelmingly Positive (>=90%)"
    ]
    df["review_band"] = pd.cut(df["positive_rate"], bins=review_bins, labels=review_labels, right=True)

    # Genres
    df["genres"] = df["genres"].fillna("").astype(str) if "genres" in df.columns else ""

    # Region fallback
    df["region"] = df.get("region", "Global")

    # Avg playtime
    play_col = next((c for c in df.columns if "playtime" in c.lower()), None)
    df["avg_playtime"] = pd.to_numeric(df[play_col], errors="coerce").fillna(0) if play_col else 0

    # Gọn hóa thêm cho các biến name & appid
    for col, default in [("name", lambda df: df.index.astype(str)),
                        ("appid", lambda df: range(1, len(df) + 1))]:
        if col not in df.columns:
            df[col] = default(df)

    return df


def clean_revenue_df(df):
    """Xử lý dữ liệu revenue CSV."""
    col_map = {
        "release_date": lambda df: pd.to_datetime(df["releaseDate"], format="%d-%m-%Y", errors='coerce') if "releaseDate" in df.columns else pd.NaT,
        "release_year": lambda df: pd.to_datetime(df["releaseDate"], format="%d-%m-%Y", errors='coerce').dt.year.fillna(0).astype(int) if "releaseDate" in df.columns else 0,
        "price": lambda df: pd.to_numeric(df["price"], errors="coerce").fillna(0.0) if "price" in df.columns else 0.0,
        "revenue": lambda df: pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0) if "revenue" in df.columns else 0.0,
        "copies_sold": lambda df: pd.to_numeric(df["copiesSold"], errors="coerce").fillna(0).astype(int) if "copiesSold" in df.columns else 0,
        "review_score": lambda df: pd.to_numeric(df["reviewScore"], errors="coerce").fillna(0).astype(int) if "reviewScore" in df.columns else 0,
        "publisher_class": lambda df: df["publisherClass"].fillna("Unknown").astype(str) if "publisherClass" in df.columns else "Unknown",
        "publishers": lambda df: df["publishers"].fillna("").astype(str) if "publishers" in df.columns else "",
        "developers": lambda df: df["developers"].fillna("").astype(str) if "developers" in df.columns else "",
        "avg_playtime": lambda df: pd.to_numeric(df["avgPlaytime"], errors="coerce").fillna(0) if "avgPlaytime" in df.columns else 0,
        "appid": lambda df: pd.to_numeric(df["steamId"], errors="coerce").fillna(0).astype(int) if "steamId" in df.columns else range(1, len(df) + 1),
        "name": lambda df: df["name"] if "name" in df.columns else df.index.astype(str),
        "genres": lambda df: "",
        "region": lambda df: "Global"
    }
    for key, func in col_map.items():
        df[key] = func(df)
    return df


def apply_filters(df, params, is_revenue=False):
    """Lọc dữ liệu theo tham số URL."""
    filtered_df = df.copy()
    date_column = "release_date" if is_revenue else DEFAULT_DATE_COL
    
    # Trích xuất các tham số filter
    start_date = params.get("start") or params.get("start_date")
    end_date = params.get("end") or params.get("end_date")
    genre_filter = params.get("genre")
    region_filter = params.get("region")
    publisher_filter = params.get("publisher")
    search_query = params.get("q")
    min_price_filter = params.get("min_price")
    max_price_filter = params.get("max_price")
    publisher_class_filter = params.get("publisher_class")

    # Áp dụng các filter theo thứ tự
    if start_date:
        filtered_df = filtered_df[filtered_df[date_column] >= pd.to_datetime(start_date, errors="coerce")]
    
    if end_date:
        filtered_df = filtered_df[filtered_df[date_column] <= pd.to_datetime(end_date, errors="coerce")]
    
    if genre_filter and not is_revenue:
        filtered_df = filtered_df[filtered_df["genres"].str.contains(genre_filter, case=False, na=False)]
    
    if region_filter:
        filtered_df = filtered_df[filtered_df["region"].str.contains(region_filter, case=False, na=False)]
    
    if publisher_filter:
        if is_revenue and "publishers" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["publishers"].astype(str).str.contains(publisher_filter, case=False, na=False)]
        elif "publisher" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["publisher"].astype(str).str.contains(publisher_filter, case=False, na=False)]
    
    if search_query:
        filtered_df = filtered_df[filtered_df["name"].astype(str).str.contains(search_query, case=False, na=False)]
    
    if min_price_filter:
        filtered_df = filtered_df[filtered_df["price"] >= float(min_price_filter)]
    
    if max_price_filter:
        filtered_df = filtered_df[filtered_df["price"] <= float(max_price_filter)]
    
    if publisher_class_filter and is_revenue and "publisher_class" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["publisher_class"].str.contains(publisher_class_filter, case=False, na=False)]
    
    return filtered_df


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
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    df = apply_filters(df, request.args, is_revenue=use_revenue)
    total_games = len(df)
    avg_price = round(df["price"].mean(), 2)
    avg_playtime = round(df["avg_playtime"].mean(), 2)
    
    if use_revenue:
        total_revenue = float(df["revenue"].sum())
        total_copies = int(df["copies_sold"].sum()) if "copies_sold" in df.columns else 0
        top_publisher_class = (
            df["publisher_class"].value_counts().idxmax()
            if not df.empty and "publisher_class" in df.columns
            else "N/A"
        )
        return jsonify(
            {
                "total_games": total_games,
                "avg_price": avg_price,
                "avg_playtime": avg_playtime,
                "total_revenue": total_revenue,
                "total_copies": total_copies,
                "top_publisher_class": top_publisher_class,
            }
        )
    else:
        total_owners = int(df["owners"].sum()) if "owners" in df.columns else 0
        top_genre = (
            df["genres"].str.split(";").explode().str.strip().value_counts().idxmax()
            if not df.empty and not df["genres"].str.strip().eq("").all()
            else "N/A"
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
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    df = apply_filters(df, request.args, is_revenue=use_revenue)
    if use_revenue:
        metric = request.args.get("metric", "revenue")
        n = int(request.args.get("n", 10))
        if metric not in df.columns:
            return jsonify({"error": f"Metric '{metric}' không tồn tại."}), 400
        top = df.sort_values(metric, ascending=False).head(n)
        top = top.assign(store_url=top["appid"].apply(lambda a: f"https://store.steampowered.com/app/{int(a)}/"))
        cols = ["appid", "name", metric, "price", "release_year", "copies_sold", "publisher_class", "store_url"]
        available_cols = [c for c in cols if c in top.columns]
        return jsonify(top[available_cols].to_dict(orient="records"))
    else:
        metric = request.args.get("metric", "popularity")
        n = int(request.args.get("n", 10))
        if metric not in df.columns:
            return jsonify({"error": f"Metric '{metric}' không tồn tại."}), 400
        top = df.sort_values(metric, ascending=False).head(n)
        top = top.assign(store_url=top["appid"].apply(lambda a: f"https://store.steampowered.com/app/{int(a)}/"))
        return jsonify(top[["appid", "name", metric, "price", "release_year", "store_url"]].to_dict(orient="records"))


@app.route("/api/series")
def series():
    """Dữ liệu time-series theo năm."""
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    filtered_df = apply_filters(df, request.args, is_revenue=use_revenue)
    
    # Lọc bỏ những năm không hợp lệ (ví dụ: năm 0)
    valid_year_df = filtered_df[filtered_df["release_year"] > 1990]
    
    metric_type = request.args.get("metric", "count")
    grouped_by_year = valid_year_df.groupby("release_year")
    
    # Mapping các metric types
    metric_handlers = {
        "avg_price": lambda: grouped_by_year["price"].mean(),
        "revenue": lambda: grouped_by_year["revenue"].sum() if use_revenue else grouped_by_year.size(),
        "copies": lambda: grouped_by_year["copies_sold"].sum() if use_revenue and "copies_sold" in filtered_df.columns else grouped_by_year.size(),
    }
    
    # Lấy kết quả theo metric
    grouped_result = metric_handlers.get(metric_type, lambda: grouped_by_year.size())()
    
    # Format output: {year: value}
    result_dict = {str(int(year)): float(value) for year, value in grouped_result.sort_index().items()}
    return jsonify(result_dict)


@app.route("/api/aggregate")
def aggregate():
    """Phân bố theo genre / region / publisher / price_band / owners_tier / review_band / publisher_class."""
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    filtered_df = apply_filters(df, request.args, is_revenue=use_revenue)
    aggregate_by = request.args.get("by", "genre")
    
    # Mapping các trường hợp aggregate
    aggregate_handlers = {
        "publisher_class": lambda: filtered_df["publisher_class"].fillna("Unknown").value_counts().to_dict(),
        "publisher": lambda: _get_publisher_aggregate(filtered_df, use_revenue),
        "region": lambda: filtered_df["region"].fillna("Unknown").value_counts().to_dict(),
        "price_band": lambda: filtered_df["price_band"].fillna("Unknown").value_counts().to_dict() if "price_band" in filtered_df.columns else {},
        "owners_tier": lambda: filtered_df["owners_tier"].fillna("Unknown").value_counts().to_dict() if "owners_tier" in filtered_df.columns else {},
        "review_band": lambda: filtered_df["review_band"].fillna("Unknown").value_counts().to_dict() if "review_band" in filtered_df.columns else {},
    }
    
    # Xử lý aggregate theo loại
    if aggregate_by in aggregate_handlers:
        aggregated_data = aggregate_handlers[aggregate_by]()
    elif use_revenue:
        # Revenue data không có genres, trả về publisher_class
        aggregated_data = filtered_df["publisher_class"].fillna("Unknown").value_counts().to_dict()
    else:
        # Tách 'genres', loại bỏ khoảng trắng, loại bỏ giá trị rỗng và đếm
        genre_list = filtered_df["genres"].str.split(";").explode().str.strip()
        aggregated_data = genre_list[genre_list != ''].value_counts().to_dict()
    
    return jsonify(aggregated_data)


def _get_publisher_aggregate(df, is_revenue):
    """Helper function để lấy aggregate theo publisher."""
    if is_revenue and "publishers" in df.columns:
        return df["publishers"].fillna("Unknown").value_counts().to_dict()
    elif "publisher" in df.columns:
        return df["publisher"].fillna("Unknown").value_counts().to_dict()
    return {}


@app.route("/api/game/<int:appid>")
def game_detail(appid: int):
    """Chi tiết 1 game + liên kết ngoài (Steam store)."""
    df = load_df()
    df = apply_filters(df, request.args)
    game_row = df[df["appid"] == appid]
    if game_row.empty:
        return jsonify({"error": "Game không tồn tại"}), 404
    
    game_data = game_row.iloc[0].to_dict()
    game_data["store_url"] = f"https://store.steampowered.com/app/{int(game_data['appid'])}/"
    return jsonify(game_data)


@app.route("/api/segments")
def segments():
    """Phân tích segment cho mục tiêu business: phân phối theo price_band, owners_tier, review_band
    và một số chỉ số tổng hợp theo genre/publisher: count, avg_price, avg_positive, revenue_proxy.
    """
    df = load_df()
    df = apply_filters(df, request.args)

    # Phân phối segment
    result = {
        "price_band": df["price_band"].value_counts().to_dict(),
        "owners_tier": df["owners_tier"].value_counts().to_dict(),
        "review_band": df["review_band"].value_counts().to_dict(),
    }

    # Tổng hợp theo genre (top 15)
    exploded_genres = df["genres"].str.split(";").explode().str.strip()
    non_empty_genres = exploded_genres[exploded_genres != ""]
    df_with_genres = df.join(non_empty_genres.rename("genre_exp"))
    
    genre_summary = df_with_genres.groupby("genre_exp").agg(
        count=("appid", "count"),
        avg_price=("price", "mean"),
        avg_positive=("positive_rate", "mean"),
        revenue_proxy=("revenue_proxy", "sum"),
    ).reset_index().sort_values(["revenue_proxy", "count"], ascending=False).head(15)
    result["genre_summary"] = genre_summary.to_dict(orient="records")

    # Tổng hợp theo publisher nếu có
    if "publisher" in df.columns:
        publisher_summary = df.groupby("publisher").agg(
            count=("appid", "count"),
            avg_price=("price", "mean"),
            avg_positive=("positive_rate", "mean"),
            revenue_proxy=("revenue_proxy", "sum"),
        ).reset_index().sort_values(["revenue_proxy", "count"], ascending=False).head(15)
        result["publisher_summary"] = publisher_summary.to_dict(orient="records")

    return jsonify(result)


@app.route("/api/revenue/analytics")
def revenue_analytics():
    """Analytics cho revenue data: Top games by revenue và Revenue by publisher class."""
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    if not use_revenue:
        return jsonify({"error": "Endpoint này chỉ dùng cho revenue mode"}), 400
    
    df = load_df(use_revenue=True)
    df = apply_filters(df, request.args, is_revenue=True)
    
    # Top games by revenue
    n = int(request.args.get("n", 10))
    top_by_revenue = df.nlargest(n, "revenue")[["appid", "name", "revenue", "copies_sold", "price", "publisher_class", "release_year"]].copy()
    top_by_revenue["revenue"] = top_by_revenue["revenue"].astype(float)
    top_by_revenue["store_url"] = top_by_revenue["appid"].apply(lambda a: f"https://store.steampowered.com/app/{int(a)}/")
    
    # Revenue by publisher class
    revenue_by_class = df.groupby("publisher_class")["revenue"].sum().sort_values(ascending=False).to_dict()
    
    return jsonify({
        "top_by_revenue": top_by_revenue.to_dict(orient="records"),
        "revenue_by_class": revenue_by_class
    })


@app.route("/api/reviews")
def reviews_summary():
    """Tổng hợp đánh giá: scatter (positive_rate vs total_reviews) và histogram theo positive_rate.
    - Trả về điểm scatter (tối đa 50 game có total_reviews cao nhất)
    - Trả về histogram theo bin của positive_rate (0-100, bước 10) cộng gộp theo số lượt review
    """
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    df = apply_filters(df, request.args, is_revenue=use_revenue)

    if df.empty:
        return jsonify({"points": [], "hist": {}})

    top_n = int(request.args.get("n", 50))
    
    if use_revenue:
        return _get_revenue_reviews_data(df, top_n)
    else:
        return _get_normal_reviews_data(df, top_n)


def _get_revenue_reviews_data(df, top_n):
    """Xử lý dữ liệu reviews cho revenue mode."""
    # Scatter: revenue vs copies sold
    top_games_by_revenue = df.sort_values("revenue", ascending=False).head(top_n).copy()
    top_games_by_revenue["store_url"] = top_games_by_revenue["appid"].apply(
        lambda app_id: f"https://store.steampowered.com/app/{int(app_id)}/"
    )
    scatter_points = top_games_by_revenue[
        ["appid", "name", "revenue", "copies_sold", "price", "review_score", "store_url"]
    ].copy()
    scatter_points = scatter_points.to_dict(orient="records")

    # Histogram theo review score (0-100, step 10)
    score_bins = list(range(0, 101, 10))
    review_scores = df["review_score"].fillna(0).clip(0, 100)
    binned_scores = pd.cut(review_scores, bins=score_bins, right=False, include_lowest=True)
    histogram_data = df.groupby(binned_scores)["revenue"].sum().to_dict()
    
    formatted_histogram = _format_histogram_intervals(histogram_data, is_float=True)
    
    return jsonify({"points": scatter_points, "hist": formatted_histogram})


def _get_normal_reviews_data(df, top_n):
    """Xử lý dữ liệu reviews cho normal mode."""
    # Scatter: positive_rate vs total_reviews
    top_games_by_reviews = df.sort_values("total_reviews", ascending=False).head(top_n).copy()
    top_games_by_reviews["store_url"] = top_games_by_reviews["appid"].apply(
        lambda app_id: f"https://store.steampowered.com/app/{int(app_id)}/"
    )
    scatter_points = top_games_by_reviews[
        ["appid", "name", "positive_rate", "total_reviews", "owners", "price", "store_url"]
    ]
    scatter_points["positive_rate_pct"] = (scatter_points["positive_rate"] * 100.0).clip(0, 100)
    scatter_points = scatter_points.to_dict(orient="records")

    # Histogram theo bin của positive_rate (0-100, step 10)
    score_bins = list(range(0, 101, 10))
    positive_rate_percent = (df["positive_rate"].fillna(0) * 100.0).clip(0, 100)
    binned_rates = pd.cut(positive_rate_percent, bins=score_bins, right=False, include_lowest=True)
    histogram_data = df.groupby(binned_rates)["total_reviews"].sum().to_dict()
    
    formatted_histogram = _format_histogram_intervals(histogram_data, is_float=False)
    
    return jsonify({"points": scatter_points, "hist": formatted_histogram})


def _format_histogram_intervals(histogram_data, is_float=False):
    """Format histogram intervals thành dictionary với key dạng "left-right"."""
    formatted_histogram = {}
    for interval, value in histogram_data.items():
        if pd.isna(value):
            continue
        left_bound = int(interval.left)
        right_bound = int(interval.right)
        formatted_histogram[f"{left_bound}-{right_bound}"] = float(value) if is_float else int(value)
    return formatted_histogram


@app.route("/api/suggest")
def suggest():
    """Gợi ý tên game dựa trên từ khóa 'q'. Trả về tối đa n kết quả theo total_reviews hoặc popularity.
    Tôn trọng các filter khác (genre/start/end/price...).
    """
    use_revenue = request.args.get("revenue_mode", "false").lower() == "true"
    df = load_df(use_revenue=use_revenue)
    # Áp dụng các filter khác trước (ngoại trừ q, vì q đang dùng cho gợi ý)
    params = request.args.to_dict(flat=True).copy()
    search_query = params.pop("q", None)
    filtered_df = apply_filters(df, params, is_revenue=use_revenue)
    
    if not search_query:
        return jsonify([])
    
    max_results = int(request.args.get("n", 8))
    name_filter_mask = filtered_df["name"].astype(str).str.contains(search_query, case=False, na=False)
    matching_games = filtered_df[name_filter_mask].copy()
    
    if matching_games.empty:
        return jsonify([])
    
    # Sắp xếp ưu tiên nhiều review hơn, sau đó popularity (độ nổi tiếng, phổ biến trò chơi)
    sort_columns = ["total_reviews", "popularity"] if "total_reviews" in matching_games.columns else ["popularity"]
    matching_games = matching_games.sort_values(sort_columns, ascending=False).head(max_results)
    matching_games["store_url"] = matching_games["appid"].apply(
        lambda app_id: f"https://store.steampowered.com/app/{int(app_id)}/"
    )
    
    return jsonify(matching_games[["appid", "name", "price", "store_url"]].to_dict(orient="records"))


@app.route("/api/export")
def export_csv():
    """Xuất dữ liệu lọc hiện tại ra CSV."""
    df = load_df()
    df = apply_filters(df, request.args)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
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