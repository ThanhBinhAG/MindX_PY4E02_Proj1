from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import os
import re
from datetime import datetime
from functools import lru_cache

# ===================== CONFIG =====================
CSV_PATH = os.environ.get("STEAM_CSV", "data/steam.csv")
REVENUE_CSV_PATH = os.environ.get("REVENUE_CSV", "data/Steam_2024_bestRevenue_1500.csv")
PORT = int(os.environ.get("PORT", 5000))
# ==================================================

app = Flask(__name__, static_folder="frontend", static_url_path="/")
CORS(app)


# ----------------- Utilities -----------------
@lru_cache(maxsize=2)
def get_cached_df(use_revenue: bool):
    """Cache dataset đã clean."""
    path = REVENUE_CSV_PATH if use_revenue else CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    df = pd.read_csv(path, low_memory=False)
    return clean_df(df, use_revenue)


def clean_df(df, is_revenue=False):
    """Xử lý chung cho cả 2 dataset."""
    df = df.copy()

    # release_date
    if is_revenue:
        if "releaseDate" in df.columns:
            df["release_date"] = pd.to_datetime(df["releaseDate"], format="%d-%m-%Y", errors='coerce')
        else:
            df["release_date"] = pd.NaT
    else:
        date_src = next((c for c in df.columns if 'release' in c.lower()), None)
        df["release_date"] = pd.to_datetime(df[date_src], errors='coerce') if date_src else pd.NaT
    df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)

    # price
    df["price"] = pd.to_numeric(df["price"], errors='coerce').fillna(0.0) if "price" in df.columns else 0.0

    # appid & name
    appid_col = "steamId" if "steamId" in df.columns else ("appid" if "appid" in df.columns else None)
    if appid_col:
        df["appid"] = pd.to_numeric(df[appid_col], errors='coerce').fillna(0).astype(int)
        if df["appid"].eq(0).all():
            df["appid"] = range(1, len(df) + 1)
    else:
        df["appid"] = range(1, len(df) + 1)
    
    df["name"] = df["name"].fillna("Unknown") if "name" in df.columns else df.index.astype(str)

    if is_revenue:
        # Revenue columns
        for col, src in [("revenue", "revenue"), ("copies_sold", "copiesSold"), ("avg_playtime", "avgPlaytime")]:
            df[col] = pd.to_numeric(df[src], errors='coerce').fillna(0) if src in df.columns else 0
        
        # String columns
        df["publisher_class"] = df["publisherClass"].fillna("Unknown").astype(str) if "publisherClass" in df.columns else "Unknown"
        df["publishers"] = df["publishers"].fillna("").astype(str) if "publishers" in df.columns else ""
        df["genres"] = ""
        df["region"] = "Global"
        return df

    # Normal dataset
    owner_col = next((c for c in df.columns if "owner" in c.lower()), None)
    df["owners"] = df[owner_col].apply(parse_owner_range).fillna(0).astype(int) if owner_col else 0

    df["popularity"] = np.log1p(df["owners"])
    df["revenue_proxy"] = df["owners"] * df["price"]

    play_col = next((c for c in df.columns if "playtime" in c.lower()), None)
    df["avg_playtime"] = pd.to_numeric(df[play_col], errors='coerce').fillna(0) if play_col else 0

    df["genres"] = df["genres"].fillna("").astype(str) if "genres" in df.columns else ""
    df["region"] = df["region"] if "region" in df.columns else "Global"

    # Bins
    df["price_band"] = pd.cut(df["price"], [-np.inf, 0, 5, 15, 30, np.inf],
                            labels=["Free", "<$5", "$5-$15", "$15-$30", ">$30"], right=False)
    df["owners_tier"] = pd.cut(df["owners"], [-np.inf, 50_000, 200_000, 1_000_000, np.inf],
                            labels=["Indie (<50k)", "Mid (50k-200k)", "Hit (200k-1M)", "Blockbuster (>=1M)"], right=False)

    return df


def parse_owner_range(s): #ép kiểu chuỗi để in ra giá trị trung bình của range
    """Chuyển '100,000 - 200,000' → trung bình."""
    try:
        nums = re.findall(r'\d+', str(s).replace(',', ''))
        if not nums: return 0
        if len(nums) == 1: return int(nums[0])
        return (int(nums[0]) + int(nums[1])) / 2
    except:
        return 0


def apply_filters(df, params, is_revenue=False): #Filter cho người dùng
    """Lọc dữ liệu."""
    df = df.copy()
    date_col = "release_date"

    start = params.get("start") or params.get("start_date") #Theo thời gian
    end = params.get("end") or params.get("end_date")
    if start: df = df[df[date_col] >= pd.to_datetime(start, errors='coerce')]
    if end: df = df[df[date_col] <= pd.to_datetime(end, errors='coerce')]

    q = params.get("q") #Theo tên
    if q: df = df[df["name"].astype(str).str.contains(q, case=False, na=False)]

    genre = params.get("genre") #Theo thể loại
    if genre and not is_revenue: df = df[df["genres"].str.contains(genre, case=False, na=False)]

    region = params.get("region") #Theo vùng 
    if region: df = df[df["region"].str.contains(region, case=False, na=False)]

    pub = params.get("publisher") #Theo nhà xuất bản
    if pub:
        col = "publishers" if is_revenue and "publishers" in df.columns else "publisher"
        if col in df.columns: df = df[df[col].astype(str).str.contains(pub, case=False, na=False)]

    min_p = params.get("min_price") #Theo giá
    max_p = params.get("max_price") #Theo giá
    if min_p: df = df[df["price"] >= float(min_p)]
    if max_p: df = df[df["price"] <= float(max_p)]

    if is_revenue and params.get("publisher_class") and "publisher_class" in df.columns:
        df = df[df["publisher_class"].str.contains(params["publisher_class"], case=False, na=False)]

    return df


def load_and_filter(use_revenue=False, params=None): #Load và filter dữ liệu
    return apply_filters(get_cached_df(use_revenue), params or request.args.to_dict(), use_revenue)

def get_store_url(appid): #Lấy URL của game
    return f"https://store.steampowered.com/app/{int(appid)}/"


# ----------------- API Endpoints -----------------
@app.route("/api/stats/summary")
def summary():
    params = request.args
    use_revenue = params.get("revenue_mode", "false").lower() == "true" #Mode doanh thu
    df = load_and_filter(use_revenue, params)

    base = {
        "total_games": len(df),
        "avg_price": round(df["price"].mean(), 2),
        "avg_playtime": round(df["avg_playtime"].mean(), 2),
    }

    if use_revenue:
        mode_result = df["publisher_class"].mode()
        base.update({
            "total_revenue": float(df["revenue"].sum()),
            "total_copies": int(df["copies_sold"].sum()) if "copies_sold" in df.columns else 0,
            "top_publisher_class": mode_result.iloc[0] if not df.empty and "publisher_class" in df.columns and not mode_result.empty else "N/A"
        })
    else:
        genres_mode = df["genres"].str.split(";").explode().str.strip().mode()
        base.update({
            "total_owners": int(df["owners"].sum()),
            "top_genre": genres_mode.iloc[0] if not df.empty and not genres_mode.empty else "N/A"
        })
    return jsonify(base)


@app.route("/api/top")
def top_games():
    params = request.args
    use_revenue = params.get("revenue_mode", "false").lower() == "true"
    df = load_and_filter(use_revenue, params)
    n = int(params.get("n", 10))
    metric = params.get("metric", "revenue" if use_revenue else "popularity")

    if metric not in df.columns:
        return jsonify({"error": f"Metric '{metric}' không tồn tại."}), 400

    top = df.nlargest(n, metric).copy()
    top["store_url"] = top["appid"].apply(get_store_url)

    if use_revenue:
        cols = ["appid", "name", "revenue", "copies_sold", "price", "publisher_class", "release_year", "store_url"]
    else:
        cols = ["appid", "name", metric, "price", "release_year", "store_url"]

    return jsonify(top[[c for c in cols if c in top.columns]].to_dict("records"))


@app.route("/api/series")
def series():
    params = request.args
    use_revenue = params.get("revenue_mode", "false").lower() == "true"
    df = load_and_filter(use_revenue, params)
    df = df[df["release_year"] > 1990]

    metric = params.get("metric", "count")
    handlers = {
        "avg_price": lambda: df.groupby("release_year")["price"].mean(),
        "revenue": lambda: df.groupby("release_year")["revenue"].sum() if use_revenue else df.groupby("release_year").size(),
        "copies": lambda: df.groupby("release_year")["copies_sold"].sum() if use_revenue else df.groupby("release_year").size(),
    }
    result = handlers.get(metric, lambda: df.groupby("release_year").size())()
    return jsonify({str(int(k)): float(v) for k, v in result.sort_index().items()})


@app.route("/api/aggregate")
def aggregate():
    params = request.args
    use_revenue = params.get("revenue_mode", "false").lower() == "true"
    df = load_and_filter(use_revenue, params)
    by = params.get("by", "genre")

    handlers = {
        "publisher_class": lambda: df["publisher_class"].fillna("Unknown").value_counts(),
        "publisher": lambda: df["publishers"].fillna("Unknown").value_counts() if use_revenue and "publishers" in df.columns else (df["publisher"].fillna("Unknown").value_counts() if "publisher" in df.columns else pd.Series()),
        "region": lambda: df["region"].fillna("Unknown").value_counts(),
        "price_band": lambda: df["price_band"].fillna("Unknown").value_counts(),
        "owners_tier": lambda: df["owners_tier"].fillna("Unknown").value_counts(),
    }

    if by in handlers:
        data = handlers[by]()
    elif use_revenue:
        data = df["publisher_class"].fillna("Unknown").value_counts()
    else:
        genres = df["genres"].str.split(";").explode().str.strip()
        data = genres[genres != ""].value_counts()

    return jsonify(data.to_dict())


@app.route("/api/game/<int:appid>")
def game_detail(appid):
    df = load_and_filter(use_revenue=False)
    game = df[df["appid"] == appid]
    if game.empty:
        return jsonify({"error": "Game không tồn tại"}), 404
    data = game.iloc[0].to_dict()
    data["store_url"] = get_store_url(data['appid'])
    return jsonify(data)


@app.route("/api/segments")
def segments():
    df = load_and_filter(use_revenue=False)
    result = {
        "price_band": df["price_band"].value_counts().to_dict(),
        "owners_tier": df["owners_tier"].value_counts().to_dict(),
    }

    # Genre summary
    exploded = df["genres"].str.split(";").explode().str.strip()
    df_genre = df.join(exploded.rename("genre")).dropna(subset=["genre"])
    if not df_genre.empty:
        genre_summary = df_genre.groupby("genre").agg(
            count=("appid", "count"),
            avg_price=("price", "mean"),
            revenue_proxy=("revenue_proxy", "sum")
        ).round(2).sort_values(["revenue_proxy", "count"], ascending=False).head(15)
        result["genre_summary"] = genre_summary.reset_index().to_dict("records")

    # Publisher summary
    if "publisher" in df.columns:
        pub_summary = df.groupby("publisher").agg(
            count=("appid", "count"),
            avg_price=("price", "mean"),
            revenue_proxy=("revenue_proxy", "sum")
        ).round(2).sort_values(["revenue_proxy", "count"], ascending=False).head(15)
        result["publisher_summary"] = pub_summary.reset_index().to_dict("records")

    return jsonify(result)


@app.route("/api/revenue/analytics")
def revenue_analytics():
    params = request.args
    if params.get("revenue_mode", "false").lower() != "true":
        return jsonify({"error": "Endpoint này chỉ dùng cho revenue mode"}), 400
    
    df = load_and_filter(use_revenue=True, params=params)
    n = int(params.get("n", 10))
    top = df.nlargest(n, "revenue")[["appid", "name", "revenue", "copies_sold", "price", "publisher_class", "release_year"]].copy()
    top["store_url"] = top["appid"].apply(get_store_url)
    
    return jsonify({
        "top_by_revenue": top.to_dict("records"),
        "revenue_by_class": df.groupby("publisher_class")["revenue"].sum().sort_values(ascending=False).to_dict()
    })


@app.route("/api/suggest")
def suggest():
    params = request.args.to_dict()
    q = params.pop("q", None)
    use_revenue = params.get("revenue_mode", "false").lower() == "true"
    df = load_and_filter(use_revenue, params)

    if not q: return jsonify([])
    matches = df[df["name"].astype(str).str.contains(q, case=False, na=False)]
    if matches.empty: return jsonify([])

    n = int(request.args.get("n", 8))
    if use_revenue and "revenue" in matches.columns:
        sort_by = ["revenue"]
    else:
        sort_by = ["popularity"] if "popularity" in matches.columns else []
    
    matches = matches.sort_values(sort_by, ascending=False).head(n) if sort_by else matches.head(n)
    matches["store_url"] = matches["appid"].apply(get_store_url)
    return jsonify(matches[["appid", "name", "price", "store_url"]].to_dict("records"))


@app.route("/api/export")
def export_csv():
    df = load_and_filter(use_revenue=False)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="export.csv"
    )

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

# ----------------- Serve frontend -----------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ----------------- Run -----------------
if __name__ == "__main__":
    status = "found" if os.path.exists(CSV_PATH) else "missing"
    print("=" * 60)
    print("Starting Mini Game Analytics API")
    print(f"Dataset: {CSV_PATH} → {status}")
    print(f"Running on: http://127.0.0.1:{PORT}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=True)