# app.py — 도시가스 공급·판매 분석 (Poly-3)
# 요구 반영:
# - 폰트: NanumGothic(동봉) → 시스템 후보(맑은고딕/애플고딕/Noto CJK) → 안내
# - 예측 설정: 시작/종료 연·월(기본 1~12), '예측 시작' 버튼 누르면만 실행
# - 그래프: 최근 5개년 실적 + 예측 1개(총 6개), X축=1~12월, Y축='양'
# - 판매량(냉방용): 파일 자동매핑(날짜 추정, '냉방' 포함 수치열 자동 선택. '취사용' 자동 제외)
# - 검침기간 평균기온=(m-1월 16일 ~ m월 15일) 명시 표기
# - 표: 가운데 정렬, '당월평균기온' & '기간평균기온(m-1월16~m15)' 동시 표시, 소수 1자리, 천단위

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# ── 기본 ─────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── 한글 폰트(강제) ──────────────────────────────────────────
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    local_fonts = [
        here / "fonts" / "NanumGothic.ttf",
        here / "assets" / "fonts" / "NanumGothic.ttf",
        here / "assets" / "fonts" / "NotoSansKR-Regular.otf",
    ]
    system_fonts = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    chosen = None
    for p in local_fonts + system_fonts:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                chosen = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                break
        except Exception:
            continue
    families = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    if chosen: families.insert(0, chosen)
    plt.rcParams["font.family"] = families
    plt.rcParams["font.sans-serif"] = families
    plt.rcParams["axes.unicode_minus"] = False
    return chosen is not None

font_ok = set_korean_font()
if not font_ok:
    st.warning("한글 폰트를 찾지 못했습니다. 프로젝트 루트에 fonts/NanumGothic.ttf를 두면 가장 안정적으로 표시됩니다.")

# ── 공통 유틸 ────────────────────────────────────────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
KNOWN_PRODUCT_ORDER = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False), errors="ignore")
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others  = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet: str = "데이터"):
    try:
        xls = pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {}
        for c in df.columns:
            if "평균기온" in c or str(c).strip().lower() in ["temp","temperature"]:
                rename_map[c] = "평균기온"
        if rename_map: df = df.rename(columns=rename_map)
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜","일자","date"]:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise"); date_col = c; break
                except Exception:
                    continue
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c).replace("(℃)","")) or (str(c).lower() in ["temp","temperature"]):
                temp_col = c; break
        if date_col is None or temp_col is None: return None
        out = pd.DataFrame({
            "일자": pd.to_datetime(df[date_col], errors="coerce"),
            "기온": pd.to_numeric(df[temp_col], errors="coerce"),
        }).dropna()
        return out.sort_values("일자").reset_index(drop=True)

    name = getattr(file, "name", str(file))
    if name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row = None
    for i in range(len(head)):
        row_vals = [str(v) for v in head.iloc[i].tolist()]
        has_date = any(v in ["날짜","일자","date","Date"] for v in row_vals)
        has_temp = any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row_vals)
        if has_date and has_temp: header_row = i; break
    df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

def month_start(x) -> pd.Timestamp:
    x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(start_m: pd.Timestamp, end_m: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=month_start(start_m), end=month_start(end_m), freq="MS")

def fit_poly3_and_predict(x_train: np.ndarray, y_train: np.ndarray, x_future: np.ndarray):
    mask = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train = x_train[mask]; y_train = y_train[mask]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측(기간평균기온)이 포함되어 있습니다.")
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2

def ym_picker_block(label_prefix: str, default_year: int, default_month: int):
    c1, c2 = st.columns(2)
    with c1:
        year = st.selectbox(f"{label_prefix} — 연", options=list(range(2010, 2036)), index=list(range(2010,2036)).index(int(default_year)))
    with c2:
        month = st.selectbox(f"{label_prefix} — 월", options=list(range(1, 13)), index=int(default_month)-1)
    return int(year), int(month)

def plot_recent5_plus_forecast(ax, hist_df, year_col, month_col, value_col,
                               forecast_df, forecast_start, forecast_value_col,
                               title):
    years_all = sorted(pd.Series(hist_df[year_col].dropna().astype(int).unique()).tolist())
    recent5 = years_all[-5:] if len(years_all) >= 5 else years_all
    months = list(range(1, 13))
    for y in recent5:
        s = (hist_df[hist_df[year_col] == y].set_index(month_col)[value_col]).reindex(months)
        ax.plot(months, s.values, label=f"{y} 실적")
    pred_vals = []
    y, m = int(forecast_start.year), int(forecast_start.month)
    fdf = forecast_df.copy()
    if "연" in fdf.columns: fdf["연"] = fdf["연"].astype(int)
    if "월" in fdf.columns: fdf["월"] = fdf["월"].astype(int)
    for _ in range(12):
        row = fdf[(fdf["연"] == y) & (fdf["월"] == m)]
        pred_vals.append(row.iloc[0][forecast_value_col] if len(row) else np.nan)
        if m == 12: y += 1; m = 1
        else: m += 1
    ax.plot(months, pred_vals, linestyle="--", label="예측")
    ax.set_xlim(1, 12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
    ax.set_xlabel("월"); ax.set_ylabel("양")
    ax.set_title(title); ax.legend(loc="best")

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    """HTML로 가운데 정렬 + 형식 지정 표 출력"""
    float1_cols = float1_cols or []
    int_cols = int_cols or []
    show = df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round().astype("Int64").map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    # 중앙 정렬 CSS
    st.markdown("""
    <style>
      table.centered-table {width:100%;}
      table.centered-table th, table.centered-table td { text-align:center !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

# ── 좌측바: 분석 유형 ───────────────────────────────────────
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# ======================================================================
# A) 공급량 분석
# ======================================================================
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        data_src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)
        df = None
        if data_src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if len(repo_files) == 0:
                st.info("data 폴더에 엑셀 파일이 없습니다. 업로드 탭을 사용하세요.")
            default_idx = next((i for i, p in enumerate(repo_files) if "상품별공급량_MJ" in p), 0) if repo_files else 0
            file_choice = st.selectbox("실적 파일(Excel)", repo_files if repo_files else ["<None>"], index=default_idx if repo_files else 0)
            if repo_files: df = read_excel_sheet(file_choice, prefer_sheet="데이터")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트 사용", type=["xlsx"])
            if up is not None: df = read_excel_sheet(up, prefer_sheet="데이터")
        if df is None or len(df) == 0: st.stop()

        st.subheader("학습 데이터 연도 선택")
        years = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        sel_years = st.multiselect("연도 선택", years, default=years)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 컬럼을 자동으로 찾지 못했습니다. 엑셀 기온 열 이름에 '평균기온' 또는 '기온'을 포함시켜 주세요.")
            st.stop()

        st.subheader("예측할 상품 선택")
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        sel_products = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

        st.subheader("예측 설정")
        last_year = int(df["연"].max())
        y1, m1 = ym_picker_block("예측 시작(연·월)", default_year=last_year, default_month=1)   # 기본 1월
        y2, m2 = ym_picker_block("예측 종료(연·월)", default_year=last_year, default_month=12)  # 기본 12월
        scen = st.radio("기온 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사", "사용자 업로드(월·기온)"], index=0)
        delta = st.slider("기온 보정(Δ°C)", -5.0, 5.0, 0.0, step=0.1)
        scen_df = None
        if scen == "사용자 업로드(월·기온)":
            up_scen = st.file_uploader("CSV/XLSX 업로드 (열: 월, 기온 또는 month, temp)", type=["csv","xlsx"], key="temp_scen_upload")
            if up_scen is not None:
                scen_df = pd.read_csv(up_scen) if up_scen.name.lower().endswith(".csv") else pd.read_excel(up_scen)
                scen_df.columns = [str(c).strip().lower() for c in scen_df.columns]
                if "월" in scen_df.columns: scen_df["month"] = scen_df["월"]
                if "기온" in scen_df.columns: scen_df["temp"] = scen_df["기온"]
        run_forecast = st.button("예측 시작", type="primary")

    if not run_forecast:
        st.info("좌측에서 연·월을 선택하고 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    # 전처리/학습
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
    train_df = df[df["연"].isin(sel_years)].copy()

    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

    forecast_start = pd.Timestamp(year=y1, month=m1, day=1)
    forecast_end   = pd.Timestamp(year=y2, month=m2, day=1)
    if forecast_end < forecast_start:
        st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

    future_index = month_range_inclusive(forecast_start, forecast_end)
    future_df = pd.DataFrame({"연": future_index.year.astype(int), "월": future_index.month.astype(int)})

    if scen == "학습기간 월별 평균":
        future_df = future_df.merge(monthly_avg_temp.reset_index(), on="월", how="left")
    elif scen == "학습 마지막해 월별 복사":
        last_train_year = int(train_df["연"].max()) if len(train_df) else int(df["연"].max())
        base = df[df["연"] == last_train_year][["월", temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
        future_df = future_df.merge(base, on="월", how="left")
    else:
        if scen_df is None:
            st.error("월·기온 시나리오 파일을 올려주세요."); st.stop()
        scen_df["month"] = pd.to_numeric(scen_df["month"], errors="coerce").astype(int)
        scen_df = scen_df[["month","temp"]].dropna()
        future_df = future_df.merge(scen_df, left_on="월", right_on="month", how="left").drop(columns=["month"])

    fallback_monthly = df.groupby("월")[temp_col].mean()
    future_df["temp"] = (future_df["temp"] + delta)\
        .fillna(future_df["월"].map(monthly_avg_temp["temp"]))\
        .fillna(future_df["월"].map(fallback_monthly))

    # 예측 + 그래프
    results = []
    x_train = train_df[temp_col].astype(float).values
    x_future = future_df["temp"].astype(float).values
    if np.isnan(x_future).any():
        st.error("예측 기간의 기온 시나리오에 결측이 있습니다."); st.stop()

    for col in sel_products:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]): 
            continue
        y_train = train_df[col].astype(float).values
        y_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
        y_future = np.rint(y_future).astype(np.int64)

        tmp = future_df[["연","월"]].copy()
        tmp["상품"] = col
        tmp["예측공급량"] = np.clip(y_future, a_min=0, a_max=None)
        tmp["R2(Train)"] = r2
        results.append(tmp)

        hist = df[["연","월", col]].copy()
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
        fig = plt.figure(figsize=(9, 3.6)); ax = plt.gca()
        plot_recent5_plus_forecast(
            ax=ax,
            hist_df=hist.rename(columns={col: "val"}),
            year_col="연", month_col="월", value_col="val",
            forecast_df=tmp.rename(columns={"예측공급량":"pred"}),
            forecast_start=forecast_start, forecast_value_col="pred",
            title=f"{col} — Poly-3 (Train R²={r2:.3f})"
        )
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    if not results:
        st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다."); st.stop()

    pred_df = pd.concat(results, ignore_index=True)
    pred_pivot = pred_df.pivot_table(index=["연","월"], columns="상품", values="예측공급량").reset_index()
    if "연" in pred_pivot.columns:
        pred_pivot["연"] = pred_pivot["연"].astype(int).astype(str)   # 오타 방지: astype(str)
    if "월" in pred_pivot.columns:
        pred_pivot["월"] = pred_pivot["월"].astype("Int64")
    for c in pred_pivot.columns:
        if c not in ["연","월"] and (pd.api.types.is_float_dtype(pred_pivot[c]) or pd.api.types.is_integer_dtype(pred_pivot[c])):
            pred_pivot[c] = pd.to_numeric(pred_pivot[c], errors="coerce").round().astype("Int64")

    st.subheader("예측 결과 미리보기")
    render_centered_table(pred_pivot.head(24), int_cols=[c for c in pred_pivot.columns if c not in ["연","월"]])

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=pred_pivot.to_csv(index=False).encode("utf-8-sig"),
        file_name="citygas_supply_forecast.csv",
        mime="text/csv",
    )

# ======================================================================
# B) 판매량 분석(냉방용)
# ======================================================================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
    st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 업로드하세요. 판매월의 평균기온은 *(전월16일~당월15일)* 창구 평균으로 계산합니다.")

    c1, c2 = st.columns(2)
    with c1:
        sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
    with c2:
        temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 업로드하세요."); st.stop()

    # 판매 실적: 자동 매핑 (날짜/냉방)
    try:
        xls_s = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls_s.sheet_names else xls_s.sheet_names[0]
        raw_sales = pd.read_excel(xls_s, sheet_name=sheet)
    except Exception:
        raw_sales = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(raw_sales)

    # 날짜 컬럼 추정
    date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
    if date_candidates:
        date_col = date_candidates[0]
    else:
        # 파싱 성공 비율이 가장 높은 열
        parse_scores = {}
        for c in sales_df.columns:
            try:
                s = pd.to_datetime(sales_df[c], errors="coerce")
                parse_scores[c] = s.notna().mean()
            except Exception:
                continue
        date_col = max(parse_scores, key=parse_scores.get) if parse_scores else None
    # 냉방 수치 열 추정
    cool_cols = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    if date_col is None or not cool_cols:
        st.error("판매 실적에서 날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다. 파일의 열 이름을 확인하세요."); st.stop()
    value_col = None
    # '냉방용' 우선
    for c in cool_cols:
        if "냉방용" in str(c): value_col = c; break
    value_col = value_col or cool_cols[0]

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    # 기온 RAW
    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 컬럼을 찾지 못했습니다."); st.stop()

    # 사이드바: 학습/예측 + 버튼
    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        sales_years = sorted(sales_df["연"].unique().tolist())
        sel_years = st.multiselect("연도 선택", options=sales_years, default=sales_years)

        st.subheader("예측 설정")
        last_year = int(sales_df["연"].max())
        y1, m1 = ym_picker_block("예측 시작(연·월)", default_year=last_year, default_month=1)
        y2, m2 = ym_picker_block("예측 종료(연·월)", default_year=last_year, default_month=12)
        run_forecast = st.button("예측 시작", type="primary")

    if not run_forecast:
        st.info("좌측에서 연·월을 선택하고 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    # 평균기온 계산(정의 고정)
    def billing_period_bounds(label_month: pd.Timestamp):
        m = month_start(label_month)
        start = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
        end   = m + pd.DateOffset(days=14)                                # 당월15
        return start, end

    def billing_period_avg_temp(daily_df: pd.DataFrame, label_month: pd.Timestamp) -> float:
        start, end = billing_period_bounds(label_month)
        mask = (daily_df["일자"] >= start) & (daily_df["일자"] <= end)
        return daily_df.loc[mask, "기온"].mean()

    # 달력월 평균(당월평균) & 동월 평균(보완용)
    temp_raw["연"] = temp_raw["일자"].dt.year
    temp_raw["월"] = temp_raw["일자"].dt.month
    monthly_temp_calendar = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
    fallback_by_M = temp_raw.groupby("월")["기온"].mean()

    # 학습 조인
    train_sales = sales_df[sales_df["연"].isin(sel_years)].copy()
    rows = []
    for m in train_sales["판매월"].unique():
        rows.append({"판매월": m, "기간평균기온": billing_period_avg_temp(temp_raw, m)})
    sales_join = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
    miss = sales_join["기간평균기온"].isna()
    if miss.any():
        sales_join.loc[miss, "기간평균기온"] = sales_join.loc[miss, "판매월"].dt.month.map(fallback_by_M)
    sales_join = sales_join.dropna(subset=["기간평균기온","판매량"])
    if len(sales_join) < 6:
        st.error("학습 샘플이 부족합니다. 기온 RAW 기간을 늘리거나 학습연도를 조정하세요."); st.stop()

    x_train = sales_join["기간평균기온"].astype(float).values
    y_train = sales_join["판매량"].astype(float).values
    _tmp, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)

    # 예측 입력
    forecast_start = pd.Timestamp(year=y1, month=m1, day=1)
    forecast_end   = pd.Timestamp(year=y2, month=m2, day=1)
    if forecast_end < forecast_start:
        st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

    months = month_range_inclusive(forecast_start, forecast_end)
    pred_rows = []
    for m in months:
        avg_period = billing_period_avg_temp(temp_raw, m)
        avg_month = monthly_temp_calendar.loc[(monthly_temp_calendar["연"]==m.year)&(monthly_temp_calendar["월"]==m.month), "기온"].mean()
        pred_rows.append({"연": int(m.year), "월": int(m.month), "기간평균기온": avg_period, "당월평균기온": avg_month})
    pred_input = pd.DataFrame(pred_rows)
    for col in ["기간평균기온","당월평균기온"]:
        miss = pred_input[col].isna()
        if miss.any():
            pred_input.loc[miss, col] = pred_input.loc[miss, "월"].map(fallback_by_M)
    pred_input = pred_input.dropna(subset=["기간평균기온"]).reset_index(drop=True)

    x_future = pred_input["기간평균기온"].astype(float).values
    y_future, _ = fit_poly3_and_predict(x_train, y_train, x_future)
    pred_input["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

    # 적용기온 안내(예: 첫 예측월)
    eg = forecast_start
    s, e = billing_period_bounds(eg)
    st.caption(f"적용 기온 정의: **전월 16일 ~ 당월 15일 평균**. 예) {eg.strftime('%Y-%m')} → {s.strftime('%Y-%m-%d')} ~ {e.strftime('%Y-%m-%d')}")

    # 그래프
    fig = plt.figure(figsize=(9, 3.6)); ax = plt.gca()
    plot_recent5_plus_forecast(
        ax=ax,
        hist_df=sales_df.rename(columns={"판매량":"val"}),
        year_col="연", month_col="월", value_col="val",
        forecast_df=pred_input.rename(columns={"예측판매량":"pred"}),
        forecast_start=forecast_start, forecast_value_col="pred",
        title=f"냉방용 판매량 — Poly-3 (Train R²={r2_fit:.3f})"
    )
    plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # 표(가운데 정렬 + 형식)
    out = pred_input.copy()
    out["연"] = out["연"].astype(int).astype(str)
    out["월"] = out["월"].astype("Int64")
    out = out.rename(columns={"기간평균기온":"기간평균기온(m-1월16~m15)"})
    st.subheader("판매량 예측(요약)")
    render_centered_table(
        out[["연","월","당월평균기온","기간평균기온(m-1월16~m15)","예측판매량"]],
        float1_cols=["당월평균기온","기간평균기온(m-1월16~m15)"],
        int_cols=["예측판매량"],
        index=False
    )
    st.download_button(
        "판매량 예측 CSV 다운로드",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="cooling_sales_forecast.csv",
        mime="text/csv",
    )
