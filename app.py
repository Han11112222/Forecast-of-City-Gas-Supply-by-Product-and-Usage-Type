# app.py — 도시가스 공급·판매 분석 (Poly-3)
# 분석 유형: ① 공급량 분석(기온↔공급량) ② 판매량 분석(냉방용, 검침기간 평균기온)
# 변경 요약:
#  - 판매량 분석도 사이드바에 "예측 설정(연/월)" + "학습 데이터 연도 선택" 배치
#  - 최근 5개년 실적 + 예측 1개 = 총 6개 라인 (X축: 1~12월, Y축: '양')
#  - 기온 RAW 헤더 자동탐지, 한글 폰트 강제 적용
#  - 예측 구간의 기간평균기온 결측을 동월 평균으로 보완, 그래도 없으면 해당 월 제외

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

# ─────────────────────────────────────────────────────────────
# 기본 세팅
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): 검침기간 평균기온 기반")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─────────────────────────────────────────────────────────────
# 한글 폰트(강제)
# ─────────────────────────────────────────────────────────────
def set_korean_font_strict():
    local_candidates = [
        "assets/fonts/NanumGothic.ttf",
        "assets/fonts/NotoSansKR-Regular.otf",
        "fonts/NanumGothic.ttf",
        "fonts/NotoSansKR-Regular.otf",
    ]
    system_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    chosen = None
    for p in local_candidates + system_candidates:
        try:
            if os.path.exists(p):
                mpl.font_manager.fontManager.addfont(p)
                chosen = mpl.font_manager.FontProperties(fname=p).get_name()
                break
        except Exception:
            pass
    plt.rcParams["font.family"] = chosen or "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font_strict()

# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
KNOWN_PRODUCT_ORDER = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜 생성
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
    # 연/월 파생
    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    # 문자열 숫자화
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False), errors="ignore")
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]): return c
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

# 기온 RAW 전용(헤더 자동탐지)
@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        # 평균기온 명칭 통일
        rename_map = {}
        for c in df.columns:
            if "평균기온" in c or str(c).strip().lower() in ["temp","temperature"]:
                rename_map[c] = "평균기온"
        if rename_map: df = df.rename(columns=rename_map)
        # 날짜/기온 열 확정
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
    # 입력 방어: NaN 제거 & x/y 길이 정합
    mask = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train = x_train[mask]; y_train = y_train[mask]
    # 미래 입력 NaN 방어(예측 전 확인)
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

def ym_picker(label: str, default: pd.Timestamp):
    c1, c2 = st.columns(2)
    with c1:
        year = st.selectbox(f"{label} — 연", options=list(range(2010, 2036)), index=list(range(2010,2036)).index(int(default.year)))
    with c2:
        month = st.selectbox(f"{label} — 월", options=list(range(1, 13)), index=int(default.month)-1)
    return pd.Timestamp(year=int(year), month=int(month), day=1)

# 최근 5개년 실적 + 예측 12개월 출력 유틸 (X축 1~12월)
def plot_recent5_plus_forecast(ax, hist_df, year_col, month_col, value_col,
                               forecast_df, forecast_start, forecast_value_col,
                               title):
    years_all = sorted(pd.Series(hist_df[year_col].dropna().astype(int).unique()).tolist())
    recent5 = years_all[-5:] if len(years_all) >= 5 else years_all
    months = list(range(1, 13))
    # 실적 5개년
    for y in recent5:
        s = (hist_df[hist_df[year_col] == y]
             .set_index(month_col)[value_col]
             .reindex(months))
        ax.plot(months, s.values, label=f"{y} 실적")
    # 예측(시작월부터 12개월)
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
    # 축/레이블
    ax.set_xlim(1, 12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
    ax.set_xlabel("월"); ax.set_ylabel("양")
    ax.set_title(title); ax.legend(loc="best")

# ─────────────────────────────────────────────────────────────
# 좌측바: 분석 유형
# ─────────────────────────────────────────────────────────────
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

        # 학습 연도
        st.subheader("학습 데이터 연도 선택")
        years = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        sel_years = st.multiselect("연도 선택", years, default=years)

        # 기온 자동탐지
        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 컬럼을 자동으로 찾지 못했습니다. 엑셀 기온 열 이름에 '평균기온' 또는 '기온'을 포함시켜 주세요.")
            st.stop()

        # 예측할 상품
        st.subheader("예측할 상품 선택")
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        sel_products = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

        # 예측 범위(연·월)
        st.subheader("예측 설정")
        last_hist = month_start(pd.to_datetime(df["날짜"].max()))
        default_start = last_hist + pd.offsets.MonthBegin(1)
        default_end   = last_hist + pd.DateOffset(months=11)  # 12개월
        forecast_start = ym_picker("예측 시작", default_start)
        forecast_end   = ym_picker("예측 종료", default_end)
        if forecast_end < forecast_start:
            st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

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
                if "month" not in scen_df.columns or "temp" not in scen_df.columns:
                    st.error("업로드 형식: '월/month'와 '기온/temp' 열이 필요합니다."); st.stop()

    # 전처리/학습 집계
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
    train_df = df[df["연"].isin(sel_years)].copy()
    if len(train_df) < 6: st.warning("학습 샘플이 적습니다. 연도를 더 선택하세요.")

    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

    # 예측 입력(연·월 + temp)
    future_index = month_range_inclusive(forecast_start, forecast_end)
    future_df = pd.DataFrame({"연": future_index.year.astype(int), "월": future_index.month.astype(int)})
    if scen == "학습기간 월별 평균":
        future_df = future_df.merge(monthly_avg_temp.reset_index(), on="월", how="left")
    elif scen == "학습 마지막해 월별 복사":
        last_train_year = int(train_df["연"].max()) if len(train_df) else int(df["연"].max())
        base = df[df["연"] == last_train_year][["월", temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
        future_df = future_df.merge(base, on="월", how="left")
    else:
        scen_df["month"] = pd.to_numeric(scen_df["month"], errors="coerce").astype(int)
        scen_df = scen_df[["month","temp"]].dropna()
        future_df = future_df.merge(scen_df, left_on="월", right_on="month", how="left").drop(columns=["month"])
    # 보정 + 결측 보강
    fallback_monthly = df.groupby("월")[temp_col].mean()
    future_df["temp"] = (future_df["temp"] + delta).fillna(future_df["월"].map(monthly_avg_temp["temp"])).fillna(future_df["월"].map(fallback_monthly))

    # 학습 + 예측 + 그래프(최근5 + 예측)
    results = []
    x_train = train_df[temp_col].astype(float).values
    x_future = future_df["temp"].astype(float).values
    if np.isnan(x_future).any():
        st.error("예측 기간의 기온 시나리오에 결측이 있습니다. 시나리오/기간을 조정하세요."); st.stop()

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

        # 그래프
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

    # 총공급량(별도 모델)
    if "총공급량" in df.columns and pd.api.types.is_numeric_dtype(df["총공급량"]):
        y_total_train = train_df["총공급량"].astype(float).values
        y_tot, r2_tot = fit_poly3_and_predict(x_train, y_total_train, x_future)
        pred_pivot["총공급량"] = np.clip(np.rint(y_tot).astype(np.int64), a_min=0, a_max=None)

    pred_pivot["연"] = pred_pivot["연"].astype(int).astype[str]
    pred_pivot["월"] = pred_pivot["월"].astype("Int64")
    for c in pred_pivot.columns:
        if c not in ["연","월"] and (pd.api.types.is_float_dtype(pred_pivot[c]) or pd.api.types.is_integer_dtype(pred_pivot[c])):
            pred_pivot[c] = pd.to_numeric(pred_pivot[c], errors="coerce").round().astype("Int64")

    st.subheader("예측 결과 미리보기")
    st.dataframe(pred_pivot.head(24), use_container_width=True, column_config={"연": st.column_config.TextColumn("연")})
    st.download_button("예측 결과 CSV 다운로드", pred_pivot.to_csv(index=False).encode("utf-8-sig"),
                       file_name="citygas_supply_forecast.csv", mime="text/csv")

# ======================================================================
# B) 판매량 분석(냉방용)
# ======================================================================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
    st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 업로드하세요. 판매월의 평균기온은 *(전월16일~당월15일)* 창구 평균으로 계산합니다.")

    # 파일 업로드
    c1, c2 = st.columns(2)
    with c1:
        sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)** (가능하면 시트명 '냉방용')", type=["xlsx"])
    with c2:
        temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 업로드하세요."); st.stop()

    # 판매 실적 로딩
    try:
        xls_s = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls_s.sheet_names else xls_s.sheet_names[0]
        sales_df = pd.read_excel(xls_s, sheet_name=sheet)
    except Exception:
        sales_df = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(sales_df)

    # 컬럼 매핑
    cand_month = [c for c in sales_df.columns if str(c) in ["날짜","판매월","월","date"] or "월" in str(c)]
    cand_value = [c for c in sales_df.columns if ("냉방" in str(c) or "판매" in str(c) or "사용" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    st.subheader("판매 실적 컬럼 매핑")
    s1, s2 = st.columns(2)
    with s1:
        sales_month_col = st.selectbox("판매월(날짜) 컬럼", options=cand_month if cand_month else list(sales_df.columns))
    with s2:
        sales_value_col = st.selectbox("냉방용 판매량 컬럼", options=cand_value if cand_value else list(sales_df.columns))

    sales_df["판매월"] = pd.to_datetime(sales_df[sales_month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[sales_value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    # 기온 RAW 로딩
    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 컬럼을 찾지 못했습니다. (예: 날짜, 평균기온)"); st.stop()

    # 사이드바: 학습연도 + 예측 설정 (공급량 분석과 동일 UI)
    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        sales_years = sorted(sales_df["연"].unique().tolist())
        sel_years = st.multiselect("연도 선택", options=sales_years, default=sales_years)

        st.subheader("예측 설정")
        default_sale_start = (sales_df["판매월"].max() + pd.offsets.MonthBegin(1))
        default_sale_end   = default_sale_start + pd.DateOffset(months=11)  # 12개월
        sale_start = ym_picker("예측 시작(판매월)", default_sale_start)
        sale_end   = ym_picker("예측 종료(판매월)", default_sale_end)
        if sale_end < sale_start:
            st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

    # 검침기간 평균기온 계산
    def billing_period_avg_temp(daily_df: pd.DataFrame, label_month: pd.Timestamp) -> float:
        m = month_start(label_month)
        start = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
        end   = m + pd.DateOffset(days=14)                                # 당월15
        mask = (daily_df["일자"] >= start) & (daily_df["일자"] <= end)
        return daily_df.loc[mask, "기온"].mean()

    # 학습 데이터 조인(선택 연도만)
    train_sales = sales_df[sales_df["연"].isin(sel_years)].copy()
    rows = []
    for m in train_sales["판매월"].unique():
        rows.append({"판매월": m, "기간평균기온": billing_period_avg_temp(temp_raw, m)})
    period_df = pd.DataFrame(rows)
    sales_join = pd.merge(train_sales[["판매월","판매량"]], period_df, on="판매월", how="left")

    # 동월 평균으로 결측 보완
    # 1) 일별 → 월 라벨(판매월) 기준 동월 평균 생성
    temp_raw["Y"] = temp_raw["일자"].dt.year
    temp_raw["M"] = temp_raw["일자"].dt.month
    monthly_temp = temp_raw.groupby("M")["기온"].mean()  # 동월 평균
    # 2) 결측 채우기
    miss_mask = sales_join["기간평균기온"].isna()
    if miss_mask.any():
        fill_vals = sales_join.loc[miss_mask, "판매월"].dt.month.map(monthly_temp)
        sales_join.loc[miss_mask, "기간평균기온"] = fill_vals

    sales_join = sales_join.dropna(subset=["기간평균기온","판매량"])
    if len(sales_join) < 6:
        st.error("학습 샘플이 부족합니다. 기온 RAW 기간을 늘리거나 학습연도를 조정하세요."); st.stop()

    # 회귀 학습
    x_train = sales_join["기간평균기온"].astype(float).values
    y_train = sales_join["판매량"].astype(float).values
    _tmp, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)  # R² 산출용

    # 예측 입력(12개월 권장)
    months = month_range_inclusive(sale_start, sale_end)
    pred_rows = []
    for m in months:
        avg = billing_period_avg_temp(temp_raw, m)
        pred_rows.append({"연": int(m.year), "월": int(m.month), "기간평균기온": avg})
    pred_input = pd.DataFrame(pred_rows)

    # 예측 입력 결측 보완(동월 평균), 그래도 없으면 제외
    miss_mask = pred_input["기간평균기온"].isna()
    if miss_mask.any():
        pred_input.loc[miss_mask, "기간평균기온"] = pred_input.loc[miss_mask, "월"].map(monthly_temp)
    dropped = pred_input[pred_input["기간평균기온"].isna()][["연","월"]].copy()
    pred_input = pred_input.dropna(subset=["기간평균기온"]).reset_index(drop=True)
    if len(pred_input) == 0:
        st.error("예측에 사용할 기간평균기온이 없습니다. 기온 RAW를 더 긴 구간으로 제공하세요."); st.stop()

    # 예측
    x_future = pred_input["기간평균기온"].astype(float).values
    y_future, r2_model = fit_poly3_and_predict(x_train, y_train, x_future)
    pred_input["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

    # 그래프: 최근 5개년 실적 + 예측(12개월)
    fig = plt.figure(figsize=(9, 3.6)); ax = plt.gca()
    plot_recent5_plus_forecast(
        ax=ax,
        hist_df=sales_df.rename(columns={"판매량":"val"}),
        year_col="연", month_col="월", value_col="val",
        forecast_df=pred_input.rename(columns={"예측판매량":"pred"}),
        forecast_start=sale_start, forecast_value_col="pred",
        title=f"냉방용 판매량 — Poly-3 (Train R²={r2_fit:.3f})"
    )
    plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # 안내: 제외된 월 표시
    if not dropped.empty:
        dropped["라벨"] = dropped["연"].astype(str) + "-" + dropped["월"].astype(str).str.zfill(2)
        st.warning("예측 입력에 기온 데이터가 부족했던 월을 제외했습니다: " + ", ".join(dropped["라벨"].tolist()))

    # 표/다운로드
    out_df = pred_input.copy()
    out_df["연"] = out_df["연"].astype(int).astype(str)
    out_df["월"] = out_df["월"].astype("Int64")
    st.subheader("판매량 예측(요약)")
    st.dataframe(out_df[["연","월","기간평균기온","예측판매량"]], use_container_width=True, column_config={"연": st.column_config.TextColumn("연")})
    st.download_button("판매량 예측 CSV 다운로드", out_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="cooling_sales_forecast.csv", mime="text/csv")
