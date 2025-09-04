# app.py — 도시가스 공급·판매 분석 (Poly-3)
# 분석 유형: ① 공급량 분석(기온↔공급량) ② 판매량 분석(냉방용, 검침기간 평균기온)
# 반영사항:
#  - 기온 RAW 엑셀: 헤더(열 제목) 자동 추정해서 '날짜'·'평균기온(℃)/기온' 잡기
#  - 한글 폰트 강제 적용(로컬 fonts/ 및 OS 폰트 탐색)
#  - 모든 시계열 그래프 X축을 1월~12월 표기로 표시
#  - 예측 시작/종료 '연·월' UI 유지, 연도 표시는 콤마 없이

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# ───────── 기본 페이지 ─────────
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): 검침기간 평균기온 기반")

# Matplotlib 캐시(권한 이슈 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ───────── 한글 폰트(강제) ─────────
def set_korean_font_strict():
    # 1) 로컬 리포 내 폰트 우선
    local_candidates = [
        "assets/fonts/NanumGothic.ttf",
        "assets/fonts/NotoSansKR-Regular.otf",
        "fonts/NanumGothic.ttf",
        "fonts/NotoSansKR-Regular.otf",
    ]
    # 2) OS 폰트
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
    if chosen is None:
        # 폰트를 못 찾으면 시스템 기본(영문)이라도 설정
        chosen = plt.rcParams.get("font.family", ["DejaVu Sans"])[0]
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font_strict()

# ───────── 공통 유틸 ─────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]

KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜 열 생성
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
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month

    # 숫자형 변환(콤마/공백 제거)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore"
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
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

# ▶ 기온 RAW 전용: 헤더 자동 추정
@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        # 다양한 표기 대응: 평균기온(℃) → 평균기온
        rename_map = {}
        for c in df.columns:
            if "평균기온" in c:
                rename_map[c] = "평균기온"
            elif c.strip().lower() in ["temp", "temperature"]:
                rename_map[c] = "평균기온"
        if rename_map:
            df = df.rename(columns=rename_map)
        # 날짜/기온 열 찾기
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜","일자","date"]:
                date_col = c; break
        if date_col is None:
            # 날짜처럼 보이는 첫 번째 열 사용 시도
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c; break
                except Exception:
                    continue
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c).replace("(℃)","")) or (str(c).lower() in ["temp","temperature"]):
                temp_col = c; break
        if date_col is None or temp_col is None:
            return None
        out = pd.DataFrame({
            "일자": pd.to_datetime(df[date_col], errors="coerce"),
            "기온": pd.to_numeric(df[temp_col], errors="coerce")
        }).dropna()
        return out.sort_values("일자").reset_index(drop=True)

    name = getattr(file, "name", str(file))
    if name.lower().endswith(".csv"):
        raw = pd.read_csv(file)
        done = _finalize(raw)
        return done

    # Excel: 헤더 자동탐지
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    # 먼저 헤더 없이 읽고(상위 50행만 스캔)
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row = None
    for i in range(len(head)):
        row_vals = [str(v) for v in head.iloc[i].tolist()]
        has_date = any(v in ["날짜", "일자", "date", "Date"] for v in row_vals)
        has_temp = any(("평균기온" in v) or ("기온" in v) or (v.lower() in ["temp","temperature"]) for v in row_vals if isinstance(v, str))
        if has_date and has_temp:
            header_row = i
            break
    if header_row is None:
        # 헤더가 못 잡히면 기본 읽기
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

def month_start(x) -> pd.Timestamp:
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(start_m: pd.Timestamp, end_m: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=month_start(start_m), end=month_start(end_m), freq="MS")

def fit_poly3_and_predict(x_train: np.ndarray, y_train: np.ndarray, x_future: np.ndarray):
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

# X축을 1~12월로 표시
def apply_month_ticks(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)

# ───────── 좌측바: 분석 유형 ─────────
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
            if repo_files:
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트 사용", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df) == 0:
            st.stop()

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
        default_end   = last_hist + pd.DateOffset(months=12)
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

    # 전처리
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
    train_df = df[df["연"].isin(sel_years)].copy()
    if len(train_df) < 6: st.warning("학습 샘플이 적습니다. 연도를 더 선택하세요.")

    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

    # 예측 타임라인(연·월)
    future_index = month_range_inclusive(forecast_start, forecast_end)
    future_df = pd.DataFrame({"연": future_index.year.astype(int), "월": future_index.month.astype(int)})
    # temp merge
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

    # Δ°C 보정 & 결측 보강
    fallback_monthly = df.groupby("월")[temp_col].mean()
    future_df["temp"] = (future_df["temp"] + delta).fillna(future_df["월"].map(monthly_avg_temp["temp"])).fillna(future_df["월"].map(fallback_monthly))

    # 학습 + 예측
    results = []
    x_train = train_df[temp_col].astype(float).values
    x_future = future_df["temp"].astype(float).values

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

        # 차트(과거 vs 예측)
        hist = df[["연","월", col]].copy()
        hist["YM"] = pd.to_datetime(hist["연"].astype(int).astype(str) + "-" + hist["월"].astype(int).astype(str) + "-01")
        pred = tmp.copy()
        pred["YM"] = pd.to_datetime(pred["연"].astype(int).astype(str) + "-" + pred["월"].astype(int).astype(str) + "-01")
        merged = pd.merge(hist[["YM", col]].rename(columns={col:"실제"}), pred[["YM","예측공급량"]].rename(columns={"예측공급량":"예측"}), on="YM", how="outer").sort_values("YM")

        fig = plt.figure(figsize=(9, 3.6))
        ax = plt.gca()
        ax.plot(merged["YM"], merged["실제"], label=f"{col} 실제")
        ax.plot(merged["YM"], merged["예측"], linestyle="--", label=f"{col} 예측")
        apply_month_ticks(ax)
        ax.set_title(f"{col} — Poly-3 Forecast (Train R²={r2:.3f})")
        ax.set_xlabel("월"); ax.set_ylabel("공급량"); ax.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    if not results:
        st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다."); st.stop()

    pred_df = pd.concat(results, ignore_index=True)
    pred_pivot = pred_df.pivot_table(index=["연","월"], columns="상품", values="예측공급량").reset_index()

    # 총공급량: 별도 모델
    if "총공급량" in df.columns and pd.api.types.is_numeric_dtype(df["총공급량"]):
        y_total_train = train_df["총공급량"].astype(float).values
        y_tot, r2_tot = fit_poly3_and_predict(x_train, y_total_train, x_future)
        pred_pivot["총공급량"] = np.clip(np.rint(y_tot).astype(np.int64), a_min=0, a_max=None)
    else:
        st.warning("원본 데이터에 '총공급량' 열이 없어 총량 예측을 표시하지 않습니다.")

    # 표: 연은 문자열(콤마 방지), 나머지는 정수
    pred_pivot["연"] = pred_pivot["연"].astype(int).astype(str)
    pred_pivot["월"] = pred_pivot["월"].astype("Int64")
    for c in pred_pivot.columns:
        if c not in ["연","월"] and (pd.api.types.is_float_dtype(pred_pivot[c]) or pd.api.types.is_integer_dtype(pred_pivot[c])):
            pred_pivot[c] = pd.to_numeric(pred_pivot[c], errors="coerce").round().astype("Int64")

    st.subheader("예측 결과 미리보기")
    st.dataframe(pred_pivot.head(24), use_container_width=True, column_config={"연": st.column_config.TextColumn("연")})

    st.download_button("예측 결과 CSV 다운로드", data=pred_pivot.to_csv(index=False).encode("utf-8-sig"),
                       file_name="citygas_supply_forecast.csv", mime="text/csv")

# ======================================================================
# B) 판매량 분석(냉방용)
# ======================================================================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
    st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 업로드하세요. 판매월의 평균기온은 *(전월16일~당월15일)* 창구 평균으로 계산합니다.")

    c1, c2 = st.columns(2)
    with c1:
        sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)** 업로드 (가능하면 시트명 '냉방용')", type=["xlsx"])
    with c2:
        temp_raw_file = st.file_uploader("**기온 RAW(일별)** 업로드 (xlsx/csv)", type=["xlsx", "csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 업로드하세요."); st.stop()

    # 판매 실적(시트 '냉방용' 우선)
    try:
        xls_s = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls_s.sheet_names else xls_s.sheet_names[0]
        sales_df = pd.read_excel(xls_s, sheet_name=sheet)
    except Exception:
        sales_df = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(sales_df)

    # 판매월/판매량 컬럼 자동 후보 + 선택 UI
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

    # ▶ 기온 RAW: 헤더 자동 추정 사용
    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 컬럼을 찾지 못했습니다. (예: 날짜, 평균기온)"); st.stop()

    # 학습 연도 선택(UI)
    st.subheader("학습 데이터 연도 선택")
    sales_years = sorted(sales_df["판매월"].dt.year.unique().astype(int))
    sel_years = st.multiselect("연도 선택", options=sales_years, default=sales_years)

    # 검침기간 평균기온 함수: 판매월 m → (m-1월16일 ~ m월15일)
    def billing_period_avg_temp(daily_df: pd.DataFrame, label_month: pd.Timestamp) -> tuple[float, pd.Timestamp, pd.Timestamp]:
        m = month_start(label_month)
        start = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
        end   = m + pd.DateOffset(days=14)                                # 당월15
        mask = (daily_df["일자"] >= start) & (daily_df["일자"] <= end)
        return daily_df.loc[mask, "기온"].mean(), start, end

    # 학습 데이터 조인(선택 연도만)
    train_sales = sales_df[sales_df["판매월"].dt.year.isin(sel_years)].copy()
    rows = []
    for m in train_sales["판매월"].unique():
        avg, s, e = billing_period_avg_temp(temp_raw, m)
        rows.append({"판매월": m, "기간시작": s, "기간끝": e, "기간평균기온": avg})
    period_df = pd.DataFrame(rows)
    sales_join = pd.merge(train_sales[["판매월","판매량"]], period_df, on="판매월", how="left").dropna(subset=["기간평균기온","판매량"])

    if len(sales_join) < 6:
        st.warning("학습에 사용할 샘플이 적습니다. 판매월 데이터를 더 제공하세요."); st.stop()

    # 회귀 학습
    x_train = sales_join["기간평균기온"].astype(float).values
    y_train = sales_join["판매량"].astype(float).values
    _fit, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)

    # 예측 범위(연·월)
    st.subheader("예측 설정")
    default_sale_start = (sales_df["판매월"].max() + pd.offsets.MonthBegin(1))
    default_sale_end   = default_sale_start + pd.DateOffset(months=5)
    sale_start = ym_picker("예측 시작(판매월)", default_sale_start)
    sale_end   = ym_picker("예측 종료(판매월)", default_sale_end)
    if sale_end < sale_start:
        st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

    months = month_range_inclusive(sale_start, sale_end)
    pred_rows = []
    for m in months:
        avg, s, e = billing_period_avg_temp(temp_raw, m)
        pred_rows.append({"연": int(m.year), "월": int(m.month), "기간시작": s, "기간끝": e, "기간평균기온": avg})
    pred_input = pd.DataFrame(pred_rows)

    x_future = pred_input["기간평균기온"].astype(float).values
    y_future, r2_model = fit_poly3_and_predict(x_train, y_train, x_future)
    pred_input["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

    # 과거 구간 비교(있으면)
    compare = pd.merge(
        sales_join.assign(연=lambda d: d["판매월"].dt.year.astype(int), 월=lambda d: d["판매월"].dt.month.astype(int))[["연","월","판매량"]],
        pred_input[["연","월","기간평균기온","예측판매량"]],
        on=["연","월"], how="right"
    )
    compare["오차"] = (compare["예측판매량"] - compare["판매량"]).astype("Int64")

    # 표시용: 연은 문자열(콤마 방지)
    compare["연"] = compare["연"].astype(int).astype(str)
    compare["월"] = compare["월"].astype("Int64")

    st.subheader("냉방용 판매량 결과")
    st.dataframe(compare[["연","월","기간평균기온","판매량","예측판매량","오차"]], use_container_width=True, column_config={"연": st.column_config.TextColumn("연")})

    st.download_button(
        "판매량 결과 CSV 다운로드",
        data=compare[["연","월","기간평균기온","판매량","예측판매량","오차"]].to_csv(index=False).encode("utf-8-sig"),
        file_name="cooling_sales_analysis.csv",
        mime="text/csv",
    )

    # 차트(실제 vs 예측) — X축 월 표기
    chart_df = compare.copy()
    chart_df["YM"] = pd.to_datetime(chart_df["연"].astype(str) + "-" + chart_df["월"].astype(str) + "-01")
    fig = plt.figure(figsize=(9, 3.6))
    ax = plt.gca()
    ax.plot(chart_df["YM"], chart_df["판매량"], label="실제 판매량")
    ax.plot(chart_df["YM"], chart_df["예측판매량"], linestyle="--", label="예측 판매량")
    apply_month_ticks(ax)
    ax.set_title(f"냉방용 판매량 — Poly-3 (Train R²={r2_fit:.3f})")
    ax.set_xlabel("월"); ax.set_ylabel("판매량"); ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
