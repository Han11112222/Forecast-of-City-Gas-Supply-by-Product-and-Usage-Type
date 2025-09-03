# app.py — Forecast of City Gas Supply by Product & Usage Type (Poly-3)
# 기능: 연도 선택, 예측 상품 선택, Repo 파일/업로드 선택, 3차 다항식(기온↔공급량) 예측

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 0) 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 상품별 공급량 예측 (Poly-3)", layout="wide")
st.title("Forecast of City Gas Supply by Product and Usage Type")
st.caption("3차 다항식(기온↔공급량) 기반 월별 예측 · 데이터 소스: Excel")

# Matplotlib 캐시(권한 이슈 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ─────────────────────────────────────────────────────────────
# 1) 한글 폰트(가능하면 적용)
# ─────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",           # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",    # Linux (Noto)
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",                # macOS
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                mpl.font_manager.fontManager.addfont(p)
                plt.rcParams["font.family"] = mpl.font_manager.FontProperties(fname=p).get_name()
                break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────────────────────
# 2) 유틸
# ─────────────────────────────────────────────────────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]

KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 생성/보정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # 연/월 -> 날짜
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")

    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month

    # 숫자형 변환(천단위 콤마 등 제거)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.replace(" ", ""), errors="ignore"
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # 메타컬럼 제외
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def load_repo_excel(path: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
    return normalize_cols(df)

@st.cache_data(ttl=600)
def load_uploaded_excel(file, sheet_name: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl", sheet_name=sheet_name)
    return normalize_cols(df)

def month_start_after(ts: pd.Timestamp) -> pd.Timestamp:
    y = ts.year + (1 if ts.month == 12 else 0)
    m = 1 if ts.month == 12 else ts.month + 1
    return pd.Timestamp(y, m, 1)

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

# ─────────────────────────────────────────────────────────────
# 3) 사이드바 — 데이터 소스/옵션
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 불러오기 방식")
    data_src = st.radio("방식 선택", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    df = None
    sheet_choice = None

    if data_src == "Repo 내 파일 사용":
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
        if len(repo_files) == 0:
            st.info("data 폴더에 엑셀 파일이 없습니다. 업로드 탭을 사용하세요.")
        # 기본: '상품별공급량_MJ'가 있으면 그걸 선택
        default_idx = 0
        for i, p in enumerate(repo_files):
            if "상품별공급량_MJ" in p:
                default_idx = i
                break
        file_choice = st.selectbox("실적 파일(Excel)", repo_files if repo_files else ["<None>"], index=default_idx if repo_files else 0)
        # 시트 선택(선택사항): 미리보기 위해 헤더만 읽어 시트 목록 획득
        if repo_files:
            try:
                xls = pd.ExcelFile(file_choice, engine="openpyxl")
                sheet_choice = st.selectbox("시트 선택", xls.sheet_names, index=0)
            except Exception as e:
                st.warning(f"시트 목록을 읽는 중 문제가 발생했습니다: {e}")
                sheet_choice = 0
            df = load_repo_excel(file_choice, sheet_name=sheet_choice)

    else:
        up = st.file_uploader("엑셀 파일 업로드 (xlsx)", type=["xlsx"])
        if up is not None:
            try:
                xls = pd.ExcelFile(up, engine="openpyxl")
                sheet_choice = st.selectbox("시트 선택", xls.sheet_names, index=0)
            except Exception as e:
                st.warning(f"시트 목록을 읽는 중 문제가 발생했습니다: {e}")
                sheet_choice = 0
            df = load_uploaded_excel(up, sheet_name=sheet_choice)

    if df is None or len(df) == 0:
        st.stop()

    # 학습 연도 선택
    st.subheader("학습 데이터 연도 선택")
    years = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
    sel_years = st.multiselect("연도 선택", years, default=years, help="필요 없는 연도는 ×로 제거하세요.")

    # 기온 컬럼 선택
    st.subheader("기온 컬럼 선택")
    temp_auto = detect_temp_col(df)
    if temp_auto and temp_auto in df.columns:
        temp_col = st.selectbox("기온(°C) 컬럼", [temp_auto] + [c for c in df.columns if c != temp_auto])
    else:
        temp_col = st.selectbox("기온(°C) 컬럼", list(df.columns))
    if not pd.api.types.is_numeric_dtype(df[temp_col]):
        st.error(f"선택한 기온 컬럼({temp_col})이 숫자형이 아닙니다.")
        st.stop()

    # 예측할 상품 선택
    st.subheader("예측할 상품 선택")
    product_cols = guess_product_cols(df)
    default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
    sel_products = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

    # 예측 범위 & 기온 시나리오
    st.subheader("예측 설정")
    horizon = st.slider("예측 개월 수", 3, 24, 12, step=1)
    scen = st.radio("기온 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사", "사용자 업로드(월·기온)"], index=0)
    delta = st.slider("기온 보정(Δ°C)", -5.0, 5.0, 0.0, step=0.1)

    scen_df = None
    if scen == "사용자 업로드(월·기온)":
        up_scen = st.file_uploader("CSV/XLSX 업로드 (열: 월, 기온 또는 month, temp)", type=["csv","xlsx"], key="temp_scen_upload")
        if up_scen is not None:
            if up_scen.name.lower().endswith(".csv"):
                scen_df = pd.read_csv(up_scen)
            else:
                scen_df = pd.read_excel(up_scen)
            scen_df.columns = [str(c).strip().lower() for c in scen_df.columns]
            if "월" in scen_df.columns:
                scen_df["month"] = scen_df["월"]
            if "기온" in scen_df.columns:
                scen_df["temp"] = scen_df["기온"]
            if "month" not in scen_df.columns or "temp" not in scen_df.columns:
                st.error("업로드 형식: '월/month'와 '기온/temp' 열이 필요합니다.")
                st.stop()

# ─────────────────────────────────────────────────────────────
# 4) 데이터 전처리 & 시나리오 기온 생성
# ─────────────────────────────────────────────────────────────
df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
train_df = df[df["연"].isin(sel_years)].copy()
if len(train_df) < 6:
    st.warning("학습 샘플이 적습니다. 연도를 더 선택하세요.")

# 월별 평균 기온(학습기간)
monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

# 미래 타임라인
last_date = df["날짜"].max()
start = pd.Timestamp(last_date.year, last_date.month, 1)
future_index = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
future_df = pd.DataFrame({"날짜": future_index})
future_df["연"] = future_df["날짜"].dt.year
future_df["월"] = future_df["날짜"].dt.month

# 기온 시나리오
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
future_df["temp"] = (future_df["temp"] + delta).fillna(future_df["월"].map(monthly_avg_temp["temp"]))

# ─────────────────────────────────────────────────────────────
# 5) 학습 + 예측
# ─────────────────────────────────────────────────────────────
results = []
charts = []

x_train = train_df[temp_col].astype(float).values
x_future = future_df["temp"].astype(float).values

for col in sel_products:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        continue
    y_train = train_df[col].astype(float).values
    y_pred_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)

    tmp = future_df[["날짜","연","월"]].copy()
    tmp["상품"] = col
    tmp["예측공급량"] = np.clip(y_pred_future, a_min=0, a_max=None)
    tmp["R2(Train)"] = r2
    results.append(tmp)

    # 과거 실제 + 미래 예측 라인
    hist = df[["날짜", col]].copy().rename(columns={col:"실제"})
    pred = tmp[["날짜","예측공급량"]].rename(columns={"예측공급량":"예측"})
    merged = pd.merge(hist, pred, on="날짜", how="outer").sort_values("날짜")

    fig = plt.figure(figsize=(9, 3.6))
    plt.plot(merged["날짜"], merged["실제"], label=f"{col} 실제")
    plt.plot(merged["날짜"], merged["예측"], linestyle="--", label=f"{col} 예측")
    plt.title(f"{col} — Poly-3 Forecast (Train R²={r2:.3f})")
    plt.xlabel("날짜")
    plt.ylabel("공급량")
    plt.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)

if not results:
    st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다.")
    st.stop()

pred_df = pd.concat(results, ignore_index=True)
pred_pivot = pred_df.pivot_table(index=["날짜","연","월"], columns="상품", values="예측공급량").reset_index()

# ─────────────────────────────────────────────────────────────
# 6) 출력
# ─────────────────────────────────────────────────────────────
st.subheader("예측 결과 미리보기")
st.dataframe(pred_pivot.head(24), use_container_width=True)

csv = pred_pivot.to_csv(index=False).encode("utf-8-sig")
st.download_button("예측 결과 CSV 다운로드", data=csv, file_name="citygas_forecast.csv", mime="text/csv")

with st.expander("사용된 기온 시나리오 보기"):
    st.dataframe(future_df[["날짜","월","temp"]].rename(columns={"temp":"기온(°C)"}), use_container_width=True)

st.caption("Tip: 예측 월수·기온 보정(Δ°C)·학습 연도/상품을 바꿔가며 감도 점검하세요.")
