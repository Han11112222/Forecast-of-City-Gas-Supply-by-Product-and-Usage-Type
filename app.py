# app.py — City Gas Supply Forecast by Product & Usage Type (Poly-3)
# 요구사항: 연도 선택 칩, 예측 상품 선택, repo 파일/업로드 선택, 3차 다항식(기온-공급량) 예측
import os, io, glob, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 상품별 공급량 예측 (Poly-3)", layout="wide")
st.title("Forecast of City Gas Supply by Product and Usage Type")
st.caption("3차 다항식(기온↔공급량) 기반 월별 예측 · 데이터 소스: Excel")

# Matplotlib 폰트 캐시 경로(권한 문제 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"

# ─────────────────────────────────────────────────────────────
# 한글 폰트(가능하면 적용, 실패해도 실행)
# ─────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
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
# 유틸
# ─────────────────────────────────────────────────────────────
META_COL_CANDIDATES = [
    "날짜","일자","date","연","년","월","평균기온","평균 기온","기온","평균기온(°C)","평균기온(섭씨)","평균기온(℃)"
]

KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"
]

TEMP_NAME_HINTS = ["평균기온", "기온", "temperature", "temp"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 공백/개행 제거
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜/연월 보정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # 날짜가 없으면 연·월에서 생성 시도
        if ("연" in df.columns or "년" in df.columns) and ("월" in df.columns):
            y = df["연"] if "연" in df.columns else df["년"]
            m = df["월"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + m.astype(str) + "-01", errors="coerce")
    if "연" not in df.columns and "년" in df.columns:
        df["연"] = df["년"]
    if "연" not in df.columns and "날짜" in df.columns:
        df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    # 숫자화 시도
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace(" ", ""), errors="ignore")
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns]
    # 우선순위: 힌트 문자열 포함 & 숫자형
    for c in cols:
        lc = str(c).lower()
        if any(h in lc for h in [h.lower() for h in TEMP_NAME_HINTS]):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    # 없으면 숫자형 & 이름에 '온' 포함
    for c in cols:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    meta = set()
    for name in META_COL_CANDIDATES:
        if name in df.columns:
            meta.add(name)
    # 숫자형 & 메타가 아닌 컬럼을 후보로
    candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in meta]
    # 알려진 순서를 우선 정렬
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def load_repo_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def load_uploaded_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl")
    return normalize_cols(df)

def month_start_after(d: pd.Timestamp) -> pd.Timestamp:
    # 다음달 1일
    y = d.year + (1 if d.month == 12 else 0)
    m = 1 if d.month == 12 else d.month + 1
    return pd.Timestamp(y, m, 1)

# ─────────────────────────────────────────────────────────────
# 사이드바 — 데이터 소스
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 불러오기 방식")
    data_src = st.radio("방식 선택", ["Repo 내 파일 사용", "파일 업로드"], index=0, horizontal=False)

    df = None
    if data_src == "Repo 내 파일 사용":
        # data 폴더에서 .xlsx 나열
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
        if len(repo_files) == 0:
            st.info("data 폴더에 엑셀 파일이 없습니다. 업로드 탭을 사용하세요.")
        default_idx = 0
        file_choice = st.selectbox("실적 파일(Excel)", repo_files if repo_files else ["<None>"], index=default_idx if repo_files else 0)
        if repo_files:
            df = load_repo_excel(file_choice)
    else:
        up = st.file_uploader("엑셀 파일 업로드 (xlsx)", type=["xlsx"])
        if up is not None:
            df = load_uploaded_excel(up)

    # 데이터가 없으면 종료
    if df is None or len(df) == 0:
        st.stop()

    # 학습 연도 선택
    st.subheader("학습 데이터 연도 선택")
    years = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
    default_years = years  # 기본: 모두 선택
    sel_years = st.multiselect("연도 선택", years, default=default_years, help="필요 없는 연도는 x로 제거하세요.")

    # 온도 컬럼 자동 탐지 + 선택
    temp_col_auto = detect_temp_col(df)
    st.subheader("기온 컬럼 선택")
    temp_col = st.selectbox("기온(°C) 컬럼", [temp_col_auto] + [c for c in df.columns if c != temp_col_auto]) if temp_col_auto else st.selectbox("기온(°C) 컬럼", list(df.columns))
    if not pd.api.types.is_numeric_dtype(df[temp_col]):
        st.error(f"선택한 기온 컬럼({temp_col})이 숫자형이 아닙니다. 엑셀 파일을 확인하세요.")
        st.stop()

    # 예측할 상품 선택
    st.subheader("예측할 상품 선택")
    prod_cols_all = guess_product_cols(df)
    default_products = [c for c in KNOWN_PRODUCT_ORDER if c in prod_cols_all] or prod_cols_all[:6]
    sel_products = st.multiselect("상품(용도) 선택", prod_cols_all, default=default_products)

    # 예측 범위 및 기온 시나리오
    st.subheader("예측 설정")
    horizon = st.slider("예측 개월 수", 3, 24, 12, step=1)
    scen = st.radio("기온 시나리오", ["학습기간 월별 평균", "전년도 월별 기온 복사", "사용자 업로드(월·기온)"], index=0)
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
            # month / temp 이름 표준화
            if "월" in scen_df.columns:
                scen_df["month"] = scen_df["월"]
            if "기온" in scen_df.columns:
                scen_df["temp"] = scen_df["기온"]
            if "month" not in scen_df.columns or "temp" not in scen_df.columns:
                st.error("업로드 형식이 올바르지 않습니다. 열 이름에 '월'/'month'와 '기온'/'temp'가 있어야 합니다.")
                st.stop()

# ─────────────────────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────────────────────
df = df.dropna(subset=["날짜"])
df = df.sort_values("날짜").reset_index(drop=True)

# 학습 데이터 필터
train_df = df[df["연"].isin(sel_years)].copy()
if len(train_df) < 6:
    st.warning("학습에 사용할 샘플이 너무 적습니다. 연도를 더 선택하세요.")

# 월별 평균 기온(학습기간)
monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

# 예측용 날짜축 생성
last_date = df["날짜"].max()
start = pd.Timestamp(last_date.year, last_date.month, 1)
future_index = pd.date_range(month_start_after(start), periods=horizon, freq="MS")
future_df = pd.DataFrame({"날짜": future_index})
future_df["연"] = future_df["날짜"].dt.year
future_df["월"] = future_df["날짜"].dt.month

# 기온 시나리오 생성
if scen == "학습기간 월별 평균":
    future_df = future_df.merge(monthly_avg_temp.reset_index(), on="월", how="left")
elif scen == "전년도 월별 기온 복사":
    # 전년도(마지막 연도)의 해당 월 기온 사용
    last_year = last_date.year
    base = df[df["연"] == last_year][["월", temp_col]].groupby("월")[temp_col].mean().rename("temp").to_frame().reset_index()
    future_df = future_df.merge(base, on="월", how="left")
else:
    # 사용자 업로드
    scen_df["month"] = pd.to_numeric(scen_df["month"], errors="coerce").astype(int)
    scen_df = scen_df[["month","temp"]].dropna()
    future_df = future_df.merge(scen_df, left_on="월", right_on="month", how="left").drop(columns=["month"])

# 보정 Δ°C 적용
future_df["temp"] = future_df["temp"] + delta

# 만약 일부 월의 temp가 비었으면 학습기간 월평균으로 보강
future_df["temp"] = future_df["temp"].fillna(future_df["월"].map(monthly_avg_temp["temp"]))

# ─────────────────────────────────────────────────────────────
# 학습 + 예측
# ─────────────────────────────────────────────────────────────
def fit_poly3_and_predict(x_train: np.ndarray, y_train: np.ndarray, x_future: np.ndarray):
    # x,y shape
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    # Poly-3
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(Xtr, y_train)
    # 성능(R2) — 학습셋 기준
    r2 = model.score(Xtr, y_train)
    # 미래 예측
    Xf = poly.transform(x_future)
    y_future = model.predict(Xf)
    return y_future, r2

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
    tmp["예측공급량"] = np.clip(y_pred_future, a_min=0, a_max=None)  # 음수 방지
    tmp["R2(Train)"] = r2
    results.append(tmp)

    # 차트 생성 (시간축: 과거 실제 + 미래 예측)
    hist = df[["날짜", col]].copy().rename(columns={col:"실제"})
    pred = tmp[["날짜","예측공급량"]].copy().rename(columns={"예측공급량":"예측"})
    merged = pd.merge(hist, pred, on="날짜", how="outer").sort_values("날짜")

    fig = plt.figure(figsize=(9, 3.6))
    plt.plot(merged["날짜"], merged["실제"], label=f"{col} 실제")
    plt.plot(merged["날짜"], merged["예측"], linestyle="--", label=f"{col} 예측")
    plt.title(f"{col} — Poly-3 Forecast  (Train R²={r2:.3f})")
    plt.xlabel("날짜")
    plt.ylabel("공급량")
    plt.legend(loc="best")
    plt.tight_layout()
    charts.append(fig)

if not results:
    st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다.")
    st.stop()

pred_df = pd.concat(results, ignore_index=True)
pred_pivot = pred_df.pivot_table(index=["날짜","연","월"], columns="상품", values="예측공급량").reset_index()

# ─────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────
st.subheader("예측 결과 미리보기")
st.dataframe(pred_pivot.head(24), use_container_width=True)

# 다운로드
csv = pred_pivot.to_csv(index=False).encode("utf-8-sig")
st.download_button("예측 결과 CSV 다운로드", data=csv, file_name="citygas_forecast.csv", mime="text/csv")

# 차트 섹션
st.subheader("상품별 예측 차트")
for fig in charts:
    st.pyplot(fig)

# 기온 시나리오 표
with st.expander("사용된 기온 시나리오 보기"):
    st.dataframe(future_df[["날짜","월","temp"]].rename(columns={"temp":"기온(°C)"}) , use_container_width=True)

st.caption("Tips: 예측 월수, 기온 보정(Δ°C), 학습 연도/상품을 바꿔가며 감도 확인하세요.")
