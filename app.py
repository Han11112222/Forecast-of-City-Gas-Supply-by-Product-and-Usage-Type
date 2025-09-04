# app.py — Forecast of City Gas Supply by Product & Usage Type (Poly-3)
# 기능: 연도 선택, 예측 상품 선택, Repo 파일/업로드 선택, 3차 다항식(기온↔공급량) 예측
# 업데이트: (1) 기온 컬럼 자동탐지(선택 UI 제거) (2) 예측 시작/종료월 지정 (3) 표는 정수 표시
#          (4) '총공급량'은 합산이 아니라 별도 Poly-3 모델로 예측하여 마지막 열에 표기

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
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace(" ", ""), errors="ignore")
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    # 힌트 기반 자동탐지
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # '온' 포함 숫자형
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
def load_repo_excel(path: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
    return normalize_cols(df)

@st.cache_data(ttl=600)
def load_uploaded_excel(file, sheet_name: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl", sheet_name=sheet_name)
    return normalize_cols(df)

def month_start(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return pd.Timestamp(ts.year, ts.month, 1)

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
        data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
        repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
        if len(repo_files) == 0:
            st.info("data 폴더에 엑셀 파일이 없습니다. 업로드 탭을 사용하세요.")
        default_idx = next((i for i, p in enumerate(repo_files) if "상품별공급량_MJ" in p), 0) if repo_files else 0
        file_choice = st.selectbox("실적 파일(Excel)", repo_files if repo_files else ["<None>"], index=default_idx if repo_files else 0)
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

    # (삭제) 기온 컬럼 선택 → 자동탐지 사용
    temp_col = detect_temp_col(df)
    if temp_col is None:
        st.error("기온 컬럼을 자동으로 찾지 못했습니다. 엑셀의 기온 열 이름에 '평균기온' 또는 '기온'을 포함시켜 주세요.")
        st.stop()

    # 예측할 상품 선택
    st.subheader("예측할 상품 선택")
    product_cols = guess_product_cols(df)
    default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
    sel_products = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

    # 예측 범위 & 기온 시나리오
    st.subheader("예측 설정")
    # 예측 시작/종료 월 선택(월초 기준으로 정규화)
    last_hist_m = month_start(pd.to_datetime(df["날짜"].max()))
    default_start = (last_hist_m + pd.offsets.MonthBegin(1)).date()  # 다음달 1일
    default_end   = (last_hist_m + pd.DateOffset(months=12)).date()  # 기본 13개월 뒤(포함 범위)
    c1, c2 = st.columns(2)
    with c1:
        forecast_start = st.date_input("예측 시작월", value=default_start)
    with c2:
        forecast_end = st.date_input("예측 종료월", value=default_end)

    if pd.to_datetime(forecast_end) < pd.to_datetime(forecast_start):
        st.error("예측 종료월이 시작월보다 빠릅니다. 범위를 다시 지정하세요.")
        st.stop()

    # 기온 시나리오
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
            if "월" in scen_df.columns: scen_df["month"] = scen_df["월"]
            if "기온" in scen_df.columns: scen_df["temp"] = scen_df["기온"]
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

# 예측 타임라인 (시작~종료월, 월초 기준, 양끝 포함)
start_m = month_start(forecast_start)
end_m   = month_start(forecast_end)
future_index = pd.date_range(start=start_m, end=end_m, freq="MS")
future_df = pd.DataFrame({"날짜": future_index})
future_df["연"] = future_df["날짜"].dt.year.astype(int)
future_df["월"] = future_df["날짜"].dt.month.astype(int)

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

# Δ°C 보정 & 결측 보강(학습기간 월평균→전체 월평균 순서)
future_df["temp"] = future_df["temp"] + delta
future_df["temp"] = future_df["temp"].fillna(future_df["월"].map(monthly_avg_temp["temp"]))
fallback_monthly = df.groupby("월")[temp_col].mean()
future_df["temp"] = future_df["temp"].fillna(future_df["월"].map(fallback_monthly))

# ─────────────────────────────────────────────────────────────
# 5) 학습 + 예측
# ─────────────────────────────────────────────────────────────
results = []

x_train = train_df[temp_col].astype(float).values
x_future = future_df["temp"].astype(float).values

# 선택된 상품 예측
for col in sel_products:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        continue
    y_train = train_df[col].astype(float).values
    y_pred_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
    # 정수화(소수점 제거)
    y_pred_future = np.rint(y_pred_future).astype(np.int64)

    tmp = future_df[["날짜","연","월"]].copy()
    tmp["상품"] = col
    tmp["예측공급량"] = np.clip(y_pred_future, a_min=0, a_max=None)
    tmp["R2(Train)"] = r2
    results.append(tmp)

    # 차트(정수 예측 사용)
    hist = df[["날짜", col]].copy().rename(columns={col:"실제"})
    pred = tmp[["날짜","예측공급량"]].rename(columns={"예측공급량":"예측"})
    merged = pd.merge(hist, pred, on="날짜", how="outer").sort_values("날짜")

    fig = plt.figure(figsize=(9, 3.6))
    plt.plot(merged["날짜"], merged["실제"], label=f"{col} 실제")
    plt.plot(merged["날짜"], merged["예측"], linestyle="--", label=f"{col} 예측")
    plt.title(f"{col} — Poly-3 Forecast (Train R²={r2:.3f})")
    plt.xlabel("날짜"); plt.ylabel("공급량")
    plt.legend(loc="best"); plt.tight_layout()
    st.pyplot(fig)

if not results:
    st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다.")
    st.stop()

pred_df = pd.concat(results, ignore_index=True)
pred_pivot = pred_df.pivot_table(index=["날짜","연","월"], columns="상품", values="예측공급량").reset_index()

# ── 총공급량: 합산이 아니라 '총공급량' 열을 별도 모델로 예측해 마지막 열로 배치
if "총공급량" in df.columns and pd.api.types.is_numeric_dtype(df["총공급량"]):
    y_total_train = train_df["총공급량"].astype(float).values
    y_total_pred, r2_tot = fit_poly3_and_predict(x_train, y_total_train, x_future)
    y_total_pred = np.rint(y_total_pred).astype(np.int64)
    total_series = pd.Series(y_total_pred, index=pred_pivot.index, name="총공급량")
    pred_pivot["총공급량"] = np.clip(total_series, a_min=0, a_max=None)
else:
    st.warning("원본 데이터에 '총공급량' 열이 없어 합산 기반 총량을 표시하지 않습니다. (요청 사항: 별도 모델 예측)")

# 컬럼 정렬: 날짜/연/월 → 선택된 상품들 → 총공급량(마지막)
ordered_cols = ["날짜","연","월"] + [c for c in sel_products if c in pred_pivot.columns and c != "총공급량"]
if "총공급량" in pred_pivot.columns:
    ordered_cols += ["총공급량"]
pred_pivot = pred_pivot[ordered_cols]

# 정수 표시(연/월 포함 숫자형 전부)
for c in pred_pivot.columns:
    if pd.api.types.is_float_dtype(pred_pivot[c]) or pd.api.types.is_integer_dtype(pred_pivot[c]):
        pred_pivot[c] = pd.to_numeric(pred_pivot[c], errors="coerce").round().astype("Int64")

# ─────────────────────────────────────────────────────────────
# 6) 출력
# ─────────────────────────────────────────────────────────────
st.subheader("예측 결과 미리보기")
st.dataframe(pred_pivot.head(24), use_container_width=True)

csv = pred_pivot.to_csv(index=False).encode("utf-8-sig")
st.download_button("예측 결과 CSV 다운로드", data=csv, file_name="citygas_forecast.csv", mime="text/csv")

with st.expander("사용된 기온 시나리오 보기"):
    scen_view = future_df[["날짜","월","temp"]].copy()
    scen_view["월"] = scen_view["월"].astype("Int64")
    st.dataframe(scen_view.rename(columns={"temp":"기온(°C)"}), use_container_width=True)

st.caption("Tip: 예측 시작/종료월·Δ°C·학습 연도/상품을 바꿔가며 감도 점검하세요.")
