# app.py — 도시가스 공급·판매 분석 (Poly-3, 최종)

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# 파일 경로 (필요 시 이름만 바꿔주세요)
# -------------------------------------------------------
SALES_PATH = "data/상품별판매량_MJ.xlsx"    # 시트: 실적_월합, 컬럼: 날짜, 냉방용(필수)
TEMP_PATH  = "data/기온.xlsx"               # 시트: 기온,  위 8줄 설명 → skiprows=8, 컬럼: 날짜, 평균기온(℃)

# -------------------------------------------------------
# 공통 도우미
# -------------------------------------------------------
def style_table(df: pd.DataFrame) -> str:
    # 가운데 정렬 + 세로 가운데
    return (
        "<style>"
        "table.dataframe td, table.dataframe th {"
        " text-align:center !important; vertical-align:middle !important;"
        "}"
        "</style>"
    )

def fmt_comma(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    try:
        return f"{int(round(float(x))):,}"
    except:
        return str(x)

def fmt_temp(x):
    if pd.isna(x): 
        return ""
    try:
        return f"{float(x):.1f}"
    except:
        return str(x)

def year_month_str(y, m):
    return f"{int(y)}.{int(m):02d}"

# -------------------------------------------------------
# 데이터 로딩
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sales(path=SALES_PATH):
    """
    월별 합계가 들어 있는 '실적_월합' 시트를 읽는다.
    필수: 날짜, 냉방용
    """
    df = pd.read_excel(path, sheet_name="실적_월합")
    if "날짜" not in df.columns:
        raise ValueError("판매 실적파일에 '날짜' 컬럼이 없습니다.")
    if "냉방용" not in df.columns:
        raise ValueError("판매 실적파일에 '냉방용' 컬럼이 없습니다.")

    # 날짜 파싱 및 연/월
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month

    # 정렬
    df = df.sort_values(["연", "월"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_temp_raw(path=TEMP_PATH):
    """
    기온 RAW (일별). 시트 '기온', skiprows=8, 컬럼 '날짜','평균기온(℃)'
    """
    temp = pd.read_excel(path, sheet_name="기온", skiprows=8)
    if "날짜" not in temp.columns or "평균기온(℃)" not in temp.columns:
        raise ValueError("기온 RAW 파일에 '날짜', '평균기온(℃)' 컬럼이 필요합니다.")

    temp["날짜"] = pd.to_datetime(temp["날짜"])
    temp = temp.dropna(subset=["날짜"]).copy()

    # 당월 평균
    temp["연"] = temp["날짜"].dt.year
    temp["월"] = temp["날짜"].dt.month
    month_avg = (
        temp.groupby(["연", "월"], as_index=False)["평균기온(℃)"].mean()
        .rename(columns={"평균기온(℃)": "당월평균기온"})
    )

    # '전월16 ~ 당월15' 기준 → 날짜 +15일 후의 month로 버킷화
    bucket = temp.copy()
    bucket["버킷연월"] = (bucket["날짜"] + pd.Timedelta(days=15)).dt.to_period("M")
    period_avg = (
        bucket.groupby("버킷연월", as_index=False)["평균기온(℃)"].mean()
        .rename(columns={"평균기온(℃)": "기간평균기온"})
    )
    period_avg["연"] = period_avg["버킷연월"].dt.year.astype(int)
    period_avg["월"] = period_avg["버킷연월"].dt.month.astype(int)
    period_avg = period_avg.drop(columns=["버킷연월"])

    base = pd.merge(month_avg, period_avg, on=["연", "월"], how="outer").sort_values(["연", "월"])
    base = base.reset_index(drop=True)
    return base

# -------------------------------------------------------
# 학습/예측 모델 (Poly-3)
# -------------------------------------------------------
def train_poly3(X: np.ndarray, y: np.ndarray):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(Xp, y)
    return model, poly

def predict_poly3(model, poly, X: np.ndarray):
    Xp = poly.transform(X.reshape(-1, 1))
    return model.predict(Xp)

def poly_equation_text(model, poly):
    # y = ax^3 + bx^2 + cx + d (d는 intercept)
    # scikit의 경우 include_bias=False라 intercept_ 따로 존재
    coefs = model.coef_
    # coefs 길이는 3차 기준 [x, x^2, x^3] 순서
    if len(coefs) >= 3:
        c1, c2, c3 = coefs[0], coefs[1], coefs[2]
    else:
        # 방어
        c1 = coefs[0] if len(coefs) > 0 else 0.0
        c2 = coefs[1] if len(coefs) > 1 else 0.0
        c3 = coefs[2] if len(coefs) > 2 else 0.0

    d = model.intercept_
    return f"y = {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

# -------------------------------------------------------
# UI 구성
# -------------------------------------------------------
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.markdown("<h1 style='margin-top:0'>도시가스 공급·판매 분석 (Poly-3)</h1>", unsafe_allow_html=True)
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

with st.sidebar:
    st.subheader("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=1)

    st.subheader("데이터 불러오기")
    how = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    # 파일 업로드 옵션(원하실 때만)
    if how == "파일 업로드":
        st.info("업로드를 사용하지 않으시면 Repo 경로의 기본 파일을 씁니다.")
        up_sales = st.file_uploader("판매 실적파일", type=["xlsx"])
        up_temp  = st.file_uploader("기온 RAW 파일", type=["xlsx"])
    else:
        up_sales = None
        up_temp = None

    st.subheader("학습 데이터 연도 선택")
    # 학습 가능한 연도는 파일에서 동적으로 계산 (아래에서 활용)
    sel_years = st.multiselect("연도 선택", [], default=[], key="year_pick")

    st.subheader("예측 기간")
    colA, colB = st.columns(2)
    with colA:
        start_year = st.selectbox("예측 시작(연)", list(range(2015, 2031)), index= list(range(2015,2031)).index(2025))
        start_month = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
    with colB:
        end_year   = st.selectbox("예측 종료(연)", list(range(2015, 2031)), index= list(range(2015,2031)).index(2025))
        end_month  = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

    st.subheader("예측 시작")
    st.button("예측 시작", use_container_width=True, key="run_button")

# ΔT 시나리오 — 바로 반영되도록 세션 상태
for key in ["dt_normal", "dt_best", "dt_consv"]:
    if key not in st.session_state:
        st.session_state[key] = 0.0

def inc(name, step=0.5):
    st.session_state[name] = round(st.session_state[name] + step, 2)

def dec(name, step=0.5):
    st.session_state[name] = round(st.session_state[name] - step, 2)

st.subheader("ΔT 시나리오 (℃)")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**ΔT(Normal)**")
    b1, b2, b3 = st.columns([1,1,3])
    with b1:  st.button("−", on_click=dec, args=("dt_normal",), key="n_minus")
    with b2:  st.button("+", on_click=inc, args=("dt_normal",), key="n_plus")
    st.metric("기온 보정(℃)", f"{st.session_state['dt_normal']:+.2f}")

with c2:
    st.markdown("**ΔT(Best)**")
    b1, b2, b3 = st.columns([1,1,3])
    with b1:  st.button("−", on_click=dec, args=("dt_best",), key="b_minus")
    with b2:  st.button("+", on_click=inc, args=("dt_best",), key="b_plus")
    st.metric("기온 보정(℃)", f"{st.session_state['dt_best']:+.2f}")

with c3:
    st.markdown("**ΔT(Conservative)**")
    b1, b2, b3 = st.columns([1,1,3])
    with b1:  st.button("−", on_click=dec, args=("dt_consv",), key="c_minus")
    with b2:  st.button("+", on_click=inc, args=("dt_consv",), key="c_plus")
    st.metric("기온 보정(℃)", f"{st.session_state['dt_consv']:+.2f}")

# -------------------------------------------------------
# 데이터 준비
# -------------------------------------------------------
# 파일 로딩 (업로드가 있으면 업로드 사용)
if how == "파일 업로드" and up_sales is not None:
    sales_df = load_sales(up_sales)
else:
    sales_df = load_sales(SALES_PATH)

if how == "파일 업로드" and up_temp is not None:
    temp_base = load_temp_raw(up_temp)
else:
    temp_base = load_temp_raw(TEMP_PATH)

# 사이드바 학습연도 목록 채우기 (최초 1회만)
all_years = sorted(sales_df["연"].unique().tolist())
if not st.session_state.get("year_initialized", False):
    # 기본: 최근 3년
    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    st.session_state["year_pick"] = default_years
    st.session_state["year_initialized"] = True

with st.sidebar:
    st.multiselect("연도 선택", options=all_years, default=st.session_state["year_pick"], key="year_pick")

train_years = sorted(st.session_state["year_pick"])

# -------------------------------------------------------
# 판매량(냉방용) 분석 탭
# -------------------------------------------------------
if mode == "판매량 분석(냉방용)":
    st.markdown("---")
    st.markdown("### 예측 결과 — Normal")

    # 학습 데이터 만들기 (선택 연도 & 냉방용 & 기간평균기온)
    train = pd.merge(sales_df[["연","월","냉방용"]],
                     temp_base[["연","월","기간평균기온"]],
                     on=["연","월"],
                     how="inner")
    train = train.dropna(subset=["냉방용","기간평균기온"]).copy()
    if train_years:
        train = train[train["연"].isin(train_years)]

    if train.empty:
        st.warning("선택한 학습 연도로 학습할 데이터가 없습니다.")
        st.stop()

    X = train["기간평균기온"].values
    y = train["냉방용"].values
    model, poly = train_poly3(X, y)
    eq_text = poly_equation_text(model, poly)
    r2 = model.score(poly.transform(X.reshape(-1,1)), y)

    # 예측 대상 연월 만들기
    rng = pd.period_range(start=pd.Period(year=start_year, month=start_month, freq="M"),
                          end=pd.Period(year=end_year, month=end_month, freq="M"),
                          freq="M")
    target = pd.DataFrame({"연":[p.year for p in rng], "월":[p.month for p in rng]})

    # ΔT 반영된 기온 표 생성 함수
    def make_forecast_table(dt_offset: float):
        base = pd.merge(target, temp_base, on=["연","월"], how="left")
        base["월평균기온(적용)"] = base["당월평균기온"] + dt_offset
        base["기간평균기온(적용)"] = base["기간평균기온"] + dt_offset

        # 예측
        yhat = np.clip(
            np.rint(predict_poly3(model, poly, base["기간평균기온(적용)"].values)),
            a_min=0, a_max=None
        ).astype(int)
        base["예측판매량"] = yhat

        show = base[["연","월","월평균기온(적용)","기간평균기온(적용)","예측판매량"]].copy()
        show["연월"] = show.apply(lambda r: year_month_str(r["연"], r["월"]), axis=1)
        show = show[["연월","월평균기온(적용)","기간평균기온(적용)","예측판매량"]]

        # 합계행
        total = pd.DataFrame(
            [["종계","", "", show["예측판매량"].sum()]],
            columns=show.columns
        )
        out = pd.concat([show, total], ignore_index=True)

        # 포맷팅
        out["월평균기온(적용)"] = out["월평균기온(적용)"].apply(lambda v: "" if v=="" else fmt_temp(v))
        out["기간평균기온(적용)"] = out["기간평균기온(적용)"].apply(lambda v: "" if v=="" else fmt_temp(v))
        out["예측판매량"] = out["예측판매량"].apply(fmt_comma)

        return out

    normal_tbl = make_forecast_table(st.session_state["dt_normal"])
    best_tbl   = make_forecast_table(st.session_state["dt_best"])
    consv_tbl  = make_forecast_table(st.session_state["dt_consv"])

    # 표시
    st.markdown(style_table(normal_tbl), unsafe_allow_html=True)
    st.dataframe(normal_tbl, use_container_width=True, hide_index=True)

    st.markdown("### 예측 결과 — Best / Conservative")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Best")
        st.dataframe(best_tbl, use_container_width=True, hide_index=True)
        st.download_button("Best CSV", normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="best_forecast.csv", use_container_width=True)
    with col2:
        st.caption("Conservative")
        st.dataframe(consv_tbl, use_container_width=True, hide_index=True)
        st.download_button("Conservative CSV", normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="conservative_forecast.csv", use_container_width=True)

    # 검증표 (실적 vs 예측)
    st.markdown("### 판매량 예측 검증")
    valid = pd.merge(
        sales_df[["연","월","냉방용"]],
        normal_tbl[["연월","예측판매량"]],
        left_on=sales_df.apply(lambda r: year_month_str(r["연"], r["월"]), axis=1),
        right_on="연월",
        how="right"
    )
    # 위 merge는 조금 복잡 → 간단히 다시 구성
    base2 = pd.merge(
        sales_df[["연","월","냉방용"]],
        pd.DataFrame({
            "연":[int(s.split(".")[0]) for s in normal_tbl["연월"] if s!="종계"],
            "월":[int(s.split(".")[1]) for s in normal_tbl["연월"] if s!="종계"],
            "예측판매량":[int(x.replace(",","")) for x in normal_tbl["예측판매량"] if x!=""]
        }),
        on=["연","월"], how="right"
    )
    base2["연월"] = base2.apply(lambda r: year_month_str(r["연"], r["월"]), axis=1)
    base2["오차"] = base2["냉방용"] - base2["예측판매량"]
    base2["오차율(%)"] = np.where(base2["냉방용"]>0, base2["오차"]/base2["냉방용"]*100, np.nan)

    vb = base2[["연월","냉방용","예측판매량","오차","오차율(%)"]].copy()
    vb.loc[len(vb)] = ["종계",
                       base2["냉방용"].sum(),
                       base2["예측판매량"].sum(),
                       base2["오차"].sum(),
                       np.nan]
    for col in ["냉방용","예측판매량","오차"]:
        vb[col] = vb[col].apply(fmt_comma)
    vb["오차율(%)"] = vb["오차율(%)"].apply(lambda v: "" if pd.isna(v) else f"{v:.1f}")

    st.dataframe(vb, use_container_width=True, hide_index=True)

    # 상관 그래프 (Train 기준, Poly-3 식 포함)
    st.markdown("### 기온-냉방용 실적 상관관계 (Train)")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5,5))
    ax.scatter(train["기간평균기온"], train["냉방용"], alpha=0.6, label="학습 샘플")
    xx = np.linspace(train["기간평균기온"].min()-1, train["기간평균기온"].max()+1, 200)
    yy = predict_poly3(model, poly, xx)
    ax.plot(xx, yy, lw=2.5, label=f"Poly-3\n{eq_text}", color="#1f77b4")
    ax.set_xlabel("기간평균기온 ( )")
    ax.set_ylabel("판매량 (MJ)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.caption(f"Train R² = {r2:.3f}")

# -------------------------------------------------------
# (선택) 공급량 분석 자리 – 필요 시 기존 로직 연결
# -------------------------------------------------------
else:
    st.info("공급량 분석 화면은 판매량 분석과 동일한 방식으로 연결 가능합니다. (요청하신 냉방용 판매 분석을 우선 반영했습니다.)")
