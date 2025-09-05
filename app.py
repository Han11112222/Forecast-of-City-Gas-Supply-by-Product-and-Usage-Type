# app.py — 도시가스 공급·판매 분석 (Poly-3)
# -----------------------------------------------------------
# 요구사항 반영:
# - 좌측 'Repo 내 파일 사용'이면 파일 드롭다운 숨김, '파일 업로드'일 때만 노출
# - [예측 시작] 버튼 복구, 이후 ΔT(±0.5)는 누를 때 즉시 반영
# - 표: 연월(예: 2025.01) 1열, 중앙 정렬, 기온 소수점1자리/나머지 천단위 콤마
# - Normal / Best / Conservative 3표 + CSV
# - 기온–냉방용 실적 상관관계(Train, Poly-3, R²) 그래프
# - 기온 RAW(일자료) → 당월평균기온, 전월16~당월15 기간평균기온 정확 계산
# - 판매 실적 파일: '연','월', '냉방용'(이름에 '냉' 포함시 자동 인식)
# -----------------------------------------------------------

import os
import io
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

DEFAULT_SALES_PATH = DATA_DIR / "상품별판매량.xlsx"  # 연/월/냉방용(또는 '냉' 포함 열)
DEFAULT_TEMP_PATH  = DATA_DIR / "기온.xlsx"        # 시트에 날짜, 평균기온(℃)

# -----------------------------
# 공통 유틸
# -----------------------------
def set_session_defaults():
    ss = st.session_state
    ss.setdefault("view", "판매량 분석(냉방용)")  # 기본 탭
    ss.setdefault("use_repo", True)
    ss.setdefault("train_years", [2023, 2024, 2025])
    ss.setdefault("start_year", 2025)
    ss.setdefault("start_month", 1)
    ss.setdefault("end_year",   2025)
    ss.setdefault("end_month",  12)
    # ΔT 시나리오
    ss.setdefault("dt_normal", 0.0)
    ss.setdefault("dt_best",   0.0)
    ss.setdefault("dt_cons",   0.0)
    # 예측 시작 상태
    ss.setdefault("ready", False)

def hyphen_to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def yyyymm_str(y:int, m:int)->str:
    return f"{y}.{m:02d}"

def month_iter(y1, m1, y2, m2):
    y, m = y1, m1
    while (y < y2) or (y == y2 and m <= m2):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1

def center_style_html(df: pd.DataFrame) -> str:
    # 중앙 정렬 + 세로 중앙
    css = """
    <style>
      table.centered-table {margin: 0 auto;}
      .centered-table th, .centered-table td {
        text-align: center !important;
        vertical-align: middle !important;
        padding: 6px 10px;
        font-size: 14px;
      }
    </style>
    """
    html = df.to_html(index=False, classes="centered-table")
    return css + html

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False, encoding="utf-8-sig")
    return b.getvalue()

# -----------------------------
# 데이터 로드 & 가공
# -----------------------------
@st.cache_data(show_spinner=False)
def load_sales(path: Path) -> pd.DataFrame:
    # 예상 포맷: 연, 월, (냉방용/냉난방/냉…)
    df = pd.read_excel(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # 연/월 파생
    if "연" not in df.columns and "년" in df.columns:
        df["연"] = df["년"]
    if "월" not in df.columns and "month" in df.columns:
        df["월"] = df["month"]

    # 열 이름 정규화
    if "연" not in df.columns or "월" not in df.columns:
        raise ValueError("판매 실적 파일에는 '연'과 '월' 열이 필요합니다.")

    # 냉방용 열 찾기(이름에 '냉' 포함)
    cooling_cols = [c for c in df.columns if ("냉" in c and c not in ["연","월","연월"])]
    if not cooling_cols:
        raise ValueError("판매 실적 파일에서 '냉'이 포함된 실적 열을 찾지 못했습니다. (예: 냉방용)")
    # 첫 번째 냉 관련 열 사용
    cool_col = cooling_cols[0]

    keep = ["연","월", cool_col]
    out = df[keep].copy()
    out.rename(columns={cool_col:"냉방용"}, inplace=True)
    # 숫자 보정
    out["냉방용"] = pd.to_numeric(out["냉방용"], errors="coerce").fillna(0)
    return out

@st.cache_data(show_spinner=False)
def load_temp_daily(path: Path) -> pd.DataFrame:
    # 포맷(예시): 날짜, 지점, 평균기온(℃)
    # 날짜는 datetime으로 파싱
    df = pd.read_excel(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # 날짜 열 자동 탐색
    date_col = None
    for c in df.columns:
        if "일자" in c or "날짜" in c or "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError("기온 파일에서 '날짜/일자' 열을 찾지 못했습니다.")

    temp_col = None
    for c in df.columns:
        if "평균기온" in c or "기온" in c:
            temp_col = c
            break
    if temp_col is None:
        raise ValueError("기온 파일에서 '평균기온' 열을 찾지 못했습니다.")

    t = df[[date_col, temp_col]].copy()
    t[date_col] = pd.to_datetime(t[date_col])
    t.rename(columns={date_col:"날짜", temp_col:"평균기온"}, inplace=True)
    return t

def make_monthly_temp(daily: pd.DataFrame) -> pd.DataFrame:
    """당월평균기온 + 기간평균기온(m-1,16~m,15) 계산"""
    t = daily.copy()
    t["연"] = t["날짜"].dt.year
    t["월"] = t["날짜"].dt.month
    t["일"] = t["날짜"].dt.day

    # 당월평균
    mon = t.groupby(["연","월"])["평균기온"].mean().reset_index()
    mon.rename(columns={"평균기온":"당월평균기온"}, inplace=True)

    # 기간평균기온(m-1, 16 ~ m, 15)
    res = []
    years = sorted(t["연"].unique())
    for y in years:
        months = sorted(t.loc[t["연"]==y, "월"].unique())
        for m in months:
            # 전월
            prev_y, prev_m = (y-1, 12) if m==1 else (y, m-1)

            prev_mask = (t["연"]==prev_y) & (t["월"]==prev_m) & (t["일"]>=16)
            curr_mask = (t["연"]==y) & (t["월"]==m) & (t["일"]<=15)

            seg = pd.concat([t.loc[prev_mask, ["평균기온"]],
                             t.loc[curr_mask, ["평균기온"]]], axis=0)
            if len(seg)==0:
                per = np.nan
            else:
                per = seg["평균기온"].mean()
            res.append((y,m, per))
    per_df = pd.DataFrame(res, columns=["연","월","기간평균기온"])

    out = pd.merge(mon, per_df, on=["연","월"], how="outer").sort_values(["연","월"])
    return out.reset_index(drop=True)

def select_train(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return df[df["연"].isin(years)].copy()

def build_poly3_model(x: np.ndarray, y: np.ndarray):
    poly = PolynomialFeatures(degree=3, include_bias=True)
    Xp = poly.fit_transform(x.reshape(-1,1))
    lr = LinearRegression()
    lr.fit(Xp, y)
    return poly, lr

def poly_predict(poly, lr, x):
    Xp = poly.transform(np.array(x).reshape(-1,1))
    return lr.predict(Xp)

def poly_equation_text(poly, lr) -> str:
    # y = a3 x^3 + a2 x^2 + a1 x + a0
    coefs = lr.coef_
    a0 = lr.intercept_
    # sklearn + include_bias=True → X: [1, x, x^2, x^3]
    a1, a2, a3 = coefs[1], coefs[2], coefs[3]
    def s(v): return f"{v:+.5e}"
    return f"y = {s(a3)}x³ {s(a2)}x² {s(a1)}x {s(a0)}"

def r2_score_manual(y_true, y_pred):
    ssr = np.sum((y_pred - np.mean(y_true))**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 0.0 if sst==0 else float(ssr/sst)

# -----------------------------
# 예측 테이블 생성
# -----------------------------
def make_forecast_table(month_temp: pd.DataFrame,
                        sales: pd.DataFrame,
                        years_train: list[int],
                        y1:int, m1:int, y2:int, m2:int,
                        dt: float) -> tuple[pd.DataFrame, dict]:
    # 학습데이터(냉방용 & 기간평균기온)
    base = pd.merge(sales, month_temp, on=["연","월"], how="inner")
    tr = select_train(base, years_train)
    tr = tr.dropna(subset=["기간평균기온","냉방용"])

    if len(tr) < 6:
        raise ValueError("학습 샘플이 너무 적습니다. (학습 연도를 더 선택하세요)")

    x = tr["기간평균기온"].values
    y = tr["냉방용"].values

    poly, lr = build_poly3_model(x, y)
    eq  = poly_equation_text(poly, lr)

    # R²
    yr_hat = poly_predict(poly, lr, x)
    r2 = r2_score_manual(y, yr_hat)

    # 예측용 기간
    fr_rows = []
    for yy, mm in month_iter(y1, m1, y2, m2):
        row = month_temp[(month_temp["연"]==yy) & (month_temp["월"]==mm)]
        if len(row)==0:
            fr_rows.append((yy,mm, np.nan, np.nan))
        else:
            fr_rows.append((yy,mm, row["당월평균기온"].values[0],
                                 row["기간평균기온"].values[0]))
    fr = pd.DataFrame(fr_rows, columns=["연","월","당월평균기온","기간평균기온"])

    # ΔT 적용은 '기간평균기온'에만
    x_pred = (fr["기간평균기온"].values + dt).astype(float)
    y_pred = np.clip(np.rint(poly_predict(poly, lr, x_pred)), 0, None)

    tbl = fr.copy()
    tbl["연월"] = [yyyymm_str(a,b) for a,b in zip(tbl["연"], tbl["월"])]
    tbl["월평균기온(적용)"] = np.round(fr["기간평균기온"] + dt, 1)  # 표시에 1자리
    tbl["예측판매량"] = y_pred.astype(int)

    # 보기용 정렬/컬럼
    show = tbl[["연월","월평균기온(적용)","예측판매량"]].copy()

    # 숫자 포맷
    show["예측판매량"] = show["예측판매량"].map(lambda v: f"{int(v):,}")
    show["월평균기온(적용)"] = show["월평균기온(적용)"].map(lambda v: f"{v:.1f}")

    # 총계
    tot = int(np.nansum(y_pred))
    show.loc[len(show)] = ["총계", "", f"{tot:,}"]

    info = {"r2": r2, "equation": eq, "poly": poly, "lr": lr, "train": tr}
    return show, info

# -----------------------------
# Streamlit UI
# -----------------------------
set_session_defaults()
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")

# 좌측 패널 ---------------------------------------------------
with st.sidebar:
    st.markdown("### 분석 유형")
    view = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=1, label_visibility="collapsed",
                    key="view_radio")
    st.session_state["view"] = view

    st.markdown("### 데이터를 불러오기")
    method = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)
    st.session_state["use_repo"] = (method == "Repo 내 파일 사용")

    # 파일 선택 (업로드 모드에서만 노출)
    sales_path = DEFAULT_SALES_PATH
    temp_path  = DEFAULT_TEMP_PATH

    if not st.session_state["use_repo"]:
        st.caption("실적파일(공급·판매·기온 중 필요한 것만 사용)")
    if (view == "판매량 분석(냉방용)") and (not st.session_state["use_repo"]):
        up_sales = st.file_uploader("판매 실적파일", type=["xlsx","csv"], key="up_sales")
        if up_sales is not None:
            sales_path = Path("/tmp/_sales.xlsx")
            with open(sales_path, "wb") as f:
                f.write(up_sales.read())
        up_temp = st.file_uploader("기온 파일", type=["xlsx","csv"], key="up_temp")
        if up_temp is not None:
            temp_path = Path("/tmp/_temp.xlsx")
            with open(temp_path, "wb") as f:
                f.write(up_temp.read())

    st.markdown("### 학습 데이터 연도 선택")
    years_all = list(range(2017, 2031))
    _years = st.multiselect("연도 선택", years_all, default=st.session_state["train_years"])
    st.session_state["train_years"] = sorted(_years)

    st.markdown("### 예측 기간")
    cols = st.columns(2)
    with cols[0]:
        y1 = st.selectbox("예측 시작(연)", list(range(2017, 2031)), index=list(range(2017,2031)).index(st.session_state["start_year"]))
    with cols[1]:
        m1 = st.selectbox("예측 시작(월)", list(range(1,13)), index=st.session_state["start_month"]-1)
    cols = st.columns(2)
    with cols[0]:
        y2 = st.selectbox("예측 종료(연)", list(range(2017, 2031)), index=list(range(2017,2031)).index(st.session_state["end_year"]))
    with cols[1]:
        m2 = st.selectbox("예측 종료(월)", list(range(1,13)), index=st.session_state["end_month"]-1)

    st.session_state["start_year"]  = y1
    st.session_state["start_month"] = m1
    st.session_state["end_year"]    = y2
    st.session_state["end_month"]   = m2

    st.markdown("---")
    # 예측 시작 버튼 복구
    if st.button("예측 시작", use_container_width=True):
        st.session_state["ready"] = True
        st.toast("예측 준비 완료! 상단 ΔT를 조정하면 즉시 반영됩니다.", icon="✅")

# 본문 타이틀 -------------------------------------------------
st.markdown("# 도시가스 공급·판매 분석 (Poly-3)")
st.caption("판매량 분석(냉방용): (전월16~당월15) 평균기온 기반 / ΔT는 기간평균기온에 적용")

# ΔT 시나리오 슬랩 -------------------------------------------
st.markdown("### ΔT 시나리오 (℃)")
g = st.columns(3)

def make_dt_box(col, label, key_minus, key_plus, ss_key):
    with col:
        st.markdown(f"**{label}**")
        c = st.columns([1,1,5,1,1])
        with c[1]:
            if st.button("−", key=key_minus):  # unique key
                st.session_state[ss_key] = round(st.session_state[ss_key] - 0.5, 2)
        with c[3]:
            if st.button("+", key=key_plus):
                st.session_state[ss_key] = round(st.session_state[ss_key] + 0.5, 2)
        st.metric("기온 보정(℃)", f"{st.session_state[ss_key]:+.2f}")

make_dt_box(g[0], "ΔT(Normal)", "m_norm", "p_norm", "dt_normal")
make_dt_box(g[1], "ΔT(Best)",   "m_best", "p_best", "dt_best")
make_dt_box(g[2], "ΔT(Conservative)", "m_cons", "p_cons", "dt_cons")

# -----------------------------
# 판매량 분석(냉방용)
# -----------------------------
if st.session_state["view"] == "판매량 분석(냉방용)":
    # 파일 로드
    try:
        sales_df = load_sales(sales_path if not st.session_state["use_repo"] else DEFAULT_SALES_PATH)
        temp_daily = load_temp_daily(temp_path if not st.session_state["use_repo"] else DEFAULT_TEMP_PATH)
    except Exception as e:
        st.error(str(e))
        st.stop()

    month_temp = make_monthly_temp(temp_daily)

    if not st.session_state["ready"]:
        st.info("좌측에서 기간/학습연도를 설정하고 **[예측 시작]**을 눌러주세요.")
        st.stop()

    # 세 가지 시나리오 테이블
    st.markdown("## 예측 결과 — Normal")
    try:
        normal_tbl, info_n = make_forecast_table(
            month_temp, sales_df,
            st.session_state["train_years"],
            st.session_state["start_year"], st.session_state["start_month"],
            st.session_state["end_year"],   st.session_state["end_month"],
            st.session_state["dt_normal"]
        )
        st.markdown(center_style_html(normal_tbl), unsafe_allow_html=True)
        st.download_button("Normal CSV", df_to_csv_bytes(normal_tbl), "normal.csv", use_container_width=False)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown("## 예측 결과 — Best")
    best_tbl, info_b = make_forecast_table(
        month_temp, sales_df,
        st.session_state["train_years"],
        st.session_state["start_year"], st.session_state["start_month"],
        st.session_state["end_year"],   st.session_state["end_month"],
        st.session_state["dt_best"]
    )
    st.markdown(center_style_html(best_tbl), unsafe_allow_html=True)
    st.download_button("Best CSV", df_to_csv_bytes(best_tbl), "best.csv", use_container_width=False)

    st.markdown("## 예측 결과 — Conservative")
    cons_tbl, info_c = make_forecast_table(
        month_temp, sales_df,
        st.session_state["train_years"],
        st.session_state["start_year"], st.session_state["start_month"],
        st.session_state["end_year"],   st.session_state["end_month"],
        st.session_state["dt_cons"]
    )
    st.markdown(center_style_html(cons_tbl), unsafe_allow_html=True)
    st.download_button("Conservative CSV", df_to_csv_bytes(cons_tbl), "conservative.csv", use_container_width=False)

    # 하단 그래프 (Train)
    st.markdown("---")
    st.markdown("## 기온-냉방용 실적 상관관계 (Train, Poly-3)")

    tr = info_n["train"].copy()
    x = tr["기간평균기온"].values
    y = tr["냉방용"].values
    poly, lr = info_n["poly"], info_n["lr"]
    r2 = info_n["r2"]

    xs = np.linspace(np.nanmin(x)-1, np.nanmax(x)+1, 200)
    yhat = poly_predict(poly, lr, xs)

    # 단순 표준편차 기반 ±1.96 밴드(참고용)
    resid = y - poly_predict(poly, lr, x)
    s = np.std(resid)
    ylow, yhigh = yhat - 1.96*s, yhat + 1.96*s

    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(x, y, alpha=0.5, label="학습 샘플")
    ax.plot(xs, yhat, linewidth=2.5, label="Poly-3")
    ax.fill_between(xs, ylow, yhigh, alpha=0.15, label="±1.96")

    ax.set_xlabel("기간평균기온 (m-1,16 ~ m,15)")
    ax.set_ylabel("판매량 (MJ)")
    ax.set_title(f"기온-냉방용 실적 상관관계 (Train, R²={r2:.3f})")
    ax.legend(loc="best")
    st.pyplot(fig)
    st.caption(f"회귀식: {info_n['equation']}")

# -----------------------------
# 공급량 분석(요청이 많아 간단 안내)
# -----------------------------
else:
    st.info("공급량 분석 화면은 판매량(냉방용)과 동일한 ΔT 조작 UX를 유지합니다. "
            "현재 데이터·모델 정의(공급 vs 기온)가 프로젝트마다 달라, "
            "판매량 분석 구조를 기준으로 기온–공급량 Poly-3를 원하시면 공급 엑셀의 "
            "열 이름(연/월/공급열명) 규칙만 알려주세요. 그 규칙에 맞춰 동일한 표/그래프를 바로 붙일 수 있습니다.")
