# app.py — 도시가스 공급·판매 분석 (Poly-3, 통합 최종)

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# =========================================
# 파일 경로 (필요 시 이름만 바꾸세요)
# =========================================
SUPPLY_PATH = "data/상품별공급량_MJ.xlsx"   # 시트: 실적_월합, 날짜 + 제품별 공급량
SALES_PATH  = "data/상품별판매량.xlsx"      # 시트: 실적_월합, 날짜 + 냉방용
TEMP_PATH   = "data/기온.xlsx"               # 시트: 기온, skiprows=8, (날짜, 평균기온(℃))

# =========================================
# 공통 유틸
# =========================================
def year_month_str(y, m):
    return f"{int(y)}.{int(m):02d}"

def fmt_comma(x):
    if pd.isna(x): 
        return ""
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

def style_table_center():
    return """
<style>
table td, table th {
  text-align:center !important;
  vertical-align:middle !important;
}
</style>
"""

# =========================================
# 로딩/가공
# =========================================
@st.cache_data(show_spinner=False)
def load_monthly(path, sheet="실적_월합"):
    df = pd.read_excel(path, sheet_name=sheet)
    if "날짜" not in df.columns:
        raise ValueError(f"{path} - '{sheet}' 시트에 '날짜' 컬럼이 없습니다.")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    # 숫자형만 남기고 정렬
    for c in df.columns:
        if c not in ["날짜","연","월"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["연","월"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_temp_base(path=TEMP_PATH):
    # sheet '기온', skiprows=8, 컬럼 '날짜','평균기온(℃)'
    t = pd.read_excel(path, sheet_name="기온", skiprows=8)
    if "날짜" not in t.columns or "평균기온(℃)" not in t.columns:
        raise ValueError("기온 파일에는 '날짜','평균기온(℃)' 컬럼이 필요합니다.")
    t["날짜"] = pd.to_datetime(t["날짜"])
    t = t.dropna(subset=["날짜"]).copy()
    t["연"] = t["날짜"].dt.year
    t["월"] = t["날짜"].dt.month

    # 당월 평균
    month_avg = (
        t.groupby(["연","월"], as_index=False)["평균기온(℃)"].mean()
         .rename(columns={"평균기온(℃)":"당월평균기온"})
    )
    # '전월16 ~ 당월15' 평균 → 날짜 +15일 후 기준 월
    bucket = t.copy()
    bucket["버킷월"] = (bucket["날짜"] + pd.Timedelta(days=15)).dt.to_period("M")
    period_avg = (
        bucket.groupby("버킷월", as_index=False)["평균기온(℃)"].mean()
              .rename(columns={"평균기온(℃)":"기간평균기온"})
    )
    period_avg["연"] = period_avg["버킷월"].dt.year.astype(int)
    period_avg["월"] = period_avg["버킷월"].dt.month.astype(int)
    period_avg = period_avg.drop(columns=["버킷월"])

    base = pd.merge(month_avg, period_avg, on=["연","월"], how="outer")
    base = base.sort_values(["연","월"]).reset_index(drop=True)
    return base

# =========================================
# Poly-3 모델
# =========================================
def train_poly3(X: np.ndarray, y: np.ndarray):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(X.reshape(-1,1))
    model = LinearRegression()
    model.fit(Xp, y)
    return model, poly

def predict_poly3(model, poly, X: np.ndarray):
    Xp = poly.transform(X.reshape(-1,1))
    return model.predict(Xp)

def poly_equation_text(model):
    # scikit: include_bias=False → coef_ = [c1(x), c2(x^2), c3(x^3)]
    c = model.coef_
    c1 = c[0] if len(c)>0 else 0.0
    c2 = c[1] if len(c)>1 else 0.0
    c3 = c[2] if len(c)>2 else 0.0
    d  = model.intercept_
    return f"y = {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

# =========================================
# 페이지 설정
# =========================================
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.markdown("<h1 style='margin:0'>도시가스 공급·판매 분석 (Poly-3)</h1>", unsafe_allow_html=True)
st.caption("공급량: 기온↔공급량(당월평균) Poly-3 · 판매량(냉방용): (전월16~당월15) 평균기온 기반 Poly-3")

# 사이드바 — 공통 옵션
with st.sidebar:
    st.subheader("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

    st.subheader("데이터 파일")
    st.caption("Repo 기본 경로를 사용합니다. 필요하면 경로만 app.py 위쪽 상수에서 수정하세요.")
    st.text_input("공급량 파일", SUPPLY_PATH)
    st.text_input("판매량 파일", SALES_PATH)
    st.text_input("기온 RAW 파일", TEMP_PATH)

# 데이터 로딩
supply_df = load_monthly(SUPPLY_PATH, "실적_월합")
sales_df  = load_monthly(SALES_PATH,  "실적_월합")
temp_base = load_temp_base(TEMP_PATH)

# 학습 연도 목록
all_years = sorted(supply_df["연"].unique().tolist())
if "pick_years" not in st.session_state:
    st.session_state["pick_years"] = all_years[-3:] if len(all_years)>=3 else all_years

with st.sidebar:
    st.subheader("학습 데이터 연도 선택")
    st.session_state["pick_years"] = st.multiselect(
        "연도", options=all_years, default=st.session_state["pick_years"]
    )

with st.sidebar:
    st.subheader("예측 기간")
    cy = list(range(2015,2031))
    cm = list(range(1,13))
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.selectbox("시작(연)", cy, index=cy.index(max(min(all_years), 2025)))
        start_month= st.selectbox("시작(월)", cm, index=0)
    with c2:
        end_year   = st.selectbox("종료(연)", cy, index=cy.index(max(min(all_years), 2025)))
        end_month  = st.selectbox("종료(월)", cm, index=11)

# ΔT 시나리오 (즉시 반영)
if "dt_normal" not in st.session_state: st.session_state["dt_normal"] = 0.0
if "dt_best"   not in st.session_state: st.session_state["dt_best"]   = -1.0
if "dt_cons"   not in st.session_state: st.session_state["dt_cons"]   = +1.0

def inc(key, step=0.5): st.session_state[key] = round(st.session_state[key]+step, 2)
def dec(key, step=0.5): st.session_state[key] = round(st.session_state[key]-step, 2)

st.markdown("### 시나리오 Δ°C (평균기온 보정)")
co1, co2, co3 = st.columns(3)

with co1:
    st.caption("Normal Δ°C")
    a,b,c = st.columns([1,2,1])
    with a: st.button("−", on_click=dec, args=("dt_normal",), key="n_minus")
    with b: st.markdown(f"<div style='text-align:center;font-size:22px'><b>{st.session_state['dt_normal']:+.2f}</b></div>", unsafe_allow_html=True)
    with c: st.button("+", on_click=inc, args=("dt_normal",), key="n_plus")

with co2:
    st.caption("Best Δ°C")
    a,b,c = st.columns([1,2,1])
    with a: st.button("−", on_click=dec, args=("dt_best",), key="b_minus")
    with b: st.markdown(f"<div style='text-align:center;font-size:22px'><b>{st.session_state['dt_best']:+.2f}</b></div>", unsafe_allow_html=True)
    with c: st.button("+", on_click=inc, args=("dt_best",), key="b_plus")

with co3:
    st.caption("Conservative Δ°C")
    a,b,c = st.columns([1,2,1])
    with a: st.button("−", on_click=dec, args=("dt_cons",), key="c_minus")
    with b: st.markdown(f"<div style='text-align:center;font-size:22px'><b>{st.session_state['dt_cons']:+.2f}</b></div>", unsafe_allow_html=True)
    with c: st.button("+", on_click=inc, args=("dt_cons",), key="c_plus")

# 예측 대상 연월
pr = pd.period_range(
    start=pd.Period(year=start_year, month=start_month, freq="M"),
    end  =pd.Period(year=end_year,   month=end_month,   freq="M"),
    freq="M",
)
target = pd.DataFrame({"연":[p.year for p in pr], "월":[p.month for p in pr]})

# =========================================
# 1) 공급량 분석
# =========================================
if mode == "공급량 분석":
    # 예측에 사용할 제품(열) 목록 추출
    numeric_cols = [c for c in supply_df.columns if c not in ["날짜","연","월"]]
    # 예시 기본 선택 (파일 컬럼명 기준으로 자동)
    default_prods = [x for x in ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"] if x in numeric_cols]
    with st.sidebar:
        st.subheader("예측 대상 제품(공급량) 선택")
        prods = st.multiselect("제품", options=numeric_cols, default=default_prods)

    if not prods:
        st.warning("예측할 제품을 선택하세요.")
        st.stop()

    # 학습 연도 필터
    train_years = sorted(st.session_state["pick_years"])

    # 기능: ΔT(당월평균기온) 보정값을 넣어 테이블 만들기
    def make_supply_table(dt):
        base = pd.merge(target, temp_base[["연","월","당월평균기온"]], on=["연","월"], how="left")
        base["월평균기온(적용)"] = base["당월평균기온"] + dt

        # 각 제품별 Poly-3 학습( X=당월평균기온, y=공급량 )
        for p in prods:
            tr = pd.merge(
                supply_df[["연","월",p]],
                temp_base[["연","월","당월평균기온"]],
                on=["연","월"], how="inner"
            )
            tr = tr.dropna(subset=[p,"당월평균기온"]).copy()
            if train_years:
                tr = tr[tr["연"].isin(train_years)]
            if tr.empty:
                base[p] = 0
                continue
            X = tr["당월평균기온"].values
            y = tr[p].values
            mdl, ply = train_poly3(X, y)
            yhat = np.clip(np.rint(predict_poly3(mdl, ply, base["월평균기온(적용)"].values)), 0, None).astype(int)
            base[p] = yhat

        show = base[["연","월","월평균기온(적용)"] + prods].copy()
        show["연월"] = show.apply(lambda r: year_month_str(r["연"], r["월"]), axis=1)
        show = show.drop(columns=["연","월"])
        show = show[["연월","월평균기온(적용)"] + prods]
        # 종계
        tot_row = ["종계", ""]
        for p in prods:
            tot_row.append(int(show[p].replace("",0).dropna().astype(float).sum()))
        show.loc[len(show)] = tot_row

        # 포맷
        show["월평균기온(적용)"] = show["월평균기온(적용)"].apply(fmt_temp)
        for p in prods:
            show[p] = show[p].apply(fmt_comma)
        return show

    st.markdown("### Normal")
    normal_tbl = make_supply_table(st.session_state["dt_normal"])
    best_tbl   = make_supply_table(st.session_state["dt_best"])
    cons_tbl   = make_supply_table(st.session_state["dt_cons"])

    st.markdown(style_table_center(), unsafe_allow_html=True)
    st.dataframe(normal_tbl, use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("Normal CSV", normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_normal.csv", use_container_width=True)
    with c2:
        st.download_button("Best CSV", best_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_best.csv", use_container_width=True)
    with c3:
        st.download_button("Conservative CSV", cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_conservative.csv", use_container_width=True)

    # ------- 하단 그래프 (Normal 기준) ----------
    st.markdown("### 그래프 (Normal 기준)")
    # 그래프에 사용할 대표 제품 선택(없으면 첫 번째)
    rep_product = prods[0]
    plot_years = st.multiselect("표시할 실적 연도", options=all_years, default=all_years[-5:], key="supply_plot_years")

    # 학습: rep_product, X=당월평균기온, y=공급량
    tr = pd.merge(
        supply_df[["연","월",rep_product]],
        temp_base[["연","월","당월평균기온"]],
        on=["연","월"], how="inner"
    )
    tr = tr.dropna(subset=[rep_product,"당월평균기온"]).copy()
    if train_years:
        tr_fit = tr[tr["연"].isin(train_years)]
    else:
        tr_fit = tr.copy()
    X = tr_fit["당월평균기온"].values
    y = tr_fit[rep_product].values
    mdl, ply = train_poly3(X, y)
    r2 = mdl.score(ply.transform(X.reshape(-1,1)), y)

    # 라인 그래프: 선택 연도의 실제 월별 라인 + 예측 Normal(점선)
    fig, ax = plt.subplots(figsize=(10,5.5))

    for yv in plot_years:
        one = supply_df[supply_df["연"]==yv][["월",rep_product]].dropna()
        if not one.empty:
            ax.plot(one["월"], one[rep_product], label=f"{yv} 실적", alpha=0.9)

    # 예측 Normal: 대상 연월 -> 월평균기온(적용) = 당월평균+dt
    base = pd.merge(target, temp_base[["연","월","당월평균기온"]], on=["연","월"], how="left")
    base["월평균기온(적용)"] = base["당월평균기온"] + st.session_state["dt_normal"]
    yhat = np.clip(np.rint(predict_poly3(mdl, ply, base["월평균기온(적용)"].values)), 0, None)
    ax.plot(base["월"], yhat, "--", lw=2.5, label="예측(Normal)")

    ax.set_title(f"{rep_product} — Poly-3 (Train R²={r2:.3f})")
    ax.set_xlabel("월")
    ax.set_ylabel("공급량 (MJ)")   # 요청 1: y축 라벨 '공급량'으로
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    # 3차 다항식 식을 그래프 시작부에 표시(좌상단)
    eq_text = poly_equation_text(mdl)
    ax.text(0.02, 0.96, f"Poly-3: {eq_text}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
    st.pyplot(fig)

# =========================================
# 2) 판매량 분석(냉방용)
# =========================================
else:
    train_years = sorted(st.session_state["pick_years"])

    # 냉방용 학습 테이블
    join = pd.merge(
        sales_df[["연","월","냉방용"]],
        temp_base[["연","월","기간평균기온","당월평균기온"]],
        on=["연","월"], how="inner"
    ).dropna(subset=["냉방용","기간평균기온"])

    # 학습 데이터(선택 연도)
    trn = join[join["연"].isin(train_years)] if train_years else join
    X = trn["기간평균기온"].values
    y = trn["냉방용"].values
    model, poly = train_poly3(X, y)
    r2 = model.score(poly.transform(X.reshape(-1,1)), y)

    # 예측 테이블 생성 함수 (dt는 기간평균/당월평균 둘 다 보정해서 보여줌)
    def make_sales_table(dt):
        base = pd.merge(target, temp_base, on=["연","월"], how="left")
        base["월평균기온(적용)"] = base["당월평균기온"] + dt
        base["기간평균기온(적용)"] = base["기간평균기온"] + dt
        yhat = np.clip(np.rint(predict_poly3(model, poly, base["기간평균기온(적용)"].values)), 0, None).astype(int)
        base["예측판매량"] = yhat

        show = base[["연","월","월평균기온(적용)","기간평균기온(적용)","예측판매량"]].copy()
        show["연월"] = show.apply(lambda r: year_month_str(r["연"], r["월"]), axis=1)
        show = show.drop(columns=["연","월"])
        show = show[["연월","월평균기온(적용)","기간평균기온(적용)","예측판매량"]]
        # 종계
        show.loc[len(show)] = ["종계","", "", int(show["예측판매량"].replace("",0).dropna().astype(float).sum())]
        # 포맷
        show["월평균기온(적용)"] = show["월평균기온(적용)"].apply(fmt_temp)
        show["기간평균기온(적용)"] = show["기간평균기온(적용)"].apply(fmt_temp)
        show["예측판매량"] = show["예측판매량"].apply(fmt_comma)
        return show

    st.markdown("### Normal")
    normal_tbl = make_sales_table(st.session_state["dt_normal"])
    best_tbl   = make_sales_table(st.session_state["dt_best"])
    cons_tbl   = make_sales_table(st.session_state["dt_cons"])

    st.markdown(style_table_center(), unsafe_allow_html=True)
    st.dataframe(normal_tbl, use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("Normal CSV", normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sales_normal.csv", use_container_width=True)
    with c2:
        st.download_button("Best CSV", best_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sales_best.csv", use_container_width=True)
    with c3:
        st.download_button("Conservative CSV", cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sales_conservative.csv", use_container_width=True)

    # ---------------------- 하단 그래프 (요청 2: 2번째 사진과 같은 구성) ----------------------
    st.markdown("### 그래프 (Normal 기준)")
    plot_years = st.multiselect("표시할 실적 연도", options=all_years, default=all_years[-5:], key="sales_plot_years")

    # 학습(냉방용, 기간평균기온)
    Xf = trn["기간평균기온"].values
    yf = trn["냉방용"].values
    mdl, ply = train_poly3(Xf, yf)
    r2_line = mdl.score(ply.transform(Xf.reshape(-1,1)), yf)

    # 라인 그래프: 연도별 냉방용 실적 + 예측 Normal(점선)
    fig2, ax2 = plt.subplots(figsize=(10,5.5))
    for yv in plot_years:
        one = sales_df[sales_df["연"]==yv][["월","냉방용"]].dropna()
        if not one.empty:
            ax2.plot(one["월"], one["냉방용"], label=f"{yv} 실적", alpha=0.9)

    base = pd.merge(target, temp_base[["연","월","기간평균기온"]], on=["연","월"], how="left")
    base["기간평균기온(적용)"] = base["기간평균기온"] + st.session_state["dt_normal"]
    yhat = np.clip(np.rint(predict_poly3(mdl, ply, base["기간평균기온(적용)"].values)), 0, None)
    ax2.plot(base["월"], yhat, "--", lw=2.5, label="예측(Normal)")

    ax2.set_title(f"냉방용 — Poly-3 (Train R²={r2_line:.3f})")
    ax2.set_xlabel("월")
    ax2.set_ylabel("판매량 (MJ)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    # 3차 다항식 식(대표)을 그래프 시작부 표시
    eq_line = poly_equation_text(mdl)
    ax2.text(0.02, 0.96, f"Poly-3: {eq_line}", transform=ax2.transAxes,
             ha="left", va="top", fontsize=10, color="#1f77b4",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
    st.pyplot(fig2)

    # ---------------------- 마지막 상관 그래프 (요청 3 + 3차식 표기) ----------------------
    st.markdown("### 기온-냉방용 실적 상관관계 (Train)")
    # 산점도 + 회귀곡선 + ±1.96*σ band
    fig3, ax3 = plt.subplots(figsize=(10,6))
    ax3.scatter(trn["기간평균기온"], trn["냉방용"], alpha=0.6, label="학습 샘플")

    xx = np.linspace(trn["기간평균기온"].min()-1, trn["기간평균기온"].max()+1, 200)
    y_pred = predict_poly3(model, poly, xx)
    ax3.plot(xx, y_pred, lw=2.5, label="Poly-3", color="#1f77b4")

    # 근사 불확실성 밴드 (잔차 표준편차 사용)
    resid = trn["냉방용"].values - predict_poly3(model, poly, trn["기간평균기온"].values)
    s = np.std(resid)
    ax3.fill_between(xx, y_pred-1.96*s, y_pred+1.96*s, color="#1f77b4", alpha=0.15, label="±1.96")

    # 온도별 중앙값(참고용) — 온도 구간 binning
    bins = np.linspace(trn["기간평균기온"].min(), trn["기간평균기온"].max(), 15)
    gb = trn.groupby(pd.cut(trn["기간평균기온"], bins))["냉방용"].median().reset_index()
    gb["x"] = [i.mid for i in gb["기간평균기온"]]
    ax3.scatter(gb["x"], gb["냉방용"], edgecolors="none", s=65, color="#ff7f0e", label="온도별 중앙값")

    ax3.set_xlabel("기간평균기온 (℃)")
    ax3.set_ylabel("판매량 (MJ)")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")

    # 3차 다항식 식을 왼쪽 아래(그래프 시작 쪽)에 표시
    eq_text = poly_equation_text(model)
    xmin, xmax = ax3.get_xlim()
    ymin, ymax = ax3.get_ylim()
    ax3.text(xmin + 0.02*(xmax-xmin), ymin + 0.06*(ymax-ymin),
             f"Poly-3: {eq_text}",
             fontsize=10, color="#1f77b4",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
    st.pyplot(fig3)

    st.caption(f"Train R² = {r2:.3f}")
