# app.py — 도시가스 공급·판매 분석 (Poly-3, 안정화 버전)

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ===== 기본 설정 =====
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
warnings.filterwarnings("ignore")

# ===== 유틸 =====
DATA_DIR = Path(__file__).parent / "data"

# 화면용 CSS: 표 중앙정렬/세로정렬, index 숨김
st.markdown("""
<style>
table.dataframe td, table.dataframe th {
  text-align:center !important; vertical-align:middle !important;
}
thead th { text-align:center !important; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


def center_format(df: pd.DataFrame) -> pd.DataFrame:
    """연·월을 'YYYY.MM'으로 만들고, 모든 숫자 서식 통일(기온 한 자리, 수치 콤마)"""
    show = df.copy()

    # 중복 컬럼 제거(에러의 주범)
    show = show.loc[:, ~show.columns.duplicated()]

    # 연월 가공
    if "연" in show.columns and "월" in show.columns:
        show.insert(0, "연월", show["연"].astype(int).astype(str) + "." + show["월"].astype(int).astype(str).str.zfill(2))
        show.drop(columns=["연", "월"], inplace=True, errors="ignore")

    # 숫자형 강제 변환(문자→숫자)
    for c in show.columns:
        if c.find("기온") >= 0:  # 기온은 소수1자리
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
        else:
            # 수치(판매/공급량) → 콤마 포맷
            show[c] = pd.to_numeric(show[c], errors="coerce")
            show[c] = show[c].apply(lambda x: f"{int(round(x)):,}" if pd.notna(x) else "")

    return show


def csv_download(df: pd.DataFrame, label: str):
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(label=label, data=csv, file_name=f"{label}.csv", mime="text/csv")


# ===== 파일 로더 (동의어/유연성 강화) =====
def load_excel(path_or_file):
    if path_or_file is None:
        return None
    if isinstance(path_or_file, (str, Path)):
        return pd.read_excel(path_or_file)
    # UploadedFile
    return pd.read_excel(path_or_file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """자주 쓰는 한국어 컬럼 동의어를 정규화"""
    if df is None or df.empty:
        return df

    mapping = {
        # 공통
        "년": "연", "년도": "연",
        "month": "월", "월(숫자)": "월",

        # 판매
        "냉방용": "냉방", "실제판매량": "냉방", "판매량": "냉방",

        # 기온
        "평균기온": "당월평균기온",
        "기간평균기온(전월16~당월15)": "기간평균기온",
        "기간 평균기온": "기간평균기온",
        "당월 평균기온": "당월평균기온",
        "월평균기온(적용)": "월평균기온(적용)",  # 그대로
        "일시": "일자",

        # 공급(샘플과 다를 수 있으므로 필요시 추가)
        "개별난방": "개별난방용",
        "중앙난방": "중앙난방용",
        "자가열": "자가열전용",
        "일반용2": "일반용(2)",
        "업무난방": "업무난방용",
        "총합": "총공급량",
    }

    new = {}
    for c in df.columns:
        cc = c.strip()
        cc = mapping.get(cc, cc)
        new[c] = cc
    out = df.rename(columns=new)

    # 일자 → 연/월 파생
    if "일자" in out.columns and ("연" not in out.columns or "월" not in out.columns):
        d = pd.to_datetime(out["일자"], errors="coerce")
        out["연"] = d.dt.year
        out["월"] = d.dt.month

    return out


# ===== 데이터 로딩 (좌측 사이드바) =====
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

    st.header("데이터 불러오기")
    load_kind = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    if mode == "공급량 분석":
        st.write("실적 파일(공급·판매·기온 중 필요한 것만 사용)")
        if load_kind == "Repo 내 파일 사용":
            supply_file = st.selectbox(
                "공급 파일",
                options=[DATA_DIR / "상품별공급량_MJ.xlsx", None],
                format_func=lambda p: str(p) if p else "선택 안 함",
            )
            temp_file = st.selectbox(
                "기온 파일(월별)",
                options=[DATA_DIR / "기온.xlsx", None],
                format_func=lambda p: str(p) if p else "선택 안 함",
            )
            sales_file = None
        else:
            supply_file = st.file_uploader("공급 파일 업로드(.xlsx)", type=["xlsx"], key="up_supply")
            temp_file = st.file_uploader("기온 파일(월별) 업로드(.xlsx)", type=["xlsx"], key="up_temp")
            sales_file = None

        # 학습연도
        st.header("학습 데이터 연도 선택")
        years = st.multiselect("연도 선택", list(range(2017, 2031)), default=[2021, 2022, 2023, 2024])
        years = sorted(years)

        # 예측 기간(가로 UI)
        st.header("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            s_year = st.selectbox("예측 시작(연)", list(range(2017, 2031)), index=list(range(2017, 2031)).index(2025))
        with c2:
            s_month = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0)

        c3, c4 = st.columns(2)
        with c3:
            e_year = st.selectbox("예측 종료(연)", list(range(2017, 2031)), index=list(range(2017, 2031)).index(2025))
        with c4:
            e_month = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11)

    else:
        # 판매량
        if load_kind == "Repo 내 파일 사용":
            sales_file = st.selectbox(
                "판매 실적 파일",
                options=[DATA_DIR / "상품별판매량.xlsx", None],
                format_func=lambda p: str(p) if p else "선택 안 함",
            )
            temp_file = st.selectbox(
                "기온 파일(월별)",
                options=[DATA_DIR / "기온.xlsx", None],
                format_func=lambda p: str(p) if p else "선택 안 함",
            )
        else:
            sales_file = st.file_uploader("판매 실적 업로드(.xlsx)", type=["xlsx"], key="up_sales")
            temp_file = st.file_uploader("기온 파일(월별) 업로드(.xlsx)", type=["xlsx"], key="up_temp2")

        st.header("학습 데이터 연도 선택")
        years = st.multiselect("연도 선택", list(range(2017, 2031)), default=[2021, 2022, 2023, 2024])
        years = sorted(years)

        st.header("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            s_year = st.selectbox("예측 시작(연)", list(range(2017, 2031)), index=list(range(2017, 2031)).index(2025))
        with c2:
            s_month = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0)

        c3, c4 = st.columns(2)
        with c3:
            e_year = st.selectbox("예측 종료(연)", list(range(2017, 2031)), index=list(range(2017, 2031)).index(2025))
        with c4:
            e_month = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11)

# ========== 공통 로딩 ==========
temp_df = normalize_columns(load_excel(temp_file)) if temp_file else None
supply_df = normalize_columns(load_excel(supply_file)) if supply_file else None
sales_df = normalize_columns(load_excel(sales_file)) if sales_file else None


def select_year_range(df, y1, m1, y2, m2):
    if df is None or df.empty:
        return df
    df = df.copy()
    # 보조 키(연*100+월)
    key = df["연"].astype(int) * 100 + df["월"].astype(int)
    start = int(y1) * 100 + int(m1)
    end = int(y2) * 100 + int(m2)
    return df[(key >= start) & (key <= end)].copy()


def train_poly3(x, y):
    """x 1D → poly3 fit"""
    X = np.array(x).reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(Xp, y)
    return model, poly


def predict_poly3(model, poly, x):
    X = np.array(x).reshape(-1, 1)
    Xp = poly.transform(X)
    return model.predict(Xp)


# ===== 델타 T(±0.5) — 즉시 반영 버튼 =====
st.title("도시가스 공급·판매 분석 (Poly-3)")

# ===== ΔT(±0.5) — 즉시 반영 버튼 (DuplicateWidgetID 방지) =====
st.subheader("ΔT 시나리오 (℃)")

for k in ["dt_normal", "dt_best", "dt_cons"]:
    if k not in st.session_state:
        st.session_state[k] = 0.0

def minus(key):
    st.session_state[key] = round(st.session_state[key] - 0.5, 2)

def plus(key):
    st.session_state[key] = round(st.session_state[key] + 0.5, 2)

cols = st.columns(3)

with cols[0]:
    st.markdown("**ΔT(Normal)**")
    c1, c2, _ = st.columns([1,1,3])
    with c1:
        st.button("−", key="btn_norm_minus", on_click=minus, args=("dt_normal",))
    with c2:
        st.button("+", key="btn_norm_plus",  on_click=plus,  args=("dt_normal",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_normal:+.2f}")

with cols[1]:
    st.markdown("**ΔT(Best)**")
    c1, c2, _ = st.columns([1,1,3])
    with c1:
        st.button("−", key="btn_best_minus", on_click=minus, args=("dt_best",))
    with c2:
        st.button("+", key="btn_best_plus",  on_click=plus,  args=("dt_best",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_best:+.2f}")

with cols[2]:
    st.markdown("**ΔT(Conservative)**")
    c1, c2, _ = st.columns([1,1,3])
    with c1:
        st.button("−", key="btn_cons_minus", on_click=minus, args=("dt_cons",))
    with c2:
        st.button("+", key="btn_cons_plus",  on_click=plus,  args=("dt_cons",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_cons:+.2f}")

# ===== 판매량 분석 =====
if mode == "판매량 분석(냉방용)":
    if sales_df is None or temp_df is None:
        st.warning("좌측에서 **판매 실적 파일**과 **기온 파일**을 선택/업로드 해 주세요.")
        st.stop()

    # 필수 컬럼 존재 확인 및 보정
    need_cols_sales = ["연", "월"]
    for c in need_cols_sales:
        if c not in sales_df.columns:
            st.error(f"판매 실적 파일에 '{c}' 컬럼이 없습니다.")
            st.stop()

    # 냉방 컬럼 후보
    value_col = None
    for cand in ["냉방", "냉방용", "실제판매량", "판매량", "냉방 실적"]:
        if cand in sales_df.columns:
            value_col = cand
            break
    if value_col is None:
        st.error("판매 실적: '냉방/냉방용/실제판매량/판매량' 중 하나의 **숫자 컬럼**이 필요합니다.")
        st.stop()

    # 기온(월별)
    if ("연" not in temp_df.columns) or ("월" not in temp_df.columns):
        st.error("기온 파일(월별)에는 '연','월' 컬럼이 필요합니다.")
        st.stop()

    # 기간평균기온 컬럼 만들기(없으면 당월평균기온 사용)
    if "기간평균기온" not in temp_df.columns:
        if "월평균기온(적용)" in temp_df.columns:
            temp_df["기간평균기온"] = temp_df["월평균기온(적용)"]
        elif "당월평균기온" in temp_df.columns:
            temp_df["기간평균기온"] = temp_df["당월평균기온"]
        else:
            st.error("기온 파일에 '기간평균기온' 또는 '월평균기온(적용)' 또는 '당월평균기온'이 필요합니다.")
            st.stop()

    # 학습데이터 준비
    train_sales = sales_df.merge(temp_df, on=["연", "월"], how="inner")
    train_sales = train_sales[train_sales["연"].isin(years)].copy()

    if train_sales.empty:
        st.error("선택한 학습 연도에 해당하는 데이터가 없습니다.")
        st.stop()

    # Poly-3 학습 (X: 기간평균기온, y: 냉방)
    x = pd.to_numeric(train_sales["기간평균기온"], errors="coerce")
    y = pd.to_numeric(train_sales[value_col], errors="coerce")
    mask = x.notna() & y.notna()
    model, poly = train_poly3(x[mask], y[mask])

    # 예측용 베이스(기간 선택)
    base = select_year_range(temp_df, s_year, s_month, e_year, e_month)
    if base is None or base.empty:
        st.error("예측 구간에 해당하는 기온 데이터가 없습니다.")
        st.stop()

    def make_table_with_delta(delta, tag):
        tmp = base.copy()
        # 적용 기온(기간평균기온 + ΔT)
        tmp["월평균기온(적용)"] = pd.to_numeric(tmp["기간평균기온"], errors="coerce") + float(delta)
        tmp["예측판매량"] = np.clip(
            np.rint(predict_poly3(model, poly, tmp["월평균기온(적용)"].values)), 0, None
        )
        tbl = tmp[["연", "월", "당월평균기온", "기간평균기온", "월평균기온(적용)", "예측판매량"]].copy()
        # 합계행
        s = pd.DataFrame({"연": [np.nan], "월": [np.nan],
                          "당월평균기온": [np.nan], "기간평균기온": [np.nan],
                          "월평균기온(적용)": [np.nan], "예측판매량": [tbl["예측판매량"].sum()]})
        tbl = pd.concat([tbl, s], ignore_index=True)
        st.markdown(f"### 예측 결과 — {tag}")
        st.dataframe(center_format(tbl), use_container_width=True)
        csv_download(tbl, f"{tag}CSV")
        return tbl

    # Normal / Best / Conservative
    t1 = make_table_with_delta(st.session_state.dt_normal, "Normal")
    t2 = make_table_with_delta(st.session_state.dt_best, "Best")
    t3 = make_table_with_delta(st.session_state.dt_cons, "Conservative")

    # 상관관계 그래프 (Train, Poly-3)
    import matplotlib.pyplot as plt

    # 그래프 데이터
    xx = np.linspace(x.min()-2, x.max()+2, 200)
    yy = predict_poly3(model, poly, xx)

    fig, ax = plt.subplots(figsize=(9,6), dpi=140)
    ax.scatter(x, y, alpha=0.35, label="학습 샘플")
    ax.plot(xx, yy, lw=3, label="Poly-3")
    ax.set_xlabel("기간평균기온 (℃)")
    ax.set_ylabel("판매량 (MJ)")
    # R2
    r2 = model.score(poly.transform(np.array(x).reshape(-1,1)), y)
    ax.set_title(f"기온-냉방용 실적 상관관계 (Train, R²={r2:.3f})")
    ax.legend()
    st.pyplot(fig)

# ===== 공급량 분석 (온도-공급량 Poly-3; 기온파일과 공급파일 병합) =====
else:
    if supply_df is None or temp_df is None:
        st.warning("좌측에서 **공급 파일**과 **기온 파일**을 선택/업로드 해 주세요.")
        st.stop()

    # 공급 파일 필수
    need_cols = ["연", "월"]
    for c in need_cols:
        if c not in supply_df.columns:
            st.error(f"공급 파일에 '{c}' 컬럼이 없습니다.")
            st.stop()

    # 예시 제품 컬럼 후보(파일 상황에 맞게 존재하는 것만 사용)
    product_candidates = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]
    prod_cols = [c for c in product_candidates if c in supply_df.columns]
    if not prod_cols:
        st.error("공급 파일에서 제품별 컬럼을 찾지 못했습니다. (예: 개별난방용, 중앙난방용 …)")
        st.stop()

    # 기온 컬럼 준비
    if "당월평균기온" not in temp_df.columns:
        st.error("기온 파일에 '당월평균기온' 컬럼이 필요합니다.")
        st.stop()

    # 학습 데이터(병합)
    train = supply_df.merge(temp_df[["연","월","당월평균기온"]], on=["연","월"], how="inner")
    train = train[train["연"].isin(years)].copy()

    if train.empty:
        st.error("선택한 학습 연도에 해당하는 데이터가 없습니다.")
        st.stop()

    # 제품별로 Poly-3 학습 후, 예측기간 temp로 예측
    base = select_year_range(temp_df, s_year, s_month, e_year, e_month)
    if base is None or base.empty:
        st.error("예측 구간에 해당하는 기온 데이터가 없습니다.")
        st.stop()

    def make_supply_table(delta, tag):
        out = base[["연","월","당월평균기온"]].copy()
        out["월평균기온(적용)"] = pd.to_numeric(out["당월평균기온"], errors="coerce") + float(delta)

        for p in prod_cols:
            # 학습 데이터
            xx = pd.to_numeric(train["당월평균기온"], errors="coerce")
            yy = pd.to_numeric(train[p], errors="coerce")
            msk = xx.notna() & yy.notna()
            if msk.sum() < 4:
                # 데이터 너무 적으면 스킵
                out[p] = np.nan
                continue
            m, poly = train_poly3(xx[msk], yy[msk])
            out[p] = np.clip(np.rint(predict_poly3(m, poly, out["월평균기온(적용)"].values)), 0, None)

        # 총합 없으면 만들어주기
        if "총공급량" not in out.columns:
            out["총공급량"] = 0
            for p in prod_cols:
                out["총공급량"] = out["총공급량"] + pd.to_numeric(out[p], errors="coerce").fillna(0)

        # 합계행
        s = {"연": np.nan, "월": np.nan, "당월평균기온": np.nan, "월평균기온(적용)": np.nan}
        for p in prod_cols + (["총공급량"] if "총공급량" in out.columns else []):
            s[p] = out[p].sum()
        out = pd.concat([out, pd.DataFrame([s])], ignore_index=True)

        st.markdown(f"### 예측 결과 — {tag}")
        st.dataframe(center_format(out), use_container_width=True)
        csv_download(out, f"{tag}CSV")
        return out

    _ = make_supply_table(st.session_state.dt_normal, "Normal")
    _ = make_supply_table(st.session_state.dt_best, "Best")
    _ = make_supply_table(st.session_state.dt_cons, "Conservative")
