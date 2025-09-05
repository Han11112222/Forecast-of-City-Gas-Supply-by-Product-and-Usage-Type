# app.py — 도시가스 공급·판매 분석 (Poly-3)

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

# =============== 기본 환경 ===============
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# =============== 한글 폰트 ===============
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =============== 공통 유틸 ===============
TEMP_HINTS = ["평균기온","기온","temp","temperature"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """엑셀을 표준 형태로 보정: 연/월 생성, 숫자 문자열 정리 등"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜/일자 인식
    date_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("날짜","일자","date"):
            date_col = c
            break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 연/월 생성
    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = pd.to_numeric(df["년"], errors="coerce")
        elif date_col is not None:
            df["연"] = df[date_col].dt.year
    if "월" not in df.columns and date_col is not None:
        df["월"] = df[date_col].dt.month

    # 문자열 숫자 클린
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if any(h in lc for h in TEMP_HINTS) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def to_yearmonth(y, m) -> str:
    try:
        return f"{int(y)}.{int(m):02d}"
    except Exception:
        return ""

def format_table(df: pd.DataFrame, temp_cols: list[str] = None):
    """기온(소수1), 나머지 천단위 콤마, 중앙정렬 HTML 표로 출력"""
    if temp_cols is None:
        temp_cols = []

    show = df.copy()
    for c in show.columns:
        if c in temp_cols:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )
        elif c not in ["연월"]:
            show[c] = pd.to_numeric(show[c], errors="ignore")
            if pd.api.types.is_numeric_dtype(show[c]):
                show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{int(round(x)):,}")
    st.markdown("""
    <style>
      table.centered-table {width:100%;}
      table.centered-table th, table.centered-table td {
        text-align:center !important; vertical-align:middle !important;
      }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(show.to_html(index=False, classes="centered-table"), unsafe_allow_html=True)

# =============== 기온 RAW 읽기 ===============
@st.cache_data(ttl=600)
def read_temperature_raw(file):
    """
    CSV/XLSX 모두 지원. 컬럼 유추:
    - 날짜/일자/date 중 하나 → '일자'
    - 기온/평균기온/temp/temperature 중 하나 → '기온'
    """
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        xls = pd.ExcelFile(file, engine="openpyxl")
        raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    raw.columns = [str(c).strip() for c in raw.columns]

    # 날짜/기온 컬럼 탐색
    date_col = None
    for c in raw.columns:
        if str(c).lower() in ("날짜","일자","date"):
            date_col = c
            break
    if date_col is None:
        for c in raw.columns:
            try:
                pd.to_datetime(raw[c], errors="raise")
                date_col = c
                break
            except Exception:
                pass
    if date_col is None:
        st.error("기온 RAW: 날짜(일자) 컬럼을 찾지 못했습니다.")
        return pd.DataFrame(columns=["일자","기온"])

    temp_col = None
    for c in raw.columns:
        lc = str(c).lower()
        if ("기온" in str(c)) or (lc in ("temp","temperature","평균기온")):
            temp_col = c
            break
    if temp_col is None:
        st.error("기온 RAW: 기온(평균기온) 컬럼을 찾지 못했습니다.")
        return pd.DataFrame(columns=["일자","기온"])

    out = pd.DataFrame({
        "일자": pd.to_datetime(raw[date_col], errors="coerce"),
        "기온": pd.to_numeric(
            raw[temp_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False),
            errors="coerce"
        )
    }).dropna(subset=["일자","기온"]).sort_values("일자")
    return out.reset_index(drop=True)

# =============== Poly-3 ===============
def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train = x_train[m].reshape(-1, 1)
    y_train = y_train[m]
    Xf = x_future.reshape(-1, 1)

    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)

    y_future = model.predict(poly.transform(Xf))
    return y_future, r2, model, poly

def poly_equation_str(model, poly, decimals=5):
    # model.coef_ : [a1, a2, a3] (x, x^2, x^3) 순서
    a1, a2, a3 = model.coef_
    a0 = model.intercept_
    return f"y = {a3:+.{decimals}e}x³ {a2:+.{decimals}e}x² {a1:+.{decimals}e}x {a0:+.{decimals}e}"

# =============== 상관관계 그래프(판매) ===============
def show_sales_scatter(train_x, train_y, model, poly, r2, title="기온-냉방용 실적 상관관계 (Train)"):
    xs = np.linspace(np.nanmin(train_x)-1, np.nanmax(train_x)+1, 200)
    ys = model.predict(poly.transform(xs.reshape(-1,1)))

    fig = plt.figure(figsize=(9,4.8))
    ax = plt.gca()
    ax.scatter(train_x, train_y, alpha=0.6, label="학습 샘플")
    ax.plot(xs, ys, linewidth=2.5, label="Poly-3")

    eq = poly_equation_str(model, poly, decimals=5)
    ax.text(0.99, 0.02, f"{eq}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_title(f"{title} (R²={r2:.3f})")
    ax.set_xlabel("기간평균기온 (m-1, 16일 ~ m15일)")
    ax.set_ylabel("판매량 (MJ)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# =============== ΔT 컨트롤 ===============
def scenario_controls():
    st.markdown("### ΔT 시나리오 (°C)")

    def _btns(key):
        c1, c2, c3 = st.columns([1,1,6])
        with c1:
            if st.button("−", key=f"minus_{key}"):
                st.session_state[key] = max(-5.0, round(st.session_state.get(key,0.0)-0.5,2))
        with c2:
            if st.button("+", key=f"plus_{key}"):
                st.session_state[key] = min( 5.0, round(st.session_state.get(key,0.0)+0.5,2))
        with c3:
            st.metric("기온 보정(°C)", f"{st.session_state.get(key,0.0):.2f}")

    colN, colB, colC = st.columns(3)
    with colN:
        st.subheader("ΔT(Normal)")
        if "dT_norm" not in st.session_state: st.session_state["dT_norm"] = 0.0
        _btns("dT_norm")
    with colB:
        st.subheader("ΔT(Best)")
        if "dT_best" not in st.session_state: st.session_state["dT_best"] = 0.0
        _btns("dT_best")
    with colC:
        st.subheader("ΔT(Conservative)")
        if "dT_cons" not in st.session_state: st.session_state["dT_cons"] = 0.0
        _btns("dT_cons")

# =============== 사이드바 ===============
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

    st.header("데이터 불러오기")
    source = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    data_dir = here / "data"
    data_dir.mkdir(exist_ok=True)

# =============== 공급량 분석 ===============
if mode == "공급량 분석":
    with st.sidebar:
        st.subheader("실적 파일(Excel)")
        if source == "Repo 내 파일 사용":
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx") if "공급" in p.name or "상품" in p.name])
            file_path = st.selectbox("파일 선택", repo_files) if repo_files else None
            file_obj = open(file_path, "rb") if file_path else None
        else:
            file_obj = st.file_uploader("공급 실적 엑셀 업로드", type=["xlsx","xls"])

        df = None
        if file_obj is not None:
            try:
                xls = pd.ExcelFile(file_obj, engine="openpyxl")
                df = pd.read_excel(xls, sheet_name=0)
            except Exception:
                df = pd.read_excel(file_obj, engine="openpyxl")

        if df is None:
            st.info("공급 실적 파일을 선택/업로드해주세요.")
            st.stop()

        df = normalize_cols(df)
        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("실적 파일에서 기온 컬럼을 찾지 못했습니다. (예: 평균기온/기온/temp)")
            st.stop()

        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("학습 데이터 연도", years_all, default=years_all[-5:] if years_all else [])

        st.subheader("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            sy = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(df["연"].max())))
        with c2:
            sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        c3, c4 = st.columns(2)
        with c3:
            ey = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(df["연"].max())))
        with c4:
            em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("예측 시작", type="primary")

    st.title("도시가스 공급·판매 분석 (Poly-3)")
    st.caption("공급량: 기온↔공급량 3차 다항식")

    scenario_controls()

    if not run_btn and "supply_ctx" not in st.session_state:
        st.info("좌측에서 학습 연도/예측 기간/파일을 선택 후 **예측 시작**을 눌러주세요.")
        st.stop()

    if run_btn:
        # 학습 데이터 준비
        base = df.dropna(subset=["연","월",temp_col]).copy()
        base = base.sort_values(["연","월"]).reset_index(drop=True)
        # 상품(용도) 후보: 숫자형 & 메타 제외
        meta = {"연","월",temp_col,"날짜","일자","date"}
        product_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c]) and c not in meta]
        # 학습 연도 필터
        train_df = base[base["연"].isin(years_sel)].copy()

        # 모델 per 상품
        ctx = {"models":{}, "poly":{}, "r2":{}, "temp_col":temp_col, "product_cols":product_cols}
        x_tr = train_df[temp_col].astype(float).values
        for col in product_cols:
            y_tr = train_df[col].astype(float).values
            _, r2, model, poly = fit_poly3_and_predict(x_tr, y_tr, x_tr)
            ctx["models"][col] = model
            ctx["poly"][col]   = poly
            ctx["r2"][col]     = r2

        st.session_state["supply_ctx"] = ctx
        st.session_state["supply_base"] = base

        # 예측 인덱스
        f_start = pd.Timestamp(year=int(sy), month=int(sm), day=1)
        f_end   = pd.Timestamp(year=int(ey), month=int(em), day=1)
        fut_idx = pd.date_range(start=f_start, end=f_end, freq="MS")
        st.session_state["supply_fut_idx"] = fut_idx

    # -------- 시나리오 테이블 생성 함수 --------
    def make_supply_table(dT: float):
        base = st.session_state["supply_base"]
        ctx  = st.session_state["supply_ctx"]
        fut_idx = st.session_state["supply_fut_idx"]

        # 월평균기온(학습 데이터 기준)
        monthly_avg = base.groupby("월")[ctx["temp_col"]].mean()

        fut = pd.DataFrame({
            "연": fut_idx.year,
            "월": fut_idx.month,
            "연월": [to_yearmonth(y,m) for y,m in zip(fut_idx.year, fut_idx.month)],
            "월평균기온(적용)": (fut_idx.month.map(monthly_avg) + dT).values
        })

        # 상품별 예측
        for col in ctx["product_cols"]:
            Xf = fut["월평균기온(적용)"].astype(float).values
            yf = ctx["models"][col].predict(ctx["poly"][col].transform(Xf.reshape(-1,1)))
            fut[col] = np.clip(np.rint(yf).astype(np.int64), a_min=0, a_max=None)

        # 총공급량(있으면 그대로, 없으면 합계)
        if "총공급량" not in fut.columns:
            fut["총공급량"] = fut[ctx["product_cols"]].sum(axis=1)

        # 정렬 및 표시
        show = fut[["연월","월평균기온(적용)"] + ctx["product_cols"] + ["총공급량"]].copy()
        return show

    # -------- 표시 --------
    st.markdown("## 예측 결과 — Normal")
    tblN = make_supply_table(st.session_state["dT_norm"])
    format_table(tblN, temp_cols=["월평균기온(적용)"])
    st.download_button("Normal CSV", data=tblN.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="supply_normal.csv")

    st.markdown("## 예측 결과 — Best")
    tblB = make_supply_table(st.session_state["dT_best"])
    format_table(tblB, temp_cols=["월평균기온(적용)"])
    st.download_button("Best CSV", data=tblB.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="supply_best.csv")

    st.markdown("## 예측 결과 — Conservative")
    tblC = make_supply_table(st.session_state["dT_cons"])
    format_table(tblC, temp_cols=["월평균기온(적용)"])
    st.download_button("Conservative CSV", data=tblC.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="supply_conservative.csv")

# =============== 판매량 분석 (냉방용) ===============
else:
    with st.sidebar:
        st.subheader("판매 실적 & 기온 RAW")
        if source == "Repo 내 파일 사용":
            repo_sales = sorted([str(p) for p in data_dir.glob("*.xlsx") if "판매" in p.name or "상품" in p.name])
            repo_temp  = sorted([str(p) for p in data_dir.glob("*.xlsx") if "기온" in p.name] +
                                [str(p) for p in data_dir.glob("*.csv") if "temp" in p.name or "기온" in p.name])
            spath = st.selectbox("판매 실적 파일", repo_sales) if repo_sales else None
            tpath = st.selectbox("기온 RAW 파일",  repo_temp ) if repo_temp  else None
            sales_file = open(spath, "rb") if spath else None
            temp_file  = open(tpath, "rb") if tpath else None
        else:
            sales_file = st.file_uploader("판매 실적 엑셀 업로드", type=["xlsx","xls"])
            temp_file  = st.file_uploader("기온 RAW 업로드 (xlsx/csv)", type=["xlsx","xls","csv"])

        if sales_file is None or temp_file is None:
            st.info("두 파일을 모두 선택/업로드해주세요.")
            st.stop()

        # 판매 실적 읽기
        try:
            xls = pd.ExcelFile(sales_file, engine="openpyxl")
            sraw = pd.read_excel(xls, sheet_name=0)
        except Exception:
            sraw = pd.read_excel(sales_file, engine="openpyxl")

        sraw = normalize_cols(sraw)

        # 날짜 / 냉방용 찾기
        # 날짜: 판매월/날짜/일자/date 중 1개
        date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sraw.columns]
        date_col = date_candidates[0] if date_candidates else None
        if date_col is None:
            # 날짜성 컬럼 자동 추정
            for c in sraw.columns:
                try:
                    pd.to_datetime(sraw[c], errors="raise")
                    date_col = c; break
                except Exception:
                    pass
        if date_col is None:
            st.error("판매 실적: 날짜(판매월/날짜/일자) 컬럼을 찾지 못했습니다.")
            st.stop()

        # 냉방용 값 추정
        cool_cols = [c for c in sraw.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sraw[c])]
        if not cool_cols:
            st.error("판매 실적: '냉방' 수치 컬럼을 찾지 못했습니다.")
            st.stop()
        value_col = cool_cols[0]
        for c in cool_cols:
            if "냉방용" in str(c):
                value_col = c; break

        # 정리
        sales_df = pd.DataFrame({
            "판매월": pd.to_datetime(sraw[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
            "판매량": pd.to_numeric(sraw[value_col], errors="coerce")
        }).dropna().copy()
        sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
        sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("학습 데이터 연도", years_all, default=years_all[-5:] if years_all else [])

        st.subheader("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            sy = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(sales_df["연"].max())))
        with c2:
            sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        c3, c4 = st.columns(2)
        with c3:
            ey = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(sales_df["연"].max())))
        with c4:
            em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("예측 시작", type="primary")

    st.title("도시가스 공급·판매 분석 (Poly-3)")
    st.caption("판매량(냉방용): (전월16~당월15) 기간평균기온 기반 3차 다항식")

    scenario_controls()

    if not run_btn and "sales_ctx" not in st.session_state:
        st.info("좌측에서 학습 연도/예측 기간/파일을 선택 후 **예측 시작**을 눌러주세요.")
        st.stop()

    if run_btn:
        temp_raw = read_temperature_raw(temp_file)
        if temp_raw is None or temp_raw.empty:
            st.error("기온 RAW를 읽지 못했습니다.")
            st.stop()

        # 월별 평균기온도 계산(백업용)
        temp_raw["연"] = temp_raw["일자"].dt.year
        temp_raw["월"] = temp_raw["일자"].dt.month
        monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
        fallback_by_M = temp_raw.groupby("월")["기온"].mean()

        # 전월16~당월15 평균
        def period_avg(month_label: pd.Timestamp) -> float | None:
            m = pd.Timestamp(year=month_label.year, month=month_label.month, day=1)
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            vals = pd.to_numeric(temp_raw.loc[mask,"기온"], errors="coerce")
            if vals.size==0 or vals.isna().all():
                return float(fallback_by_M.loc[m.month])
            return float(vals.mean())

        train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
        rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
        sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
        miss = sj["기간평균기온"].isna()
        if miss.any():
            sj.loc[miss,"기간평균기온"] = sj.loc[miss,"판매월"].dt.month.map(fallback_by_M)
        sj = sj.dropna(subset=["기간평균기온","판매량"])

        x_train = sj["기간평균기온"].astype(float).values
        y_train = sj["판매량"].astype(float).values
        _, r2_fit, model, poly = fit_poly3_and_predict(x_train, y_train, x_train)

        st.session_state["sales_ctx"] = {
            "model":model, "poly":poly, "r2":r2_fit,
            "fallback_by_M": fallback_by_M,
            "period_avg_func": period_avg
        }
        st.session_state["sales_base"] = sales_df

        f_start = pd.Timestamp(year=int(sy), month=int(sm), day=1)
        f_end   = pd.Timestamp(year=int(ey), month=int(em), day=1)
        fut_idx = pd.date_range(start=f_start, end=f_end, freq="MS")
        st.session_state["sales_fut_idx"] = fut_idx

    # -------- 시나리오 테이블 생성 함수 --------
    def make_sales_table(dT: float):
        base = st.session_state["sales_base"]
        ctx  = st.session_state["sales_ctx"]
        fut_idx = st.session_state["sales_fut_idx"]
        period_avg = ctx["period_avg_func"]

        rows = []
        for m in fut_idx:
            pavg = period_avg(m)
            rows.append({"연월": to_yearmonth(m.year,m.month),
                         "당월평균기온": pd.to_numeric(base.loc[(base['연']==m.year)&(base['월']==m.month),'판매량']).mean()*0, # dummy for column order
                         "기간평균기온": pavg + dT})

        fut = pd.DataFrame(rows)
        # 당월평균기온은 보기용(월 평균). RAW가 있으면 계산, 없으면 NaN
        # 여기선 시각적 정합 위해 기간평균만 사용

        Xf = fut["기간평균기온"].astype(float).values
        yf = ctx["model"].predict(ctx["poly"].transform(Xf.reshape(-1,1)))
        fut["예측판매량"] = np.clip(np.rint(yf).astype(np.int64), a_min=0, a_max=None)

        # 실제 및 오차(검증용: 예측 구간에 실제가 있으면 표시)
        actual = base.copy()
        actual["연월"] = [to_yearmonth(y,m) for y,m in zip(actual["연"], actual["월"])]
        actual = actual[["연월","판매량"]].rename(columns={"판매량":"실제판매량"})
        out = pd.merge(fut, actual, on="연월", how="left")
        out["오차"] = (out["예측판매량"] - out["실제판매량"]).astype("Int64")
        out["오차율(%)"] = (out["오차"]/out["실제판매량"]*100).round(1).astype("Float64")
        out = out[["연월","당월평균기온","기간평균기온","예측판매량","실제판매량","오차","오차율(%)"]]
        return out

    # -------- 표시 --------
    st.markdown("## 예측 결과 — Normal")
    tblN = make_sales_table(st.session_state["dT_norm"])
    format_table(tblN, temp_cols=["당월평균기온","기간평균기온"])
    st.download_button("Normal CSV", data=tblN.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="sales_normal.csv")

    st.markdown("## 예측 결과 — Best")
    tblB = make_sales_table(st.session_state["dT_best"])
    format_table(tblB, temp_cols=["당월평균기온","기간평균기온"])
    st.download_button("Best CSV", data=tblB.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="sales_best.csv")

    st.markdown("## 예측 결과 — Conservative")
    tblC = make_sales_table(st.session_state["dT_cons"])
    format_table(tblC, temp_cols=["당월평균기온","기간평균기온"])
    st.download_button("Conservative CSV", data=tblC.to_csv(index=False).encode("utf-8-sig"),
                       mime="text/csv", file_name="sales_conservative.csv")

    # 상관관계 그래프 (학습 데이터 기준)
    ctx = st.session_state["sales_ctx"]
    base = st.session_state["sales_base"]
    x_train = []
    y_train = []
    for m in pd.to_datetime(base["판매월"].unique()):
        # 학습 구간만 표시
        if int(m.year) in years_sel:
            x_train.append(ctx["period_avg_func"](m))
            y_train.append(float(base.loc[(base["연"]==m.year)&(base["월"]==m.month),"판매량"].mean()))
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    show_sales_scatter(x_train, y_train, ctx["model"], ctx["poly"], ctx["r2"],
                       title="기온-냉방용 실적 상관관계 (Train)")
