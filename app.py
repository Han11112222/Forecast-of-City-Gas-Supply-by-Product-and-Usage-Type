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

# ─────────────────────────────────────────────────────────────
# 기본 & 한글 폰트
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
        here / "data" / "fonts" / "NanumGothic.ttf",
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
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False
set_korean_font()

# ─────────────────────────────────────────────────────────────
# 공통 유틸

KNOWN_PRODUCT_ORDER = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜 파생
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")

    # 연/월 채우기
    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = pd.to_numeric(df["년"], errors="coerce").astype("Int64")
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year

    if "월" not in df.columns:
        if "날짜" in df.columns:
            df["월"] = df["날짜"].dt.month

    # 숫자화
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore"
            )
    return df

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet="데이터"):
    try:
        xls = pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        low = str(c).lower()
        if ("평균기온" in c) or ("기온" in c) or ("temp" in low) or ("temperature" in low):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    # 한글 '온' 포함
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    meta = {"연","년","월","날짜","일자","date"}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in meta]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

def poly3_fit(x: np.ndarray, y: np.ndarray):
    x = x.reshape(-1,1).astype(float)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x)
    model = LinearRegression().fit(Xtr, y.astype(float))
    r2 = model.score(Xtr, y.astype(float))
    return model, poly, r2

def poly3_predict(model, poly, x):
    X = np.array(x).reshape(-1,1).astype(float)
    return model.predict(poly.transform(X))

# 날짜/기온 RAW 자동 인식 로더 (판매용)
def load_temperature_raw(file_or_path) -> pd.DataFrame:
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(file_or_path)
    else:
        xls = pd.ExcelFile(file_or_path, engine="openpyxl")
        raw = pd.read_excel(xls, sheet_name=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    # 날짜열 탐지
    date_candidates = [c for c in raw.columns if str(c).lower() in ["일자","날짜","date","일시","time","datetime","timestamp"]]
    date_col = date_candidates[0] if date_candidates else None
    if date_col is None:
        best, best_rate = None, 0
        for c in raw.columns:
            s = pd.to_datetime(raw[c], errors="coerce")
            rate = s.notna().mean()
            if rate > best_rate:
                best, best_rate = c, rate
        date_col = best
    if date_col is None:
        raise ValueError("기온 RAW에서 날짜 열을 찾지 못했습니다.")

    # 기온열 탐지
    temp_col = None
    for c in raw.columns:
        low = c.lower()
        if ("평균기온" in c) or ("기온" in c) or ("temp" in low) or ("temperature" in low):
            temp_col = c; break
    if temp_col is None:
        raise ValueError("기온 RAW에서 기온 열을 찾지 못했습니다. (예: '평균기온', '기온', 'temp')")

    out = raw[[date_col, temp_col]].copy()
    out.columns = ["일자","기온"]
    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    out = out.dropna(subset=["일자","기온"]).sort_values("일자").reset_index(drop=True)
    out["연"] = out["일자"].dt.year
    out["월"] = out["일자"].dt.month
    return out

# 표 중앙 정렬 + 포맷
def fmt_centered_table(df: pd.DataFrame, temp_cols=("당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)", "월평균기온(적용)")) -> str:
    df = df.copy()

    # 연월 앞에
    if set(["연","월"]).issubset(df.columns):
        if "연월" not in df.columns:
            df.insert(0, "연월", df["연"].astype(int).astype(str) + "." + df["월"].astype(int).astype(str).str.zfill(2))
        df = df.drop(columns=["연","월"])

    # 포맷
    for c in df.columns:
        if c in temp_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{int(round(x)):,}")

    sty = (df.style
           .set_table_attributes('class="dataframe centered-table"')
           .set_properties(**{"text-align":"center"})
           .set_table_styles([{"selector":"th", "props":"text-align:center; vertical-align:middle;"}])
          )
    return sty.to_html()

def show_table_centered(df: pd.DataFrame):
    html = fmt_centered_table(df)
    st.markdown("""
    <style>
    table.centered-table {width:100%;}
    table.centered-table th, table.centered-table td { text-align:center !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 시나리오 ΔT 버튼 (±0.5, 즉시 반영)

def scenario_delta_controls():
    for key, init in [("dT_norm", 0.0), ("dT_best", -0.5), ("dT_cons", 0.5)]:
        if key not in st.session_state:
            st.session_state[key] = init

    st.subheader("ΔT 시나리오 (°C)")
    c1, c2, c3 = st.columns(3)

    def block(title, key_dec, key_inc, key_value):
        st.caption(title)
        cols = st.columns([1,2,1])
        if cols[0].button("−", key=key_dec): st.session_state[key_value] = round(st.session_state[key_value] - 0.5, 2)
        cols[1].markdown(f"<h3 style='text-align:center'>{st.session_state[key_value]:+.2f}</h3>", unsafe_allow_html=True)
        if cols[2].button("+", key=key_inc): st.session_state[key_value] = round(st.session_state[key_value] + 0.5, 2)

    with c1: block("ΔT(Normal)", "n_dec", "n_inc", "dT_norm")
    with c2: block("ΔT(Best)",   "b_dec", "b_inc", "dT_best")
    with c3: block("ΔT(Conservative)", "c_dec", "c_inc", "dT_cons")

    return st.session_state.dT_norm, st.session_state.dT_best, st.session_state.dT_cons

# ─────────────────────────────────────────────────────────────
# 사이드바 (공통)

with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

    st.header("데이터 불러오기")
    src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

# ─────────────────────────────────────────────────────────────
# =============== A) 공급량 분석 ==========================================

if mode == "공급량 분석":

    with st.sidebar:
        # 파일
        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data")
            candidates = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            default_idx = 0
            if candidates:
                file_choice = st.selectbox("실적 파일(Excel)", candidates, index=default_idx, key="supply_repo_file")
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("data 폴더에 *.xlsx 파일이 없습니다. 업로드로 진행하세요.")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="supply_up")
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df) == 0:
            st.info("좌측에서 파일을 선택/업로드 해주세요.")
            st.stop()

        # 연도 선택
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        st.header("학습 데이터 연도 선택")
        years_sel = st.multiselect("연도 선택", years_all, default=years_all, key="supply_years")

        # 예측 기간 (가로로)
        st.header("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            y_start = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(df["연"].max())), key="sup_y_start")
            y_end   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(int(df["연"].max())), key="sup_y_end")
        with c2:
            m_start = st.selectbox("예측 시작(월)", list(range(1,13)), index=0,  key="sup_m_start")
            m_end   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="sup_m_end")

        run_btn = st.button("예측 시작", type="primary", key="supply_run")

    # 메인 화면
    st.title("도시가스 공급·판매 분석 (Poly-3)")
    st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

    # ΔT 버튼
    dT_norm, dT_best, dT_cons = scenario_delta_controls()

    # 데이터 준비
    base = df.dropna(subset=["연","월"]).copy()
    temp_col = detect_temp_col(base)
    if temp_col is None:
        st.error("⚠️ 기온(평균기온/기온/temp 등) 열을 찾지 못했습니다.")
        st.stop()

    product_cols = guess_product_cols(base)
    product_default = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]

    with st.sidebar:
        st.header("예측할 상품 선택")
        prods = st.multiselect("상품(용도)", product_cols, default=product_default, key="supply_products")

    if not prods:
        st.info("예측할 상품을 1개 이상 선택하세요.")
        st.stop()

    # 학습 데이터
    train_df = base[base["연"].isin(years_sel)].copy().sort_values(["연","월"])

    # 월별 평균기온(학습)
    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("월평균기온").to_frame()

    # 예측 달 목록
    f_start = pd.Timestamp(year=int(y_start), month=int(m_start), day=1)
    f_end   = pd.Timestamp(year=int(y_end),   month=int(m_end),   day=1)
    if f_end < f_start:
        st.error("⚠️ 예측 종료가 시작보다 빠릅니다.")
        st.stop()

    fut_idx = month_range_inclusive(f_start, f_end)
    fut = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

    # 시나리오별 기온 (월평균 기반 + ΔT)
    fut_base = fut.merge(monthly_avg_temp.reset_index(), on="월", how="left")
    if fut_base["월평균기온"].isna().any():
        # 전체 데이터로 보강
        fallback = base.groupby("월")[temp_col].mean()
        fut_base["월평균기온"] = fut_base["월"].map(fallback)

    fut_norm = fut_base.copy(); fut_norm["월평균기온(적용)"] = fut_norm["월평균기온"] + dT_norm
    fut_best = fut_base.copy(); fut_best["월평균기온(적용)"] = fut_best["월평균기온"] + dT_best
    fut_cons = fut_base.copy(); fut_cons["월평균기온(적용)"] = fut_cons["월평균기온"] + dT_cons

    # 모델 학습(상품별 하나씩)
    models = {}
    for col in prods:
        if col not in train_df.columns or not pd.api.types.is_numeric_dtype(train_df[col]): 
            continue
        x = train_df[temp_col].astype(float).values
        y = train_df[col].astype(float).values
        model, poly, r2 = poly3_fit(x, y)
        models[col] = (model, poly, r2)

    def predict_table(fut_temp_df: pd.DataFrame, label="예측공급량"):
        rows = []
        for col in prods:
            if col not in models: 
                continue
            model, poly, r2 = models[col]
            x_future = fut_temp_df["월평균기온(적용)"].values
            y_future = np.clip(np.rint(poly3_predict(model, poly, x_future)), a_min=0, a_max=None).astype(np.int64)
            tmp = fut_temp_df[["연","월"]].copy()
            tmp[col] = y_future
            rows.append(tmp.set_index(["연","월"]))
        if not rows:
            return None
        pivot = pd.concat(rows, axis=1)
        pivot = pivot.loc[:, ~pivot.columns.duplicated()]  # 중복 제거
        pivot = pivot.reset_index()
        return pivot

    normal_tbl = predict_table(fut_norm)
    best_tbl   = predict_table(fut_best)
    cons_tbl   = predict_table(fut_cons)

    # ─ 표시
    st.markdown("### 예측 결과 — Normal")
    if normal_tbl is not None:
        show_table_centered(normal_tbl.rename(columns={"월평균기온":"월평균기온(적용)"}))
        st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_normal.csv", mime="text/csv")
    st.markdown("### 예측 결과 — Best")
    if best_tbl is not None:
        show_table_centered(best_tbl.rename(columns={"월평균기온":"월평균기온(적용)"}))
        st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_best.csv", mime="text/csv")
    st.markdown("### 예측 결과 — Conservative")
    if cons_tbl is not None:
        show_table_centered(cons_tbl.rename(columns={"월평균기온":"월평균기온(적용)"}))
        st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_conservative.csv", mime="text/csv")

    # ─ 그래프 (각 상품)
    st.markdown("---")
    st.subheader("연도별 실적 & 예측(12개월 미리보기) — 공급량 (MJ)")

    years_view = sorted([int(y) for y in pd.Series(base["연"]).dropna().unique()])[-5:] if len(base)>0 else []
    months = list(range(1,13))

    for prod in prods:
        if prod not in models: 
            continue
        model, poly, r2 = models[prod]
        fig = plt.figure(figsize=(9,3.8)); ax = plt.gca()
        # 실적 라인
        for y in years_view:
            s = (base[base["연"]==y].set_index("월")[prod] if prod in base.columns else pd.Series(index=months, dtype=float))
            s = s.reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")
        # 예측(12개월)
        P = normal_tbl.copy() if normal_tbl is not None else pd.DataFrame(columns=["연","월",prod])
        pred_vals, y, m = [], int(f_start.year), int(f_start.month)
        for _ in range(12):
            row = P[(P["연"]==y)&(P["월"]==m)]
            pred_vals.append(row.iloc[0][prod] if len(row) and prod in row.columns else np.nan)
            if m==12: y+=1; m=1
            else: m+=1
        ax.plot(months, pred_vals, linestyle="--", label="예측(Normal)")

        # 폴리노미얼 식(소수5자리)
        coef = model.coef_
        intercept = model.intercept_
        terms = dict(zip(PolynomialFeatures(degree=3, include_bias=False).get_feature_names_out(), coef))
        a = float(terms.get("x0^3", 0.0))
        b = float(terms.get("x0^2", 0.0))
        c = float(terms.get("x0",   0.0))
        d = float(intercept)
        eq = f"y = {a:+.5e}x³ {b:+.5e}x² {c:+.5e}x {d:+.5e}"
        ax.text(0.01, 0.02, eq, transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=.7, edgecolor="none"))

        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
        ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
        ax.set_title(f"{prod} — Poly-3 (Train R²={r2:.3f})"); ax.legend(loc="best")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

# ─────────────────────────────────────────────────────────────
# =============== B) 판매량 분석(냉방용) =====================================

else:
    with st.sidebar:
        # 파일
        sales_file = None
        temp_raw_file = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data")
            sales_candidates = [
                data_dir / "상품별판매량.xlsx",
                *[p for p in data_dir.glob("*판매*.xlsx")]
            ]
            temp_candidates = [
                data_dir / "기온.xlsx",
                *[p for p in data_dir.glob("*기온*.xlsx")],
                *[p for p in data_dir.glob("*temp*.csv")]
            ]
            sales_path = next((p for p in sales_candidates if p.exists()), None)
            temp_path  = next((p for p in temp_candidates  if p.exists()), None)
            if sales_path: 
                sales_file = open(sales_path, "rb")
            if temp_path:
                temp_raw_file = open(temp_path, "rb")
            if not sales_file or not temp_raw_file:
                st.info("repo에 '상품별판매량*.xlsx' 또는 '기온' 파일이 없습니다. 업로드로 진행하세요.")
        else:
            c1, c2 = st.columns(2)
            with c1: sales_file = st.file_uploader("판매 실적(xlsx)", type=["xlsx"], key="sales_up")
            with c2: temp_raw_file = st.file_uploader("기온 RAW(xlsx/csv)", type=["xlsx","csv"], key="temp_up")

        if sales_file is None or temp_raw_file is None:
            st.info("두 파일을 모두 준비하세요.")
            st.stop()

        # 학습 연도 선택
        raw_sales = read_excel_sheet(sales_file)  # 시트 자동
        # 날짜/냉방 열 추정
        sales_df = normalize_cols(raw_sales)
        date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            score = {}
            for c in sales_df.columns:
                try: score[c] = pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
                except Exception: pass
            date_col = max(score, key=score.get) if score else None
        cool_cols = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
        value_col = None
        for c in cool_cols:
            if "냉방용" in str(c): value_col = c; break
        value_col = value_col or (cool_cols[0] if cool_cols else None)
        if date_col is None or value_col is None:
            st.error("⚠️ 날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

        sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
        sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
        sales_df["연"] = sales_df["판매월"].dt.year.astype(int); sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

        years_all = sorted(sales_df["연"].unique().tolist())
        st.header("학습 데이터 연도 선택")
        years_sel = st.multiselect("연도 선택", options=years_all, default=years_all, key="sales_years")

        # 예측 기간 (가로)
        st.header("예측 기간")
        c1, c2 = st.columns(2)
        last_year = int(sales_df["연"].max())
        with c1:
            y_start = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year), key="sal_y_start")
            y_end   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year), key="sal_y_end")
        with c2:
            m_start = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="sal_m_start")
            m_end   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="sal_m_end")

        run_btn = st.button("예측 시작", type="primary", key="sales_run")

    # 메인
    st.title("도시가스 공급·판매 분석 (Poly-3)")
    st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

    # ΔT
    dT_norm, dT_best, dT_cons = scenario_delta_controls()

    # 기온 RAW
    temp_raw = load_temperature_raw(temp_raw_file)

    # 월평균/기간평균 계산 보조
    temp_raw["연"] = temp_raw["일자"].dt.year; temp_raw["월"] = temp_raw["일자"].dt.month
    monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
    fallback_by_M = temp_raw.groupby("월")["기온"].mean()

    def period_avg(label_m: pd.Timestamp) -> float:
        m = month_start(label_m)
        s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
        e = m + pd.DateOffset(days=14)                                # 당월15
        mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
        return temp_raw.loc[mask,"기온"].mean()

    # 학습 표준화
    train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
    rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
    sj = pd.merge(train_sales[["판매월","판매량","연","월"]], pd.DataFrame(rows), on="판매월", how="left")
    miss = sj["기간평균기온"].isna()
    if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"월"].map(fallback_by_M)
    sj = sj.dropna(subset=["기간평균기온","판매량"])

    # 모델 학습
    x_train = sj["기간평균기온"].astype(float).values
    y_train = sj["판매량"].astype(float).values
    s_model, s_poly, s_r2 = poly3_fit(x_train, y_train)

    # 예측 입력
    f_start = pd.Timestamp(year=int(y_start), month=int(m_start), day=1)
    f_end   = pd.Timestamp(year=int(y_end),   month=int(m_end),   day=1)
    if f_end < f_start:
        st.error("⚠️ 예측 종료가 시작보다 빠릅니다.")
        st.stop()

    months_idx = month_range_inclusive(f_start, f_end)
    rows = []
    for m in months_idx:
        s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
        e = m + pd.DateOffset(days=14)                                # 당월15
        mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
        avg_period = temp_raw.loc[mask,"기온"].mean()
        avg_month  = monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
        rows.append({"연":int(m.year),"월":int(m.month),
                     "당월평균기온":avg_month,"기간평균기온 (m-1, 16일 ~ m15일)":avg_period})
    pred_base = pd.DataFrame(rows)
    for c in ["당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)"]:
        miss = pred_base[c].isna()
        if miss.any(): pred_base.loc[miss,c] = pred_base.loc[miss,"월"].map(fallback_by_M)

    def predict_sales(tbl: pd.DataFrame, dT: float):
        tmp = tbl.copy()
        tmp["기간평균기온 (m-1, 16일 ~ m15일)"] = tmp["기간평균기온 (m-1, 16일 ~ m15일)"] + dT
        y_future = np.clip(np.rint(poly3_predict(s_model, s_poly, tmp["기간평균기온 (m-1, 16일 ~ m15일)"].values)),
                           a_min=0, a_max=None).astype(np.int64)
        out = tmp[["연","월","당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)"]].copy()
        out["예측판매량"] = y_future
        return out

    normal_tbl = predict_sales(pred_base, dT_norm)
    best_tbl   = predict_sales(pred_base, dT_best)
    cons_tbl   = predict_sales(pred_base, dT_cons)

    # ─ 표
    st.markdown("### 예측 결과 — Normal")
    show_table_centered(normal_tbl)
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_normal.csv", mime="text/csv")

    st.markdown("### 예측 결과 — Best")
    show_table_centered(best_tbl)
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_best.csv", mime="text/csv")

    st.markdown("### 예측 결과 — Conservative")
    show_table_centered(cons_tbl)
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_conservative.csv", mime="text/csv")

    # ─ 예측 검증 (실제/오차/오차율)
    st.markdown("---")
    st.subheader("판매량 예측 검증 (Normal)")
    actual = sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
    verify = pd.merge(normal_tbl, actual, on=["연","월"], how="left")
    verify["오차"] = (verify["예측판매량"] - verify["실제판매량"]).astype("Int64")
    verify["오차율(%)"] = ( (verify["오차"] / verify["실제판매량"]) * 100 ).round(1).astype("Float64")
    show_table_centered(verify)
    st.download_button("검증 CSV", data=verify.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_verify.csv", mime="text/csv")

    # ─ 상관 그래프
    st.markdown("---")
    st.subheader("기온-냉방용 실적 상관관계 (Train, 3차 다항식)")

    def draw_sales_correlation(df: pd.DataFrame, xcol="기간평균기온", ycol="판매량", title="기온-냉방용 실적 상관관계 (Train)"):
        X = df[xcol].values.reshape(-1,1).astype(float)
        y = df[ycol].astype(float).values
        poly = PolynomialFeatures(degree=3, include_bias=False)
        Xtr = poly.fit_transform(X)
        reg = LinearRegression().fit(Xtr, y)
        r2 = reg.score(Xtr, y)

        xx = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
        yy = reg.predict(poly.transform(xx))

        resid = y - reg.predict(Xtr)
        sigma = resid.std(ddof=1)
        bandU = yy + 1.96*sigma
        bandL = yy - 1.96*sigma

        fig = plt.figure(figsize=(9,5.2))
        ax  = plt.gca()
        ax.scatter(X, y, s=40, alpha=.6, label="학습 샘플")
        ax.plot(xx, yy, linewidth=3, label="Poly-3")
        ax.fill_between(xx.ravel(), bandL, bandU, alpha=.15, label="±1.96")

        bins = np.round(df[xcol], 0)
        med  = df.groupby(bins)[ycol].median()
        ax.scatter(med.index, med.values, s=60, color="orange", label="온도별 중앙값", zorder=5)

        coef = reg.coef_; intercept = reg.intercept_
        terms = dict(zip(poly.get_feature_names_out(), coef))
        a = float(terms.get("x0^3", 0.0))
        b = float(terms.get("x0^2", 0.0))
        c = float(terms.get("x0",   0.0))
        d = float(intercept)
        eq = f"y = {a:+.5e}x³ {b:+.5e}x² {c:+.5e}x {d:+.5e}"
        ax.text(0.01, 0.02, eq, transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=.7, edgecolor="none"))

        ax.set_title(f"{title} (Train, R²={r2:.3f})")
        ax.set_xlabel("기간평균기온 (°C)")
        ax.set_ylabel("판매량 (MJ)")
        ax.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # 학습 데이터프레임 생성
    corr_df = sj.rename(columns={"기간평균기온":"기간평균기온", "판매량":"판매량"})
    draw_sales_correlation(corr_df, xcol="기간평균기온", ycol="판매량")
