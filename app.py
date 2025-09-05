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
from glob import glob

# ─────────────────────────────────────────────────────────────
# 기본
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─────────────────────────────────────────────────────────────
# 한글 폰트: 레포의 data/fonts/NanumGothic-Regular.ttf 우선 적용
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "data" / "fonts" / "NanumGothic.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
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

# 공통 유틸
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
# ‘중앙난방용’을 ‘개별난방용’ 바로 뒤로 이동
KNOWN_PRODUCT_ORDER = [
    "개별난방용", "중앙난방용",
    "자가열전용", "일반용(2)", "업무난방용", "냉난방용",
    "주한미군", "총공급량"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str)+"-"+df["월"].astype(str)+"-01", errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore"
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        nm = str(c).lower()
        if any(h in nm for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others  = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet="데이터"):
    try:
        xls = pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜","일자","date"]:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try: pd.to_datetime(df[c], errors="raise"); date_col = c; break
                except Exception: pass
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp","temperature"]):
                temp_col = c; break
        if date_col is None or temp_col is None: return None
        out = pd.DataFrame({"일자": pd.to_datetime(df[date_col], errors="coerce"),
                            "기온": pd.to_numeric(df[temp_col], errors="coerce")}).dropna()
        return out.sort_values("일자").reset_index(drop=True)

    name = getattr(file, "name", str(file))
    if name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    head  = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row = None
    for i in range(len(head)):
        row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜","일자","date","Date"] for v in row) and any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
            header_row = i; break
    df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any(): raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1,1); x_future = x_future.reshape(-1,1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols = float1_cols or []; int_cols = int_cols or []
    show = df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round().astype("Int64").map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    st.markdown("""
    <style>
      table.centered-table {width:100%; table-layout: fixed;}
      table.centered-table th, table.centered-table td { text-align:center !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 분석 유형
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# =============== A) 공급량 분석 ==========================================
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i,p in enumerate(repo_files) if "상품별공급량" in Path(p).stem), 0)
                file_choice = st.selectbox("실적 파일(Excel)", repo_files, index=default_idx)
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None: df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df)==0: st.stop()

        st.subheader("학습 데이터 연도 선택")
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("연도 선택", years_all, default=years_all)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()

        st.subheader("예측할 상품 선택")
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

        st.subheader("예측 설정")
        last_year = int(df["연"].max())
        col1, col2 = st.columns(2)
        with col1:
            start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
            end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        with col2:
            start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("예측 시작", type="primary")

    # 계산은 버튼 클릭 시에만 (기본 미래 월/기온의 ‘베이스’를 만든 후, 아래에서 Δ°C만 즉시 반영)
    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        # 베이스 월평균 기온(학습기간 월평균)
        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

        fut_idx = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})
        fut_base = fut_base.merge(monthly_avg_temp.reset_index(), on="월", how="left")

        # 학습 입력(모든 상품 공통 x_train)
        x_train_base = train_df[temp_col].astype(float).values

        # 결과 객체 저장 (모델 재적합 없이 Δ°C로 즉시 재계산할 수 있도록 재료 보관)
        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, x_train=x_train_base,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end,
        )
        st.success("공급량 예측(베이스) 준비 완료! 아래에서 시나리오 Δ°C를 조절하세요.")

    if "supply_materials" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    # ── 시나리오 즉시 반영 UI (본문)
    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    x_train, fut_base = mats["x_train"], mats["fut_base"]
    months = list(range(1,13))

    # Δ°C 조절(0.5 간격, 즉시 반영)
    st.subheader("시나리오 Δ°C (평균기온 보정)")
    c1, c2, c3 = st.columns(3)
    with c1:
        d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="s_norm")
    with c2:
        d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="s_best")
    with c3:
        d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="s_cons")

    def _forecast_table_for_delta(delta: float) -> pd.DataFrame:
        # 미래 월평균기온(보정 포함)
        x_future = (fut_base["temp"] + float(delta)).astype(float).values

        # 각 상품 예측
        pred_rows = []
        r2_dict = {}
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
            r2_dict[col] = r2
            tmp = fut_base[["연","월"]].copy()
            tmp["월평균기온"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)

        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연","월","월평균기온"], columns="상품", values="예측").reset_index()

        # 열 정렬(지정 순서 + 나머지)
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연","월","월평균기온"] + ordered)]
        pivot = pivot[["연","월","월평균기온"] + ordered + others]

        # 종계 행 추가
        tot = {}
        for c in pivot.columns:
            if c in ["연","월","월평균기온"]:
                tot[c] = "" if c!="월평균기온" else ""
            else:
                tot[c] = pd.to_numeric(pivot[c], errors="coerce").sum()
        tot["연"] = ""; tot["월"] = "종계"; tot["월평균기온"] = ""
        pivot_total = pd.concat([pivot, pd.DataFrame([tot])], ignore_index=True)

        return pivot_total

    # ── 출력(표: Normal → Best → Conservative)
    st.markdown("### Normal")
    tbl_n = _forecast_table_for_delta(d_norm)
    render_centered_table(tbl_n, float1_cols=["월평균기온"], int_cols=[c for c in tbl_n.columns if c not in ["연","월","월평균기온"]], index=False)

    st.markdown("### Best")
    tbl_b = _forecast_table_for_delta(d_best)
    render_centered_table(tbl_b, float1_cols=["월평균기온"], int_cols=[c for c in tbl_b.columns if c not in ["연","월","월평균기온"]], index=False)

    st.markdown("### Conservative")
    tbl_c = _forecast_table_for_delta(d_cons)
    render_centered_table(tbl_c, float1_cols=["월평균기온"], int_cols=[c for c in tbl_c.columns if c not in ["연","월","월평균기온"]], index=False)

    st.download_button("예측 결과 CSV 다운로드 (Normal)", data=tbl_n.to_csv(index=False).encode("utf-8-sig"),
                       file_name="citygas_supply_forecast_normal.csv", mime="text/csv")

# =============== B) 판매량 분석(냉방용) =====================================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
    st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 준비하세요.")

    # 파일 선택(Repo/업로드)
    with st.sidebar:
        st.header("데이터 불러오기")
        sales_src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    def _find_repo_sales_and_temp():
        here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        data_dir = here / "data"
        sales_candidates = [
            data_dir / "상품별판매량.xlsx",
            *[Path(p) for p in glob(str(data_dir / "*판매*.xlsx"))],
        ]
        temp_candidates = [
            data_dir / "기온.xlsx",
            *[Path(p) for p in glob(str(data_dir / "*기온*.xlsx"))],
            *[Path(p) for p in glob(str(data_dir / "*temp*.csv"))],
        ]
        sales_path = next((p for p in sales_candidates if p.exists()), None)
        temp_path  = next((p for p in temp_candidates  if p.exists()), None)
        return sales_path, temp_path

    c1, c2 = st.columns(2)
    if sales_src == "Repo 내 파일 사용":
        repo_sales_path, repo_temp_path = _find_repo_sales_and_temp()
        if not repo_sales_path or not repo_temp_path:
            with c1: sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
            with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])
        else:
            st.success(f"레포 파일 사용: {repo_sales_path.name} · {repo_temp_path.name}")
            sales_file = open(repo_sales_path, "rb")
            temp_raw_file = open(repo_temp_path, "rb")
    else:
        with c1: sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 준비하세요."); st.stop()

    # 판매 실적 자동 매핑
    try:
        xls = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
        raw_sales = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw_sales = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(raw_sales)

    # 날짜/냉방 열 추정
    date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
    if date_candidates: date_col = date_candidates[0]
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
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int); sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()

    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("연도 선택", options=years_all, default=years_all)

        st.subheader("예측 설정")
        last_year = int(sales_df["연"].max())
        col1, col2 = st.columns(2)
        with col1:
            start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
            end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        with col2:
            start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)
        run_btn = st.button("예측 시작", type="primary")

    if run_btn:
        # 보조 집계
        temp_raw["연"] = temp_raw["일자"].dt.year; temp_raw["월"] = temp_raw["일자"].dt.month
        monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
        fallback_by_M = temp_raw.groupby("월")["기온"].mean()

        def period_avg(label_m: pd.Timestamp) -> float:
            m = month_start(label_m)
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
            e = m + pd.DateOffset(days=14)                                # 당월15
            mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            return temp_raw.loc[mask,"기온"].mean()

        # 학습 데이터
        train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
        rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
        sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
        miss = sj["기간평균기온"].isna()
        if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"판매월"].dt.month.map(fallback_by_M)
        sj = sj.dropna(subset=["기간평균기온","판매량"])

        x_train = sj["기간평균기온"].astype(float).values
        y_train = sj["판매량"].astype(float).values
        _, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)

        # 예측 입력
        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

        months = month_range_inclusive(f_start, f_end)
        rows = []
        for m in months:
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            avg_period = temp_raw.loc[mask,"기온"].mean()
            avg_month  = monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
            rows.append({"연":int(m.year),"월":int(m.month),"기간평균기온":avg_period,"당월평균기온":avg_month})
        pred = pd.DataFrame(rows)
        for c in ["기간평균기온","당월평균기온"]:
            miss = pred[c].isna()
            if miss.any(): pred.loc[miss,c] = pred.loc[miss,"월"].map(fallback_by_M)

        x_future = pred["기간평균기온"].astype(float).values
        y_future, _ = fit_poly3_and_predict(x_train, y_train, x_future)
        pred["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

        # 실제/오차
        actual = sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
        out = pd.merge(pred, actual, on=["연","월"], how="left")
        out["오차"] = (out["예측판매량"] - out["실제판매량"]).astype("Int64")
        out["오차율(%)"] = ((out["오차"] / out["실제판매량"]) * 100).round(1).astype("Float64")

        # 총계
        sum_row = {
            "연": "", "월": "종계",
            "당월평균기온": "", "기간평균기온": "",
            "예측판매량": out["예측판매량"].sum(),
            "실제판매량": out["실제판매량"].sum(),
            "오차": out["오차"].sum(),
            "오차율(%)": ""
        }
        out_sum = pd.concat([out, pd.DataFrame([sum_row])], ignore_index=True)

        st.session_state["sales_result"] = {
            "hist": sales_df.rename(columns={"판매량":"val"})[["연","월","val"]],
            "pred": out_sum,
            "r2": r2_fit
        }
        st.success("냉방용 판매량 예측을 완료했습니다.")

    if "sales_result" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    res = st.session_state["sales_result"]

    show = res["pred"].rename(columns={"기간평균기온":"기간평균기온 (m-1, 16일 ~ m15일)"})
    show["연"] = show["연"].astype(str)
    st.subheader("판매량 예측(요약)")
    render_centered_table(
        show[["연","월","당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)","예측판매량"]],
        float1_cols=["당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)"],
        int_cols=["예측판매량"], index=False
    )
    st.download_button(
        "판매량 예측 CSV 다운로드",
        data=show.to_csv(index=False).encode("utf-8-sig"),
        file_name="cooling_sales_forecast.csv", mime="text/csv"
    )

    st.subheader("판매량 예측 검증")
    render_centered_table(
        res["pred"][["연","월","실제판매량","예측판매량","오차","오차율(%)"]],
        int_cols=["실제판매량","예측판매량","오차"], index=False
    )

    # 상관관계(R2) 시각화
    st.subheader(f"기온-냉방용 실적 상관관계 (Train, R²={res['r2']:.3f})")
    fig = plt.figure(figsize=(6.5,4)); ax = plt.gca()
    hist = res["hist"]
    ax.scatter(hist["월"], hist["val"], label="학습 샘플", alpha=0.7)
    ax.set_xlabel("월"); ax.set_ylabel("판매량")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
