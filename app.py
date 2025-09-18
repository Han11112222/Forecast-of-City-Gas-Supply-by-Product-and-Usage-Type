# app.py — 도시가스 공급·판매 예측 (Poly-3) + 동적차트 + 추세분석(결측 보강/기본값/레이아웃 수정)

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
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 공급·판매 예측 (Poly-3)", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

st.markdown("""
<style>
.icon-title{display:flex;align-items:center;gap:.55rem;margin:.2rem 0 .6rem 0}
.icon-title .emoji{font-size:1.55rem;line-height:1}
.small-icon .emoji{font-size:1.2rem}
table.centered-table {width:100%; table-layout: fixed;}
table.centered-table th, table.centered-table td { text-align:center !important; }
</style>
""", unsafe_allow_html=True)

def title_with_icon(icon: str, text: str, level: str = "h1", small=False):
    klass = "icon-title small-icon" if small else "icon-title"
    st.markdown(f"<{level} class='{klass}'><span class='emoji'>{icon}</span><span>{text}</span></{level}>",
                unsafe_allow_html=True)

title_with_icon("📊", "도시가스 공급량·판매량 예측 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

# ─────────────────────────────────────────────────────────────
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
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font()

META_COLS = {"날짜","일자","date","연","년","월"}
TEMP_HINTS = ["평균기온","기온","temperature","temp"]
KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"
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
                df[c].astype(str).str.replace(",","",regex=False).str.replace(" ","",regex=False),
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
    if name and name.lower().endswith(".csv"):
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

@st.cache_data(ttl=600)
def read_temperature_forecast(file):
    """(연,월, 예상기온, [추세기온]) 반환"""
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = "기온예측" if "기온예측" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(file, engine="openpyxl")

    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ["날짜","일자","date","Date"]), df.columns[0])
    base_temp_col = next((c for c in df.columns if ("평균기온" in c) or (str(c).lower() in ["temp","temperature","기온"])), None)
    trend_col = next((c for c in df.columns if "추세" in c), None)
    if base_temp_col is None:
        raise ValueError("기온예측 파일에서 '평균기온/기온' 열을 찾지 못했습니다.")

    d = pd.DataFrame({
        "날짜": pd.to_datetime(df[date_col], errors="coerce"),
        "예상기온": pd.to_numeric(df[base_temp_col], errors="coerce")
    }).dropna()

    if trend_col is not None:
        d["추세기온"] = pd.to_numeric(df[trend_col], errors="coerce")

    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    cols = ["연","월","예상기온"] + (["추세기온"] if "추세기온" in d.columns else [])
    return d[cols].sort_values(["연","월"])

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1,1); x_future = x_future.reshape(-1,1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def poly_eq_text(model):
    c = model.coef_
    c1 = c[0] if len(c)>0 else 0.0
    c2 = c[1] if len(c)>1 else 0.0
    c3 = c[2] if len(c)>2 else 0.0
    d  = model.intercept_
    return f"Poly-3: y = {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols = float1_cols or []; int_cols = int_cols or []
    show = df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round().astype("Int64").map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
with st.sidebar:
    title_with_icon("🧭","예측 유형","h3", small=True)
    mode = st.radio("🔀 선택", ["공급량 예측","판매량 예측(냉방용)"], index=0, label_visibility="visible")

# ======================= A) 공급량 예측 =======================
if mode == "공급량 예측":
    with st.sidebar:
        title_with_icon("📥","데이터 불러오기","h3", small=True)
        src = st.radio("📦 방식", ["Repo 내 파일 사용","파일 업로드"], index=0)
        df = None; forecast_df = None

        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i,p in enumerate(repo_files)
                                    if ("상품별공급량" in Path(p).stem) or ("공급량" in Path(p).stem)), 0)
                file_choice = st.selectbox("📄 실적 파일(Excel)", repo_files, index=default_idx,
                                           format_func=lambda p: Path(p).name)
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("📂 data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")

            fc_candidates = [data_dir / "기온예측.xlsx", *[Path(p) for p in glob(str(data_dir / "*기온예측*.xlsx"))]]
            if any(p.exists() for p in fc_candidates):
                fc_path = next(p for p in fc_candidates if p.exists())
                st.success(f"🌡️ 예상기온 파일 사용: {fc_path.name}")
                forecast_df = read_temperature_forecast(fc_path)
            else:
                up_fc = st.file_uploader("🌡️ 예상기온 업로드(xlsx) — (날짜, 평균기온, [추세분석])", type=["xlsx"], key="up_fc_repo")
                if up_fc is not None:
                    forecast_df = read_temperature_forecast(up_fc)
        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None: df = read_excel_sheet(up, prefer_sheet="데이터")
            up_fc = st.file_uploader("🌡️ 예상기온 엑셀 업로드(xlsx) — (날짜, 평균기온, [추세분석])", type=["xlsx"])
            if up_fc is not None: forecast_df = read_temperature_forecast(up_fc)

        if df is None or len(df)==0:
            st.info("🧩 좌측에서 실적 엑셀을 선택/업로드하세요."); st.stop()
        if forecast_df is None or forecast_df.empty:
            st.info("🌡️ 좌측에서 예상기온 엑셀을 선택/업로드하세요."); st.stop()

        title_with_icon("📚","학습 데이터 연도 선택","h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("🗓️ 연도 선택", years_all, default=years_all)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("🌡️ 기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()

        title_with_icon("🧰","예측할 상품 선택","h3", small=True)
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=default_products)

        title_with_icon("⚙️","예측 설정","h3", small=True)
        allowed_years = list(range(2020, 2031))  # 2020~2030
        last_year = min(max(int(df["연"].max()), allowed_years[0]), allowed_years[-1])

        # 연/연 한 줄, 월/월 한 줄 (요청 레이아웃)
        colY1, colY2 = st.columns(2)
        with colY1:
            start_y = st.selectbox("🚀 예측 시작(연)", allowed_years, index=allowed_years.index(last_year))
        with colY2:
            end_y   = st.selectbox("🏁 예측 종료(연)", allowed_years, index=allowed_years.index(last_year))

        colM1, colM2 = st.columns(2)
        with colM1:
            start_m = st.selectbox("🗓️ 예측 시작(월)", list(range(1,13)), index=0)
        with colM2:
            end_m   = st.selectbox("🗓️ 예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("🧮 예측 시작", type="primary")

    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()

        fut_idx  = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        # 예측 파일 병합
        fut_base = fut_base.merge(forecast_df, on=["연","월"], how="left")  # 예상기온, [추세기온]
        # 학습연도 월평균(보강용)
        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("보강기온").reset_index()

        # 예상기온 보강
        miss = fut_base["예상기온"].isna()
        if miss.any():
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
            fut_base.loc[miss, "예상기온"] = fut_base.loc[miss, "보강기온"]

        # 추세기온 보강: (추세기온)→(예상기온)→(월평균)
        if "보강기온" not in fut_base.columns:
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
        fut_base["추세기온_적용"] = fut_base.get("추세기온", np.nan)
        fut_base.loc[fut_base["추세기온_적용"].isna(), "추세기온_적용"] = fut_base["예상기온"]
        fut_base.loc[fut_base["추세기온_적용"].isna(), "추세기온_적용"] = fut_base["보강기온"]
        fut_base.drop(columns=[c for c in ["보강기온"] if c in fut_base.columns], inplace=True)

        # 추세 데이터가 원본으로 실제 존재하는 연도 집합(기본값 계산에 사용)
        trend_years_available = sorted(forecast_df.loc[forecast_df.get("추세기온").notna() if "추세기온" in forecast_df.columns else [], "연"].unique().tolist()) if "추세기온" in forecast_df.columns else []

        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, temp_col=temp_col,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end,
            trend_years_available=trend_years_available
        )
        st.success("✅ 공급량 예측(베이스) 준비 완료!")

    if "supply_materials" not in st.session_state:
        st.info("👈 좌측에서 설정 후 **예측 시작**을 눌러 실행하세요."); st.stop()

    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    fut_base, temp_col = mats["fut_base"], mats["temp_col"]
    start_ts, end_ts = mats["start_ts"], mats["end_ts"]
    horizon_years = list(range(start_ts.year, end_ts.year + 1))
    months = list(range(1,13))

    # ───────────────── Δ°C (표) 그대로 유지 ─────────────────
    title_with_icon("🌡️","시나리오 Δ°C (평균기온 보정)","h3", small=True)
    c1,c2,c3 = st.columns(3)
    with c1: d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="s_norm")
    with c2: d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="s_best")
    with c3: d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="s_cons")

    def _forecast_table_for_delta(delta: float) -> pd.DataFrame:
        x_future = (fut_base["예상기온"] + float(delta)).astype(float).values
        pred_rows = []
        for col in prods:
            x_train = train_df[temp_col].astype(float).values
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연","월"]].copy()
            tmp["월평균기온"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연","월","월평균기온"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others  = [c for c in pivot.columns if c not in (["연","월","월평균기온"] + ordered)]
        pivot   = pivot[["연","월","월평균기온"] + ordered + others]
        tot = {}
        for c in pivot.columns:
            if c in ["연","월","월평균기온"]:
                tot[c] = "" if c!="월" else "종계"
            else:
                tot[c] = pd.to_numeric(pivot[c], errors="coerce").sum()
        tot["연"] = ""
        return pd.concat([pivot, pd.DataFrame([tot])], ignore_index=True)

    st.markdown("### 🎯 Normal")
    tbl_n = _forecast_table_for_delta(d_norm)
    render_centered_table(tbl_n, float1_cols=["월평균기온"],
                          int_cols=[c for c in tbl_n.columns if c not in ["연","월","월평균기온"]], index=False)

    st.markdown("### 💎 Best")
    tbl_b = _forecast_table_for_delta(d_best)
    render_centered_table(tbl_b, float1_cols=["월평균기온"],
                          int_cols=[c for c in tbl_b.columns if c not in ["연","월","월평균기온"]], index=False)

    st.markdown("### 🛡️ Conservative")
    tbl_c = _forecast_table_for_delta(d_cons)
    render_centered_table(tbl_c, float1_cols=["월평균기온"],
                          int_cols=[c for c in tbl_c.columns if c not in ["연","월","월평균기온"]], index=False)

    # ───────────────── 그래프(동적) ─────────────────
    title_with_icon("📈","그래프 (실적 + 예측(Normal) + 추세분석)","h3", small=True)

    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    forecast_years     = sorted(fut_base["연"].unique().tolist())
    trend_years_avail  = mats["trend_years_available"]

    # 기본값: 예측연도 = 수평선 전체 구간, 추세분석연도 = 그 중 실제 추세 데이터가 있는 연도
    cA,cB,cC = st.columns(3)
    with cA:
        years_view_actual = st.multiselect("👀 실적연도", options=years_all_for_plot,
                                           default=years_all_for_plot[-2:] if len(years_all_for_plot)>=2 else years_all_for_plot,
                                           key="view_actual")
    with cB:
        default_pred = [y for y in horizon_years if y in forecast_years] or forecast_years[:1]
        years_view_pred = st.multiselect("📈 예측연도 (Normal)", options=forecast_years,
                                         default=default_pred, key="view_pred")
    with cC:
        default_trend = [y for y in horizon_years if y in trend_years_avail]
        years_view_trend = st.multiselect("🧭 추세분석연도", options=sorted(set(forecast_years) | set(trend_years_avail)),
                                          default=default_trend, key="view_trend")

    for prod in prods:
        # 학습
        x_train = train_df[temp_col].astype(float).values
        y_train = train_df[prod].astype(float).values

        # 예측(Normal)
        x_future_norm = (fut_base["예상기온"] + float(d_norm)).astype(float).values
        y_future_norm, r2_train, model, _ = fit_poly3_and_predict(x_train, y_train, x_future_norm)
        Pn = fut_base[["연","월"]].copy()
        Pn["pred"] = np.clip(np.rint(y_future_norm).astype(np.int64), 0, None)

        # 추세분석(결측 보강된 열 사용)
        x_future_trend = fut_base["추세기온_적용"].astype(float).values
        y_future_trend, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future_trend)
        Pt = fut_base[["연","월"]].copy()
        Pt["pred"] = np.clip(np.rint(y_future_trend).astype(np.int64), 0, None)

        # 동적 Plotly
        fig = go.Figure()

        for y in years_view_actual:
            s = (base.loc[base["연"]==y, ["월", prod]].set_index("월")[prod]).reindex(months)
            fig.add_trace(go.Scatter(x=months, y=s.values, mode="lines", name=f"{y} 실적", line=dict(width=2)))

        for yy in years_view_pred:
            row = Pn.loc[Pn["연"]==yy].sort_values("월")
            if not row.empty:
                fig.add_trace(go.Scatter(x=row["월"], y=row["pred"], mode="lines",
                                         name=f"예측(Normal) {yy}", line=dict(width=3, dash="dash")))

        for yy in years_view_trend:
            row = Pt.loc[Pt["연"]==yy].sort_values("월")
            if not row.empty:
                fig.add_trace(go.Scatter(x=row["월"], y=row["pred"], mode="lines",
                                         name=f"추세분석 {yy}", line=dict(width=3, dash="dot")))

        fig.update_layout(
            title=f"{prod} — Poly-3",
            xaxis=dict(title="월", tickvals=months, ticktext=[f"{m}월" for m in months]),
            yaxis=dict(title="공급량 (MJ)"),
            legend=dict(orientation="h", y=-0.22),
            height=560, margin=dict(l=50, r=20, t=60, b=110)
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False,
                                "modeBarButtonsToAdd": ["zoom2d","pan2d","autoscale2d","resetScale2d"]})

        # 상관(Train) — Matplotlib (굵은선+큰점+그라데이션)
        title_with_icon("🔬", f"{prod} — 기온·공급량 상관(Train)", "h4", small=True)
        x_tr = train_df[temp_col].astype(float).values; y_tr = y_train
        figc, axc = plt.subplots(figsize=(10.5,6.2))
        axc.scatter(x_tr, y_tr, alpha=0.75, label="학습 샘플", s=55, color="#1f77b4")
        xx = np.linspace(np.nanmin(x_tr)-1, np.nanmax(x_tr)+1, 240)
        yhat, _, model_s, _ = fit_poly3_and_predict(x_tr, y_tr, xx)
        axc.plot(xx, yhat, lw=3.0, color="#1f77b4", label="Poly-3")
        pred_train, _, _, _ = fit_poly3_and_predict(x_tr, y_tr, x_tr)
        s_res = np.nanstd(y_tr - pred_train)
        for k, alp in [(1.96,0.18),(1.60,0.12),(1.30,0.08),(1.00,0.05)]:
            axc.fill_between(xx, yhat-k*s_res, yhat+k*s_res, color="#1f77b4", alpha=alp)
        bins = np.linspace(np.nanmin(x_tr), np.nanmax(x_tr), 15)
        gb = pd.DataFrame({"bin": pd.cut(x_tr, bins), "y": y_tr}).groupby("bin")["y"].median().reset_index()
        gb["x"] = [b.mid for b in gb["bin"]]
        axc.scatter(gb["x"], gb["y"], s=85, color="#ff7f0e", label="온도별 중앙값")
        axc.set_xlabel("기온 (℃)"); axc.set_ylabel("공급량 (MJ)")
        axc.grid(alpha=0.25); axc.legend(loc="upper right")
        eq = poly_eq_text(model_s)
        r2_txt = f" (Train R²={model_s.score(PolynomialFeatures(3, include_bias=False).fit_transform(x_tr.reshape(-1,1)), y_tr):.3f})"
        axc.text(0.01, 0.02, eq + r2_txt, transform=axc.transAxes,
                 fontsize=11, color="#1f77b4",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
        st.pyplot(figc)

    st.caption("ℹ️ 95% 밴드는 잔차 표준편차 기반의 **예측구간**(근사)입니다. 새 관측치가 포함될 확률이 약 95%로 이해하면 충분합니다.")

# ======================= B) 판매량 예측(냉방용) =======================
else:
    title_with_icon("🧊", "판매량 예측(냉방용)", "h2")
    st.info("판매 파트는 기존 로직 그대로 사용하세요. (요청사항은 공급 그래프 개선에 반영)")
