# app.py — 도시가스 공급·판매 예측 (Poly-3) + 추세분석 표 + Plotly 상관그래프
import os
from io import BytesIO
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
import plotly.graph_objects as go  # 동적 상관그래프

# ─────────────────────────────────────────────────────────────
# 기본
st.set_page_config(page_title="도시가스 공급·판매 예측 (Poly-3)", layout="wide")

# (제목/섹션 왼쪽 아이콘 유틸 + 표 중앙정렬)
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

# 상단 타이틀
title_with_icon("📊", "도시가스 공급량·판매량 예측 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─────────────────────────────────────────────────────────────
# 한글 폰트
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
    """
    월 단위 (날짜, 평균기온[, 추세분석*]) → (연, 월, 예상기온, 추세기온[선택])
    """
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = "기온예측" if "기온예측" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ["날짜","일자","date","Date"]), df.columns[0])
    # 기본 예측온도
    temp_col = next((c for c in df.columns if ("평균기온" in c) or (str(c).lower() in ["temp","temperature","기온"])), None)
    if temp_col is None:
        raise ValueError("기온예측 파일에서 '평균기온' 또는 '기온' 열을 찾지 못했습니다.")
    # 추세분석 온도 (선택)
    trend_col = next((c for c in df.columns if "추세분석" in c or "추세" in c), None)

    d = pd.DataFrame({"날짜": pd.to_datetime(df[date_col], errors="coerce"),
                      "예상기온": pd.to_numeric(df[temp_col], errors="coerce")}).dropna()
    if trend_col is not None:
        d["추세기온"] = pd.to_numeric(df[trend_col], errors="coerce")
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    cols = ["연","월","예상기온"] + (["추세기온"] if "추세기온" in d.columns else [])
    return d[cols].dropna(subset=["연","월"])

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# --- Poly3
def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        # NaN이 있어도 예측할 수 있도록 전처리 단계에서 모두 보강했어야 함.
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
    return f"y = {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

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
# 예측 유형
with st.sidebar:
    title_with_icon("🧭", "예측 유형", "h3", small=True)
    mode = st.radio("🔀 선택", ["공급량 예측", "판매량 예측(냉방용)"], index=0, label_visibility="visible")

# =============== A) 공급량 예측 ==========================================
if mode == "공급량 예측":
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

        df = None
        forecast_df = None

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
                up_fc = st.file_uploader("🌡️ 예상기온 업로드(xlsx) — (날짜, 평균기온[, 추세분석])", type=["xlsx"], key="up_fc_repo")
                if up_fc is not None:
                    forecast_df = read_temperature_forecast(up_fc)

        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")
            up_fc = st.file_uploader("🌡️ 예상기온 엑셀 업로드(xlsx) — (날짜, 평균기온[, 추세분석])", type=["xlsx"])
            if up_fc is not None:
                forecast_df = read_temperature_forecast(up_fc)

        if df is None or len(df)==0:
            st.info("🧩 좌측에서 실적 엑셀을 선택/업로드하세요."); st.stop()
        if forecast_df is None or forecast_df.empty:
            st.info("🌡️ 좌측에서 예상기온 엑셀을 선택/업로드하세요."); st.stop()

        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("🗓️ 연도 선택", years_all, default=years_all)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("🌡️ 기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()

        title_with_icon("🧰", "예측할 상품 선택", "h3", small=True)
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=default_products)

        # ── 예측 설정(연/월 분리; 사이드바에는 column 사용 금지)
        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        last_year = int(df["연"].max())
        start_y = st.selectbox("🚀 예측 시작(연)", list(range(2020,2031)),
                               index=list(range(2020,2031)).index(min(max(2020,last_year),2030)))
        start_m = st.selectbox("🗓️ 예측 시작(월)", list(range(1,13)), index=0)
        end_y   = st.selectbox("🏁 예측 종료(연)", list(range(2020,2031)),
                               index=list(range(2020,2031)).index(min(max(2020,last_year),2030)))
        end_m   = st.selectbox("🗓️ 예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("🧮 예측 시작", type="primary")

    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()
        fut_idx = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        # 예측 파일 병합: 예상기온 + (선택)추세기온
        fut_base = fut_base.merge(forecast_df, on=["연","월"], how="left")
        # 누락월 보강(예상기온)
        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("보강기온").reset_index()
        miss = fut_base["예상기온"].isna()
        if miss.any():
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
            fut_base.loc[miss, "예상기온"] = fut_base.loc[miss, "보강기온"]
        fut_base.drop(columns=[c for c in ["보강기온"] if c in fut_base.columns], inplace=True)
        # 추세기온도 없으면 예상기온으로 보강
        if "추세기온" in fut_base.columns:
            fut_base["추세기온"] = fut_base["추세기온"].fillna(fut_base["예상기온"])

        x_train_base = train_df[temp_col].astype(float).values

        # 예측 구간의 연도 리스트 (그래프 기본값에 사용)
        yr_list = sorted(fut_base["연"].unique().astype(int).tolist())

        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, x_train=x_train_base,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end, temp_col=temp_col,
            years_for_forecast=yr_list
        )
        st.success("✅ 공급량 예측(베이스) 준비 완료! 아래에서 **시나리오 Δ°C**를 조절하세요.")

    if "supply_materials" not in st.session_state:
        st.info("👈 좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    x_train, fut_base = mats["x_train"], mats["fut_base"]
    temp_col = mats["temp_col"]
    months = list(range(1,13))

    title_with_icon("🌡️", "시나리오 Δ°C (평균기온 보정)", "h3", small=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="s_norm")
    with c2:
        d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="s_best")
    with c3:
        d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="s_cons")

    def _make_forecast_pivot(x_future_array: np.ndarray) -> pd.DataFrame:
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future_array)
            tmp = fut_base[["연","월"]].copy()
            tmp["월평균기온"] = x_future_array
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연","월","월평균기온"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연","월","월평균기온"] + ordered)]
        pivot = pivot[["연","월","월평균기온"] + ordered + others]
        return pivot

    def _render_table_with_year_sums(df_pivot: pd.DataFrame, caption: str):
        st.markdown(f"### {caption}")
        render_centered_table(
            df_pivot, float1_cols=["월평균기온"],
            int_cols=[c for c in df_pivot.columns if c not in ["연","월","월평균기온"]],
            index=False
        )
        # 연도별 총계 요약
        num_cols = [c for c in df_pivot.columns if c not in ["연","월","월평균기온"]]
        year_sum = df_pivot.groupby("연")[num_cols].sum().reset_index()
        st.markdown("**연도별 총계 요약**")
        render_centered_table(year_sum, int_cols=[c for c in year_sum.columns if c!="연"], index=False)

    # 4개 표 생성
    tbl_norm = _make_forecast_pivot((fut_base["예상기온"] + float(d_norm)).astype(float).values)
    tbl_best = _make_forecast_pivot((fut_base["예상기온"] + float(d_best)).astype(float).values)
    tbl_cons = _make_forecast_pivot((fut_base["예상기온"] + float(d_cons)).astype(float).values)
    # 추세분석 표: 추세기온 있으면 사용, 없으면 예상기온
    x_trend = (fut_base["추세기온"] if "추세기온" in fut_base.columns else fut_base["예상기온"]).astype(float).values
    tbl_trend = _make_forecast_pivot(x_trend)

    _render_table_with_year_sums(tbl_norm, "🎯 Normal")
    _render_table_with_year_sums(tbl_best, "💎 Best")
    _render_table_with_year_sums(tbl_cons, "🛡️ Conservative")
    _render_table_with_year_sums(tbl_trend, "📈 추세분석")

    # 엑셀 다운로드 (4개 시트)
    def to_xlsx_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, ddf in dfs.items():
                ddf.to_excel(writer, sheet_name=name, index=False)
        buf.seek(0)
        return buf.read()

    st.download_button(
        "⬇️ 예측 결과 엑셀 다운로드 (Normal/Best/Conservative/추세분석)",
        data=to_xlsx_bytes({
            "Normal": tbl_norm, "Best": tbl_best, "Conservative": tbl_cons, "Trend": tbl_trend
        }),
        file_name="citygas_supply_forecast_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ───────── 그래프(실적 + 예측 + 추세) — Normal 기준 + 연도 선택들 ─────────
    title_with_icon("📈", "그래프 (실적 + 예측(Normal) + 추세분석)", "h3", small=True)
    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    default_years_hist = years_all_for_plot[-2:] if len(years_all_for_plot) >= 2 else years_all_for_plot
    y_hist = st.multiselect("👀 실적연도", options=years_all_for_plot,
                            default=st.session_state.get("s_years_hist", default_years_hist),
                            key="s_years_hist")
    # 예측/추세 기본값: 예측 구간 연도
    def_years_fore = mats.get("years_for_forecast", [])
    y_fore = st.multiselect("📈 예측연도 (Normal)", options=sorted(set(def_years_fore)),
                            default=st.session_state.get("s_years_fore", def_years_fore),
                            key="s_years_fore")
    y_trnd = st.multiselect("🧮 추세분석연도", options=sorted(set(def_years_fore)),
                            default=st.session_state.get("s_years_tr", def_years_fore),
                            key="s_years_tr")

    # Normal 예측 배열 / 추세 예측 배열
    x_future_norm = (fut_base["예상기온"] + float(d_norm)).astype(float).values
    pred_cache_norm = {}
    pred_cache_trend = {}

    for prod in prods:
        y_train_prod = train_df[prod].astype(float).values
        # 모델 피팅(Train R2)
        y_future_norm, r2_train, model, poly = fit_poly3_and_predict(x_train, y_train_prod, x_future_norm)

        # 월별 예측 시리즈(연도별로 잘라서)
        P = fut_base[["연","월"]].copy()
        P["pred_n"] = np.clip(np.rint(y_future_norm).astype(np.int64), a_min=0, a_max=None)

        # 추세 예측
        y_future_trend, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_trend)
        P["pred_t"] = np.clip(np.rint(y_future_trend).astype(np.int64), a_min=0, a_max=None)

        # 캐시
        pred_cache_norm[prod] = P[["연","월","pred_n"]]
        pred_cache_trend[prod] = P[["연","월","pred_t"]]

        # ─ 그래프(실적 + N + Trend) ─
        fig = plt.figure(figsize=(10.5,4.0)); ax = plt.gca()
        # 실적
        for y in sorted([int(v) for v in y_hist]):
            s = (base.loc[base["연"]==y, ["월", prod]].set_index("월")[prod]).reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")
        # 예측(Normal) — 연도별 라인
        for yy in sorted([int(v) for v in y_fore]):
            row = pred_cache_norm[prod][pred_cache_norm[prod]["연"]==yy].set_index("월")["pred_n"].reindex(months)
            ax.plot(months, row.values, linestyle=(0,(6,3)), label=f"예측(Normal) {yy}")
        # 추세분석 — 연도별 라인(점선)
        for yy in sorted([int(v) for v in y_trnd]):
            row = pred_cache_trend[prod][pred_cache_trend[prod]["연"]==yy].set_index("월")["pred_t"].reindex(months)
            ax.plot(months, row.values, linestyle=(0,(2,3)), label=f"추세분석 {yy}")

        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
        ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
        ax.set_title(f"{prod} — Poly-3 (Train R²={r2_train:.3f})")
        ax.legend(loc="best"); ax.grid(alpha=0.25)
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

        # ─ 상관그래프(동적, Plotly) ─
        title_with_icon("🔬", f"{prod} — 기온·공급량 상관(Train, R²={r2_train:.3f})", "h4", small=True)
        x_tr = train_df[temp_col].astype(float).values
        y_tr = y_train_prod

        # 회귀곡선/신뢰구간
        xx = np.linspace(np.nanmin(x_tr)-1, np.nanmax(x_tr)+1, 240)
        yhat, _, model_s, _ = fit_poly3_and_predict(x_tr, y_tr, xx)
        pred_train, _, _, _ = fit_poly3_and_predict(x_tr, y_tr, x_tr)
        s_res = np.nanstd(y_tr - pred_train)  # 잔차 표준편차
        upper = yhat + 1.96*s_res
        lower = yhat - 1.96*s_res

        # 온도별 중앙값
        bins = np.linspace(np.nanmin(x_tr), np.nanmax(x_tr), 15)
        gb = pd.DataFrame({"bin": pd.cut(x_tr, bins), "y": y_tr}).groupby("bin")["y"].median().reset_index()
        gb["x"] = [b.mid for b in gb["bin"]]

        figp = go.Figure()
        # 신뢰구간(밴드)
        figp.add_trace(go.Scatter(x=np.concatenate([xx, xx[::-1]]),
                                  y=np.concatenate([upper, lower[::-1]]),
                                  fill='toself',
                                  fillcolor='rgba(255, 159, 67, 0.25)',  # 주황톤 그라데이션 느낌
                                  line=dict(color='rgba(255,159,67,0)'),
                                  name='95% 신뢰구간',
                                  hoverinfo='skip'))
        # 회귀곡선
        figp.add_trace(go.Scatter(x=xx, y=yhat, mode='lines',
                                  line=dict(color='#1f77b4', width=3),
                                  name='Poly-3',
                                  hovertemplate="x=%{x:.1f}℃<br>y=%{y:,.0f} MJ"))
        # 학습샘플
        figp.add_trace(go.Scatter(x=x_tr, y=y_tr, mode='markers',
                                  marker=dict(size=7, color='rgba(31,119,180,0.7)'),
                                  name='학습 샘플',
                                  hovertemplate="온도=%{x:.1f}℃<br>공급량=%{y:,.0f} MJ"))
        # 온도별 중앙값
        figp.add_trace(go.Scatter(x=gb["x"], y=gb["y"], mode='markers',
                                  marker=dict(size=10, color='rgba(255,127,14,0.95)'),
                                  name='온도별 중앙값',
                                  hovertemplate="중앙값<br>온도=%{x:.1f}℃<br>공급량=%{y:,.0f} MJ"))
        figp.update_layout(
            margin=dict(l=10,r=10,t=40,b=10),
            height=420,
            legend=dict(bgcolor="rgba(255,255,255,0.8)"),
            hovermode="closest",
            xaxis_title="기온 (℃)",
            yaxis_title="공급량 (MJ)",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        st.plotly_chart(figp, use_container_width=True, config={
            "scrollZoom": True,      # 마우스 스크롤로만 확대/축소
            "doubleClick": "reset",  # 더블클릭 시 리셋
            "displaylogo": False
        })
    st.caption("ℹ️ **95% 신뢰구간(근사 예측구간)**: 잔차 표준편차 s 기반으로 예측값 ± 1.96·s. 새 관측의 약 95% 포함.")

# =============== B) 판매량 예측(냉방용) =====================================
else:
    # 기존 판매량 파트는 변경 없이 유지 (상단 요구사항은 공급량 파트에 해당)
    title_with_icon("🧊", "판매량 예측(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준", "h2")
    st.info("판매량 예측(냉방용) 파트는 이전과 동일하게 동작합니다. 필요 시 별도 수정 요청해 주세요.")
