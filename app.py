# app.py — 도시가스 공급·판매 예측 (Poly-3) + 추세분석 + 상단 Plotly 그래프

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
from io import BytesIO

# ============== 새로 추가 ==============
try:
    import plotly.graph_objects as go
except Exception:
    go = None  # requirements에 plotly>=5.18,<6 추가 필요
# =====================================

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
    st.markdown(
        f"<{level} class='{klass}'><span class='emoji'>{icon}</span><span>{text}</span></{level}>",
        unsafe_allow_html=True
    )

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
    월 단위 (날짜, 평균기온[, 추세분석]) → (연, 월, 예상기온, 추세기온)
    """
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = "기온예측" if "기온예측" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in df.columns if c in ["날짜","일자","date","Date"]), df.columns[0])
    base_temp_col = next((c for c in df.columns if ("평균기온" in c) or (str(c).lower() in ["temp","temperature","기온"])), None)

    # 추세 열(이름 다양한 경우 대응)
    trend_cols = [c for c in df.columns if any(k in str(c) for k in ["추세분석", "추세기온"])]
    trend_col = trend_cols[0] if trend_cols else None

    if base_temp_col is None:
        raise ValueError("기온예측 파일에서 '평균기온' 또는 '기온' 열을 찾지 못했습니다.")

    d = pd.DataFrame({
        "날짜": pd.to_datetime(df[date_col], errors="coerce"),
        "예상기온": pd.to_numeric(df[base_temp_col], errors="coerce")
    }).dropna(subset=["날짜"])
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)

    if trend_col:
        d["추세기온"] = pd.to_numeric(df[trend_col], errors="coerce")
    else:
        d["추세기온"] = np.nan  # 없을 수 있음

    return d[["연","월","예상기온","추세기온"]]

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# --- Poly3
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
    # columns.name 제거(“상품” 헤더 숨김)
    show.columns.name = None
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

        # =========== 예측 설정 (사이드바 맨 아래, 연/월 가로 배치) ===========
        title_with_icon("⚙️", "예측 설정", "h3", small=True)

        last_year = int(df["연"].max())
        # 시작(연/월)
        sy, sm = st.columns(2)
        with sy:
            start_y = st.selectbox("🚀 예측 시작(연)", list(range(2010,2036)),
                                   index=list(range(2010,2036)).index(last_year), key="start_y")
        with sm:
            start_m = st.selectbox("🗓️ 예측 시작(월)", list(range(1,13)), index=0, key="start_m")

        # 종료(연/월)
        ey, em = st.columns(2)
        with ey:
            end_y = st.selectbox("🏁 예측 종료(연)", list(range(2010,2036)),
                                 index=list(range(2010,2036)).index(last_year), key="end_y")
        with em:
            end_m = st.selectbox("🗓️ 예측 종료(월)", list(range(1,13)), index=11, key="end_m")

        run_btn = st.button("🧮 예측 시작", type="primary")

    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()
        fut_idx = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        # 예상기온/추세기온 병합
        fut_base = fut_base.merge(forecast_df, on=["연","월"], how="left")  # '예상기온','추세기온'

        # 빈 월 보강: 우선 추세기온→예상기온→월평균(temp_col)
        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("월평균").reset_index()
        # 1) 예상기온 보강
        miss1 = fut_base["예상기온"].isna()
        if miss1.any():
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
            fut_base.loc[miss1, "예상기온"] = fut_base.loc[miss1, "월평균"]
        # 2) 추세기온 보강
        miss2 = fut_base["추세기온"].isna()
        if miss2.any():
            fut_base.loc[miss2, "추세기온"] = fut_base.loc[miss2, "예상기온"]

        fut_base.drop(columns=[c for c in ["월평균"] if c in fut_base.columns], inplace=True)

        x_train_base = train_df[temp_col].astype(float).values

        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, x_train=x_train_base,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end, temp_col=temp_col,
            default_pred_years=list(range(int(start_y), int(end_y)+1))
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

    # ─── 시나리오 Δ°C ───
    title_with_icon("🌡️", "시나리오 Δ°C (평균기온 보정)", "h3", small=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="s_norm")
    with c2:
        d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="s_best")
    with c3:
        d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="s_cons")

    # 공통 예측 테이블 빌더
    def _forecast_table(delta: float) -> pd.DataFrame:
        x_future = (fut_base["예상기온"] + float(delta)).astype(float).values
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연","월"]].copy()
            tmp["월평균기온"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연","월","월평균기온"], columns="상품", values="예측").reset_index()
        # “상품” 헤더 숨김
        pivot.columns.name = None
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연","월","월평균기온"] + ordered)]
        pivot = pivot[["연","월","월평균기온"] + ordered + others]
        return pivot.sort_values(["연","월"]).reset_index(drop=True)

    def _forecast_table_trend() -> pd.DataFrame:
        # 추세기온 NaN은 이미 fut_base 구성시 보강됨
        x_future = fut_base["추세기온"].astype(float).values
        if np.isnan(x_future).any():
            # 그래도 남아있다면 마지막 안전장치로 월평균으로 대체
            back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
            x_future = np.where(np.isnan(x_future), back, x_future)

        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연","월"]].copy()
            tmp["월평균기온(추세)"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연","월","월평균기온(추세)"], columns="상품", values="예측").reset_index()
        pivot.columns.name = None  # “상품” 헤더 숨김
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연","월","월평균기온(추세)"] + ordered)]
        pivot = pivot[["연","월","월평균기온(추세)"] + ordered + others]
        return pivot.sort_values(["연","월"]).reset_index(drop=True)

    # ─── 표 4종 (월별표 + 연도별 총계표) ───
    def _render_with_year_sums(title, table, temp_col_name):
        st.markdown(f"### {title}")
        # 월별표(총계행 없이)
        table = table.copy()
        table.columns.name = None  # “상품” 헤더 숨김
        render_centered_table(
            table,
            float1_cols=[temp_col_name],
            int_cols=[c for c in table.columns if c not in ["연","월",temp_col_name]],
            index=False
        )
        # 연도별 총계
        sums = table.groupby("연").sum(numeric_only=True).reset_index()
        # ‘월’은 "1~12월", temp는 공란으로
        sums["월"] = "1~12월"
        if temp_col_name not in sums.columns:
            sums.insert(2, temp_col_name, "")
        else:
            sums[temp_col_name] = ""
        # 컬럼 정렬
        fixed = ["연", "월", temp_col_name]
        rest = [c for c in sums.columns if c not in fixed]
        sums = sums[fixed + rest]
        sums.columns.name = None
        st.markdown("#### 연도별 총계")
        render_centered_table(sums, int_cols=[c for c in rest], index=False)
        return sums

    tbl_n   = _forecast_table(d_norm)
    tbl_b   = _forecast_table(d_best)
    tbl_c   = _forecast_table(d_cons)
    tbl_trd = _forecast_table_trend()

    sum_n   = _render_with_year_sums("🎯 Normal",        tbl_n,   "월평균기온")
    sum_b   = _render_with_year_sums("💎 Best",          tbl_b,   "월평균기온")
    sum_c   = _render_with_year_sums("🛡️ Conservative", tbl_c,   "월평균기온")
    sum_t   = _render_with_year_sums("📈 추세분석",       tbl_trd, "월평균기온(추세)")

    # 다운로드(4개 시나리오 합본 CSV/Excel)
    def add_scenario(df, name, temp_name):
        out = df.copy()
        out.insert(0, "시나리오", name)
        out.rename(columns={temp_name:"월평균기온"}, inplace=True)
        out.columns.name = None
        return out

    to_dl = pd.concat([
        add_scenario(tbl_n,   "Normal",      "월평균기온"),
        add_scenario(tbl_b,   "Best",        "월평균기온"),
        add_scenario(tbl_c,   "Conservative","월평균기온"),
        add_scenario(tbl_trd, "추세분석",    "월평균기온(추세)"),
    ], ignore_index=True)

    st.download_button(
        "⬇️ 예측 결과 CSV 다운로드 (Normal/Best/Cons/추세분석)",
        data=to_dl.to_csv(index=False).encode("utf-8-sig"),
        file_name="citygas_supply_forecast_all_scenarios.csv",
        mime="text/csv",
        key="dl_csv"
    )

    # Excel (시나리오별 시트)
   excel_buf = BytesIO()
try:
    # 기본: openpyxl 사용 (대부분 환경에 설치되어 있음)
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as xw:
        tbl_n.to_excel(xw, sheet_name="Normal(月별)", index=False)
        sum_n.to_excel(xw, sheet_name="Normal(연도별)", index=False)
        tbl_b.to_excel(xw, sheet_name="Best(月별)", index=False)
        sum_b.to_excel(xw, sheet_name="Best(연도별)", index=False)
        tbl_c.to_excel(xw, sheet_name="Cons(月별)", index=False)
        sum_c.to_excel(xw, sheet_name="Cons(연도별)", index=False)
        tbl_trd.to_excel(xw, sheet_name="추세(月별)", index=False)
        sum_t.to_excel(xw, sheet_name="추세(연도별)", index=False)
except ModuleNotFoundError:
    # 드물게 openpyxl이 없을 때만 xlsxwriter로 폴백
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as xw:
        tbl_n.to_excel(xw, sheet_name="Normal(月별)", index=False)
        sum_n.to_excel(xw, sheet_name="Normal(연도별)", index=False)
        tbl_b.to_excel(xw, sheet_name="Best(月별)", index=False)
        sum_b.to_excel(xw, sheet_name="Best(연도별)", index=False)
        tbl_c.to_excel(xw, sheet_name="Cons(月별)", index=False)
        sum_c.to_excel(xw, sheet_name="Cons(연도별)", index=False)
        tbl_trd.to_excel(xw, sheet_name="추세(月별)", index=False)
        sum_t.to_excel(xw, sheet_name="추세(연도별)", index=False)

st.download_button(
    "⬇️ 예측 결과 Excel 다운로드 (시나리오별 시트)",
    data=excel_buf.getvalue(),
    file_name="citygas_supply_forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_xlsx"
)


    # ─── 상단 그래프 (Plotly) ───
    title_with_icon("📈", "그래프 (실적 + 예측(Normal) + 추세분석)", "h3", small=True)

    # 선택칩 3개 가로 배치
    cyr, cpr, ctr = st.columns(3)
    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    default_years = years_all_for_plot[-2:] if len(years_all_for_plot)>=2 else years_all_for_plot

    with cyr:
        years_view = st.multiselect("👀 실적연도", options=years_all_for_plot, default=default_years, key="supply_years_view")

    pred_default = mats.get("default_pred_years", [])
    with cpr:
        years_pred = st.multiselect("📈 예측연도 (Normal)", options=sorted(list(set(fut_base["연"].tolist()))),
                                    default=[y for y in pred_default if y in fut_base["연"].unique()], key="years_pred")
    with ctr:
        years_trnd = st.multiselect("📊 추세분석연도", options=sorted(list(set(fut_base["연"].tolist()))),
                                    default=[y for y in pred_default if y in fut_base["연"].unique()], key="years_trnd")

    months_txt = [f"{m}월" for m in months]

    for prod in prods:
        # 학습 성능
        y_train_prod = train_df[prod].astype(float).values
        # Normal 예측 벡터
        x_future_norm = (fut_base["예상기온"] + float(d_norm)).astype(float).values
        y_future_norm, r2_train, model, poly = fit_poly3_and_predict(x_train, y_train_prod, x_future_norm)
        P_norm = fut_base[["연","월"]].copy(); P_norm["pred"] = np.clip(np.rint(y_future_norm).astype(np.int64), a_min=0, a_max=None)

        # Trend 예측 벡터
        x_future_trend = fut_base["추세기온"].astype(float).values
        if np.isnan(x_future_trend).any():
            back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
            x_future_trend = np.where(np.isnan(x_future_trend), back, x_future_trend)
        y_future_trd, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_trend)
        P_trend = fut_base[["연","월"]].copy(); P_trend["pred"] = np.clip(np.rint(y_future_trd).astype(np.int64), a_min=0, a_max=None)

        # ─ Plotly 라인 ─
        if go is None:
            st.warning("Plotly가 설치되지 않아 상단 그래프는 Matplotlib로 대체됩니다.")
            fig = plt.figure(figsize=(9,3.6)); ax = plt.gca()
            for y in sorted([int(v) for v in years_view]):
                s = (base.loc[base["연"]==y, ["월", prod]].set_index("월")[prod]).reindex(months)
                ax.plot(months, s.values, label=f"{y} 실적")
            for y in years_pred:
                pred_vals = P_norm[P_norm["연"]==int(y)].sort_values("월")["pred"].reindex(range(1,13)).values
                ax.plot(months, pred_vals, linestyle="--", label=f"예측(Normal) {y}")
            for y in years_trnd:
                pred_vals = P_trend[P_trend["연"]==int(y)].sort_values("월")["pred"].reindex(range(1,13)).values
                ax.plot(months, pred_vals, linestyle=":", label=f"추세분석 {y}")
            ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels(months_txt)
            ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
            ax.set_title(f"{prod} — Poly-3 (Train R²={r2_train:.3f})"); ax.legend(loc="best"); st.pyplot(fig, clear_figure=True)
        else:
            fig = go.Figure()

            # 실적 연도
            for y in sorted([int(v) for v in years_view]):
                one = base[base["연"]==y][["월", prod]].dropna().sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in one["월"]], y=one[prod], mode="lines+markers",
                    name=f"{y} 실적", hovertemplate="%{x} %{y:,} MJ"
                ))

            # Normal 예측
            for y in years_pred:
                row = P_norm[P_norm["연"]==int(y)].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]], y=row["pred"], mode="lines",
                    name=f"예측(Normal) {y}", line=dict(dash="dash"), hovertemplate="%{x} %{y:,} MJ"
                ))

            # 추세 예측
            for y in years_trnd:
                row = P_trend[P_trend["연"]==int(y)].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]], y=row["pred"], mode="lines",
                    name=f"추세분석 {y}", line=dict(dash="dot"), hovertemplate="%{x} %{y:,} MJ"
                ))

            fig.update_layout(
                title=f"{prod} — Poly-3 (Train R²={r2_train:.3f})",
                xaxis=dict(title="월"),
                yaxis=dict(title="공급량 (MJ)", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
            )
            # 마우스휠 확대만 허용(드래그는 팬)
            config = dict(scrollZoom=True, doubleClick=False, displaylogo=False, displayModeBar=True)
            fig.update_layout(dragmode="pan")
            st.plotly_chart(fig, use_container_width=True, config=config)

        # ─ 하단 상관도 (Matplotlib 고정) ─
        title_with_icon("🔎", f"{prod} — 기온·공급량 상관(Train, R²={r2_train:.3f})", "h3", small=True)
        figc, axc = plt.subplots(figsize=(10,5.2))
        x_tr = train_df[temp_col].astype(float).values
        y_tr = y_train_prod
        axc.scatter(x_tr, y_tr, alpha=0.65, label="학습 샘플")

        xx = np.linspace(np.nanmin(x_tr)-1, np.nanmax(x_tr)+1, 200)
        yhat, _, model_s, _ = fit_poly3_and_predict(x_tr, y_tr, xx)
        axc.plot(xx, yhat, lw=2.8, color="#1f77b4", label="Poly-3")

        pred_train, _, _, _ = fit_poly3_and_predict(x_tr, y_tr, x_tr)
        resid = y_tr - pred_train
        s = np.nanstd(resid)
        axc.fill_between(xx, yhat-1.96*s, yhat+1.96*s, color="#ff7f0e", alpha=0.25, label="95% 신뢰구간")

        # 온도 구간 중앙값
        bins = np.linspace(np.nanmin(x_tr), np.nanmax(x_tr), 15)
        gb = pd.DataFrame({"bin": pd.cut(x_tr, bins), "y": y_tr}).groupby("bin")["y"].median().reset_index()
        gb["x"] = [b.mid for b in gb["bin"]]
        axc.scatter(gb["x"], gb["y"], label="온도별 중앙값", s=65, color="#ff7f0e")
        axc.set_xlabel("기온 (℃)"); axc.set_ylabel("공급량 (MJ)")
        axc.grid(alpha=0.25); axc.legend(loc="best")
        xmin, xmax = axc.get_xlim(); ymin, ymax = axc.get_ylim()
        axc.text(xmin + 0.02*(xmax-xmin), ymin + 0.04*(ymax-ymin),
                 f"Poly-3: {poly_eq_text(model_s)}",
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        st.pyplot(figc)

    st.caption("ℹ️ **95% 신뢰구간**: 잔차 표준편차 기준 근사 예측구간으로, 새로운 관측이 약 95% 확률로 이 음영 안에 들어옵니다.")

# =============== B) 판매량 예측(냉방용) =====================================
else:
    # (기존 로직 그대로 — 변경 없음 / 사이드바 구성만 유지)
    title_with_icon("🧊", "판매량 예측(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준", "h2")
    st.write("🗂️ 냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 준비하세요.")

    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        sales_src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    def _find_repo_sales_and_temp():
        here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        data_dir = here / "data"
        sales_candidates = [data_dir / "상품별판매량.xlsx", *[Path(p) for p in glob(str(data_dir / "*판매*.xlsx"))]]
        temp_candidates  = [data_dir / "기온.xlsx", *[Path(p) for p in glob(str(data_dir / "*기온*.xlsx"))],
                            *[Path(p) for p in glob(str(data_dir / "*temp*.csv"))]]
        sales_path = next((p for p in sales_candidates if p.exists()), None)
        temp_path  = next((p for p in temp_candidates  if p.exists()), None)
        return sales_path, temp_path

    c1, c2 = st.columns(2)
    if sales_src == "Repo 내 파일 사용":
        repo_sales_path, repo_temp_path = _find_repo_sales_and_temp()
        if not repo_sales_path or not repo_temp_path:
            with c1: sales_file = st.file_uploader("📄 냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
            with c2: temp_raw_file = st.file_uploader("🌡️ **기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])
        else:
            with c1: st.info(f"📄 레포 파일: {repo_sales_path.name}")
            with c2: st.info(f"🌡️ 레포 파일: {repo_temp_path.name}")
            sales_file = open(repo_sales_path, "rb")
            temp_raw_file = open(repo_temp_path, "rb")
    else:
        with c1: sales_file = st.file_uploader("📄 냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file = st.file_uploader("🌡️ **기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("👈 두 파일을 모두 준비하세요."); st.stop()

    try:
        xls = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "실적_월합" if "실적_월합" in xls.sheet_names else ("냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0])
        raw_sales = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw_sales = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(raw_sales)

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
        st.error("⛔ 날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int); sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("⛔ 기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()

    with st.sidebar:
        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("🗓️ 연도 선택", options=years_all, default=years_all)

        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        last_year = int(sales_df["연"].max())
        sy, sm = st.columns(2)
        with sy:
            start_y = st.selectbox("🚀 예측 시작(연)", list(range(2010,2036)),
                                   index=list(range(2010,2036)).index(last_year))
        with sm:
            start_m = st.selectbox("🗓️ 예측 시작(월)", list(range(1,13)), index=0)
        ey, em = st.columns(2)
        with ey:
            end_y   = st.selectbox("🏁 예측 종료(연)", list(range(2010,2036)),
                                   index=list(range(2010,2036)).index(last_year))
        with em:
            end_m   = st.selectbox("🗓️ 예측 종료(월)", list(range(1,13)), index=11)
        run_btn = st.button("🧮 예측 시작", type="primary")

    # (판매 파트 계산 로직은 기존과 동일하게 이어서 사용)
    # ...
