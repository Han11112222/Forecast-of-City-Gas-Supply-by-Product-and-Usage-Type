# app.py — 도시가스 공급·판매 예측 (3섹션 분리 + 추천 학습 데이터 기간 패널)
# A) 공급량 예측        : Poly-3 기반 + Normal/Best/Conservative + 기온추세분석
# B) 판매량 예측(냉방용) : 전월16~당월15 평균기온 + Poly-3/4 비교
# C) 공급량 추세분석     : 연도별 총합 OLS/CAGR/Holt/SES + ARIMA/SARIMA(12)
# 추천 학습 데이터 기간 : 예측유형 라디오 위 패널 (R² 상위 2개 구간, 소수 4자리, 배경 하이라이트)

import os
from io import BytesIO
from pathlib import Path
import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# Plotly (있으면 사용)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# statsmodels (ARIMA/SARIMA)
_HAS_SM = True
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    _HAS_SM = False

# ───────────── 공통 초기설정/스타일 ─────────────
st.set_page_config(page_title="도시가스 공급·판매량 예측", layout="wide")
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
    st.markdown(
        f"<{level} class='{klass}'><span class='emoji'>{icon}</span><span>{text}</span></{level}>",
        unsafe_allow_html=True,
    )

# ───────────── 한글 폰트 ─────────────
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

# ───────────── 공통 상수/유틸 ─────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
KNOWN_PRODUCT_ORDER = [
    "개별난방용", "중앙난방용",
    "자가열전용", "일반용(2)", "업무난방용", "냉난방용",
    "주한미군", "취사용", "총공급량",
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
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore",
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        nm = str(c).lower()
        if any(h in nm for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
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
            if str(c).lower() in ["날짜", "일자", "date"]:
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c
                    break
                except Exception:
                    pass
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp", "temperature"]):
                temp_col = c
                break
        if date_col is None or temp_col is None:
            return None
        out = pd.DataFrame(
            {"일자": pd.to_datetime(df[date_col], errors="coerce"), "기온": pd.to_numeric(df[temp_col], errors="coerce")}
        ).dropna()
        return out.sort_values("일자").reset_index(drop=True)

    name = getattr(file, "name", str(file))
    if name and name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row = None
    for i in range(len(head)):
        row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜", "일자", "date", "Date"] for v in row) and any(
            ("평균기온" in v) or ("기온" in v) or (isinstance(v, str) and v.lower() in ["temp", "temperature"])
            for v in row
        ):
            header_row = i
            break
    df = (
        pd.read_excel(xls, sheet_name=sheet)
        if header_row is None
        else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    )
    return _finalize(df)

@st.cache_data(ttl=600)
def read_temperature_forecast(file):
    """월 단위 (날짜, 평균기온[, 추세분석]) → (연, 월, 예상기온, 추세기온)"""
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = "기온예측" if "기온예측" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ["날짜", "일자", "date", "Date"]), df.columns[0])
    base_temp_col = next(
        (c for c in df.columns if ("평균기온" in c) or (str(c).lower() in ["temp", "temperature", "기온"])), None
    )
    trend_cols = [c for c in df.columns if any(k in str(c) for k in ["추세분석", "추세기온"])]
    trend_col = trend_cols[0] if trend_cols else None
    if base_temp_col is None:
        raise ValueError("기온예측 파일에서 '평균기온' 또는 '기온' 열을 찾지 못했습니다.")
    d = pd.DataFrame(
        {"날짜": pd.to_datetime(df[date_col], errors="coerce"), "예상기온": pd.to_numeric(df[base_temp_col], errors="coerce")}
    ).dropna(subset=["날짜"])
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    d["추세기온"] = pd.to_numeric(df[trend_col], errors="coerce") if trend_col else np.nan
    return d[["연", "월", "예상기온", "추세기온"]]

def month_start(x):
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(s, e):
    return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# Poly-3/4 공통
def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def fit_poly4_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=4, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

# ▼ Poly-3 방정식 표기(소수 4자리, 지수표기 금지)
def poly_eq_text(model, decimals: int = 4):
    c = model.coef_
    c1 = c[0] if len(c) > 0 else 0.0
    c2 = c[1] if len(c) > 1 else 0.0
    c3 = c[2] if len(c) > 2 else 0.0
    d = model.intercept_
    fmt = lambda v: f"{v:+,.{decimals}f}"
    return f"y = {fmt(c3)}x³ {fmt(c2)}x² {fmt(c1)}x {fmt(d)}"

def poly_eq_text4(model):
    c = model.coef_
    c1 = c[0] if len(c) > 0 else 0.0
    c2 = c[1] if len(c) > 1 else 0.0
    c3 = c[2] if len(c) > 2 else 0.0
    c4 = c[3] if len(c) > 3 else 0.0
    d = model.intercept_
    return f"y = {c4:+.5e}x⁴ {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols = float1_cols or []
    int_cols = int_cols or []
    show = df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(4).map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = (
                pd.to_numeric(show[c], errors="coerce")
                .round()
                .astype("Int64")
                .map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
            )
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

# ===========================================================
# 🔎 추천 학습 데이터 기간 (예측유형 라디오 위 패널 + 본문 섹션)
# ===========================================================
def _load_repo_supply_file():
    """레포 data 폴더에서 기본 공급량 파일을 찾아 로드."""
    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
    if not repo_files:
        return None, None
    default_idx = next((i for i, p in enumerate(repo_files)
                        if ("상품별공급량" in Path(p).stem) or ("공급량" in Path(p).stem)), 0)
    file_choice = st.selectbox("📄 실적 파일(Excel)", repo_files, index=default_idx,
                               format_func=lambda p: Path(p).name, key="rec_repo_file")
    df = read_excel_sheet(file_choice, prefer_sheet="데이터")
    return df, Path(file_choice).name

def compute_r2_by_start_year(df: pd.DataFrame, product: str):
    """종료연도=최신연도로 고정, 시작연도 1년씩 이동하며 R² 계산."""
    df = df.dropna(subset=["연", "월"]).copy()
    df["연"] = df["연"].astype(int)
    temp_col = detect_temp_col(df)
    if temp_col is None:
        raise ValueError("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요.")
    first_year = int(df["연"].min())
    last_year  = int(df["연"].max())

    rows = []
    for start in range(first_year, last_year + 1):
        part = df[(df["연"] >= start) & (df["연"] <= last_year)].copy()
        x = part[temp_col].astype(float).values
        y = part[product].astype(float).values
        if len(part) < 6 or np.allclose(y, y.mean()):
            r2 = np.nan
        else:
            _, r2, _, _ = fit_poly3_and_predict(x, y, x)
        rows.append({"시작연도": start, "종료연도": last_year, "기간": f"{start}~현재", "R2": r2})
    rec_df = pd.DataFrame(rows).dropna(subset=["R2"])
    if rec_df.empty:
        return None
    rec_df = rec_df.sort_values(["시작연도"]).reset_index(drop=True)

    # Top2 추출 (동점 시 최근 시작연도 우선)
    top2 = rec_df.sort_values(["R2", "시작연도"], ascending=[False, False]).head(2).reset_index(drop=True)
    top2.insert(0, "추천순위", [1, 2])
    return dict(rec_df=rec_df, top=top2, end=last_year)

def sidebar_recommendation_panel():
    """좌측 최상단 패널 UI (예측유형 라디오 위)"""
    with st.sidebar:
        with st.expander("🎯 추천 학습 데이터 기간", expanded=True):
            # 파일 소스 선택: 레포/업로드
            src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, horizontal=True, key="rec_src")

            df, src_name = None, None
            if src == "Repo 내 파일 사용":
                df, src_name = _load_repo_supply_file()
                if df is None:
                    st.info("📂 data 폴더에 엑셀 파일이 없습니다. 아래 다른 탭에서 업로드하고 다시 열어보세요.")
            else:
                up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="rec_up")
                if up is not None:
                    df = read_excel_sheet(up, prefer_sheet="데이터")
                    src_name = getattr(up, "name", "uploaded.xlsx")

            if df is not None and not df.empty:
                product_cols = guess_product_cols(df)
                default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:1]
                tgt = st.selectbox("대상 상품(1개)", options=product_cols,
                                   index=product_cols.index(default_products[0]) if default_products else 0,
                                   key="rec_target_prod")

                end_year = int(pd.to_numeric(df["연"], errors="coerce").max())
                st.caption(f"기준 종료연도: **{end_year}** (데이터 최신연도)")

                if st.button("🔍 추천 구간 계산", key="btn_calc_rec"):
                    rr = compute_r2_by_start_year(df, tgt)
                    if rr is None:
                        st.warning("계산 가능한 구간이 없습니다.")
                    else:
                        st.session_state["rec_result"] = dict(
                            rr=rr, product=tgt, src_name=src_name
                        )
                        st.success("추천 학습 구간 계산 완료! 아래 본문에 결과가 표시됩니다.")
            else:
                st.caption("실적 파일을 선택/업로드하면 계산할 수 있습니다.")

def body_recommendation_section():
    """본문 상단 결과 섹션 (계산 후 렌더링)"""
    if "rec_result" not in st.session_state:
        return
    product = st.session_state["rec_result"]["product"]
    rr = st.session_state["rec_result"]["rr"]
    rec_df = rr["rec_df"].copy()
    topk = rr["top"].copy()

    title_with_icon("🧠", f"추천 학습 데이터 기간 — {product}", "h2")

    # 표 (R² 4자리)
    show = topk[["추천순위", "기간", "시작연도", "종료연도", "R2"]].copy()
    render_centered_table(show, float1_cols=["R2"], index=False)

    # 그래프: 카테고리 축을 숫자 인덱스로 바꿔 세로 밴드 shape 사용
    cat_labels = rec_df.sort_values("시작연도")["기간"].tolist()
    r2_vals    = rec_df.sort_values("시작연도")["R2"].round(6).tolist()
    xpos = list(range(len(cat_labels)))
    top_periods = topk["기간"].tolist()
    top_idx = [cat_labels.index(p) for p in top_periods]

    y_min = max(0.0, float(np.nanmin(r2_vals)) - 0.01)
    y_max = min(1.0, float(np.nanmax(r2_vals)) + 0.01)

    if go is not None:
        fig = go.Figure()

        # 세로 배경 밴드 하이라이트 (Top1 진하게, Top2 연하게)
        for i, idx in enumerate(top_idx):
            band_color = "rgba(46,204,113,0.22)" if i == 0 else "rgba(52,152,219,0.16)"
            fig.add_shape(
                type="rect", xref="x", yref="paper",
                x0=idx - 0.48, x1=idx + 0.48, y0=0.0, y1=1.0,
                line=dict(width=0), fillcolor=band_color, layer="below"
            )
            fig.add_annotation(
                x=idx, y=1.0, xref="x", yref="paper",
                text=f"추천 {i+1}", showarrow=False, yshift=18,
                font=dict(size=11, color="#666")
            )

        fig.add_trace(go.Scatter(
            x=xpos, y=r2_vals,
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=7),
            name="R² (train fit)",
            hovertemplate="%{text}<br>R²=%{y:.4f}<extra></extra>",
            text=cat_labels
        ))
        fig.update_layout(
            title=f"학습 시작연도별 R² (종료연도={rr['end']})",
            xaxis=dict(title="학습 기간(시작연도~현재)", tickmode="array", tickvals=xpos, ticktext=cat_labels, tickangle=-25),
            yaxis=dict(title="R² (train fit)", range=[y_min, y_max]),
            margin=dict(t=60, b=80, l=60, r=30),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        xs = np.arange(len(cat_labels))
        fig, ax = plt.subplots(figsize=(10.8, 4.0))
        for i, idx in enumerate(top_idx):
            color = (46/255, 204/255, 113/255, 0.22) if i == 0 else (52/255, 152/255, 219/255, 0.16)
            ax.axvspan(idx-0.48, idx+0.48, color=color, zorder=0)
            ax.text(idx, 1.02, f"추천 {i+1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#666")
        ax.plot(xs, r2_vals, "-o", lw=2.8, ms=6, zorder=2)
        ax.set_title(f"학습 시작연도별 R² (종료연도={rr['end']})")
        ax.set_ylabel("R² (train fit)")
        ax.set_xticks(xs); ax.set_xticklabels(cat_labels, rotation=25, ha="right")
        ax.set_ylim(y_min, y_max); ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

# ===========================================================
# A) 공급량 예측
# ===========================================================
def render_supply_forecast():
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="sup_src")
        df, forecast_df = None, None

        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i, p in enumerate(repo_files)
                                    if ("상품별공급량" in Path(p).stem) or ("공급량" in Path(p).stem)), 0)
                file_choice = st.selectbox("📄 실적 파일(Excel)", repo_files, index=default_idx,
                                           format_func=lambda p: Path(p).name, key="sup_repo_file")
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
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="sup_up")
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")
            up_fc = st.file_uploader("🌡️ 예상기온 엑셀 업로드(xlsx) — (날짜, 평균기온[, 추세분석])", type=["xlsx"], key="sup_up_fc")
            if up_fc is not None:
                forecast_df = read_temperature_forecast(up_fc)

        if df is None or len(df) == 0:
            st.info("🧩 좌측에서 실적 엑셀을 선택/업로드하세요."); st.stop()
        if forecast_df is None or forecast_df.empty:
            st.info("🌡️ 좌측에서 예상기온 엑셀을 선택/업로드하세요."); st.stop()

        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("🗓️ 연도 선택", years_all, default=years_all, key="sup_years_sel")

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("🌡️ 기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()

        title_with_icon("🧰", "예측할 상품 선택", "h3", small=True)
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=default_products, key="sup_prods")

        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        last_year = int(df["연"].max())
        years = list(range(2010, 2036))
        col_sy, col_sm = st.columns(2)
        with col_sy:
            start_y = st.selectbox("🚀 예측 시작(연)", years, index=years.index(last_year), key="sup_sy")
        with col_sm:
            start_m = st.selectbox("📅 예측 시작(월)", list(range(1, 13)), index=0, key="sup_sm")
        col_ey, col_em = st.columns(2)
        with col_ey:
            end_y = st.selectbox("🏁 예측 종료(연)", years, index=years.index(last_year), key="sup_ey")
        with col_em:
            end_m = st.selectbox("📅 예측 종료(월)", list(range(1, 13)), index=11, key="sup_em")

        run_btn = st.button("🧮 예측 시작", type="primary", key="sup_run")

    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()
        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start:
            st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()
        fut_idx = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        # ✔️ 단순 병합
        fut_base = fut_base.merge(forecast_df, on=["연", "월"], how="left")

        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("월평균").reset_index()
        miss1 = fut_base["예상기온"].isna()
        if miss1.any():
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
            fut_base.loc[miss1, "예상기온"] = fut_base.loc[miss1, "월평균"]
        miss2 = fut_base["추세기온"].isna()
        if miss2.any():
            fut_base.loc[miss2, "추세기온"] = fut_base.loc[miss2, "예상기온"]
        fut_base.drop(columns=[c for c in ["월평균"] if c in fut_base.columns], inplace=True)

        x_train_base = train_df[temp_col].astype(float).values

        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, x_train=x_train_base,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end, temp_col=temp_col,
            default_pred_years=list(range(int(start_y), int(end_y) + 1)),
            years_sel=years_sel
        )
        st.success("✅ 공급량 예측(베이스) 준비 완료! 아래에서 **시나리오 Δ°C**를 조절하세요.")

    if "supply_materials" not in st.session_state:
        st.info("👈 좌측에서 설정 후 **예측 시작**을 눌러 실행하세요."); st.stop()

    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    x_train, fut_base = mats["x_train"], mats["fut_base"]
    temp_col = mats["temp_col"]; years_sel = mats["years_sel"]
    months = list(range(1, 13))

    # 시나리오 Δ°C
    title_with_icon("🌡️", "시나리오 Δ°C (평균기온 보정)", "h3", small=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="s_norm")
    with c2:
        d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="s_best")
    with c3:
        d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="s_cons")

    def _forecast_table(delta: float) -> pd.DataFrame:
        x_future = (fut_base["예상기온"] + float(delta)).astype(float).values
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연", "월"]].copy()
            tmp["월평균기온"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연", "월", "월평균기온"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연", "월", "월평균기온"] + ordered)]
        pivot = pivot[["연", "월", "월평균기온"] + ordered + others]
        return pivot.sort_values(["연", "월"]).reset_index(drop=True)

    def _forecast_table_trend() -> pd.DataFrame:
        x_future = fut_base["추세기온"].astype(float).values
        if np.isnan(x_future).any():
            back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
            x_future = np.where(np.isnan(x_future), back, x_future)
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연", "월"]].copy()
            tmp["월평균기온(추세)"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연", "월", "월평균기온(추세)"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연", "월", "월평균기온(추세)"] + ordered)]
        pivot = pivot[["연", "월", "월평균기온(추세)"] + ordered + others]
        return pivot.sort_values(["연", "월"]).reset_index(drop=True)

    # 표 + 연/반기 합계
    def _render_with_year_sums(title, table, temp_col_name):
        title_with_icon("🗂️", title, "h3", small=True)
        render_centered_table(
            table,
            float1_cols=[temp_col_name],
            int_cols=[c for c in table.columns if c not in ["연", "월", temp_col_name]],
            index=False,
        )
        year_sum = table.groupby("연").sum(numeric_only=True).reset_index()
        year_sum_show = year_sum.drop(columns=[c for c in ["월", temp_col_name] if c in year_sum.columns])
        year_sum_show.insert(1, "기간", "1~12월")
        cols_int = [c for c in year_sum_show.columns if c not in ["연", "기간"]]
        title_with_icon("🗓️", "연도별 총계", "h4", small=True)
        render_centered_table(year_sum_show, int_cols=cols_int, index=False)

        tmp = table.copy()
        tmp["__half"] = np.where(tmp["월"].astype(int) <= 6, "1~6월", "7~12월")
        half = tmp.groupby(["연", "__half"]).sum(numeric_only=True).reset_index().rename(columns={"__half": "반기"})
        half_to_show = half.rename(columns={"반기": "기간"}).drop(columns=[c for c in ["월", temp_col_name] if c in half.columns])
        title_with_icon("🧮", "반기별 총계 (1~6월, 7~12월)", "h4", small=True)
        render_centered_table(
            half_to_show,
            int_cols=[c for c in half_to_show.columns if c not in ["연", "기간"]],
            index=False,
        )
        return year_sum_show, half_to_show

    tbl_n = _forecast_table(d_norm)
    tbl_b = _forecast_table(d_best)
    tbl_c = _forecast_table(d_cons)
    tbl_trd = _forecast_table_trend()

    sum_n, half_n = _render_with_year_sums("🎯 Normal", tbl_n, "월평균기온")
    sum_b, half_b = _render_with_year_sums("💎 Best", tbl_b, "월평균기온")
    sum_c, half_c = _render_with_year_sums("🛡️ Conservative", tbl_c, "월평균기온")
    sum_t, half_t = _render_with_year_sums("📈 기온추세분석", tbl_trd, "월평균기온(추세)")

    # 다운로드 (메타 포함)
    def _pack_for_download(df_list, names, temp_names):
        outs = []
        for df, nm, tnm in zip(df_list, names, temp_names):
            d = df.copy()
            d.insert(0, "시나리오", nm)
            if tnm in d.columns and tnm != "월평균기온":
                d.rename(columns={tnm: "월평균기온"}, inplace=True)
            outs.append(d)
        return pd.concat(outs, ignore_index=True)

    to_dl = _pack_for_download(
        [tbl_n, tbl_b, tbl_c, tbl_trd],
        ["Normal", "Best", "Conservative", "기온추세분석"],
        ["월평균기온", "월평균기온", "월평균기온", "월평균기온(추세)"],
    )

    learn_years = sorted([int(y) for y in mats["years_sel"]])
    meta_learn  = f"{min(learn_years)}~{max(learn_years)}년" if learn_years else "-"
    all_years = sorted([int(y) for y in base["연"].unique()])
    if learn_years:
        span = list(range(min(learn_years), max(learn_years) + 1))
        exclude_years = [y for y in span if (y in all_years and y not in learn_years)]
    else:
        exclude_years = []
    meta_excl = ", ".join(str(y) for y in exclude_years) if exclude_years else "-"

    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            startrow = 2
            to_dl.to_excel(writer, index=False, sheet_name="Forecast", startrow=startrow)
            ws = writer.sheets["Forecast"]
            ws.cell(row=1, column=1, value="학습기간"); ws.cell(row=1, column=2, value=meta_learn)
            ws.cell(row=1, column=3, value="제외기간"); ws.cell(row=1, column=4, value=meta_excl)

            def write_yearsum(sheet_name, year_df, half_df):
                ysr = 2
                year_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=ysr)
                ws2 = writer.sheets[sheet_name]
                ws2.cell(row=1, column=1, value="학습기간"); ws2.cell(row=1, column=2, value=meta_learn)
                ws2.cell(row=1, column=3, value="제외기간"); ws2.cell(row=1, column=4, value=meta_excl)
                start_half = ysr + len(year_df) + 3
                half_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_half)

            write_yearsum("YearSum_Normal",    sum_n, half_n)
            write_yearsum("YearSum_Best",      sum_b, half_b)
            write_yearsum("YearSum_Cons",      sum_c, half_c)
            write_yearsum("YearSum_TrendTemp", sum_t, half_t)

        buf.seek(0)
        st.download_button(
            "⬇️ 예측 결과 XLSX 다운로드 (연합/반기 포함 · 학습·제외기간 표기)",
            data=buf.read(),
            file_name="citygas_supply_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.download_button(
            "⬇️ 예측 결과 CSV 다운로드 (Forecast만)",
            data=to_dl.to_csv(index=False).encode("utf-8-sig"),
            file_name="citygas_supply_forecast.csv",
            mime="text/csv",
        )

    # 그래프
    title_with_icon("📈", "그래프(실적 + 예측 + 기온추세분석)", "h3", small=True)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        show_best = st.toggle("Best 표시", value=False, key="show_best_top")
        show_cons = st.toggle("Conservative 표시", value=False, key="show_cons_top")

    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    default_years = years_all_for_plot[-2:] if len(years_all_for_plot) >= 2 else years_all_for_plot
    c_y1, c_y2, c_y3 = st.columns(3)
    with c_y1:
        years_view = st.multiselect("👀 실적연도", options=years_all_for_plot, default=default_years, key="supply_years_view")
    pred_default = mats.get("default_pred_years", [])
    with c_y2:
        years_pred = st.multiselect(
            "📈 예측연도",
            options=sorted(list(set(fut_base["연"].tolist()))),
            default=[y for y in pred_default if y in fut_base["연"].unique()],
            key="years_pred",
        )
    with c_y3:
        years_trnd = st.multiselect(
            "📊 기온추세분석연도",
            options=sorted(list(set(fut_base["연"].tolist()))),
            default=[y for y in pred_default if y in fut_base["연"].unique()],
            key="years_trnd",
        )

    months_txt = [f"{m}월" for m in months]
    def _pred_series(delta): return (fut_base["예상기온"] + float(delta)).astype(float).values
    x_future_norm = _pred_series(d_norm)
    x_future_best = _pred_series(d_best)
    x_future_cons = _pred_series(d_cons)
    x_future_trend = fut_base["추세기온"].astype(float).values
    if np.isnan(x_future_trend).any():
        back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
        x_future_trend = np.where(np.isnan(x_future_trend), back, x_future_trend)

    # Plotly Hover 준비
    fut_with_t = fut_base.copy()
    fut_with_t["T_norm"] = x_future_norm
    fut_with_t["T_best"] = x_future_best
    fut_with_t["T_cons"] = x_future_cons
    fut_with_t["T_trend"] = x_future_trend

    actual_temp = (
        base.groupby(["연", "월"])[temp_col].mean().reset_index().rename(columns={temp_col: "T_actual"})
    )

    for prod in prods:
        y_train_prod = train_df[prod].astype(float).values
        y_norm, r2_train, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_norm)
        P_norm = fut_with_t[["연", "월", "T_norm"]].copy(); P_norm["pred"] = np.clip(np.rint(y_norm).astype(np.int64), 0, None)
        y_best, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_best)
        P_best = fut_with_t[["연", "월", "T_best"]].copy(); P_best["pred"] = np.clip(np.rint(y_best).astype(np.int64), 0, None)
        y_cons, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_cons)
        P_cons = fut_with_t[["연", "월", "T_cons"]].copy(); P_cons["pred"] = np.clip(np.rint(y_cons).astype(np.int64), 0, None)
        y_trd, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_trend)
        P_trend = fut_with_t[["연", "월", "T_trend"]].copy(); P_trend["pred"] = np.clip(np.rint(y_trd).astype(np.int64), 0, None)

        # ───────── 그래프 ─────────
        if go is None:
            fig = plt.figure(figsize=(9, 3.6)); ax = plt.gca()
            for y in sorted([int(v) for v in years_view]):
                s = base.loc[base["연"] == y, ["월", prod]].set_index("월")[prod].reindex(months)
                ax.plot(months, s.values, label=f"{y} 실적")
            for y in years_pred:
                pv = P_norm[P_norm["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                ax.plot(months, pv, "--", label=f"예측(Normal) {y}")
                if show_best:
                    pv = P_best[P_best["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                    ax.plot(months, pv, "--", label=f"예측(Best) {y}")
                if show_cons:
                    pv = P_cons[P_cons["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                    ax.plot(months, pv, "--", label=f"예측(Conservative) {y}")
            for y in years_trnd:
                pv = P_trend[P_trend["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                ax.plot(months, pv, ":", label=f"기온추세분석 {y}")
            ax.set_xlim(1, 12); ax.set_xticks(months); ax.set_xticklabels(months_txt)
            ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
            ax.set_title(f"{prod} — Poly-3 (Train R²={r2_train:.3f})")
            ax.legend(loc="best"); st.pyplot(fig, clear_figure=True)
        else:
            fig = go.Figure()
            # 실적
            for y in sorted([int(v) for v in years_view]):
                one = base[base["연"] == y][["월", prod]].dropna().sort_values("월")
                t_one = actual_temp[actual_temp["연"] == y].sort_values("월")
                one = one.merge(t_one[["월", "T_actual"]], on="월", how="left")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in one["월"]],
                    y=one[prod],
                    customdata=np.round(one["T_actual"].values.astype(float), 2),
                    mode="lines+markers",
                    name=f"{y} 실적",
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
            # 예측(Normal/Best/Cons)
            for y in years_pred:
                row = P_norm[P_norm["연"] == int(y)].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]],
                    y=row["pred"],
                    customdata=np.round(row["T_norm"].values.astype(float), 2),
                    mode="lines",
                    name=f"예측(Normal) {y}",
                    line=dict(dash="dash"),
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
                if show_best:
                    rb = P_best[P_best["연"] == int(y)].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in rb["월"]],
                        y=rb["pred"],
                        customdata=np.round(rb["T_best"].values.astype(float), 2),
                        mode="lines",
                        name=f"예측(Best) {y}",
                        line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                    ))
                if show_cons:
                    rc = P_cons[P_cons["연"] == int(y)].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in rc["월"]],
                        y=rc["pred"],
                        customdata=np.round(rc["T_cons"].values.astype(float), 2),
                        mode="lines",
                        name=f"예측(Conservative) {y}",
                        line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                    ))
            # 기온추세
            for y in years_trnd:
                row = P_trend[P_trend["연"] == int(y)].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]],
                    y=row["pred"],
                    customdata=np.round(row["T_trend"].values.astype(float), 2),
                    mode="lines",
                    name=f"기온추세분석 {y}",
                    line=dict(dash="dot"),
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
            fig.update_layout(
                title=f"{prod} — Poly-3 (Train R²={r2_train:.3f})",
                xaxis=dict(title="월"),
                yaxis=dict(title="공급량 (MJ)", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
                margin=dict(t=60, b=120, l=40, r=20),
                dragmode="pan",
            )
            st.plotly_chart(fig, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))

        # 월별 표
        title_with_icon("📑", f"{prod} — 월별 표 (선택 연도)", "h3", small=True)
        months_idx = list(range(1, 13))
        table = pd.DataFrame({"월": months_idx})
        for y in sorted([int(v) for v in years_view]):
            s = base.loc[base["연"] == y, ["월", prod]].set_index("월")[prod].astype(float)
            table[f"{y} 실적"] = s.reindex(months_idx).values
        for y in years_pred:
            s = P_norm[P_norm["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
            table[f"예측(Normal) {y}"] = s.reindex(months_idx).values
        if show_best:
            for y in years_pred:
                s = P_best[P_best["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
                table[f"예측(Best) {y}"] = s.reindex(months_idx).values
        if show_cons:
            for y in years_pred:
                s = P_cons[P_cons["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
                table[f"예측(Conservative) {y}"] = s.reindex(months_idx).values
        for y in years_trnd:
            s = P_trend[P_trend["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
            table[f"기온추세 {y}"] = s.reindex(months_idx).values
        sum_row = {"월": "합계"}
        for c in [col for col in table.columns if col != "월"]:
            sum_row[c] = pd.to_numeric(table[c], errors="coerce").sum()
        table_show = pd.concat([table, pd.DataFrame([sum_row])], ignore_index=True)
        render_centered_table(
            table_show,
            int_cols=[c for c in table_show.columns if c != "월"],
            index=False,
        )

        # 산점도
        title_with_icon("🔎", f"{prod} — 기온·공급량 상관(Train, R²={r2_train:.3f})", "h3", small=True)
        figc, axc = plt.subplots(figsize=(10, 5.2))
        x_tr = train_df[temp_col].astype(float).values
        y_tr = y_train_prod
        axc.scatter(x_tr, y_tr, alpha=0.65, label="학습 샘플")
        xx = np.linspace(np.nanmin(x_tr) - 1, np.nanmax(x_tr) + 1, 200)
        yhat, _, model_s, _ = fit_poly3_and_predict(x_tr, y_tr, xx)
        axc.plot(xx, yhat, lw=2.8, color="#1f77b4", label="Poly-3")
        pred_train, _, _, _ = fit_poly3_and_predict(x_tr, y_tr, x_tr)
        resid = y_tr - pred_train; s = np.nanstd(resid)
        axc.fill_between(xx, yhat - 1.96 * s, yhat + 1.96 * s, color="#ff7f0e", alpha=0.25, label="95% 신뢰구간")
        axc.set_xlabel("기온 (℃)"); axc.set_ylabel("공급량 (MJ)")
        axc.grid(alpha=0.25); axc.legend(loc="best")
        axc.text(0.02, 0.04, f"Poly-3: {poly_eq_text(model_s)}", transform=axc.transAxes,
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        st.pyplot(figc)

    st.caption("ℹ️ 95% 신뢰구간: 잔차 표준편차 기준 근사 예측구간(신규 관측 약 95% 포함).")

# ===========================================================
# B) 판매량 예측(냉방용) — Poly-3/4
# ===========================================================
def render_cooling_sales_forecast():
    title_with_icon("🧊", "판매량 예측(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준", "h2")
    st.write("🗂️ 냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 준비하세요.")

    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        sales_src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    def _find_repo_sales_and_temp():
        here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        data_dir = here / "data"
        sales_candidates = [data_dir / "상품별판매량.xlsx", *[Path(p) for p in glob(str(data_dir / "*판매*.xlsx"))]]
        temp_candidates = [data_dir / "기온.xlsx", *[Path(p) for p in glob(str(data_dir / "*기온*.xlsx"))],
                           *[Path(p) for p in glob(str(data_dir / "*temp*.csv"))]]
        sales_path = next((p for p in sales_candidates if p.exists()), None)
        temp_path = next((p for p in temp_candidates if p.exists()), None)
        return sales_path, temp_path

    c1, c2 = st.columns(2)
    if sales_src == "Repo 내 파일 사용":
        repo_sales_path, repo_temp_path = _find_repo_sales_and_temp()
        if not repo_sales_path or not repo_temp_path:
            with c1: sales_file = st.file_uploader("📄 냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
            with c2: temp_raw_file = st.file_uploader("🌡️ **기온 RAW(일별)** (xlsx/csv)", type=["xlsx", "csv"])
        else:
            with c1: st.info(f"📄 레포 파일: {repo_sales_path.name}")
            with c2: st.info(f"🌡️ 레포 파일: {repo_temp_path.name}")
            sales_file = open(repo_sales_path, "rb")
            temp_raw_file = open(repo_temp_path, "rb")
    else:
        with c1: sales_file = st.file_uploader("📄 냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file = st.file_uploader("🌡️ **기온 RAW(일별)** (xlsx/csv)", type=["xlsx", "csv"])

    if 'sales_file' not in locals() or sales_file is None or temp_raw_file is None:
        st.info("👈 두 파일을 모두 준비하세요."); st.stop()

    try:
        xls = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "실적_월합" if "실적_월합" in xls.sheet_names else ("냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0])
        raw_sales = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw_sales = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(raw_sales)

    date_candidates = [c for c in ["판매월", "날짜", "일자", "date"] if c in sales_df.columns]
    if date_candidates: date_col = date_candidates[0]
    else:
        score = {}
        for c in sales_df.columns:
            try: score[c] = pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
            except Exception: pass
        date_col = max(score, key=score.get) if score else None
    cool_cols = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    value_col = next((c for c in cool_cols if "냉방용" in str(c)), (cool_cols[0] if cool_cols else None))
    if date_col is None or value_col is None:
        st.error("⛔ 날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월", "판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("⛔ 기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()

    with st.sidebar:
        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("🗓️ 연도 선택", options=years_all, default=years_all)

        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        years = list(range(2010, 2036))
        last_year = int(sales_df["연"].max())
        col_sy, col_sm = st.columns(2)
        with col_sy:
            start_y = st.selectbox("🚀 예측 시작(연)", years, index=years.index(last_year))
        with col_sm:
            start_m = st.selectbox("📅 예측 시작(월)", list(range(1, 13)), index=0)
        col_ey, col_em = st.columns(2)
        with col_ey:
            end_y = st.selectbox("🏁 예측 종료(연)", years, index=years.index(last_year))
        with col_em:
            end_m = st.selectbox("📅 예측 종료(월)", list(range(1, 13)), index=11)
        run_btn = st.button("🧮 예측 시작", type="primary")

    if run_btn:
        temp_raw["연"] = temp_raw["일자"].dt.year; temp_raw["월"] = temp_raw["일자"].dt.month
        monthly_cal = temp_raw.groupby(["연", "월"])["기온"].mean().reset_index()
        fallback_by_M = temp_raw.groupby("월")["기온"].mean()

        def period_avg(label_m: pd.Timestamp) -> float:
            m = month_start(label_m)
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
            e = m + pd.DateOffset(days=14)                                # 당월15
            mask = (temp_raw["일자"] >= s) & (temp_raw["일자"] <= e)
            return temp_raw.loc[mask, "기온"].mean()

        train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
        rows = [{"판매월": m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
        sj = pd.merge(train_sales[["판매월", "판매량"]], pd.DataFrame(rows), on="판매월", how="left")
        miss = sj["기간평균기온"].isna()
        if miss.any():
            sj.loc[miss, "기간평균기온"] = sj.loc[miss, "판매월"].dt.month.map(fallback_by_M)
        sj = sj.dropna(subset=["기간평균기온", "판매량"])

        x_train = sj["기간평균기온"].astype(float).values
        y_train = sj["판매량"].astype(float).values
        _, r2_fit, model_fit, _ = fit_poly3_and_predict(x_train, y_train, x_train)
        _, r2_fit4, model_fit4, _ = fit_poly4_and_predict(x_train, y_train, x_train)

        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start:
            st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()

        months_rng = month_range_inclusive(f_start, f_end)
        rows = []
        for m in months_rng:
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"] >= s) & (temp_raw["일자"] <= e)
            avg_period = temp_raw.loc[mask, "기온"].mean()
            avg_month = monthly_cal.loc[(monthly_cal["연"] == m.year) & (monthly_cal["월"] == m.month), "기온"].mean()
            rows.append({"연": int(m.year), "월": int(m.month), "기간평균기온": avg_period, "당월평균기온": avg_month})
        pred_base = pd.DataFrame(rows)
        for c in ["기간평균기온", "당월평균기온"]:
            miss = pred_base[c].isna()
            if miss.any():
                pred_base.loc[miss, c] = pred_base.loc[miss, "월"].map(fallback_by_M)

        st.session_state["sales_materials"] = dict(
            sales_df=sales_df, temp_raw=temp_raw, years_all=years_all,
            train_xy=(x_train, y_train),
            r2_fit=r2_fit, model_fit=model_fit,
            r2_fit4=r2_fit4, model_fit4=model_fit4,
            pred_base=pred_base, f_start=f_start, f_end=f_end,
        )
        st.success("✅ 냉방용 판매량 예측(베이스) 준비 완료! 아래에서 시나리오 Δ°C를 조절하세요.")

    if "sales_materials" not in st.session_state:
        st.info("👈 좌측에서 설정 후 **예측 시작**을 눌러 실행하세요."); st.stop()

    sm = st.session_state["sales_materials"]
    sales_df, pred_base = sm["sales_df"], sm["pred_base"]
    x_train, y_train = sm["train_xy"]
    r2_fit, r2_fit4 = sm["r2_fit"], sm["r2_fit4"]
    years_all = sm["years_all"]

    st.markdown("#### 다항식 보기 선택")
    view_choice = st.radio("다항식", options=["3차(Poly-3)", "4차(Poly-4)", "둘 다"], index=2, horizontal=True, key="poly_view_choice")
    show_poly3 = view_choice in ["3차(Poly-3)", "둘 다"]
    show_poly4 = view_choice in ["4차(Poly-4)", "둘 다"]

    if show_poly3:
        st.subheader("시나리오 Δ°C (평균기온 보정) — Poly-3")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.5, format="%.1f", key="c_norm")
        with c2:
            d_best = st.number_input("Best Δ°C", value=-1.0, step=0.5, format="%.1f", key="c_best")
        with c3:
            d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.5, format="%.1f", key="c_cons")

        def forecast_sales_table(delta: float) -> pd.DataFrame:
            base = pred_base.copy()
            base["월평균기온(적용)"] = base["당월평균기온"] + delta
            base["기간평균기온(적용)"] = base["기간평균기온"] + delta
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, base["기간평균기온(적용)"].values.astype(float))
            base["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            out = base[["연", "월", "월평균기온(적용)", "기간평균기온(적용)", "예측판매량"]].copy()
            out.loc[len(out)] = ["", "종계", "", "", int(out["예측판매량"].sum())]
            return out

        st.markdown("### Normal"); sale_n = forecast_sales_table(d_norm)
        render_centered_table(sale_n, float1_cols=["월평균기온(적용)", "기간평균기온(적용)"], int_cols=["예측판매량"], index=False)
        st.markdown("### Best");   sale_b = forecast_sales_table(d_best)
        render_centered_table(sale_b, float1_cols=["월평균기온(적용)", "기간평균기온(적용)"], int_cols=["예측판매량"], index=False)
        st.markdown("### Conservative"); sale_c = forecast_sales_table(d_cons)
        render_centered_table(sale_c, float1_cols=["월평균기온(적용)", "기간평균기온(적용)"], int_cols=["예측판매량"], index=False)

        st.download_button(
            "판매량 예측 CSV 다운로드 (Poly-3 · Normal)",
            data=sale_n.to_csv(index=False).encode("utf-8-sig"),
            file_name="cooling_sales_forecast_poly3_normal.csv",
            mime="text/csv",
        )

        # 검증
        st.subheader("판매량 예측 검증 — Poly-3")
        valid_pred = sale_n[sale_n["월"] != "종계"].copy()
        valid_pred["연"] = pd.to_numeric(valid_pred["연"], errors="coerce").astype("Int64")
        valid_pred["월"] = pd.to_numeric(valid_pred["월"], errors="coerce").astype("Int64")
        comp = pd.merge(
            valid_pred[["연", "월", "예측판매량"]],
            sales_df[["연", "월", "판매량"]].rename(columns={"판매량": "실제판매량"}),
            on=["연", "월"],
            how="left",
        ).sort_values(["연", "월"])
        comp["오차"] = (comp["예측판매량"] - comp["실제판매량"]).astype("Int64")
        comp["오차율(%)"] = ((comp["오차"] / comp["실제판매량"]) * 100).round(1).astype("Float64")
        render_centered_table(comp[["연", "월", "실제판매량", "예측판매량", "오차", "오차율(%)"]],
                              int_cols=["실제판매량", "예측판매량", "오차"], index=False)

        # 그래프(Normal)
        st.subheader("그래프 (Normal 기준) — Poly-3")
        years_default = years_all[-5:] if len(years_all) >= 5 else years_all
        years_view = st.multiselect("표시할 실적 연도", options=years_all,
                                    default=st.session_state.get("sales_years_view", years_default),
                                    key="sales_years_view")
        base_plot = pred_base.copy()
        base_plot["기간평균기온(적용)"] = base_plot["기간평균기온"] + d_norm
        y_pred_norm, r2_line, model_line, _ = fit_poly3_and_predict(
            x_train, y_train, base_plot["기간평균기온(적용)"].values.astype(float)
        )
        base_plot["pred"] = np.clip(np.rint(y_pred_norm).astype(np.int64), 0, None)
        months = list(range(1, 13))
        fig2, ax2 = plt.subplots(figsize=(10, 4.2))
        for y in years_view:
            one = sales_df[sales_df["연"] == y][["월", "판매량"]].dropna()
            if not one.empty:
                ax2.plot(one["월"], one["판매량"], label=f"{y} 실적", alpha=0.95)
        pred_vals = []
        y, m = int(sm["f_start"].year), int(sm["f_start"].month)
        P2 = base_plot[["연", "월", "pred"]].astype(int)
        for _ in range(12):
            row = P2[(P2["연"] == y) & (P2["월"] == m)]
            pred_vals.append(row.iloc[0]["pred"] if len(row) else np.nan)
            if m == 12: y += 1; m = 1
            else: m += 1
        ax2.plot(months, pred_vals, "--", lw=2.5, label="예측(Normal)")
        ax2.set_xlim(1, 12); ax2.set_xlabel("월"); ax2.set_ylabel("판매량 (MJ)")
        ax2.set_title(f"냉방용 — Poly-3 (Train R²={r2_line:.3f})")
        ax2.legend(loc="best"); ax2.grid(alpha=0.25)
        ax2.text(0.02, 0.96, f"Poly-3: {poly_eq_text(model_line)}",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        st.pyplot(fig2)

        # 산점
        st.subheader(f"기온-냉방용 실적 상관관계 (Train, R²={r2_fit:.3f}) — Poly-3")
        fig3, ax3 = plt.subplots(figsize=(10, 5.2))
        ax3.scatter(x_train, y_train, alpha=0.65, label="학습 샘플")
        xx = np.linspace(np.nanmin(x_train) - 1, np.nanmax(x_train) + 1, 200)
        yhat, _, model_s, _ = fit_poly3_and_predict(x_train, y_train, xx)
        ax3.plot(xx, yhat, lw=2.6, color="#1f77b4", label="Poly-3")
        pred_train, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_train)
        resid = y_train - pred_train; s = np.nanstd(resid)
        ax3.fill_between(xx, yhat - 1.96 * s, yhat + 1.96 * s, color="#1f77b4", alpha=0.14, label="95% 신뢰구간")
        ax3.set_xlabel("기간평균기온 (℃)"); ax3.set_ylabel("판매량 (MJ)")
        ax3.grid(alpha=0.25); ax3.legend(loc="best")
        ax3.text(0.02, 0.06, f"Poly-3: {poly_eq_text(model_s)}",
                 transform=ax3.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        st.pyplot(fig3)

# ===========================================================
# C) 공급량 추세분석 예측 — OLS/CAGR/Holt/SES + ARIMA/SARIMA
# ===========================================================
def render_trend_forecast():
    title_with_icon("📈", "공급량 추세분석 예측 (연도별 총합 · Normal)", "h2")

    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="trend_src")

        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i, p in enumerate(repo_files)
                                   if ("상품별공급량" in Path(p).stem) or ("공급량" in Path(p).stem)), 0)
                file_choice = st.selectbox("📄 실적 파일(Excel)", repo_files, index=default_idx,
                                           format_func=lambda p: Path(p).name, key="trend_file_ch")
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("📂 data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")
                df = None
        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="trend_up")
            df = read_excel_sheet(up, prefer_sheet="데이터") if up is not None else None

        if df is None or df.empty:
            st.info("👈 좌측에서 실적 파일을 선택/업로드하세요."); st.stop()

        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("🗓️ 연도 선택(학습)", years_all, default=years_all, key="trend_years")

        title_with_icon("🧰", "분석할 상품 선택", "h3", small=True)
        product_cols = guess_product_cols(df)
        defaults = [c for c in ["개별난방용", "중앙난방용", "취사용"] if c in product_cols]
        if not defaults:
            defaults = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols][:3] or product_cols[:3]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=defaults, key="trend_prods")

        title_with_icon("⚙️", "예측 연도", "h3", small=True)
        last_year = int(df["연"].max())
        cand_years = list(range(2010, 2036))
        start_y = st.selectbox("🚀 예측 시작(연)", cand_years, index=cand_years.index(min(last_year + 1, 2035)), key="trend_sy")
        end_y = st.selectbox("🏁 예측 종료(연)", cand_years, index=cand_years.index(min(last_year + 2, 2035)), key="trend_ey")

        title_with_icon("🧪", "적용할 방법", "h3", small=True)
        method_opts = ["OLS(선형추세)", "CAGR(복리성장)", "Holt(지수평활)", "지수평활(SES)", "ARIMA", "SARIMA(12)"]
        methods_selected = st.multiselect("방법 선택(표·그래프 표시)", options=method_opts, default=method_opts, key="trend_methods")

    base = df.dropna(subset=["연", "월"]).copy()
    base["연"] = base["연"].astype(int); base["월"] = base["월"].astype(int)
    years_pred = list(range(int(start_y), int(end_y) + 1))
    yearly_all = base.groupby("연").sum(numeric_only=True).reset_index()

    def _fore_ols(years, vals, target_years):
        x = np.array(years, float).reshape(-1, 1); y = np.array(vals, float)
        mdl = LinearRegression().fit(x, y)
        return {ty: float(mdl.predict(np.array([[ty]], float))[0]) for ty in target_years}

    def _fore_cagr(years, vals, target_years):
        years = list(years); vals = list(vals)
        y0, yT = years[0], years[-1]; v0, vT = float(vals[0]), float(vals[-1])
        n = max(1, (yT - y0)); g = (vT / v0) ** (1.0 / n) - 1.0 if v0 > 0 else 0.0
        basev = vT; out = {}
        for i, ty in enumerate(target_years, start=1):
            out[ty] = basev * ((1.0 + g) ** i)
        return out

    def _fore_ses(vals, target_len, alpha=0.3):
        l = float(vals[0])
        for v in vals[1:]:
            l = alpha * float(v) + (1 - alpha) * l
        return [l for _ in range(target_len)]

    def _fore_holt(vals, target_len, alpha=0.3, beta=0.1):
        l = float(vals[0]); b = float(vals[1] - vals[0]) if len(vals) >= 2 else 0.0
        for v in vals[1:]:
            prev_l = l
            l = alpha * float(v) + (1 - alpha) * (l + b)
            b = beta * (l - prev_l) + (1 - beta) * b
        return [l + (h + 1) * b for h in range(target_len)]

    def _monthly_series_for(prod: str) -> pd.Series:
        s = base[["연", "월", prod]].dropna()
        s["날짜"] = pd.to_datetime(s["연"].astype(int).astype(str) + "-" + s["월"].astype(int).astype(str) + "-01")
        s = s.sort_values("날짜").set_index("날짜")[prod].astype(float).asfreq("MS")
        return s

    def _prepare_train_series(ts: pd.Series, train_years: list[int]) -> pd.Series:
        ser = ts[ts.index.year.isin(train_years)].copy()
        if ser.empty: return ser
        full_idx = pd.date_range(start=ser.index.min().to_period("M").to_timestamp(),
                                 end=ser.index.max().to_period("M").to_timestamp(), freq="MS")
        ser = ser.reindex(full_idx).interpolate(limit_direction="both").ffill().bfill()
        return ser

    def _yearly_series_for(prod: str) -> pd.Series:
        y = base.groupby("연")[prod].sum(numeric_only=True).sort_index().astype(float)
        full_years = pd.Index(range(int(y.index.min()), int(y.index.max()) + 1), name="연")
        y = y.reindex(full_years).interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
        return y

    def _fit_arima_candidate(series: pd.Series, orders=[(1,1,0),(0,1,1),(1,1,1)]):
        best, aic = None, np.inf
        for od in orders:
            try:
                m = ARIMA(series, order=od).fit()
                if m.aic < aic: best, aic = m, m.aic
            except Exception:
                continue
        return best

    def _fore_arima_yearsum(prod: str, target_years: list[int]) -> dict:
        if not _HAS_SM: return {y: np.nan for y in target_years}
        out = {y: np.nan for y in target_years}
        # 1) 월별
        try:
            ts_m = _monthly_series_for(prod)
            train_m = _prepare_train_series(ts_m, years_sel)
            if not train_m.empty:
                last_year = int(train_m.index[-1].year)
                steps = 12 * max(1, (max(target_years) - last_year))
                mdl = _fit_arima_candidate(train_m)
                if mdl is not None:
                    f = mdl.forecast(steps=steps)
                    fut = f.copy()
                    fut.index = pd.date_range(start=train_m.index[-1] + pd.offsets.MonthBegin(1),
                                              periods=len(fut), freq="MS")
                    df_year = fut.groupby(fut.index.year).sum()
                    for y in target_years:
                        if y in df_year.index: out[y] = float(df_year.loc[y])
        except Exception:
            pass
        # 2) 폴백: 연도합 직접
        if all(np.isnan(list(out.values()))):
            try:
                ys = _yearly_series_for(prod); ys_train = ys[ys.index.isin(years_sel)]
                if len(ys_train) >= 3:
                    mdl = _fit_arima_candidate(ys_train)
                    if mdl is not None:
                        f = mdl.forecast(steps=len(target_years))
                        start_y = int(ys_train.index.max()) + 1
                        idx = pd.Index(range(start_y, start_y + len(f)), name="연"); f.index = idx
                        for y in target_years:
                            if y in f.index: out[y] = float(f.loc[y])
            except Exception:
                pass
        return out

    def _fore_sarima_yearsum(prod: str, target_years: list[int]) -> dict:
        if not _HAS_SM: return {y: np.nan for y in target_years}
        out = {y: np.nan for y in target_years}
        # 1) 월별 SARIMA
        try:
            ts = _monthly_series_for(prod); train = _prepare_train_series(ts, years_sel)
            if not train.empty:
                last_year = int(train.index[-1].year)
                steps = 12 * max(1, (max(target_years) - last_year))
                mdl = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                f = mdl.forecast(steps=steps)
                fut = f.copy()
                fut.index = pd.date_range(start=train.index[-1] + pd.offsets.MonthBegin(1),
                                          periods=len(fut), freq="MS")
                df_year = fut.groupby(fut.index.year).sum()
                for y in target_years:
                    if y in df_year.index: out[y] = float(df_year.loc[y])
        except Exception:
            pass
        # 2) 폴백: 연도합 ARIMA
        if all(np.isnan(list(out.values()))):
            return _fore_arima_yearsum(prod, target_years)
        return out

    if not _HAS_SM:
        st.info("🔧 ARIMA/SARIMA는 statsmodels 미설치 환경에선 계산되지 않습니다.")

    # 화면: 상품별 카드
    for prod in prods:
        yearly = base.groupby("연").sum(numeric_only=True).reset_index()[["연", prod]].dropna().astype({"연": int})
        train = yearly[yearly["연"].isin(years_sel)].sort_values("연")
        if train.empty:
            st.warning(f"'{prod}' 학습 데이터가 없습니다."); continue

        yrs = train["연"].tolist()
        vals = train[prod].astype(float).tolist()

        pred_map = {}
        if "OLS(선형추세)" in methods_selected:
            pred_map["OLS(선형추세)"] = _fore_ols(yrs, vals, years_pred)
        if "CAGR(복리성장)" in methods_selected:
            pred_map["CAGR(복리성장)"] = _fore_cagr(yrs, vals, years_pred)
        if "지수평활(SES)" in methods_selected:
            pred_map["지수평활(SES)"] = dict(zip(years_pred, _fore_ses(vals, len(years_pred))))
        if "Holt(지수평활)" in methods_selected:
            pred_map["Holt(지수평활)"] = dict(zip(years_pred, _fore_holt(vals, len(years_pred))))
        if "ARIMA" in methods_selected:
            pred_map["ARIMA"] = _fore_arima_yearsum(prod, years_pred)
        if "SARIMA(12)" in methods_selected:
            pred_map["SARIMA(12)"] = _fore_sarima_yearsum(prod, years_pred)

        title_with_icon("📋", f"{prod} — 연도별 총합 예측표 (Normal)", "h3", small=True)
        df_tbl = pd.DataFrame({"연": years_pred})
        for k in methods_selected:
            if k in pred_map:
                vals_k = []
                for y in years_pred:
                    v = pred_map[k].get(y, np.nan)
                    vals_k.append(int(max(0, round(v))) if v == v else "")
                df_tbl[k] = vals_k
        render_centered_table(df_tbl, int_cols=[c for c in df_tbl.columns if c != "연"], index=False)

        st.markdown("**방법별 표시 토글(상단)**")
        toggles_top = {}
        cols_top = st.columns(min(6, len(methods_selected))) or [st]
        for i, name in enumerate(methods_selected):
            with cols_top[i % len(cols_top)]:
                toggles_top[name] = st.toggle(name, value=True, key=f"tg_top_{prod}_{name}")

        if go is None:
            fig, ax = plt.subplots(figsize=(10, 4.2))
            yd = yearly_all[["연", prod]].dropna().sort_values("연")
            ax.fill_between(yd["연"], yd[prod], step="pre", alpha=0.15)
            ax.plot(yd["연"], yd[prod], "-o", label="실적")
            markers = {"CAGR(복리성장)": "o", "Holt(지수평활)": "s", "OLS(선형추세)": "^",
                       "지수평활(SES)": "+", "ARIMA": "x", "SARIMA(12)": "D"}
            for name in methods_selected:
                if name in pred_map and toggles_top.get(name, True):
                    xs = years_pred
                    ys = [pred_map[name].get(y, np.nan) for y in xs]
                    ax.scatter(xs, ys, label=name, marker=markers.get(name, "o"))
            ax.set_title("연도별 총합(실적 라인 + 예측 포인트)")
            ax.set_xlabel("연도"); ax.set_ylabel("총합")
            ax.legend(loc="best"); ax.grid(alpha=0.25)
            st.pyplot(fig)
        else:
            fig = go.Figure()
            yd = yearly_all[["연", prod]].dropna().sort_values("연")
            fig.add_trace(go.Scatter(x=yd["연"], y=yd[prod], mode="lines", name="실적", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=yd["연"], y=yd[prod], mode="lines", name="영역",
                                     fill="tozeroy", line=dict(width=0.1), showlegend=False, hoverinfo="skip"))
            sym = {"CAGR(복리성장)": "circle", "Holt(지수평활)": "square", "OLS(선형추세)": "triangle-up",
                   "지수평활(SES)": "cross", "ARIMA": "x", "SARIMA(12)": "diamond"}
            for name in methods_selected:
                if name in pred_map and toggles_top.get(name, True):
                    xs = years_pred; ys = [pred_map[name].get(y, np.nan) for y in xs]
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="markers+text", name=name,
                        marker_symbol=sym.get(name, "circle"),
                        text=[f"{int(v):,}" if v == v else "" for v in ys],
                        textposition="top center"
                    ))
            fig.update_layout(
                title="연도별 총합(실적 라인 + 예측 포인트)",
                xaxis_title="연도", yaxis_title="총합",
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
                margin=dict(t=60, b=120, l=40, r=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        if go is not None:
            with st.expander(f"🔀 {prod} 방법별 표시 토글(동적)"):
                toggles = {}
                cols = st.columns(min(6, len(methods_selected))) or [st]
                for i, name in enumerate(methods_selected):
                    with cols[i % len(cols)]:
                        toggles[name] = st.toggle(name, value=True, key=f"tg_{prod}_{name}")
                fig2 = go.Figure()
                yd = yearly_all[["연", prod]].dropna().sort_values("연")
                fig2.add_trace(go.Scatter(x=yd["연"], y=yd[prod], mode="lines+markers", name="실적"))
                for name in methods_selected:
                    if not toggles.get(name, True):
                        continue
                    if name in pred_map:
                        xs = years_pred
                        ys = [pred_map[name].get(y, np.nan) for y in xs]
                        fig2.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", name=name,
                                                  text=[f"{int(v):,}" if v == v else "" for v in ys],
                                                  textposition="top center"))
                fig2.update_layout(title="방법별 동적 표시", xaxis_title="연도", yaxis_title="총합",
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0),
                                   margin=dict(t=60, b=110, l=40, r=20))
                st.plotly_chart(fig2, use_container_width=True)

    with st.expander("예측 방법 설명 (쉬운 설명 + 산식)"):
        st.markdown(r"""
- **선형추세(OLS)** — 해마다 늘어나는 폭을 직선으로 잡아 앞으로 그린다.  
  \( y_t = a + b t,\ \ \hat y_{t+h} = a + b (t+h) \)

- **CAGR(복리성장)** — 시작~끝 사이의 평균 복리 성장률만큼 매년 같은 비율로 증가.  
  \( g = (y_T / y_0)^{1/n} - 1,\ \ \hat y_{t+h} = y_T (1+g)^h \)

- **Holt(지수평활)** — 수준/추세를 지수 가중으로 갱신.  
  \( l_t = \alpha y_t + (1-\alpha)(l_{t-1}+b_{t-1}),\ b_t=\beta(l_t-l_{t-1})+(1-\beta)b_{t-1} \)

- **지수평활(SES)** — 최근 관측치 가중 평균화(추세/계절X).  
  \( l_t = \alpha y_t + (1-\alpha)l_{t-1},\ \hat y_{t+h}=l_T \)

- **ARIMA(p,d,q)** — 차분으로 정상화 후 AR/MA 결합. 여기선 후보 \((1,1,0),(0,1,1),(1,1,1)\) 중 AIC 최소.  

- **SARIMA(P,D,Q,12)** — 월별 계절주기 12 확장. 기본 \((1,1,1)\times(1,1,1)_{12}\).
""")

# ===========================================================
# 라우터
# ===========================================================
def main():
    title_with_icon("📊", "도시가스 공급량·판매량 예측")
    st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

    # 좌측 최상단: 추천 패널
    sidebar_recommendation_panel()

    # 예측유형
    with st.sidebar:
        title_with_icon("🧭", "예측 유형", "h3", small=True)
        mode = st.radio("🔀 선택",
                        ["공급량 예측", "판매량 예측(냉방용)", "공급량 추세분석 예측"],
                        index=0, label_visibility="visible")

    # 본문 상단: 추천 결과가 있으면 먼저 렌더
    body_recommendation_section()

    # 본문: 각 기능
    if mode == "공급량 예측":
        render_supply_forecast()
    elif mode == "판매량 예측(냉방용)":
        render_cooling_sales_forecast()
    else:
        render_trend_forecast()

if __name__ == "__main__":
    main()
