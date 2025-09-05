# app.py — 도시가스 공급·판매 분석 (Poly-3)
# ─────────────────────────────────────────────────────────────────────────────
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

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# 폰트 설정 (레포의 data/fonts 우선)
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    fam = None
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                break
        except Exception:
            pass
    if fam:
        plt.rcParams["font.family"] = [fam]
        plt.rcParams["font.sans-serif"] = [fam]
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸

KNOWN_PRODUCT_ORDER = [
    "개별난방용", "중앙난방용", "자가열전용", "일반용(2)", "업무난방용", "냉난방용", "주한미군", "총공급량"
]
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜/연월 생성
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    # 연/월 채우기
    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
    if "월" not in df.columns:
        if "날짜" in df.columns:
            df["월"] = df["날짜"].dt.month
    # 숫자형 변환
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
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): 
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in {"연","년","월","날짜"}]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others  = [c for c in candidates if c not in ordered]
    return ordered + others

def month_range_inclusive(s: pd.Timestamp, e: pd.Timestamp):
    s = pd.Timestamp(year=s.year, month=s.month, day=1)
    e = pd.Timestamp(year=e.year, month=e.month, day=1)
    return pd.date_range(start=s, end=e, freq="MS")

def poly3_fit(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x).reshape(-1,1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr  = poly.fit_transform(x)
    model = LinearRegression().fit(Xtr, y)
    r2 = model.score(Xtr, y)
    return model, poly, r2

def poly3_predict(model, poly, X1d):
    X = np.asarray(X1d).reshape(-1,1)
    return model.predict(poly.transform(X))

def format_yyyy_mm(y: int, m: int) -> str:
    return f"{int(y):04d}.{int(m):02d}"

def format_table_html(df: pd.DataFrame, temp_cols: list[str]) -> str:
    """기온(소수1) + 숫자(천단위 콤마) + 가운데 정렬 HTML 반환"""
    show = df.copy()
    for c in show.columns:
        if c in temp_cols:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1).map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )
        elif pd.api.types.is_numeric_dtype(show[c]):
            # 소수 없음, 반올림 후 천단위 콤마
            show[c] = pd.to_numeric(show[c], errors="coerce").round().astype("Int64").map(
                lambda v: "" if pd.isna(v) else format(int(v), ",")
            )
    style = """
    <style>
    table.centered-table {width: 100%; border-collapse: collapse;}
    table.centered-table th, table.centered-table td {
        text-align: center !important; vertical-align: middle !important; padding: 6px;
    }
    </style>
    """
    html = show.to_html(index=False, classes="centered-table", border=1)
    return style + html

def render_html_table(df: pd.DataFrame, temp_cols: list[str]):
    st.markdown(format_table_html(df, temp_cols=temp_cols), unsafe_allow_html=True)

def coef_text_3rd(model, poly) -> str:
    # poly: features [x, x^2, x^3] (include_bias=False)
    coef = model.coef_.ravel().tolist()  # [a1, a2, a3]
    c0 = float(model.intercept_)
    a1, a2, a3 = coef[0], coef[1], coef[2]
    # y = c3 x^3 + c2 x^2 + c1 x + c0
    return f"y = {a3:+.5e}x³ {a2:+.5e}x² {a1:+.5e}x {c0:+.5e}"

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 (Repo 기본 / 업로드 옵션)
def load_repo_file(patterns: list[str]):
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    for pat in patterns:
        for p in sorted(glob(str(here / "data" / pat))):
            return Path(p)
    return None

@st.cache_data(ttl=600)
def load_supply_df(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "name"):
        xls = pd.ExcelFile(file_or_path, engine="openpyxl")
        sheet = "데이터" if "데이터" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        xls = pd.ExcelFile(file_or_path, engine="openpyxl")
        sheet = "데이터" if "데이터" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    return normalize_cols(df)

@st.cache_data(ttl=600)
def load_sales_df(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "name"):
        xls = pd.ExcelFile(file_or_path, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        xls = pd.ExcelFile(file_or_path, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    return normalize_cols(df)

@st.cache_data(ttl=600)
def load_temp_raw(file_or_path) -> pd.DataFrame:
    # CSV/엑셀 자동 처리, '일자'/'날짜' + '기온'/평균기온 등 자동 추출
    if hasattr(file_or_path, "name"):
        name = file_or_path.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file_or_path)
        else:
            xls = pd.ExcelFile(file_or_path, engine="openpyxl")
            sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
    else:
        p = Path(file_or_path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            xls = pd.ExcelFile(p, engine="openpyxl")
            sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
    df = normalize_cols(df)
    # 날짜/기온 컬럼 결정
    date_col = None
    for c in df.columns:
        if str(c).lower() in ["날짜","일자","date"]:
            date_col = c; break
    if date_col is None:
        for c in df.columns:
            try:
                _ = pd.to_datetime(df[c], errors="raise")
                date_col = c; break
            except Exception:
                pass
    temp_col = detect_temp_col(df)
    if date_col is None or temp_col is None:
        return pd.DataFrame(columns=["일자","기온"])
    out = pd.DataFrame({"일자": pd.to_datetime(df[date_col], errors="coerce"), "기온": pd.to_numeric(df[temp_col], errors="coerce")}).dropna()
    return out.sort_values("일자").reset_index(drop=True)

def make_year_month_selectors(title: str, years: list[int]):
    st.subheader(title)
    c1, c2 = st.columns(2)
    with c1:
        y = st.selectbox("연", years, index=len(years)-1, key=title+"_Y")
    with c2:
        m = st.selectbox("월", list(range(1,13)), index=0, key=title+"_M")
    return int(y), int(m)

# ─────────────────────────────────────────────────────────────────────────────
# ΔT 버튼(±0.5) — 즉시 갱신
def temp_delta_box(label: str, key_prefix: str):
    st.markdown(f"### {label}")
    col1, col2, col3 = st.columns([1,1,6])
    # 초기화
    if key_prefix not in st.session_state:
        st.session_state[key_prefix] = 0.0
    def dec():
        st.session_state[key_prefix] = float(np.clip(st.session_state[key_prefix] - 0.5, -5.0, 5.0))
    def inc():
        st.session_state[key_prefix] = float(np.clip(st.session_state[key_prefix] + 0.5, -5.0, 5.0))
    with col1:
        st.button("−", key=key_prefix+"_dec", on_click=dec)
    with col2:
        st.button("+", key=key_prefix+"_inc", on_click=inc)
    with col3:
        st.caption("기온 보정(°C)")
        v = st.session_state[key_prefix]
        st.markdown(f"<h2 style='margin-top:-12px;'>{v:+.2f}</h2>", unsafe_allow_html=True)
    return st.session_state[key_prefix]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 페이지 기본
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16–당월15) 평균기온 기반")

with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# ─────────────────────────────────────────────────────────────────────────────
# A) 공급량 분석
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="supply_src")

        if src == "Repo 내 파일 사용":
            repo = load_repo_file(["상품별공급량_MJ.xlsx", "*공급*.xlsx"])
            if repo is None:
                st.error("data 폴더에 공급량 엑셀 파일이 없습니다.")
                st.stop()
            df = load_supply_df(repo)
        else:
            up = st.file_uploader("공급량 엑셀(xlsx) — '데이터' 시트", type=["xlsx"], key="supply_up")
            if up is None:
                st.stop()
            df = load_supply_df(up)

        # 학습연도 선택
        st.header("학습 데이터 연도 선택")
        years_all = sorted(pd.Series(df["연"]).dropna().astype(int).unique().tolist())
        years_sel = st.multiselect("연도 선택", years_all, default=years_all[-5:], key="supply_years_sel")

        # 예측 기간 (가로형)
        st.header("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            sY = st.selectbox("예측 시작(연)", years_all, index=len(years_all)-1)
            sM = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        with c2:
            eY = st.selectbox("예측 종료(연)", years_all, index=len(years_all)-1, key="supply_endY")
            eM = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="supply_endM")

    # 훈련 데이터 준비
    temp_col = detect_temp_col(df)
    if temp_col is None:
        st.error("공급량 데이터에서 기온 열(평균기온/기온 등)을 찾지 못했습니다.")
        st.stop()

    df_base = df.dropna(subset=["연","월"]).copy()
    df_base["연"] = df_base["연"].astype(int)
    df_base["월"] = df_base["월"].astype(int)
    train_df = df_base[df_base["연"].isin(years_sel)].copy()

    product_cols = [c for c in guess_product_cols(df_base) if c in df_base.columns]
    # 모델 학습(상품별)
    models = {}
    for col in product_cols:
        x = train_df[temp_col].astype(float).values
        y = train_df[col].astype(float).values
        m, p, r2 = poly3_fit(x, y)
        models[col] = {"model": m, "poly": p, "r2": r2}

    # 예측 인덱스
    f_start = pd.Timestamp(year=int(sY), month=int(sM), day=1)
    f_end   = pd.Timestamp(year=int(eY), month=int(eM), day=1)
    if f_end < f_start:
        st.error("예측 종료가 시작보다 빠릅니다.")
        st.stop()
    fut_idx = month_range_inclusive(f_start, f_end)
    fut = pd.DataFrame({"연": fut_idx.year, "월": fut_idx.month})
    # 학습기간 월평균 기온(기본 시나리오)
    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("base_temp")

    def make_supply_table(deltaT: float) -> pd.DataFrame:
        tb = fut.merge(monthly_avg_temp.reset_index(), on="월", how="left").copy()
        tb["월평균기온(적용)"] = tb["base_temp"] + float(deltaT)
        for col in product_cols:
            y_future = poly3_predict(models[col]["model"], models[col]["poly"], tb["월평균기온(적용)"].values)
            tb[col] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
        tb["연월"] = [format_yyyy_mm(y,m) for y,m in zip(tb["연"], tb["월"])]
        # 표시 순서: 연월, 월평균기온, 상품들
        show_cols = ["연월", "월평균기온(적용)"] + product_cols
        tb = tb[show_cols]
        return tb

    st.markdown("## ΔT 시나리오 (°C)")
    c1, c2, c3 = st.columns(3)
    with c1:
        dT_normal = temp_delta_box("ΔT(Normal)", "dT_normal_supply")
    with c2:
        dT_best   = temp_delta_box("ΔT(Best)", "dT_best_supply")
    with c3:
        dT_cons   = temp_delta_box("ΔT(Conservative)", "dT_cons_supply")

    # 표 3개 — 즉시 갱신
    normal_tbl = make_supply_table(dT_normal)
    best_tbl   = make_supply_table(dT_best)
    cons_tbl   = make_supply_table(dT_cons)

    st.markdown("## 예측 결과 — Normal")
    render_html_table(normal_tbl, temp_cols=["월평균기온(적용)"])
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_normal.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Best")
    render_html_table(best_tbl, temp_cols=["월평균기온(적용)"])
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_best.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Conservative")
    render_html_table(cons_tbl, temp_cols=["월평균기온(적용)"])
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_conservative.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# B) 판매량 분석(냉방용)
else:
    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="sales_src")

        if src == "Repo 내 파일 사용":
            repo_sales = load_repo_file(["상품별판매량.xlsx","*판매*.xlsx"])
            repo_temp  = load_repo_file(["기온.xlsx","*기온*.xlsx","*temp*.csv","기온.csv"])
            if repo_sales is None or repo_temp is None:
                st.error("data 폴더에 '상품별판매량.xlsx' 또는 '기온' 파일이 없습니다.")
                st.stop()
            sales_df = load_sales_df(repo_sales)
            temp_raw = load_temp_raw(repo_temp)
        else:
            up1 = st.file_uploader("판매 실적(xlsx)", type=["xlsx"], key="sales_up")
            up2 = st.file_uploader("기온 RAW(일별) (xlsx/csv)", type=["xlsx","csv"], key="temp_up")
            if up1 is None or up2 is None:
                st.stop()
            sales_df = load_sales_df(up1)
            temp_raw = load_temp_raw(up2)

        # 학습연도 선택
        st.header("학습 데이터 연도 선택")
        sales_df["판매월"] = pd.to_datetime(sales_df.get("판매월", sales_df.get("날짜", sales_df.get("일자", sales_df.get("date")))), errors="coerce")
        sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
        years_all = sorted(sales_df["연"].dropna().astype(int).unique().tolist())
        years_sel = st.multiselect("연도 선택", years_all, default=years_all[-5:], key="sales_years_sel")

        # 예측 기간 (가로형)
        st.header("예측 기간")
        c1, c2 = st.columns(2)
        with c1:
            sY = st.selectbox("예측 시작(연)", years_all, index=len(years_all)-1, key="sales_startY")
            sM = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="sales_startM")
        with c2:
            eY = st.selectbox("예측 종료(연)", years_all, index=len(years_all)-1, key="sales_endY")
            eM = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="sales_endM")

    # 판매량 실적 열 추정
    value_col = None
    cand = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    if cand:
        if "냉방용" in cand:
            value_col = "냉방용"
        else:
            value_col = cand[0]
    else:
        # 첫 번째 숫자열
        for c in sales_df.columns:
            if pd.api.types.is_numeric_dtype(sales_df[c]):
                value_col = c; break
    if value_col is None:
        st.error("판매 실적에서 '냉방' 수치 열을 찾지 못했습니다.")
        st.stop()

    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    # 기간평균기온 (전월16 ~ 당월15)
    temp_raw["연"] = temp_raw["일자"].dt.year
    temp_raw["월"] = temp_raw["일자"].dt.month
    def m_range_avg(label_m: pd.Timestamp) -> float:
        m = pd.Timestamp(year=label_m.year, month=label_m.month, day=1)
        s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
        e = m + pd.DateOffset(days=14)
        mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
        return temp_raw.loc[mask,"기온"].mean()

    # 학습 데이터(연도 필터)
    train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
    rows = [{"판매월": m, "기간평균기온": m_range_avg(m)} for m in train_sales["판매월"].unique()]
    sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left").dropna()
    x_train = sj["기간평균기온"].astype(float).values
    y_train = sj["판매량"].astype(float).values
    model, poly, r2 = poly3_fit(x_train, y_train)

    # ΔT 시나리오
    st.markdown("## ΔT 시나리오 (°C)")
    c1, c2, c3 = st.columns(3)
    with c1:
        dT_normal = temp_delta_box("ΔT(Normal)", "dT_normal_sales")
    with c2:
        dT_best   = temp_delta_box("ΔT(Best)", "dT_best_sales")
    with c3:
        dT_cons   = temp_delta_box("ΔT(Conservative)", "dT_cons_sales")

    # 예측 인덱스
    f_start = pd.Timestamp(year=int(sY), month=int(sM), day=1)
    f_end   = pd.Timestamp(year=int(eY), month=int(eM), day=1)
    if f_end < f_start:
        st.error("예측 종료가 시작보다 빠릅니다.")
        st.stop()
    fut_idx = month_range_inclusive(f_start, f_end)

    def make_sales_table(deltaT: float) -> pd.DataFrame:
        rows = []
        for m in fut_idx:
            # 당월평균도 참고로 제공
            mon_mask = (temp_raw["연"]==m.year) & (temp_raw["월"]==m.month)
            mon_avg  = temp_raw.loc[mon_mask, "기온"].mean()
            per_avg  = m_range_avg(m)
            rows.append({
                "연": int(m.year), "월": int(m.month),
                "당월평균기온": mon_avg,
                "기간평균기온(적용)": per_avg + float(deltaT)
            })
        tb = pd.DataFrame(rows)
        y_future = poly3_predict(model, poly, tb["기간평균기온(적용)"].values)
        tb["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
        # 실적/오차(있는 경우)
        act = sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
        tb  = tb.merge(act, on=["연","월"], how="left")
        tb["오차"] = (tb["예측판매량"] - tb["실제판매량"]).astype("Int64")
        tb["오차율(%)"] = ((tb["오차"] / tb["실제판매량"])*100).round(1)
        tb.loc[tb["실제판매량"].isna(),"오차율(%)"] = np.nan
        tb["연월"] = [format_yyyy_mm(y,m) for y,m in zip(tb["연"],tb["월"])]
        cols = ["연월","당월평균기온","기간평균기온(적용)","예측판매량","실제판매량","오차","오차율(%)"]
        return tb[cols]

    # 표 3개 — 즉시 갱신
    n_tbl = make_sales_table(dT_normal)
    b_tbl = make_sales_table(dT_best)
    c_tbl = make_sales_table(dT_cons)

    st.markdown("## 예측 결과 — Normal")
    render_html_table(n_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"])
    st.download_button("Normal CSV", data=n_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_normal.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Best")
    render_html_table(b_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"])
    st.download_button("Best CSV", data=b_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_best.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Conservative")
    render_html_table(c_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"])
    st.download_button("Conservative CSV", data=c_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_conservative.csv", mime="text/csv")

    # ───────── 그래프: 기온-냉방용 실적 상관관계(Train) ─────────
    st.markdown("## 기온-냉방용 실적 상관관계 (Train, R²={:.3f})".format(r2))
    fig = plt.figure(figsize=(10,6), dpi=140)
    ax = plt.gca()

    # 산점도(학습 샘플)
    ax.scatter(sj["기간평균기온"], sj["판매량"], s=28, alpha=0.7, label="학습 샘플")

    # 예측선
    xs = np.linspace(max(0, sj["기간평균기온"].min()-1), sj["기간평균기온"].max()+1, 120)
    yhat = poly3_predict(model, poly, xs)
    ax.plot(xs, yhat, linewidth=3, label="Poly-3")

    # 95% 밴드 (±1.96*RMSE)
    resid = sj["판매량"].values - poly3_predict(model, poly, sj["기간평균기온"].values)
    sigma = np.sqrt(np.mean(resid**2))
    ax.fill_between(xs, yhat-1.96*sigma, yhat+1.96*sigma, alpha=0.15, label="±1.96")

    # 온도별 중앙값(보조)
    med = sj.copy()
    med["bin"] = med["기간평균기온"].round(0)
    med = med.groupby("bin")["판매량"].median().reset_index()
    ax.scatter(med["bin"], med["판매량"], color="tab:orange", s=38, label="온도별 중앙값", zorder=5)

    # 수식(5자리)
    txt = coef_text_3rd(model, poly)
    ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

    ax.set_xlabel("기간평균기온 (°C)")
    ax.set_ylabel("판매량 (MJ)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
