# -*- coding: utf-8 -*-
# 도시가스 공급·판매 분석 (Poly-3)
# 요청 반영:
# - 표 첫열 '연월' + 중앙정렬 + 기온(소수1), 나머지 콤마
# - ΔT Normal/Best/Conservative ±0.5 버튼 (즉시 반영)
# - 사이드바 예측기간(연/월) 가로 배치
# - 판매량 상관관계 그래프(산점+Poly3+95%대역+중앙값+식 표시)
# - CSV 다운로드 유지

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
# 기본 환경
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

# Matplotlib cache dir(권한 문제 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─────────────────────────────────────────────────────────────
# 한글 폰트
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
# 유틸
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)",
    "업무난방용","냉난방용","주한미군","총공급량"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜/연/월 보정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
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
    """일별 RAW: 열(날짜/일자/date, 기온/평균기온/temp 등) 자동 인식"""
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
        out = pd.DataFrame({
            "일자": pd.to_datetime(df[date_col], errors="coerce"),
            "기온": pd.to_numeric(df[temp_col], errors="coerce")
        }).dropna()
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

def poly3_fit(x, y):
    """Poly-3 모델 학습"""
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m].reshape(-1,1)
    y = y[m]
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X = poly.fit_transform(x)
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return model, poly, r2

def poly3_predict(model, poly, x_vec):
    Xf = poly.transform(np.asarray(x_vec).reshape(-1,1))
    return model.predict(Xf)

def ym_col(df):
    """연·월 -> 연월(yyyy.mm) 생성 + 첫열로 배치"""
    out = df.copy()
    out["연"] = out["연"].astype(int)
    out["월"] = out["월"].astype(int)
    out.insert(0, "연월", out["연"].astype(str) + "." + out["월"].map(lambda m:f"{m:02d}"))
    out = out.drop(columns=["연","월"])
    return out

def center_table_html(df, temp_cols=None, thousand_cols=None):
    """표 중앙정렬 + 포맷 적용 → HTML"""
    df2 = df.copy()
    temp_cols = temp_cols or []
    thousand_cols = thousand_cols or [c for c in df2.columns if c not in temp_cols and c != "연월"]
    # 포맷
    for c in temp_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in thousand_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").round().astype("Int64").map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    # 중앙 CSS
    css = """
    <style>
      table.centered td, table.centered th {text-align:center !important; vertical-align:middle !important;}
    </style>
    """
    html = df2.to_html(index=False, classes="centered")
    return css + html

# ─────────────────────────────────────────────────────────────
# 사이드바 — 공통 UI
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# ─────────────────────────────────────────────────────────────
# ========== A) 공급량 분석 ==========
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i,p in enumerate(repo_files) if "공급" in Path(p).name), 0)
                file_choice = st.selectbox("실적 파일(Excel)", repo_files, index=default_idx)
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("data 폴더에 엑셀이 없습니다. 업로드를 사용하세요.")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None: df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df)==0:
            st.stop()

        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        st.subheader("학습 데이터 연도 선택")
        years_sel = st.multiselect("연도 선택", years_all, default=years_all)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. '평균기온' 또는 '기온' 포함 필요.")
            st.stop()

        st.subheader("예측할 상품 선택")
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

        # 예측 기간 (가로 배치)
        st.subheader("예측 기간")
        colA, colB = st.columns(2)
        with colA:
            sy = st.selectbox("예측 시작(연)", years_all, index=years_all.index(max(years_all)))
        with colB:
            sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        colC, colD = st.columns(2)
        with colC:
            ey = st.selectbox("예측 종료(연)", years_all, index=years_all.index(max(years_all)))
        with colD:
            em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        # 시나리오 기본
        st.subheader("시나리오 기본값")
        scen = st.radio("기본 온도 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사"], index=0)

        run_btn = st.button("예측 시작", type="primary")

    # 최초 실행 시 계산(학습)
    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()
        f_start = pd.Timestamp(year=int(sy), month=int(sm), day=1)
        f_end   = pd.Timestamp(year=int(ey), month=int(em), day=1)
        if f_end < f_start:
            st.error("예측 종료가 시작보다 빠릅니다.")
            st.stop()

        fut_idx = pd.date_range(start=pd.Timestamp(f_start.year, f_start.month, 1),
                                end=pd.Timestamp(f_end.year, f_end.month, 1), freq="MS")
        fut = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        if scen == "학습기간 월별 평균":
            fut = fut.merge(monthly_avg_temp.reset_index(), on="월", how="left")
        else:
            last_train_year = int(train_df["연"].max()) if len(train_df) else int(base["연"].max())
            base_temp = base[base["연"]==last_train_year][["월",temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
            fut = fut.merge(base_temp, on="월", how="left")

        # 학습 입력/모델
        x_train = train_df[temp_col].astype(float).values
        models = {}
        r2s    = {}
        for col in prods:
            y_train = train_df[col].astype(float).values
            m, poly, r2 = poly3_fit(x_train, y_train)
            models[col] = (m, poly)
            r2s[col] = r2

        # 컨텍스트 저장
        st.session_state["supply_ctx"] = {
            "base": base, "train": train_df, "models": models, "r2s": r2s,
            "fut_base": fut, "years_all": years_all, "prods": prods,
            "f_start": f_start
        }
        # ΔT 초기화
        for k in ["dT_norm","dT_best","dT_cons"]:
            st.session_state[k] = st.session_state.get(k, 0.0)

    # 재계산(ΔT 즉시 반영)과 표/그래프 출력
    if "supply_ctx" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    ctx = st.session_state["supply_ctx"]

    # ΔT 라인(세 시나리오) — 즉시 반영
    st.subheader("ΔT 시나리오 (℃)")
    col1, col2, col3 = st.columns(3)

    def dt_row(col, key, title):
        with col:
            st.markdown(f"### {title}")
            left, mid, right = st.columns([1,2,1])
            with left:
                if st.button("−", key=f"minus_{key}"):
                    st.session_state[key] = round(st.session_state.get(key,0.0) - 0.5, 2)
            with mid:
                st.metric("기온 보정", f"{st.session_state.get(key,0.0):.2f}")
            with right:
                if st.button("+", key=f"plus_{key}"):
                    st.session_state[key] = round(st.session_state.get(key,0.0) + 0.5, 2)

    dt_row(col1, "dT_norm", "ΔT(Normal)")
    dt_row(col2, "dT_best", "ΔT(Best)")
    dt_row(col3, "dT_cons", "ΔT(Conservative)")

    def make_supply_table(dT):
        fut = ctx["fut_base"].copy()
        fut["temp"] = fut["temp"] + float(dT)

        rows = []
        for col in ctx["prods"]:
            m, poly = ctx["models"][col]
            y = np.clip(np.rint(poly3_predict(m, poly, fut["temp"].values)), 0, None).astype("int64")
            rows.append(pd.DataFrame({"연":fut["연"], "월":fut["월"], col:y}))
        tbl = rows[0][["연","월"]].copy()
        for r in rows:
            tbl = tbl.merge(r, on=["연","월"], how="left")
        # 월평균기온(적용) 붙이기
        tbl = tbl.merge(fut[["연","월","temp"]].rename(columns={"temp":"월평균기온(적용)"}), on=["연","월"], how="left")
        # 연월 → 포맷
        tbl = ym_col(tbl)
        # 열 순서: 연월, 월평균기온(적용), prods...
        cols = ["연월","월평균기온(적용)"] + ctx["prods"]
        return tbl[cols]

    normal_tbl = make_supply_table(st.session_state.get("dT_norm",0.0))
    best_tbl   = make_supply_table(st.session_state.get("dT_best",0.0))
    cons_tbl   = make_supply_table(st.session_state.get("dT_cons",0.0))

    # 세 표 상단 노출
    st.markdown("## 예측 결과 — Normal")
    st.markdown(center_table_html(normal_tbl, temp_cols=["월평균기온(적용)"]))
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_normal.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Best")
    st.markdown(center_table_html(best_tbl, temp_cols=["월평균기온(적용)"]))
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_best.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Conservative")
    st.markdown(center_table_html(cons_tbl, temp_cols=["월평균기온(적용)"]))
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_conservative.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# ========== B) 판매량 분석(냉방용) ==========
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")

    # 파일
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
            st.success(f"레포 파일 사용: {Path(repo_sales_path).name} · {Path(repo_temp_path).name}")
            sales_file = open(repo_sales_path, "rb")
            temp_raw_file = open(repo_temp_path, "rb")
    else:
        with c1: sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 준비하세요.")
        st.stop()

    # 실적
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
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다.")
        st.stop()

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다.")
        st.stop()

    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("연도 선택", options=years_all, default=years_all)

        st.subheader("예측 기간")
        colA, colB = st.columns(2)
        with colA:
            sy = st.selectbox("예측 시작(연)", years_all, index=years_all.index(max(years_all)))
        with colB:
            sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        colC, colD = st.columns(2)
        with colC:
            ey = st.selectbox("예측 종료(연)", years_all, index=years_all.index(max(years_all)))
        with colD:
            em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("예측 시작", type="primary")

    # 최초 학습
    if run_btn:
        # 보조 집계
        temp_raw["연"] = temp_raw["일자"].dt.year
        temp_raw["월"] = temp_raw["일자"].dt.month
        monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
        fallback_by_M = temp_raw.groupby("월")["기온"].mean()

        # 기간평균(전월16~당월15)
        def period_avg(label_m: pd.Timestamp) -> float:
            m = pd.Timestamp(label_m.year, label_m.month, 1)
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
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
        model, poly, r2_fit = poly3_fit(x_train, y_train)

        # 예측 입력 기본
        f_start = pd.Timestamp(year=int(sy), month=int(sm), day=1)
        f_end   = pd.Timestamp(year=int(ey), month=int(em), day=1)
        if f_end < f_start:
            st.error("예측 종료가 시작보다 빠릅니다.")
            st.stop()
        months = pd.date_range(start=f_start, end=f_end, freq="MS")

        # 기본 표준 컨텍스트
        st.session_state["sales_ctx"] = {
            "sales_df": sales_df,
            "temp_raw": temp_raw,
            "monthly_cal": monthly_cal,
            "fallback_by_M": fallback_by_M,
            "model": model, "poly": poly, "r2": r2_fit,
            "f_months": months, "years_all": years_all
        }
        for k in ["dT_norm","dT_best","dT_cons"]:
            st.session_state[k] = st.session_state.get(k, 0.0)

    if "sales_ctx" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    sctx = st.session_state["sales_ctx"]

    # ΔT 상단 버튼(즉시 적용)
    st.subheader("ΔT 시나리오 (℃)")
    col1, col2, col3 = st.columns(3)
    def dt_row(col, key, title):
        with col:
            st.markdown(f"### {title}")
            left, mid, right = st.columns([1,2,1])
            with left:
                if st.button("−", key=f"minus_{key}"):
                    st.session_state[key] = round(st.session_state.get(key,0.0) - 0.5, 2)
            with mid:
                st.metric("기온 보정", f"{st.session_state.get(key,0.0):.2f}")
            with right:
                if st.button("+", key=f"plus_{key}"):
                    st.session_state[key] = round(st.session_state.get(key,0.0) + 0.5, 2)
    dt_row(col1, "dT_norm", "ΔT(Normal)")
    dt_row(col2, "dT_best", "ΔT(Best)")
    dt_row(col3, "dT_cons", "ΔT(Conservative)")

    # 예측 테이블 생성(연월 첫열)
    def make_sales_table(dT):
        months = sctx["f_months"]
        mdf = []
        for m in months:
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (sctx["temp_raw"]["일자"]>=s)&(sctx["temp_raw"]["일자"]<=e)
            avg_period = sctx["temp_raw"].loc[mask,"기온"].mean()
            avg_month  = sctx["monthly_cal"].loc[
                (sctx["monthly_cal"]["연"]==m.year)&(sctx["monthly_cal"]["월"]==m.month),
                "기온"
            ].mean()
            mdf.append([m.year, m.month, avg_month, avg_period])
        pred = pd.DataFrame(mdf, columns=["연","월","당월평균기온","기간평균기온"])
        for c in ["당월평균기온","기간평균기온"]:
            miss = pred[c].isna()
            if miss.any(): pred.loc[miss,c] = pred.loc[miss,"월"].map(sctx["fallback_by_M"])
        pred["기간평균기온(적용)"] = pred["기간평균기온"] + float(dT)
        y = np.clip(np.rint(poly3_predict(sctx["model"], sctx["poly"], pred["기간평균기온(적용)"].values)), 0, None).astype("int64")
        pred["예측판매량"] = y
        pred = ym_col(pred[["연","월","당월평균기온","기간평균기온(적용)","예측판매량"]])
        # 포맷 표
        return pred

    normal_tbl = make_sales_table(st.session_state.get("dT_norm",0.0))
    best_tbl   = make_sales_table(st.session_state.get("dT_best",0.0))
    cons_tbl   = make_sales_table(st.session_state.get("dT_cons",0.0))

    # 세 표(상단)
    st.markdown("## 예측 결과 — Normal")
    st.markdown(center_table_html(normal_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"]))
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_normal.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Best")
    st.markdown(center_table_html(best_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"]))
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_best.csv", mime="text/csv")

    st.markdown("## 예측 결과 — Conservative")
    st.markdown(center_table_html(cons_tbl, temp_cols=["당월평균기온","기간평균기온(적용)"]))
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_conservative.csv", mime="text/csv")

    # ───── 상관관계 그래프(요청 형태)
    st.markdown("## 기온-냉방용 실적 상관관계 (Train)")
    # 학습 표 다시 만들기
    temp_raw = sctx["temp_raw"]; monthly_cal = sctx["monthly_cal"]; fbm = sctx["fallback_by_M"]
    def period_avg(label_m: pd.Timestamp) -> float:
        m = pd.Timestamp(label_m.year, label_m.month, 1)
        s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
        e = m + pd.DateOffset(days=14)
        mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
        return temp_raw.loc[mask,"기온"].mean()
    train_sales = sctx["sales_df"][sctx["sales_df"]["연"].isin(years_sel)].copy()
    rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
    sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
    miss = sj["기간평균기온"].isna()
    if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"판매월"].dt.month.map(fbm)
    sj = sj.dropna(subset=["기간평균기온","판매량"])

    # Poly3 곡선 / 식 / 95% 대역
    x = sj["기간평균기온"].values
    y = sj["판매량"].values
    model, poly, r2 = sctx["model"], sctx["poly"], sctx["r2"]
    x_line = np.linspace(max(0, x.min()-1), x.max()+1, 400)
    y_line = poly3_predict(model, poly, x_line)
    # 95% 대역(잔차 표준편차 사용)
    y_hat = poly3_predict(model, poly, x)
    resid = y - y_hat
    s = np.std(resid)
    upper = y_line + 1.96*s
    lower = y_line - 1.96*s
    # 온도별 중앙값(1℃ bin)
    bins = np.floor(x).astype(int)
    med_points = pd.DataFrame({"bin":bins,"y":y}).groupby("bin")["y"].median().reset_index()
    # 식 표기(소수 5자리)
    coef = np.r_[model.coef_[2::-1], model.intercept_] if len(model.coef_)==3 else None
    # 하지만 PolynomialFeatures(include_bias=False): coef 순서 [x, x^2, x^3]
    a3 = model.coef_[2] if len(model.coef_)>=3 else 0.0
    a2 = model.coef_[1] if len(model.coef_)>=2 else 0.0
    a1 = model.coef_[0] if len(model.coef_)>=1 else 0.0
    a0 = model.intercept_
    eq_txt = f"y = {a3:+.5f}x³ {a2:+.5f}x² {a1:+.5f}x {a0:+.5f}"

    fig = plt.figure(figsize=(9.5,5.6))
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.45, label="학습 샘플")
    ax.plot(x_line, y_line, linewidth=3, label="Poly-3")
    ax.fill_between(x_line, lower, upper, alpha=0.15, label="±1.96")
    ax.scatter(med_points["bin"], med_points["y"], color="orange", s=40, label="온도별 중앙값")
    ax.set_xlabel("기간평균기온 (℃)")
    ax.set_ylabel("판매량 (MJ)")
    ax.set_title(f"기온-냉방용 실적 상관관계 (Train, R²={r2:.3f})")
    ax.legend()
    # 식 텍스트
    ax.text(0.99, 0.03, eq_txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9), fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
