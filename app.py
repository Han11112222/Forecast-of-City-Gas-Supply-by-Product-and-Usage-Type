# app.py — 도시가스 공급·판매 분석 (Poly-3)
# - 공급량: 기온↔공급량 3차 다항식(상품별) / 예측 그래프 및 표(3시나리오) / CSV
# - 판매량(냉방용): (전월16~당월15) 기간평균기온 기반 3차 다항식 / 검증표·그래프 / 3시나리오 표 / CSV
# - 좌측: 데이터/기간/시나리오 기본값 설정 + [예측 시작] 버튼(학습·모델 계산)
# - 우측: ΔT 슬라이더(각 시나리오) => 즉시 표/그래프 반영
# - 숫자 포맷: 기온 소수1자리, 나머지 천단위 콤마
# - 한글 폰트: data/fonts/NanumGothic-Regular.ttf 자동 적용

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import streamlit as st

# ─────────────────────────────────────────────────────────────
# 페이지 / 한글 폰트 / 경고
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "data" / "fonts" / "NanumGothic.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
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

META_COLS = {"날짜", "일자", "date", "판매월", "연", "년", "월"}
KNOWN_PRODUCT_ORDER = ["개별난방용", "중앙난방용", "자가열전용", "일반용(2)",
                       "업무난방용", "냉난방용", "주한미군", "총공급량"]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")

    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = pd.to_datetime(df["날짜"], errors="coerce").dt.year
    if "월" not in df.columns:
        if "날짜" in df.columns: df["월"] = pd.to_datetime(df["날짜"], errors="coerce").dt.month
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False)
                                  .str.replace(" ", "", regex=False), errors="ignore")
    return df

def format_table(df: pd.DataFrame, temp_cols=("월평균기온(적용)", "기간평균기온", "당월평균기온")) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in temp_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64") \
                     .map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    return out

def month_start(ts): 
    ts = pd.to_datetime(ts)
    return pd.Timestamp(ts.year, ts.month, 1)

def month_range_inclusive(start_y, start_m, end_y, end_m):
    s = pd.Timestamp(int(start_y), int(start_m), 1)
    e = pd.Timestamp(int(end_y), int(end_m), 1)
    return pd.date_range(start=s, end=e, freq="MS")

def poly3_fit(X, Y):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y).ravel()
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, Y)
    yhat = model.predict(Xp)
    r2 = r2_score(Y, yhat)
    return model, poly, r2

def poly3_predict(model, poly, X):
    X = np.asarray(X).reshape(-1, 1)
    return model.predict(poly.transform(X))

def poly3_equation(model, poly):
    # coef shape: [a,b,c], intercept: d (y = a*x^3 + b*x^2 + c*x + d)
    # PolynomialFeatures with include_bias=False on x returns [x, x^2, x^3] in that order
    # So we need to map back: model.coef_ -> c1*x + c2*x^2 + c3*x^3
    c1, c2, c3 = model.coef_
    d = model.intercept_
    # reorder to a*x^3 + b*x^2 + c*x + d
    a, b, c = c3, c2, c1
    return a, b, c, d

def annotate_poly3(ax, model, poly, xdata):
    a, b, c, d = poly3_equation(model, poly)
    txt = f"y = {a:+.5f}x³ {b:+.5f}x² {c:+.5f}x {d:+.5f}"
    # right-bottom corner
    ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

def safe_fillna_period(avg_series, months, fallback_by_M):
    s = avg_series.copy()
    miss = s.isna()
    if miss.any():
        s.loc[miss] = months.loc[miss].map(fallback_by_M)
    return s

# ─────────────────────────────────────────────────────────────
# 데이터 불러오기 유틸

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
    # xlsx or csv with daily '일자' and '기온'(average)
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        # detect date col
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜", "일자", "date"]:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c; break
                except Exception:
                    pass
        # detect temp col
        temp_col = None
        for c in df.columns:
            s = str(c)
            if ("평균기온" in s) or ("기온" in s) or (s.lower() in ["temp", "temperature"]):
                temp_col = c; break
        if date_col is None or temp_col is None:
            return None
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
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=40)
    header_row = None
    for i in range(len(head)):
        row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜","일자","date","Date"] for v in row) and \
           any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
            header_row = i; break
    df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else \
         pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

def detect_temp_col(df):
    for c in df.columns:
        s = str(c).lower()
        if ("기온" in s) or ("temp" in s):
            if pd.api.types.is_numeric_dtype(df[c]): return c
    # fallback
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def guess_product_cols(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

# ─────────────────────────────────────────────────────────────
# 좌측 UI

with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# ============================================================
# A) 공급량 분석
# ============================================================
if mode == "공급량 분석":

    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="SUP_SRC")

        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data")
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i,p in enumerate(repo_files) if "공급" in Path(p).stem), 0)
                supply_file = st.selectbox("실적 엑셀(시트명 '데이터')", repo_files, index=default_idx)
                df = read_excel_sheet(supply_file, prefer_sheet="데이터")
            else:
                st.info("data 폴더에 엑셀이 없습니다. 업로드를 선택하세요.")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="SUP_UP")
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df)==0:
            st.info("좌측에서 파일을 선택해 주세요.")
            st.stop()

        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        st.subheader("학습 데이터 연도 선택")
        years_sel = st.multiselect("연도 선택", years_all, default=years_all, key="SUP_YEARS")

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '기온' 또는 'temp'가 필요합니다.")
            st.stop()

        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        st.subheader("예측할 상품(용도) 선택")
        prods = st.multiselect("상품", product_cols, default=default_products, key="SUP_PRODS")

        st.subheader("예측 기간")
        last_year = int(df["연"].max())
        start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)),
                               index=list(range(2010,2036)).index(last_year), key="SUP_SY")
        start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="SUP_SM")
        end_y = st.selectbox("예측 종료(연)", list(range(2010,2036)),
                             index=list(range(2010,2036)).index(last_year), key="SUP_EY")
        end_m = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="SUP_EM")

        st.subheader("시나리오 기본값")
        base_scen = st.radio("기본 온도 시나리오", ["학습기간 월별 평균","학습 마지막해 월별 복사"],
                             index=0, key="SUP_SCEN")
        run_btn = st.button("예측 시작", type="primary", key="SUP_RUN")

    # ───────── 계산 (버튼 클릭 시) ─────────
    if run_btn:
        base = df.dropna(subset=["연","월", temp_col]).copy()
        base = base.sort_values(["연","월"]).reset_index(drop=True)
        train_df = base[base["연"].isin(st.session_state["SUP_YEARS"])]

        # 모델 학습(상품별)
        models = {}
        for col in st.session_state["SUP_PRODS"]:
            if col not in base.columns or not pd.api.types.is_numeric_dtype(base[col]): 
                continue
            X = train_df[temp_col].values.astype(float)
            Y = train_df[col].values.astype(float)
            model, poly, r2 = poly3_fit(X, Y)
            models[col] = {"model": model, "poly": poly, "r2": r2}

        # 예측 입력(월·기온)
        fut_idx = month_range_inclusive(st.session_state["SUP_SY"], st.session_state["SUP_SM"],
                                        st.session_state["SUP_EY"], st.session_state["SUP_EM"])
        fut = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})
        monthly_avg = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()
        if st.session_state["SUP_SCEN"] == "학습기간 월별 평균":
            fut = fut.merge(monthly_avg.reset_index(), on="월", how="left")
        else:
            last_train_year = int(train_df["연"].max())
            last_temp = base[base["연"]==last_train_year][["월", temp_col]] \
                        .groupby("월")[temp_col].mean().rename("temp").reset_index()
            fut = fut.merge(last_temp, on="월", how="left")

        # 세션 저장
        st.session_state["SUP_CTX"] = {
            "base": base, "train": train_df, "temp_col": temp_col,
            "models": models, "prods": st.session_state["SUP_PRODS"],
            "fut_base": fut, "years_all": years_all
        }
        st.success("공급량 예측 준비 완료!")

    # ───────── 표시 ─────────
    if "SUP_CTX" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러주세요.")
        st.stop()

    ctx = st.session_state["SUP_CTX"]

    # 오른쪽 시나리오 ΔT 슬라이더 -> 표 즉시 반영
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("ΔT(Normal)")
        dT_normal = st.slider("기온 보정(°C)", -5.0, 5.0, 0.0, step=0.5, key="SUP_DT_N")
    with c2:
        st.subheader("ΔT(Best)")
        dT_best = st.slider("기온 보정(°C)", -5.0, 5.0, -0.5, step=0.5, key="SUP_DT_B")
    with c3:
        st.subheader("ΔT(Conservative)")
        dT_con = st.slider("기온 보정(°C)", -5.0, 5.0, +0.5, step=0.5, key="SUP_DT_C")

    def make_forecast_table(delta):
        fut = ctx["fut_base"].copy()
        fut["월평균기온(적용)"] = fut["temp"] + float(delta)

        tbl = fut[["연","월","월평균기온(적용)"]].copy()
        for col in ctx["prods"]:
            if col not in ctx["models"]: 
                continue
            m = ctx["models"][col]["model"]
            p = ctx["models"][col]["poly"]
            yhat = poly3_predict(m, p, tbl["월평균기온(적용)"].values)
            tbl[col] = np.clip(np.rint(yhat), a_min=0, a_max=None).astype(np.int64)
        if "총공급량" not in tbl.columns:
            sum_cols = [c for c in ctx["prods"] if c in tbl.columns and c != "총공급량"]
            if sum_cols:
                tbl["총공급량"] = tbl[sum_cols].sum(axis=1)
        return tbl

    normal_tbl = make_forecast_table(dT_normal)
    best_tbl   = make_forecast_table(dT_best)
    cons_tbl   = make_forecast_table(dT_con)

    # 총계 행
    def append_total_row(df0):
        df = df0.copy()
        total = {"연":"", "월":"총계", "월평균기온(적용)": df["월평균기온(적용)"].mean()}
        for c in df.columns:
            if c in ["연","월","월평균기온(적용)"]: 
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                total[c] = df[c].sum()
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df

    st.subheader("예측 결과 — Normal")
    show_normal = append_total_row(normal_tbl)
    st.dataframe(format_table(show_normal), use_container_width=True)
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_normal.csv", mime="text/csv")

    st.subheader("예측 결과 — Best")
    show_best = append_total_row(best_tbl)
    st.dataframe(format_table(show_best), use_container_width=True)
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_best.csv", mime="text/csv")

    st.subheader("예측 결과 — Conservative")
    show_cons = append_total_row(cons_tbl)
    st.dataframe(format_table(show_cons), use_container_width=True)
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_conservative.csv", mime="text/csv")

    # 그래프: 상품별 연도별 실적 + Normal 예측 + 다항식 방정식(5자리)
    st.subheader("연도별 실적 vs 예측 (Normal)")
    years_view = st.multiselect("표시할 실적 연도", options=ctx["years_all"],
                                default=ctx["years_all"][-5:], key="SUP_VIEW_YEARS")
    months = list(range(1,13))
    for prod in ctx["prods"]:
        if prod not in ctx["models"]:
            continue
        fig, ax = plt.subplots(figsize=(9,3.6))
        # 실적
        for y in sorted([int(v) for v in years_view]):
            s = (ctx["base"][ctx["base"]["연"]==y].set_index("월")[prod]).reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")
        # 예측(Normal)
        P = normal_tbl.copy()
        P = P.set_index("월")[prod].reindex(months)
        ax.plot(months, P.values, linestyle="--", label="예측(Normal)")
        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{m}월" for m in months])
        ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
        r2 = ctx["models"][prod]["r2"]
        ax.set_title(f"{prod} — Poly-3 (Train R²={r2:.3f})")
        annotate_poly3(ax, ctx["models"][prod]["model"], ctx["models"][prod]["poly"],
                       ctx["train"][ctx["train"]["월"]>0][ctx["train"]["월"]<13][ctx["temp_col"]].values if "월" in ctx["train"] else [])
        ax.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# ============================================================
# B) 판매량 분석(냉방용)
# ============================================================
else:
    with st.sidebar:
        st.header("데이터 불러오기")
        sales_src = st.radio("방식", ["Repo 내 파일 사용","파일 업로드"], index=0, key="SAL_SRC")

        def _find_repo():
            here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
            data_dir = here / "data"
            sales_candidates = [data_dir / "상품별판매량.xlsx"] + list(data_dir.glob("*판매*.xlsx"))
            temp_candidates  = [data_dir / "기온.xlsx"] + list(data_dir.glob("*기온*.xlsx")) + list(data_dir.glob("*temp*.csv"))
            s_path = next((p for p in sales_candidates if p.exists()), None)
            t_path = next((p for p in temp_candidates  if p.exists()), None)
            return s_path, t_path

        c1, c2 = st.columns(2)
        if sales_src == "Repo 내 파일 사용":
            s_path, t_path = _find_repo()
            if not s_path or not t_path:
                with c1: sales_file = st.file_uploader("냉방용 판매 실적(xlsx)", type=["xlsx"], key="SAL_UP1")
                with c2: temp_raw_file = st.file_uploader("기온 RAW(일별) (xlsx/csv)", type=["xlsx","csv"], key="SAL_UP2")
            else:
                st.success(f"레포 파일 사용: {s_path.name} · {t_path.name}")
                sales_file = open(s_path, "rb")
                temp_raw_file = open(t_path, "rb")
        else:
            with c1: sales_file = st.file_uploader("냉방용 판매 실적(xlsx)", type=["xlsx"], key="SAL_UP1")
            with c2: temp_raw_file = st.file_uploader("기온 RAW(일별) (xlsx/csv)", type=["xlsx","csv"], key="SAL_UP2")

        if sales_file is None or temp_raw_file is None:
            st.info("두 파일을 모두 준비해주세요.")
            st.stop()

        # 판매 실적 읽기
        try:
            xls = pd.ExcelFile(sales_file, engine="openpyxl")
            sheet = "냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
            sales_raw = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            sales_raw = pd.read_excel(sales_file, engine="openpyxl")
        s_df = normalize_cols(sales_raw)

        # 날짜/냉방 열 자동 탐지
        date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in s_df.columns]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            score = {}
            for c in s_df.columns:
                try:
                    score[c] = pd.to_datetime(s_df[c], errors="coerce").notna().mean()
                except Exception: pass
            date_col = max(score, key=score.get) if score else None
        cool_cols = [c for c in s_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(s_df[c])]
        value_col = next((c for c in cool_cols if "냉방용" in str(c)), None) or (cool_cols[0] if cool_cols else None)
        if date_col is None or value_col is None:
            st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다.")
            st.stop()

        s_df["판매월"] = pd.to_datetime(s_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        s_df["판매량"] = pd.to_numeric(s_df[value_col], errors="coerce")
        s_df = s_df.dropna(subset=["판매월","판매량"]).copy()
        s_df["연"] = s_df["판매월"].dt.year.astype(int)
        s_df["월"] = s_df["판매월"].dt.month.astype(int)

        temp_raw = read_temperature_raw(temp_raw_file)
        if temp_raw is None or temp_raw.empty:
            st.error("기온 RAW에서 날짜/기온을 읽지 못했습니다.")
            st.stop()

        st.subheader("학습 데이터 연도 선택")
        years_all = sorted(s_df["연"].unique().tolist())
        years_sel = st.multiselect("연도 선택", options=years_all, default=years_all, key="SAL_YEARS")

        st.subheader("예측 기간")
        last_year = int(s_df["연"].max())
        sy = st.selectbox("예측 시작(연)", list(range(2010,2036)),
                          index=list(range(2010,2036)).index(last_year), key="SAL_SY")
        sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="SAL_SM")
        ey = st.selectbox("예측 종료(연)", list(range(2010,2036)),
                          index=list(range(2010,2036)).index(last_year), key="SAL_EY")
        em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="SAL_EM")

        run_btn = st.button("예측 시작", type="primary", key="SAL_RUN")

    # ───────── 계산 (버튼 클릭 시) ─────────
    if run_btn:
        # 월별 백업 평균(결측 보정용)
        fallback_by_M = temp_raw.assign(월=temp_raw["일자"].dt.month) \
                                .groupby("월")["기온"].mean()

        # 학습 데이터 (기간평균기온: m-1월16일~m월15일)
        train = s_df[s_df["연"].isin(st.session_state["SAL_YEARS"])].copy()
        rows = []
        for m in sorted(train["판매월"].unique()):
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"] >= s) & (temp_raw["일자"] <= e)
            avg_period = temp_raw.loc[mask,"기온"].mean()
            rows.append({"판매월": m, "기간평균기온": avg_period})
        tmp = pd.DataFrame(rows)
        sj = pd.merge(train[["판매월","판매량","연","월"]], tmp, on="판매월", how="left")
        miss = sj["기간평균기온"].isna()
        if miss.any():
            sj.loc[miss, "기간평균기온"] = sj.loc[miss, "월"].map(fallback_by_M)
        sj = sj.dropna(subset=["기간평균기온","판매량"])
        X = sj["기간평균기온"].values.astype(float)
        Y = sj["판매량"].values.astype(float)
        model, poly, r2 = poly3_fit(X, Y)

        # 예측 입력
        months_idx = month_range_inclusive(st.session_state["SAL_SY"], st.session_state["SAL_SM"],
                                           st.session_state["SAL_EY"], st.session_state["SAL_EM"])
        rows = []
        monthly_cal = temp_raw.assign(연=temp_raw["일자"].dt.year, 월=temp_raw["일자"].dt.month) \
                               .groupby(["연","월"])["기온"].mean().reset_index()
        for m in months_idx:
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"] >= s) & (temp_raw["일자"] <= e)
            avg_period = temp_raw.loc[mask,"기온"].mean()
            avg_month  = monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
            rows.append({"연": int(m.year), "월": int(m.month),
                         "기간평균기온": avg_period, "당월평균기온": avg_month})
        pred_base = pd.DataFrame(rows)
        for c in ["기간평균기온","당월평균기온"]:
            miss = pred_base[c].isna()
            if miss.any():
                pred_base.loc[miss, c] = pred_base.loc[miss, "월"].map(fallback_by_M)

        st.session_state["SALES_CTX"] = {
            "train_df": sj, "model": model, "poly": poly, "r2": r2,
            "pred_base": pred_base, "hist": s_df[["연","월","판매량"]],
            "years_all": years_all, "fallback_by_M": fallback_by_M
        }
        st.success("판매량 예측 준비 완료!")

    if "SALES_CTX" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러주세요.")
        st.stop()

    sctx = st.session_state["SALES_CTX"]

    # 오른쪽 ΔT 슬라이더 (즉시 반영)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("ΔT(Normal)")
        dTn = st.slider("기온 보정(°C)", -5.0, 5.0, 0.0, step=0.5, key="SAL_DT_N")
    with c2:
        st.subheader("ΔT(Best)")
        dTb = st.slider("기온 보정(°C)", -5.0, 5.0, -0.5, step=0.5, key="SAL_DT_B")
    with c3:
        st.subheader("ΔT(Conservative)")
        dTc = st.slider("기온 보정(°C)", -5.0, 5.0, +0.5, step=0.5, key="SAL_DT_C")

    def make_sales_table(delta):
        base = sctx["pred_base"].copy()
        if base["기간평균기온"].isna().any():
            base["기간평균기온"] = base["기간평균기온"] \
                .fillna(base["월"].map(sctx["fallback_by_M"]))
        base["월평균기온(적용)"] = base["기간평균기온"] + float(delta)
        x = base["월평균기온(적용)"].values.astype(float)
        if np.isnan(x).any():
            x = np.where(np.isnan(x), np.nanmean(x), x)
        y = np.clip(np.rint(poly3_predict(sctx["model"], sctx["poly"], x)), a_min=0, a_max=None).astype(np.int64)
        out = base[["연","월","당월평균기온","월평균기온(적용)"]].copy()
        out = out.rename(columns={"월평균기온(적용)":"기간평균기온 (m-1, 16일 ~ m15일)"})
        out["예측판매량"] = y
        return out

    normal_tbl = make_sales_table(dTn)
    best_tbl   = make_sales_table(dTb)
    cons_tbl   = make_sales_table(dTc)

    # Normal 표 + 검증표 + 그래프(연도별 라인, 상관관계)
    st.subheader("예측 결과 — Normal")
    show = normal_tbl.copy()
    # 총계 행
    total_row = {"연":"", "월":"총계", "당월평균기온": show["당월평균기온"].mean(),
                 "기간평균기온 (m-1, 16일 ~ m15일)": show["기간평균기온 (m-1, 16일 ~ m15일)"].mean(),
                 "예측판매량": show["예측판매량"].sum()}
    show = pd.concat([show, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(format_table(show, temp_cols=("당월평균기온", "기간평균기온 (m-1, 16일 ~ m15일)")),
                 use_container_width=True)
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_normal.csv", mime="text/csv")

    st.subheader("예측 결과 — Best")
    show_b = best_tbl.copy()
    total_row = {"연":"", "월":"총계", "당월평균기온": show_b["당월평균기온"].mean(),
                 "기간평균기온 (m-1, 16일 ~ m15일)": show_b["기간평균기온 (m-1, 16일 ~ m15일)"].mean(),
                 "예측판매량": show_b["예측판매량"].sum()}
    show_b = pd.concat([show_b, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(format_table(show_b, temp_cols=("당월평균기온", "기간평균기온 (m-1, 16일 ~ m15일)")),
                 use_container_width=True)
    st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_best.csv", mime="text/csv")

    st.subheader("예측 결과 — Conservative")
    show_c = cons_tbl.copy()
    total_row = {"연":"", "월":"총계", "당월평균기온": show_c["당월평균기온"].mean(),
                 "기간평균기온 (m-1, 16일 ~ m15일)": show_c["기간평균기온 (m-1, 16일 ~ m15일)"].mean(),
                 "예측판매량": show_c["예측판매량"].sum()}
    show_c = pd.concat([show_c, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(format_table(show_c, temp_cols=("당월평균기온", "기간평균기온 (m-1, 16일 ~ m15일)")),
                 use_container_width=True)
    st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_conservative.csv", mime="text/csv")

    # 연도별 실적 vs 예측(Normal)
    st.subheader("연도별 실적 vs 예측(Normal)")
    years_view = st.multiselect("표시할 실적 연도", options=sctx["years_all"],
                                default=sctx["years_all"][-5:], key="SAL_VIEW_YEARS")
    months = list(range(1,13))
    fig, ax = plt.subplots(figsize=(9,3.6))
    for y in sorted([int(v) for v in years_view]):
        s = (sctx["hist"][sctx["hist"]["연"]==y].set_index("월")["판매량"]).reindex(months)
        ax.plot(months, s.values, label=f"{y} 실적")
    P = normal_tbl[["연","월","예측판매량"]].copy()
    P = P.set_index("월")["예측판매량"].reindex(months)
    ax.plot(months, P.values, linestyle="--", label="예측(Normal)")
    ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{m}월" for m in months])
    ax.set_xlabel("월"); ax.set_ylabel("판매량 (MJ)")
    ax.set_title("연도별 실적 vs 예측 (Normal)")
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # 상관관계 그래프 (Train)
    st.subheader(f"기온-냉방용 실적 상관관계 (Train, R²={sctx['r2']:.3f})")
    train_df = sctx["train_df"].copy()
    X = train_df["기간평균기온"].values
    Y = train_df["판매량"].values
    xx = np.linspace(np.nanmin(X), np.nanmax(X), 200)
    yy = poly3_predict(sctx["model"], sctx["poly"], xx)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.scatter(X, Y, alpha=0.7, label="학습 샘플")
    ax.plot(xx, yy, linewidth=3, label="Poly-3")
    annotate_poly3(ax, sctx["model"], sctx["poly"], X)
    ax.set_xlabel("기간평균기온 (m-1, 16일 ~ m15일)")
    ax.set_ylabel("판매량 (MJ)")
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
