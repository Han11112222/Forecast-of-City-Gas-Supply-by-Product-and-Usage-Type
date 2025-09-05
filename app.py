# app.py — 도시가스 공급·판매 분석 (Poly-3 + Scenarios)

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import streamlit as st

# ─────────────────────────────────────────────────────────────
# 기본/폰트
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

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

# ─────────────────────────────────────────────────────────────
# 공통 유틸
META_COLS = {"날짜","일자","date","연","년","월"}
TEMP_HINTS = ["평균기온","기온","temperature","temp"]
KNOWN_PRODUCT_ORDER = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]

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
        if "년" in df.columns: df["연"]=df["년"]
        elif "날짜" in df.columns: df["연"]=df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"]=df["날짜"].dt.month
    for c in df.columns:
        if df[c].dtype=="object":
            df[c]=pd.to_numeric(
                df[c].astype(str).str.replace(",","",regex=False).str.replace(" ","",regex=False),
                errors="ignore"
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str|None:
    for c in df.columns:
        nm=str(c).lower()
        if any(h in nm for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates=[c for c in numeric if c not in META_COLS]
    ordered=[c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others=[c for c in candidates if c not in ordered]
    return ordered+others

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet="데이터"):
    try:
        xls=pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet=prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df=pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df=pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def read_temperature_raw(file):
    import pandas as pd
    def _finalize(df):
        df.columns=[str(c).strip() for c in df.columns]
        date_col=None
        for c in df.columns:
            if str(c).lower() in ["날짜","일자","date"]: date_col=c; break
        if date_col is None:
            for c in df.columns:
                try: pd.to_datetime(df[c], errors="raise"); date_col=c; break
                except Exception: pass
        temp_col=None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp","temperature"]):
                temp_col=c; break
        if date_col is None or temp_col is None: return None
        out=pd.DataFrame({
            "일자":pd.to_datetime(df[date_col], errors="coerce"),
            "기온":pd.to_numeric(df[temp_col], errors="coerce")
        }).dropna()
        return out.sort_values("일자").reset_index(drop=True)
    name=getattr(file,"name",str(file))
    if name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls=pd.ExcelFile(file, engine="openpyxl")
    sheet=xls.sheet_names[0]
    head=pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row=None
    for i in range(len(head)):
        row=[str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜","일자","date","Date"] for v in row) and any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
            header_row=i; break
    df=pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

def month_start(x): x=pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s,e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

def fit_poly3_and_predict(x_train, y_train, x_future):
    m=(~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    x_train=x_train.reshape(-1,1); x_future=x_future.reshape(-1,1)
    poly=PolynomialFeatures(degree=3, include_bias=False)
    Xtr=poly.fit_transform(x_train)
    model=LinearRegression().fit(Xtr, y_train)
    r2=model.score(Xtr, y_train)
    y_future=model.predict(poly.transform(x_future))
    coef=np.polyfit(x_train.ravel(), y_train, 3)
    return y_future, r2, coef

def eq_text_from_coef(coef):
    a,b,c,d = coef
    return f"y = {a:+.3e}x³ {b:+.3e}x² {c:+.3e}x {d:+.3e}"

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols=float1_cols or []; int_cols=int_cols or []
    show=df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c]=pd.to_numeric(show[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in int_cols:
        if c in show.columns:
            show[c]=pd.to_numeric(show[c], errors="coerce").round().astype("Int64").map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    st.markdown("""
    <style>
      table.centered-table { width:100%; table-layout:fixed; }
      table.centered-table th, table.centered-table td { text-align:center !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

# 상관관계 그래프(판매)
def correlation_plot(temp_series, sales_series, y_label="판매량 (MJ)", title="기온-냉방용 실적 상관관계"):
    s=pd.DataFrame({"temp":pd.to_numeric(temp_series, errors="coerce"),
                    "sales":pd.to_numeric(sales_series, errors="coerce")}).dropna()
    x=s["temp"].values.reshape(-1,1)
    y=s["sales"].values
    poly=PolynomialFeatures(degree=3, include_bias=False)
    X=poly.fit_transform(x)
    reg=LinearRegression().fit(X,y)
    r2=reg.score(X,y)
    coef=np.polyfit(s["temp"].values,y,3)
    eq_text=eq_text_from_coef(coef)

    xg=np.linspace(float(s["temp"].min()), float(s["temp"].max()), 200)
    y_hat=reg.predict(poly.transform(xg.reshape(-1,1)))
    resid=y - reg.predict(X)
    sigma=float(np.sqrt(np.mean(resid**2)))
    y_up=y_hat+1.96*sigma; y_dn=y_hat-1.96*sigma

    s["temp_bin"]=s["temp"].round(0)
    by=s.groupby("temp_bin")["sales"].median().reset_index()

    fig=plt.figure(figsize=(7.8,4.6)); ax=plt.gca()
    ax.scatter(s["temp"], s["sales"], alpha=0.55, label="학습 샘플")
    ax.plot(xg, y_hat, color="tab:blue", linewidth=2.3, label="Poly-3")
    ax.fill_between(xg, y_dn, y_up, color="tab:blue", alpha=0.12, label="±1.96σ")
    ax.scatter(by["temp_bin"], by["sales"], color="tab:orange", edgecolor="k", s=40, linewidth=0.3, label="온도별 중앙값")
    ax.set_xlabel("기간평균기온 (℃)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} (Train, R²={r2:.3f})")
    ax.legend(loc="best")
    ax.text(0.99, 0.02, eq_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.65", alpha=0.9))
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# 사이드바
with st.sidebar:
    st.header("분석 유형")
    mode=st.radio("선택", ["공급량 분석","판매량 분석(냉방용)"], index=0)

# ============================ A) 공급량 분석 ============================
if mode=="공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        src=st.radio("방식", ["Repo 내 파일 사용","파일 업로드"], index=0)
        df=None
        if src=="Repo 내 파일 사용":
            data_dir=Path("data")
            repo_files=sorted([p for p in data_dir.glob("*.xlsx") if "공급" in p.name])
            if repo_files:
                f=st.selectbox("실적 파일(Excel)", repo_files, index=0)
                df=read_excel_sheet(f, prefer_sheet="데이터")
            else:
                st.info("data/*.xlsx 에 공급 실적을 넣어두면 기본으로 불러옵니다.")
        else:
            up=st.file_uploader("공급량 실적 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None: df=read_excel_sheet(up, prefer_sheet="데이터")
        if df is None or len(df)==0: st.stop()

        st.subheader("학습 데이터 연도 선택")
        years_all=sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel=st.multiselect("연도 선택", years_all, default=years_all)

        temp_col=detect_temp_col(df)
        if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()

        st.subheader("예측할 상품 선택")
        product_cols=guess_product_cols(df)
        # '개별난방용' 다음에 '중앙난방용'이 오도록 정렬
        ordered=[]
        for k in ["개별난방용","중앙난방용"]:
            if k in product_cols: ordered.append(k)
        for c in product_cols:
            if c not in ordered: ordered.append(c)
        prods=st.multiselect("상품(용도) 선택", ordered, default=ordered[:8])

        st.subheader("예측 기간")
        last_year=int(df["연"].max())
        c1,c2=st.columns(2)
        with c1:
            start_y=st.selectbox("예측 시작(연)", list(range(2010,2036)),
                                 index=list(range(2010,2036)).index(last_year))
            end_y=st.selectbox("예측 종료(연)", list(range(2010,2036)),
                               index=list(range(2010,2036)).index(last_year))
        with c2:
            start_m=st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m=st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        scen_radio = st.radio("기온 입력(기본)", ["학습기간 월별 평균","학습 마지막해 월별 복사"], index=0)
        run_btn=st.button("예측 시작", type="primary")

    if run_btn:
        base=df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df=base[base["연"].isin(years_sel)].copy()

        monthly_avg_temp=train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()
        f_start=pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end  =pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        fut_idx=month_range_inclusive(f_start, f_end)
        fut_base=pd.DataFrame({"연":fut_idx.year.astype(int), "월":fut_idx.month.astype(int)})

        if scen_radio=="학습기간 월별 평균":
            fut_base=fut_base.merge(monthly_avg_temp.reset_index(), on="월", how="left")
        else:
            last_train_year=int(train_df["연"].max())
            base_temp=base[base["연"]==last_train_year][["월",temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
            fut_base=fut_base.merge(base_temp, on="월", how="left")

        # 결과 저장(시나리오 재계산에 필요)
        result={
            "forecast_start": f_start,
            "years_all": sorted([int(y) for y in base["연"].dropna().unique()]),
            "per_product": {},
            "fut_base": fut_base[["연","월","temp"]].copy(),     # Scenario용 temp base
            "train_df": train_df,
            "temp_col": temp_col,
            "prods": prods
        }

        for col in prods:
            if col not in base.columns or not pd.api.types.is_numeric_dtype(base[col]): 
                continue
            x_train=train_df[temp_col].astype(float).values
            y_train=train_df[col].astype(float).values
            y_future, r2, coef = fit_poly3_and_predict(x_train, y_train, fut_base["temp"].astype(float).values)

            hist=base[["연","월",col]].rename(columns={col:"val"}).copy()
            result["per_product"][col] = {
                "x_train": x_train, "y_train": y_train,
                "hist": hist, "r2": r2, "coef": coef
            }

        st.session_state["supply_state"]=result
        st.success("공급량 예측을 준비했습니다. 아래에서 시나리오를 조절하세요.")

    if "supply_state" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    res=st.session_state["supply_state"]

    # ===== Scenario controls (즉시 반영) =====
    st.subheader("시나리오 온도 보정(℃)")
    c1,c2,c3=st.columns(3)
    with c1: off_norm=st.number_input("Normal", value=0.0, step=0.5, key="sup_off_norm")
    with c2: off_best=st.number_input("Best",   value=+0.5, step=0.5, key="sup_off_best")
    with c3: off_cons=st.number_input("Conservative", value=-0.5, step=0.5, key="sup_off_cons")

    def build_supply_table(offset: float) -> pd.DataFrame:
        fut=res["fut_base"].copy()
        fut["월평균기온(적용)"]=fut["temp"]+float(offset)
        months_df=fut[["연","월","월평균기온(적용)"]].copy()
        pred_rows=[]
        for col, pkg in res["per_product"].items():
            y_future, _, _ = fit_poly3_and_predict(pkg["x_train"], pkg["y_train"], fut["월평균기온(적용)"].values)
            tmp=months_df.copy()
            tmp[col]=np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        # merge by 연/월/월평균기온(적용)
        out=pred_rows[0]
        for t in pred_rows[1:]:
            out=out.merge(t, on=["연","월","월평균기온(적용)"], how="left")
        # 종계
        total=out.drop(columns=["연","월","월평균기온(적용)"]).sum().to_frame().T
        total.insert(0,"월","종계"); total.insert(0,"연",""); total.insert(2,"월평균기온(적용)","")
        out_w_total=pd.concat([out, total], ignore_index=True)
        return out_w_total

    tbl_norm = build_supply_table(off_norm)
    tbl_best = build_supply_table(off_best)
    tbl_cons = build_supply_table(off_cons)

    # ===== 그래프 (Normal 기준) =====
    st.caption("그래프는 **Normal** 시나리오 기준으로 표시됩니다.")
    months=list(range(1,13))
    for prod, pkg in res["per_product"].items():
        fig=plt.figure(figsize=(9.2,3.9)); ax=plt.gca()
        years_view=st.session_state.get("supply_years_view", res["years_all"][-5:])
        years_view=st.multiselect(f"[{prod}] 표시할 실적 연도", options=res["years_all"], default=years_view, key=f"supply_years_view_{prod}")
        for y in sorted([int(v) for v in years_view]):
            s=(pkg["hist"][pkg["hist"]["연"]==y].set_index("월")["val"]).reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")
        # Normal 예측 12개월
        P=tbl_norm[["연","월",prod]].copy()
        P=P[pd.to_numeric(P["월"], errors="coerce").notna()]
        y0, m0=int(res["forecast_start"].year), int(res["forecast_start"].month)
        pred_vals=[]
        y,m=y0,m0
        for _ in range(12):
            row=P[(P["연"]==y)&(P["월"]==m)]
            pred_vals.append(row.iloc[0][prod] if len(row) else np.nan)
            if m==12: y+=1; m=1
            else: m+=1
        ax.plot(months, pred_vals, linestyle="--", label="예측(Normal)")
        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
        ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
        ax.set_title(f"{prod} — Poly-3 (Train R²={pkg['r2']:.3f})")
        ax.text(0.99, 0.02, eq_text_from_coef(pkg["coef"]), ha="right", va="bottom",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.65", alpha=0.9))
        ax.legend(loc="best"); plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # ===== 표 3종 출력: Normal ▶ Best ▶ Conservative =====
    st.subheader("예측 결과 — Normal")
    render_centered_table(tbl_norm, float1_cols=["월평균기온(적용)"],
                          int_cols=[c for c in tbl_norm.columns if c not in ["연","월","월평균기온(적용)"]])
    st.subheader("예측 결과 — Best")
    render_centered_table(tbl_best, float1_cols=["월평균기온(적용)"],
                          int_cols=[c for c in tbl_best.columns if c not in ["연","월","월평균기온(적용)"]])
    st.subheader("예측 결과 — Conservative")
    render_centered_table(tbl_cons, float1_cols=["월평균기온(적용)"],
                          int_cols=[c for c in tbl_cons.columns if c not in ["연","월","월평균기온(적용)"]])

# ============================ B) 판매량 분석(냉방용) ============================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")

    with st.sidebar:
        st.header("데이터 불러오기")
        sales_src=st.radio("방식", ["Repo 내 파일 사용","파일 업로드"], index=0)

    def _find_repo_sales_and_temp():
        here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        data_dir = here / "data"
        sales_candidates = [*data_dir.glob("*판매량*.xlsx"), *data_dir.glob("상품별판매량*.xlsx")]
        temp_candidates  = [*data_dir.glob("기온.*"), *data_dir.glob("*기온*.xlsx"), *data_dir.glob("*temp*.csv")]
        sale = next((p for p in sales_candidates if p.exists()), None)
        temp = next((p for p in temp_candidates  if p.exists()), None)
        return sale, temp

    c1,c2=st.columns(2)
    if sales_src=="Repo 내 파일 사용":
        s_path, t_path = _find_repo_sales_and_temp()
        if not s_path or not t_path:
            with c1: sales_file=st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
            with c2: temp_raw_file=st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])
        else:
            st.success(f"레포 파일 사용: {s_path.name} · {t_path.name}")
            sales_file=open(s_path,"rb"); temp_raw_file=open(t_path,"rb")
    else:
        with c1: sales_file=st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file=st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 준비하세요."); st.stop()

    # 판매 실적 로드
    try:
        xls=pd.ExcelFile(sales_file, engine="openpyxl")
        sheet="냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
        raw_sales=pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw_sales=pd.read_excel(sales_file, engine="openpyxl")
    sales_df=normalize_cols(raw_sales)

    date_candidates=[c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
    if date_candidates: date_col=date_candidates[0]
    else:
        score={}
        for c in sales_df.columns:
            try: score[c]=pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
            except Exception: pass
        date_col=max(score, key=score.get) if score else None
    cool_cols=[c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    value_col=None
    for c in cool_cols:
        if "냉방용" in str(c): value_col=c; break
    value_col=value_col or (cool_cols[0] if cool_cols else None)
    if date_col is None or value_col is None:
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

    sales_df["판매월"]=pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"]=pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df=sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"]=sales_df["판매월"].dt.year.astype(int); sales_df["월"]=sales_df["판매월"].dt.month.astype(int)

    temp_raw=read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()

    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        years_all=sorted(sales_df["연"].unique().tolist())
        years_sel=st.multiselect("연도 선택", options=years_all, default=years_all)
        st.subheader("예측 기간")
        last_year=int(sales_df["연"].max())
        c3,c4=st.columns(2)
        with c3:
            start_y=st.selectbox("예측 시작(연)", list(range(2010,2036)),
                                 index=list(range(2010,2036)).index(last_year))
            end_y=st.selectbox("예측 종료(연)", list(range(2010,2036)),
                               index=list(range(2010,2036)).index(last_year))
        with c4:
            start_m=st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m=st.selectbox("예측 종료(월)", list(range(1,13)), index=11)
        run_btn=st.button("예측 시작", type="primary")

    # 학습용 temp avg
    temp_raw["연"]=temp_raw["일자"].dt.year; temp_raw["월"]=temp_raw["일자"].dt.month
    monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
    fallback_by_M = temp_raw.groupby("월")["기온"].mean()

    def period_avg(label_m: pd.Timestamp) -> float:
        m = month_start(label_m)
        s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
        e = m + pd.DateOffset(days=14)
        mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
        return temp_raw.loc[mask,"기온"].mean()

    train_sales=sales_df[sales_df["연"].isin(years_sel)].copy()
    rows=[{"판매월":m, "기간평균기온":period_avg(m)} for m in train_sales["판매월"].unique()]
    sj=pd.merge(train_sales[["판매월","판매량","연","월"]], pd.DataFrame(rows), on="판매월", how="left")
    miss=sj["기간평균기온"].isna()
    if miss.any(): sj.loc[miss,"기간평균기온"]=sj.loc[miss,"월"].map(fallback_by_M)
    sj=sj.dropna(subset=["기간평균기온","판매량"])

    x_train=sj["기간평균기온"].astype(float).values
    y_train=sj["판매량"].astype(float).values
    _, r2_fit, coef_fit = fit_poly3_and_predict(x_train, y_train, x_train)

    if run_btn:
        f_start=pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end  =pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        months_rng=month_range_inclusive(f_start, f_end)
        rows=[]
        for m in months_rng:
            s=(m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e=m + pd.DateOffset(days=14)
            mask=(temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            avg_period=temp_raw.loc[mask,"기온"].mean()
            avg_month =monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
            rows.append({"연":int(m.year),"월":int(m.month),"기간평균기온":avg_period,"당월평균기온":avg_month})
        base_pred=pd.DataFrame(rows)
        for c in ["기간평균기온","당월평균기온"]:
            miss=base_pred[c].isna()
            if miss.any(): base_pred.loc[miss,c]=base_pred.loc[miss,"월"].map(fallback_by_M)

        st.session_state["sales_state"]={
            "base_pred": base_pred, "x_train": x_train, "y_train": y_train,
            "r2": r2_fit, "coef": coef_fit, "forecast_start": f_start,
            "actual": sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
        }
        st.success("판매량 예측을 준비했습니다. 아래에서 시나리오를 조절하세요.")

    if "sales_state" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    ss=st.session_state["sales_state"]

    # ===== Scenario controls (즉시 반영) =====
    st.subheader("시나리오 온도 보정(℃)")
    d1,d2,d3=st.columns(3)
    with d1: soff_norm=st.number_input("Normal", value=0.0, step=0.5, key="sal_off_norm")
    with d2: soff_best=st.number_input("Best",   value=+0.5, step=0.5, key="sal_off_best")
    with d3: soff_cons=st.number_input("Conservative", value=-0.5, step=0.5, key="sal_off_cons")

    def build_sales_table(offset: float) -> pd.DataFrame:
        base=ss["base_pred"].copy()
        base["기간평균기온(적용)"]=base["기간평균기온"]+float(offset)
        y_future, _, _ = fit_poly3_and_predict(ss["x_train"], ss["y_train"], base["기간평균기온(적용)"].astype(float).values)
        out=base.copy()
        out["예측판매량"]=np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
        out=out.rename(columns={"기간평균기온(적용)":"기간평균기온(적용, m-1,16일~m15일)"})
        total=out[["예측판매량"]].sum().to_frame().T
        total.insert(0,"월","종계"); total.insert(0,"연","")
        total.insert(2,"기간평균기온(적용, m-1,16일~m15일)","")
        total.insert(3,"당월평균기온","")
        out_w_total=pd.concat([out, total], ignore_index=True)
        return out_w_total

    sal_norm=build_sales_table(soff_norm)
    sal_best=build_sales_table(soff_best)
    sal_cons=build_sales_table(soff_cons)

    # 검증표는 Normal 기준
    verify=pd.merge(sal_norm[["연","월","예측판매량"]],
                    ss["actual"], on=["연","월"], how="left")
    verify["오차"]=(verify["예측판매량"]-verify["실제판매량"]).astype("Int64")
    verify["오차율(%)"]=(verify["오차"]/verify["실제판매량"]*100).round(1).astype("Float64")

    # ===== 표 출력 — Normal / Best / Conservative =====
    st.subheader("판매량 예측(요약) — Normal")
    render_centered_table(
        sal_norm[["연","월","당월평균기온","기간평균기온(적용, m-1,16일~m15일)","예측판매량"]],
        float1_cols=["당월평균기온","기간평균기온(적용, m-1,16일~m15일)"],
        int_cols=["예측판매량"], index=False
    )
    st.subheader("판매량 예측(요약) — Best")
    render_centered_table(
        sal_best[["연","월","당월평균기온","기간평균기온(적용, m-1,16일~m15일)","예측판매량"]],
        float1_cols=["당월평균기온","기간평균기온(적용, m-1,16일~m15일)"],
        int_cols=["예측판매량"], index=False
    )
    st.subheader("판매량 예측(요약) — Conservative")
    render_centered_table(
        sal_cons[["연","월","당월평균기온","기간평균기온(적용, m-1,16일~m15일)","예측판매량"]],
        float1_cols=["당월평균기온","기간평균기온(적용, m-1,16일~m15일)"],
        int_cols=["예측판매량"], index=False
    )

    # 검증표
    st.subheader("판매량 예측 검증 (Normal 기준)")
    render_centered_table(
        verify[["연","월","실제판매량","예측판매량","오차","오차율(%)"]],
        int_cols=["실제판매량","예측판매량","오차"], index=False
    )

    # 월별 추이 그래프(실적 + Normal 예측)
    st.caption("아래 그래프는 선택 연도의 실적과 **Normal** 예측을 함께 보여줍니다.")
    years_view=st.multiselect("표시할 실적 연도", options=sorted(sales_df["연"].unique().tolist()),
                              default=sorted(sales_df["연"].unique().tolist())[-5:], key="sales_years_view")
    months=list(range(1,13))
    fig=plt.figure(figsize=(9.2,3.9)); ax=plt.gca()
    for y in sorted([int(v) for v in years_view]):
        s=(sales_df[sales_df["연"]==y].set_index("월")["판매량"]).reindex(months)
        ax.plot(months, s.values, label=f"{y} 실적")
    # Normal 12개월
    P=sal_norm[["연","월","예측판매량"]].copy()
    P=P[pd.to_numeric(P["월"], errors="coerce").notna()]
    y0,m0=int(ss["forecast_start"].year), int(ss["forecast_start"].month)
    pred_vals=[]; y,m=y0,m0
    for _ in range(12):
        row=P[(P["연"]==y)&(P["월"]==m)]
        pred_vals.append(row.iloc[0]["예측판매량"] if len(row) else np.nan)
        if m==12: y+=1; m=1
        else: m+=1
    ax.plot(months, pred_vals, linestyle="--", label="예측(Normal)")
    ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
    ax.set_xlabel("월"); ax.set_ylabel("판매량 (MJ)")
    ax.set_title(f"냉방용 — Poly-3 (Train R²={ss['r2']:.3f})")
    ax.text(0.99, 0.02, eq_text_from_coef(ss["coef"]), ha="right", va="bottom",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.65", alpha=0.9))
    ax.legend(loc="best"); plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # 상관관계(훈련) 그래프
    st.subheader("기온-냉방용 실적 상관관계 (Train)")
    st.pyplot(correlation_plot(sj["기간평균기온"], sj["판매량"], y_label="판매량 (MJ)"), clear_figure=True)
