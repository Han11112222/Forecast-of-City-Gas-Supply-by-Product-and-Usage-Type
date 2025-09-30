# app.py — 추천 학습 데이터 기간(Poly-3, 하이라이트, 동적 줌)
# - 시작연도별 R²을 계산해 Top 3 구간(시작연도~현재)을 추천
# - 하이라이트: 각 추천 구간 전체(시작~종료)를 배경 색상으로 강조
# - R² 표기: 소수점 4자리
# - 그래프: Plotly(줌/팬/툴팁/토글 가능)
# - 데이터 형식(권장): [연도, 월, 용도, 공급량, 기온] — 열 이름은 아래 매핑 UI로 지정 가능

import io
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.colors import qualitative

st.set_page_config(page_title="추천 학습 데이터 기간", layout="wide")

# ===================== 유틸 =====================
CURRENT_YEAR = datetime.now().year

@st.cache_data(show_spinner=False)
def _read_any(file) -> pd.DataFrame:
    if file is None:
        # 샘플 데이터(없을 때만). 월별 난방 수요가 기온과 음의 상관을 갖도록 생성
        rng = np.random.default_rng(0)
        rows = []
        for y in range(2013, CURRENT_YEAR + 1):
            for m in range(1, 13):
                t = 8 + 12*np.sin((m-1)/12*2*np.pi)  # 간단한 월별 기온 패턴
                supply = 1200 - 35*t + rng.normal(0, 25)  # 공급량
                rows.append([y, m, "개별난방용", max(100, supply), t])
        return pd.DataFrame(rows, columns=["연도","월","용도","공급량","기온"])
    content = file.read()
    file.seek(0)
    # 엑셀/CSV 모두 지원
    name = getattr(file, 'name', 'uploaded')
    if name.lower().endswith(('.xlsx','.xls')):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), encoding="utf-8")


def ensure_types(df: pd.DataFrame):
    # 숫자 변환 안전 처리
    for col in ["연도","월"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for col in ["공급량","기온"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fit_poly3_r2(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    # 결측 제거, 표준 Poly-3 학습 R²
    d = df[[x_col, y_col]].dropna()
    if len(d) < 6:  # 안전 장치
        return np.nan
    X = d[[x_col]].values
    y = d[y_col].values
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    r2 = model.score(Xp, y)
    return float(r2)


def compute_r2_by_start_year(df_all: pd.DataFrame, product: str, end_year: int, x_col: str, y_col: str) -> pd.DataFrame:
    dd = df_all[df_all["용도"]==product].copy()
    years = dd["연도"].dropna().astype(int)
    if years.empty:
        return pd.DataFrame(columns=["시작연도","종료연도","R2"])    
    min_year, max_year = int(years.min()), int(years.max())
    end_year = min(end_year, max_year)

    rows = []
    # 후보: min_year ~ end_year-1
    for s in range(min_year, end_year+1):
        mask = (dd["연도"]>=s) & (dd["연도"]<=end_year)
        r2 = fit_poly3_r2(dd.loc[mask], x_col=x_col, y_col=y_col)
        rows.append([s, end_year, r2])
    out = pd.DataFrame(rows, columns=["시작연도","종료연도","R2"]).dropna()
    out.sort_values("시작연도", inplace=True)
    return out


def pick_top_k_ranges(df_r2: pd.DataFrame, k:int=3) -> pd.DataFrame:
    # R² 상위 k개(동점은 최근 시작연도 우선)
    return (df_r2.sort_values(["R2","시작연도"], ascending=[False, False])
                  .head(k)
                  .reset_index(drop=True))


def pretty_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["기간"] = t["시작연도"].astype(int).astype(str) + "~현재"
    t["R2"] = t["R2"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    t.insert(0, "추천순위", range(1, len(t)+1))
    return t[["추천순위","기간","시작연도","종료연도","R2"]]


def make_r2_figure(df_r2: pd.DataFrame, highlights: pd.DataFrame):
    if df_r2.empty:
        return go.Figure()

    x = df_r2["시작연도"].astype(int).tolist()
    y = df_r2["R2"].astype(float).tolist()

    fig = go.Figure()

    # ---- 하이라이트(추천 구간 전체 배경) ----
    palette = ["rgba(255,179,71,0.17)", "rgba(118,214,165,0.17)", "rgba(120,180,255,0.17)"]
    for i, (_, row) in enumerate(highlights.iterrows()):
        x0 = int(row["시작연도"]) - 0.5
        x1 = int(row["종료연도"]) + 0.5
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=x0, x1=x1, y0=0, y1=1,
                      line=dict(width=0), fillcolor=palette[i % len(palette)])

    # ---- R² 라인 ----
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers+text",
                             text=[f"{v:.4f}" if pd.notna(v) else "" for v in y],
                             textposition="top center",
                             hovertemplate="시작연도=%{x}<br>R²=%{y:.4f}<extra></extra>",
                             name="R² (train)",
                             marker=dict(size=8)))

    fig.update_layout(
        title="학습 시작연도별 R² (종료연도=현재)",
        xaxis_title="학습 기간(시작연도~현재)",
        yaxis_title="R² (train-fit)",
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(range=[max(0, min(y)-0.05), min(1.0, max(y)+0.02)] if len(y)>0 else [0,1]),
        margin=dict(l=50, r=20, t=70, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===================== UI =====================
st.markdown("## 🧠 추천 학습 데이터 기간 — 개별난방용")

c1, c2 = st.columns([1.2, 1])
with c1:
    file = st.file_uploader("실적 파일(엑셀/CSV) — *용도/연도/월/공급량/기온* 포함", type=["csv","xlsx","xls"], accept_multiple_files=False)
    df = ensure_types(_read_any(file))

    # 열 매핑(사용자 데이터 호환)
    with st.expander("열 매핑(필요 시 조정)", expanded=False):
        col_year = st.selectbox("연도 컬럼", options=df.columns, index=list(df.columns).index("연도") if "연도" in df.columns else 0)
        col_mon  = st.selectbox("월 컬럼", options=df.columns, index=list(df.columns).index("월") if "월" in df.columns else 0)
        col_prod = st.selectbox("용도 컬럼", options=df.columns, index=list(df.columns).index("용도") if "용도" in df.columns else 0)
        col_y    = st.selectbox("목표(y) 컬럼(공급량)", options=df.columns, index=list(df.columns).index("공급량") if "공급량" in df.columns else 0)
        col_x    = st.selectbox("설명(x) 컬럼(기온)", options=df.columns, index=list(df.columns).index("기온") if "기온" in df.columns else 0)

    # 내부 표준 컬럼명으로 변환
    df = df.rename(columns={col_year:"연도", col_mon:"월", col_prod:"용도", col_y:"공급량", col_x:"기온"})

    prods = sorted(df["용도"].dropna().unique().tolist())
    sel_prod = st.multiselect("대상 상품(용도)", prods, default=prods[:1])

with c2:
    years = df["연도"].dropna().astype(int)
    data_min, data_max = int(years.min()), int(years.max())
    end_year = st.number_input("종료연도(현재)", min_value=data_min, max_value=min(data_max, CURRENT_YEAR), value=min(data_max, CURRENT_YEAR), step=1)
    top_k = st.slider("추천 순위 개수", 1, 5, 3, 1)

st.divider()

# ===================== 계산 & 표시 =====================
for idx, prod in enumerate(sel_prod):
    st.markdown(f"### 🔹 {prod}")

    r2_table = compute_r2_by_start_year(df, product=prod, end_year=end_year, x_col="기온", y_col="공급량")
    if r2_table.empty:
        st.warning("데이터가 부족합니다. 열 매핑 또는 업로드 파일을 확인하세요.")
        continue

    top_tbl = pick_top_k_ranges(r2_table, k=top_k)

    # ----- 표(추천순위) -----
    st.dataframe(pretty_table(top_tbl), use_container_width=True, hide_index=True)

    # ----- 그래프 -----
    fig = make_r2_figure(r2_table, highlights=top_tbl)
    st.plotly_chart(fig, use_container_width=True, config={
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline","drawrect","eraseshape"],
    })

    st.caption("그래프는 마우스 드래그로 확대/축소, 더블클릭으로 초기화, 모드바로 저장/그리기 도구 사용 가능.")

st.stop()
