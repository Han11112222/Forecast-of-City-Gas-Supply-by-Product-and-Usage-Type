# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="도시가스 공급량·판매량 예측 (Poly-3)", layout="wide")

# -----------------------------
# 기본 옵션/스타일
# -----------------------------
st.markdown("""
<style>
/* 표 제목 좌측 여백 줄이기 */
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
/* 표 헤더 굵게 */
thead tr th {font-weight: 700 !important;}
/* 멀티셀렉트(칩) 위아래 여백 줄이기 */
div[data-baseweb="select"] {margin-top: 0.35rem; margin-bottom: 0.35rem;}
/* 섹션 타이틀 */
h3, h4 { margin-top: .6rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 로드 (예: repo의 기본 파일 사용)
# -----------------------------
@st.cache_data
def load_supply(filepath: str) -> pd.DataFrame:
    # 필드명은 사용중 파일에 맞춰주세요.
    # 예시: ['연','월','평균기온','개별난방용','중앙난방용','자가열전용','일반용(2)','업무난방용','냉난방용','주한미군','총공급량','상품']
    df = pd.read_excel(filepath)
    # 숫자 컬럼 강제 변환
    for c in df.columns:
        if c not in ['연','월','상품','비고','지역','구분']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data
def load_temp_trend(filepath: str) -> pd.DataFrame:
    # 시트/컬럼명은 실제 파일에 맞춰주세요.
    # 예시 시트: '기온예측' / 컬럼: ['날짜','평균기온','추세분석(지수평활법)']
    return pd.read_excel(filepath)

DATA_FILE = "data/상품별공급량_MJ.xlsx"     # repo 경로
TEMP_FILE = "data/기온예측.xlsx"            # repo 경로

df_raw = load_supply(DATA_FILE)
df_temp = load_temp_trend(TEMP_FILE)

# -----------------------------
# 유틸
# -----------------------------
def month_label(k):
    return f"{k}월"

def format_int(x):
    if pd.isna(x): return ""
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return f"{x}"

def poly3_fit_predict(x, y, x_eval):
    # 3차 다항식 회귀
    coef = np.polyfit(x, y, 3)
    p = np.poly1d(coef)
    y_hat = p(x)
    r2 = r2_score(y, y_hat)
    y_pred = p(x_eval)
    return p, y_pred, r2

# 카테고리(열) 집합 – 실제 보유 열명에 맞춰 사용
CATE_COLS = ['개별난방용','중앙난방용','자가열전용','일반용(2)','업무난방용','냉난방용','주한미군','총공급량']

# -----------------------------
# 사이드바 – 학습 데이터 연도 선택/상품 선택(기존과 동일하게 유지)
# -----------------------------
with st.sidebar:
    st.header("예측 설정")
    # 1) 예측 시작 (연/월) — 가로 배치
    c1, c2 = st.columns(2)
    years_all = sorted(df_raw['연'].unique())
    months_all = list(range(1,13))

    with c1:
        start_year = st.selectbox("🚀 예측 시작(연)", options=years_all, index=0, key="start_year")
    with c2:
        start_month = st.selectbox("📅 예측 시작(월)", options=months_all, index=0, key="start_month")

    # 2) 예측 종료 (연/월) — 가로 배치
    c3, c4 = st.columns(2)
    with c3:
        end_year = st.selectbox("🏁 예측 종료(연)", options=years_all, index=len(years_all)-1, key="end_year")
    with c4:
        end_month = st.selectbox("📅 예측 종료(월)", options=months_all, index=11, key="end_month")

    # 예측 실행 버튼(기존 로직 그대로 사용)
    run_btn = st.button("📊 예측 시작", use_container_width=True)

# -----------------------------
# 본문 – 필터(칩) 3개를 가로로
# -----------------------------
st.markdown("### 📈 그래프 (실적 + 예측(Normal) + 추세분석)")
colA, colB, colC = st.columns(3)

# 기본 연도 후보 (필요 시 변경)
base_years = sorted(df_raw['연'].unique())
default_actual = base_years[-2:] if len(base_years)>=2 else base_years
default_pred   = [y for y in base_years if y>=base_years[-1]][:3] or default_actual
default_trend  = default_pred

with colA:
    years_actual = st.multiselect("👀 실적연도", options=base_years, default=default_actual, key="years_actual")
with colB:
    years_pred = st.multiselect("📈 예측연도 (Normal)", options=base_years, default=default_pred, key="years_pred")
with colC:
    years_trend = st.multiselect("📚 추세분석연도", options=base_years, default=default_trend, key="years_trend")

# -----------------------------
# 라인 차트 (Plotly) — 마우스오버 표시, 휠로만 확대
# -----------------------------
def make_line_fig(df, years_actual, years_pred, years_trend, title_suffix="Poly-3"):
    fig = go.Figure()

    # 실적
    for y in years_actual:
        sub = df[df['연']==y].sort_values('월')
        fig.add_trace(go.Scatter(
            x=sub['월'], y=sub['총공급량'], mode="lines",
            name=f"{y} 실적",
            hovertemplate="%{x}월<br>%{y:,} MJ<extra></extra>"
        ))

    # 예측(Normal) — 예시: 과거 기온으로 Poly-3에 대입해 만든 series라고 가정(앱 기존 로직 그대로 연결)
    # 여기서는 단순히 실적과 같은 값을 점선으로 예시
    for y in years_pred:
        sub = df[df['연']==y].sort_values('월')
        fig.add_trace(go.Scatter(
            x=sub['월'], y=sub['총공급량'], mode="lines",
            name=f"예측(Normal) {y}",
            line=dict(dash="dash"),
            hovertemplate="%{x}월<br>%{y:,} MJ<extra></extra>"
        ))

    # 추세분석 — 예시: 점선+작은 점
    for y in years_trend:
        sub = df[df['연']==y].sort_values('월')
        fig.add_trace(go.Scatter(
            x=sub['월'], y=sub['총공급량'], mode="lines",
            name=f"추세분석 {y}",
            line=dict(dash="dot"),
            hovertemplate="%{x}월<br>%{y:,} MJ<extra></extra>"
        ))

    fig.update_layout(
        title=f"개별난방용 — {title_suffix}",
        xaxis=dict(title="월", tickmode="array", tickvals=list(range(1,13)), ticktext=[f"{m}월" for m in range(1,13)]),
        yaxis=dict(title="공급량 (MJ)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        hovermode="x unified",
        dragmode="pan"  # 드래그는 이동만, 확대는 휠
    )
    return fig

line_fig = make_line_fig(df_raw, years_actual, years_pred, years_trend, title_suffix="Poly-3 (Train R² 표시는 하단 상관도)")

st.plotly_chart(
    line_fig,
    use_container_width=True,
    config=dict(
        scrollZoom=True,  # 휠로 확대/축소
        modeBarButtonsToRemove=[
            "zoom2d","select2d","lasso2d","autoScale2d",
            "zoomIn2d","zoomOut2d","resetScale2d"
        ]
    )
)

# -----------------------------
# 상관도(기온-공급량) – Matplotlib 고정형 그대로 유지 (R² 표기 + 95% 신뢰구간 안내)
# -----------------------------
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"  # 한글 가능 폰트(배포환경마다 다름)

st.markdown("### 🔬 개별난방용 — 기온·공급량 상관(Train, **R²** 포함)")
with st.expander("📎 95% 신뢰구간 설명", expanded=False):
    st.write("선형/다항 회귀의 예측 불확실성을 근거로 계산한 **95% 신뢰구간**입니다. "
             "같은 조건에서 반복 추출한다면 **약 95%**가 해당 구간 안에 들어온다는 의미예요.")

# 학습 데이터(예: 가장 최근 3개년)로 예시 작성
use_years = sorted(df_raw['연'].unique())[-3:]
train = df_raw[df_raw['연'].isin(use_years)].copy()
train['월평균기온'] = pd.to_numeric(train.get('평균기온', np.nan), errors='coerce')

x = train['월평균기온'].values
y = train['총공급량'].values
mask = ~(pd.isna(x) | pd.isna(y))
x, y = x[mask], y[mask]

x_eval = np.linspace(np.nanmin(x), np.nanmax(x), 200)
p, y_line, r2 = poly3_fit_predict(x, y, x_eval)

# 신뢰구간(간단 근사)
y_fit = p(x)
resid = y - y_fit
s = np.std(resid)
ci = 1.96 * s  # 대략적 95% CI 근사

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(x, y, s=30, alpha=.8, color="#2563eb", label="학습 샘플")  # 진한 파랑 점
ax.plot(x_eval, y_line, color="#1f5da0", linewidth=3, label="Poly-3")
ax.fill_between(x_eval, y_line-ci, y_line+ci, color="#f97316", alpha=0.25, label="95% 신뢰구간")  # 주황 그라데이션
# 온도별 중앙값(옵션)
temp_bins = pd.cut(x, bins=np.linspace(x.min(), x.max(), 10))
med = pd.Series(y).groupby(temp_bins).median()
centers = [interval.mid for interval in med.index.categories]
ax.scatter(centers, med.values, s=80, color="#fb923c", zorder=5, label="온도별 중앙값")

ax.set_title(f"개별난방용 — 기온·공급량 상관(Train, R²={r2:.3f})")
ax.set_xlabel("기온 (°C)")
ax.set_ylabel("공급량 (MJ)")
ax.legend(loc="upper right")
st.pyplot(fig, use_container_width=True)

# -----------------------------
# 표 영역 – 시나리오 세부표 + 연도별 총계(상품 삭제 / 월·월평균기온 공란)
# -----------------------------
def pretty_table(df, caption=None):
    # 숫자 포맷 적용
    df_disp = df.copy()
    for c in df_disp.columns:
        if c not in ['연','월','월평균기온','상품']:
            df_disp[c] = df_disp[c].apply(format_int)
    if caption:
        st.markdown(f"#### {caption}")
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

def yearly_total_table(df_detail, caption="연도별 총계"):
    # 상품 열 제거, 월/월평균기온 공란으로
    cols_keep = ['연'] + [c for c in CATE_COLS if c in df_detail.columns]
    g = df_detail.groupby('연', as_index=False)[cols_keep[1:]].sum(numeric_only=True)
    g.insert(1, '월', "")              # 공란
    g.insert(2, '월평균기온', "")       # 공란
    # 숫자 포맷
    g_disp = g.copy()
    for c in g_disp.columns:
        if c not in ['연','월','월평균기온']:
            g_disp[c] = g_disp[c].apply(format_int)
    st.markdown("#### 연도별 총계")
    st.dataframe(g_disp, use_container_width=True, hide_index=True)

# 예시: Normal 표(월별 상세) 만들고 바로 아래 총계(연도별) 붙이기
def build_monthly_detail(df_src, years, caption="Normal"):
    # 상품 열은 제거
    df_detail = df_src[df_src['연'].isin(years)].copy()
    df_detail = df_detail.drop(columns=['상품'], errors='ignore')
    # 보기 좋게 정렬/컬럼 순서
    cols = ['연','월','월평균기온'] + [c for c in CATE_COLS if c in df_detail.columns]
    exist = [c for c in cols if c in df_detail.columns]
    df_detail = df_detail[exist].sort_values(['연','월'])
    pretty_table(df_detail, caption=caption)
    yearly_total_table(df_detail)

# 상단 3가지 표(예: Normal / Best / Cons.)는 기존 로직 사용하되,
# 아래처럼 호출만 바꿔주면 됨. 여기서는 Normal만 예시로 보여줌.
st.markdown("### 🎯 Normal")
build_monthly_detail(df_raw, years_pred, caption="Normal (월별)")

# 필요하면 Best/Cons도 같은 방식으로 호출:
# st.markdown("### 🟢 Best")
# build_monthly_detail(df_best, years_pred, caption="Best (월별)")
# st.markdown("### 🔴 Cons.")
# build_monthly_detail(df_cons, years_pred, caption="Cons. (월별)")

# =========================================
# 엑셀 다운로드 (원래 있던 기능 복구)
# =========================================
def to_excel_bytes(df_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, dfx in df_dict.items():
            dfx.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()

st.markdown("### ⬇️ 결과 엑셀 다운로드")
export_normal_detail = df_raw[df_raw['연'].isin(years_pred)].drop(columns=['상품'], errors='ignore')
export_normal_yearly = export_normal_detail.groupby('연', as_index=False)[[c for c in CATE_COLS if c in export_normal_detail.columns]].sum(numeric_only=True)
export_normal_yearly.insert(1,'월',"")
export_normal_yearly.insert(2,'월평균기온',"")

xlsx_bytes = to_excel_bytes({
    "Normal_월별": export_normal_detail,
    "Normal_연도별총계": export_normal_yearly
})
st.download_button(
    label="📥 엑셀 내려받기",
    data=xlsx_bytes,
    file_name="도시가스_예측_결과.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
