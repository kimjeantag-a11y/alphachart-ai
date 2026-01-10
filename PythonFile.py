import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 앱 설정
st.set_page_config(
    page_title="AlphaChart AI", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 파일명 정의
ex1_name = "장대양봉 중간 지키며 상승.jpg"
ex2_name = "급락후 바닥에서 양봉.jpg" 

if 'selected_path' not in st.session_state:
    st.session_state.selected_path = ex2_name

def select_pattern(path):
    st.session_state.selected_path = path

# 2. 프리미엄 디자인 시스템 (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * { font-family: 'Pretendard', sans-serif; -webkit-tap-highlight-color: transparent; }
    .stApp { background-color: #f8fafc; }
    
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #020617 0%, #1e293b 100%);
        padding: 40px 15px; border-radius: 25px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4); text-align: center;
    }
    .candlestick-ai-logo {
        width: 80px; margin-bottom: 15px;
        filter: drop-shadow(0 0 15px rgba(0, 255, 255, 0.7));
        animation: float-glow 3s infinite ease-in-out;
    }
    @keyframes float-glow {
        0%, 100% { transform: translateY(0px); filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)); }
        50% { transform: translateY(-8px); filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.9)); }
    }
    .brand-title {
        font-size: 36px; font-weight: 900; margin: 0;
        background: linear-gradient(to right, #ffffff, #00ffff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .brand-subtitle { font-size: 14px; color: #00ffff; letter-spacing: 3px; font-weight: 700; margin-top: 10px; }

    /* 핵심 안내문(Mission) 스타일 */
    .mission-box {
        background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0;
        margin-bottom: 2rem; line-height: 1.8; color: #334155; font-size: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .mission-highlight { color: #0ea5e9; font-weight: 700; }

    .method-desc { 
        background: #f1f5f9; padding: 15px; border-radius: 12px; color: #475569; 
        font-size: 13px; border-left: 4px solid #00ffff; margin-bottom:10px; line-height: 1.6;
    }
    .menu-card { background: white; border: 1px solid #e2e8f0; border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .result-card { padding: 15px; border-radius: 18px; background: white; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    .compact-img img { max-height: 120px !important; width: auto !important; border-radius: 8px; }
    .naver-link {
        display: inline-block; margin-top: 8px; padding: 6px 14px;
        background-color: #03c75a; color: white !important;
        border-radius: 8px; font-size: 12px; font-weight: 700; text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_krx_list():
    try: return fdr.StockListing('KRX')[['Code', 'Name']].values.tolist()
    except: return [["005930", "삼성전자"]]

stock_list = get_krx_list()

# --- 분석 엔진 ---
def extract_features_engine(img_input, is_file_path=False):
    try:
        if is_file_path:
            img_array = np.fromfile(img_input, np.uint8); img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8); img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                                cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])))
        mask_b = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        mask_k = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r, mask_b), mask_k)
        height, width = combined.shape
        p_avg = []
        for x in range(width):
            px = np.where(combined[:, x] > 0)[0]
            if len(px) > 0: p_avg.append(height - np.mean(px))
        if not p_avg: return None
        res_p = np.interp(np.linspace(0, len(p_avg)-1, 50), np.arange(len(p_avg)), np.array(p_avg))
        return res_p, img
    except: return None

def analyze_stock(code, name, user_p_norm, n_days=20):
    try:
        df = fdr.DataReader(code).tail(n_days + 5)
        if len(df) < n_days: return None
        df_t = df.tail(n_days); flow = df_t['Close'].values
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        return {'code': code, 'name': name, 'sim': (pearsonr(user_p_norm, s_res)[0]+1)*50, 'price': df_t['Close'].iloc[-1]}
    except: return None

# --- UI 메인 ---
st.markdown(f"""
    <div class="brand-container">
        <img src="https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png" class="candlestick-ai-logo">
        <div class="brand-title">AlphaChart AI</div>
        <div class="brand-subtitle">CANDLE CHART DOPPELGÄNGER</div>
    </div>
    """, unsafe_allow_html=True)

# [핵심 안내문 추가]
st.markdown(f"""
    <div class="mission-box">
        오랜 주식 거래 역사를 볼 때, 최근 캔들의 단순한 형태보다는 수거래일 동안의 <span class="mission-highlight">추세와 최근 캔들의 형태를 함께 보는 것</span>이 중요하다는 것이 수많은 연구자와 투자자들로부터 검증되어 왔습니다.<br><br>
        과거의 패턴을 통한 미래의 패턴 예측, 그 중에서도 우리는 오늘까지의 차트를 보고 내일의 캔들을 알고 싶습니다. 일일이 3천여개의 모든 종목을 찾아서 내가 원하는 차트와 유사한 종목을 찾고, 증권사 PC 프로그램을 통해 원하는 그림을 그리거나 조건을 입력해서 검색해 왔죠. 그러나 시간만 오래 걸리고 불편하고 부정확했습니다.<br><br>
        이제 <span class="mission-highlight">AlphaChart AI</span>가 몇 분만에 도플갱어 종목들을 찾아줄 수 있습니다. 또, 그동안 차트 매매의 대가들이 정립해 놓은 검증된 패턴들을 기본 장착하여 사용자가 가져온 차트 업로드 없이도 바로 내일이나 모레 살만한 종목 후보군을 찾을 수 있게 되었습니다. 차트 매매를 주로 하시는 데이 트레이더, 기업가치와 함께 차트를 같이 보시는 단중기 트레이더 모두 AlphaChart AI를 통해 <span class="mission-highlight">불필요한 시간 투입 없이, 성공확률을 극대화</span> 하시길 기원합니다.
    </div>
    """, unsafe_allow_html=True)

# 1단계
st.markdown("### 🧭 1단계: 검색 범위 설정 (Free Version)")
search_limit = st.slider("무료 버전은 최대 200개 종목 스캔을 지원합니다.", 50, 200, 200, step=10)
st.markdown(f"""<div style="text-align:right; font-size:11px; color:#ef4444; margin-top:-10px; font-weight:700;">🔒 Pro 전용: 전종목 3,000개 스캔 가능</div>""", unsafe_allow_html=True)

st.markdown("---")

# 2단계 제1방법
st.markdown("### 💡 2단계 제1방법: AI 추천 패턴 선택")
st.markdown("""
    <div class="method-desc">
        이 패턴은 AlphaChart AI가 추천하는 검증된 급등 지속 또는 급락 후 반등 패턴입니다.<br>
        이러한 패턴을 보이는 <b>도플갱어 종목</b>을 찾아 단타, 스윙의 성공률을 높여보세요.
    </div>
    """, unsafe_allow_html=True)
col_p1, col_p2 = st.columns(2)
with col_p1:
    if os.path.exists(ex1_name):
        st.markdown('<div class="menu-card compact-img">', unsafe_allow_html=True)
        st.image(ex1_name, caption="패턴 A: 급등 지속"); st.button("패턴 A 선택", on_click=select_pattern, args=(ex1_name,), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with col_p2:
    if os.path.exists(ex2_name):
        st.markdown('<div class="menu-card compact-img">', unsafe_allow_html=True)
        st.image(ex2_name, caption="패턴 B: 급락 후 반등"); st.button("패턴 B 선택", on_click=select_pattern, args=(ex2_name,), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# 2단계 제2방법
st.markdown("### 📷 2단계 제2방법: 관심 차트 직접 업로드")
st.markdown('<div class="method-desc">이동평균선이 없는 캔들 위주의 깔끔한 차트일수록 정확도가 높습니다.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

input_target = uploaded_file if uploaded_file else st.session_state.selected_path
is_path = True if (not uploaded_file and st.session_state.selected_path) else False

if input_target:
    feat = extract_features_engine(input_target, is_file_path=is_path)
    if feat:
        user_p, original_img = feat
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        
        st.markdown("<div style='font-size:13px; font-weight:700; color:#0ea5e9; margin-top:10px; margin-bottom:5px;'>🎯 분석 대상 패턴 (Compact View)</div>", unsafe_allow_html=True)
        view_c1, view_c2, view_c3 = st.columns([1, 1.2, 2])
        with view_c1:
            st.markdown('<div class="compact-img">', unsafe_allow_html=True); st.image(original_img); st.markdown('</div>', unsafe_allow_html=True)
        with view_c2:
            fig, ax = plt.subplots(figsize=(2.5, 1.2)); ax.plot(user_p_norm, color='#00ffff', lw=5); ax.axis('off'); fig.patch.set_alpha(0); st.pyplot(fig)
        with view_c3:
            st.write(""); search_btn = st.button(f"🚀 AI 통합 검색 시작", type="primary", use_container_width=True)

        if search_btn:
            results = []
            prog = st.progress(0)
            with ThreadPoolExecutor(max_workers=30) as ex:
                futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm) for s in stock_list[:search_limit]]
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    if res: results.append(res)
                    if i % 10 == 0: prog.progress(min((i+1)/search_limit, 1.0))
            
            results.sort(key=lambda x: x['sim'], reverse=True)
            st.markdown(f"### 🏆 매칭 결과 Top 10")
            for i, res in enumerate(results[:10]):
                naver_url = f"https://finance.naver.com/item/main.naver?code={res['code']}"
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <b>{i+1}. {res['name']}</b><br><small>{res['code']} | {res['price']:,.0f}원</small>
                        </div>
                        <span style="color:#008080; font-weight:800; font-size:18px;">{res['sim']:.1f}% Match</span>
                    </div>
                    <a href="{naver_url}" target="_blank" class="naver-link">네이버 증권 상세 정보 ↗</a>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #0ea5e9; border-radius: 20px; padding: 25px; margin-top: 40px; text-align: center; color: #0369a1;">
        <h4 style="margin-top:0;">🚀 Upgrade to PRO Version</h4>
        <p style="font-size:14px; font-weight:500;">전종목 <b>3,000개 실시간 스캔</b>과 더 정밀한 AI 엔진, AlphaChart AI가 추천하는 급등, 급락 상황에서의 <b>검증된 다양한 패턴들</b>을 만나보세요.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("AlphaChart AI v5.2 | Candle Chart Doppelgänger")
