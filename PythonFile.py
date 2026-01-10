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

# [사용자 설정 반영] 파일명 정의
ex1_name = "장대양봉 중간 지키며 상승.jpg"
ex2_name = "급락후 바닥에서 양봉.jpg" 

# 세션 상태 초기화
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = ex2_name

def select_pattern(path):
    st.session_state.selected_path = path

# 2. 최상급 디자인 시스템 (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * { font-family: 'Pretendard', sans-serif; -webkit-tap-highlight-color: transparent; }
    
    .stApp { background-color: #f8fafc; }
    .main .block-container { padding-top: 2rem !important; }

    /* 프리미엄 헤더 & 애니메이션 심볼 */
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #020617 0%, #1e293b 100%);
        padding: 40px 15px; border-radius: 25px; color: white; margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4); text-align: center;
    }

    .candlestick-ai-logo {
        width: 80px; margin-bottom: 10px;
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
    
    .brand-subtitle {
        font-size: 14px; color: #00ffff; letter-spacing: 3px; font-weight: 700; margin-top: 10px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }

    /* 해설 텍스트 스타일 */
    .method-desc {
        background: #f1f5f9; padding: 15px; border-radius: 12px;
        color: #475569; font-size: 14px; line-height: 1.6; margin-bottom: 15px;
        border-left: 4px solid #00ffff;
    }

    /* Pro 버전 홍보 박스 */
    .pro-promo-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9; border-radius: 20px; padding: 25px;
        margin-top: 40px; text-align: center; color: #0369a1;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
    }

    .menu-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 20px; padding: 12px; margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    }
    .stImage > img { max-height: 160px !important; border-radius: 12px; }
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
            if not os.path.exists(img_input): return None
            img_array = np.fromfile(img_input, np.uint8); img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8); img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50])), cv2.inRange(hsv, np.array([170, 50, 50])))
        mask_b = cv2.bitwise_or(cv2.inRange(hsv, np.array([100, 50, 50])), cv2.inRange(hsv, np.array([70, 50, 50])))
        mask_k = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r, mask_b), mask_k)
        height, width = combined.shape
        p_avg, p_low, colors = [], [], []
        for x in range(width):
            px = np.where(combined[:, x] > 0)[0]
            if len(px) > 0:
                p_avg.append(height - np.mean(px)); p_low.append(height - np.max(px))
                rk, bk, kk = mask_r[:, x].sum(), mask_b[:, x].sum(), mask_k[:, x].sum()
                if kk > max(rk, bk): colors.append(0)
                elif rk > bk: colors.append(1)
                else: colors.append(-1)
        res_p = np.interp(np.linspace(0, len(p_avg)-1, 50), np.arange(len(p_avg)), (np.array(p_avg)*0.7 + np.array(p_low)*0.3))
        last_c = np.mean(colors[-10:]); target_color = 0 if abs(last_c) < 0.3 else (1 if last_c > 0 else -1)
        return res_p, len(p_avg)//5, target_color, img
    except: return None

def analyze_stock(code, name, user_p_norm, n_days, target_color):
    try:
        df = fdr.DataReader(code).tail(n_days + 5)
        if len(df) < n_days: return None
        df_t = df.tail(n_days); flow = (df_t['Close'].values * 0.7) + (df_t['Low'].values * 0.3)
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        return {'code': code, 'name': name, 'sim': (pearsonr(user_p_norm, s_res)[0]+1)*50, 'price': df_t['Close'].iloc[-1]}
    except: return None

# --- UI 메인 ---
st.markdown(f"""
    <div class="brand-container">
        <img src="https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png" class="candlestick-ai-logo">
        <div class="brand-title">AlphaChart AI</div>
        <div class="brand-subtitle">Candle Chart Doppelgänger</div>
    </div>
    """, unsafe_allow_html=True)

# 1단계
st.markdown("### 🧭 1단계: 검색 범위 설정 (Free Version)")
search_limit = st.slider("현재 무료 버전은 최대 200개 종목 스캔을 지원합니다.", 50, 200, 200, step=10)
st.markdown(f"""<div style="text-align:right; font-size:12px; color:#ef4444; margin-top:-10px;">🔒 Pro 전용: 201 ~ 3000 종목 (잠김)</div>""", unsafe_allow_html=True)

st.markdown("---")

# 2단계 제1방법
st.markdown("### 💡 2단계 제1방법: AI추천패턴 선택")
st.markdown("""
    <div class="method-desc">
        이 패턴은 AlphaChart AI가 추천하는 전형적인 급등 지속 또는 급락 후 반등 패턴입니다. <br>
        이러한 패턴을 보이는 <b>도플갱어 종목</b>을 찾아 단타, 스윙의 성공률을 높일 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

col_p1, col_p2 = st.columns(2)
with col_p1:
    if os.path.exists(ex1_name):
        st.markdown('<div class="menu-card">', unsafe_allow_html=True)
        st.image(ex1_name, caption="급등 지속 패턴")
        st.button("패턴 A 선택", on_click=select_pattern, args=(ex1_name,), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with col_p2:
    if os.path.exists(ex2_name):
        st.markdown('<div class="menu-card">', unsafe_allow_html=True)
        st.image(ex2_name, caption="급락 후 반등 패턴")
        st.button("패턴 B 선택", on_click=select_pattern, args=(ex2_name,), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# 2단계 제2방법
st.markdown("### 📷 2단계 제2방법: 관심차트 직접 업로드")
st.markdown("""
    <div class="method-desc">
        위 추천 차트처럼 이동평균선 등을 제외한 <b>캔들차트만 올리면</b> 일치도가 높습니다.
    </div>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("모바일 갤러리 또는 카메라 선택", type=['png', 'jpg', 'jpeg'])

# 입력 소스 결정
input_target = uploaded_file if uploaded_file else st.session_state.selected_path
is_path = True if (not uploaded_file and st.session_state.selected_path) else False

if input_target:
    feat = extract_features_engine(input_target, is_file_path=is_path)
    if feat:
        user_p, n_days, target_color, original_img = feat
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        
        st.info("✅ 분석 패턴 로드 완료")
        res_c1, res_c2 = st.columns(2)
        with res_c1: st.image(original_img, use_container_width=True)
        with res_c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(user_p_norm, color='#00ffff', lw=8); ax.axis('off'); fig.patch.set_alpha(0)
            st.pyplot(fig)

        if st.button(f"🚀 AI 통합 검색 시작 ({search_limit}종목)", type="primary", use_container_width=True):
            results = []
            prog = st.progress(0)
            with ThreadPoolExecutor(max_workers=30) as ex:
                futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm, 20, target_color) for s in stock_list[:search_limit]]
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    if res: results.append(res)
                    if i % 10 == 0: prog.progress(min((i+1)/search_limit, 1.0))
            
            results.sort(key=lambda x: x['sim'], reverse=True)
            st.markdown(f"### 🏆 매칭 결과 Top 10")
            for i, res in enumerate(results[:10]):
                st.markdown(f"""
                <div style="padding:15px; border-radius:15px; background:white; border:1px solid #e2e8f0; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <b>{i+1}. {res['name']}</b>
                        <span style="color:#008080; font-weight:800;">{res['sim']:.1f}% Match</span>
                    </div>
                    <small style="color:#64748b;">CODE: {res['code']} | {res['price']:,.0f}원</small>
                </div>
                """, unsafe_allow_html=True)

# Pro 버전 홍보 코멘트
st.markdown("""
    <div class="pro-promo-box">
        <h4 style="margin-top:0;">🚀 Upgrade to Pro Version</h4>
        <p style="font-size:15px; font-weight:500;">
            Pro 버전에서는 <b>KOSPI, KOSDAQ 전종목</b> 검색이 가능하고,<br>
            AlphaChart AI가 추천하는 다양한 급등, 급락 상황에서의 <b>높은 확률 패턴</b>을 만나볼 수 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.caption("AlphaChart AI v4.1 | Candle Chart Doppelgänger")
