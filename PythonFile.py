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
st.set_page_config(page_title="AlphaChart AI", page_icon="📈", layout="wide") # 아이콘도 캔들로 변경

# 세션 상태 초기화
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

def select_pattern(path):
    st.session_state.selected_path = path

# 2. 디자인 시스템 (캔들스틱 AI 심볼 + 프리미엄 모바일 UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * { font-family: 'Pretendard', sans-serif; }
    
    /* 배경 및 프리미엄 헤더 */
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #020617 0%, #1e293b 100%);
        padding: 45px 15px; border-radius: 30px; color: white; margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }

    /* 캔들스틱 AI 심볼 스타일 (글자 앞에 배치용) */
    .brand-header-area {
        display: flex; align-items: center; justify-content: center; gap: 15px;
        flex-wrap: wrap; /* 모바일 대응 */
    }

    .candlestick-ai-logo {
        width: 100px; /* 글자와 어울리는 크기 */
        border-radius: 15px; /* 부드러운 모서리 */
        filter: drop-shadow(0 0 15px rgba(0, 255, 255, 0.7)); /* 청록색 빛 효과 */
        animation: float-glow 3s infinite ease-in-out;
    }
    
    @keyframes float-glow {
        0% { transform: translateY(0px); filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)); }
        50% { transform: translateY(-10px); filter: drop_shadow(0 0 20px rgba(0, 255, 255, 0.9)); }
        100% { transform: translateY(0px); filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)); }
    }

    .brand-title {
        font-size: 42px; font-weight: 900; margin: 0;
        background: linear-gradient(to right, #ffffff, #00ffff); /* 청록색 그라데이션 */
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .brand-subtitle { font-size: 14px; opacity: 0.7; margin-top: 10px; letter-spacing: 2px; color: #94a3b8; }
    
    /* 카드 및 리스트 디자인 */
    .stock-card {
        padding: 20px; border-radius: 18px; background: white; margin-bottom: 12px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
    .match-badge {
        background: #008080; /* 진한 청록색 */
        color: white; padding: 4px 12px; border-radius: 10px; font-weight: 800; font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_krx_list():
    try: return fdr.StockListing('KRX')[['Code', 'Name']].values.tolist()
    except: return [["005930", "삼성전자"]]

stock_list = get_krx_list()

# --- 분석 로직 (엔진) ---
def extract_features_engine(img_input, is_file_path=False):
    try:
        if is_file_path:
            img_array = np.fromfile(img_input, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                                cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])))
        mask_b = cv2.bitwise_or(cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255])),
                                cv2.inRange(hsv, np.array([70, 50, 50]), np.array([90, 255, 255]))) # 초록색 캔들 인식 추가
        mask_k = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r, mask_b), mask_k)
        height, width = combined.shape
        p_avg, p_low, colors, valid_x = [], [], [], []
        for x in range(width):
            px = np.where(combined[:, x] > 0)[0]
            if len(px) > 0:
                p_avg.append(height - np.mean(px)); p_low.append(height - np.max(px)); valid_x.append(x)
                rk, bk, kk = mask_r[:, x].sum(), mask_b[:, x].sum(), mask_k[:, x].sum()
                if kk > max(rk, bk): colors.append(0)
                elif rk > bk: colors.append(1) # 빨간색 (하락)
                else: colors.append(-1) # 파란색/초록색 (상승)
        if not p_avg: return None
        diffs = np.diff(valid_x); candle_count = 1
        for d in diffs:
            if d > 2: candle_count += 1
        combined_p = (np.array(p_avg) * 0.7) + (np.array(p_low) * 0.3)
        res_p = np.interp(np.linspace(0, len(combined_p)-1, 50), np.arange(len(combined_p)), combined_p)
        last_c = np.mean(colors[-10:])
        target_color = 0 if abs(last_c) < 0.3 else (1 if last_c > 0 else -1) # 0: 무색, 1: 빨강, -1: 파랑/초록
        return res_p, candle_count, target_color, img
    except: return None

def analyze_stock(code, name, user_p_norm, n_days, target_color):
    try:
        df = fdr.DataReader(code).tail(n_days + 5)
        if len(df) < n_days: return None
        df_t = df.tail(n_days)
        chg = (df_t['Close'].iloc[-1] - df_t['Open'].iloc[-1]) / (df_t['Open'].iloc[-1] + 1e-5)
        act_c = 0 if abs(chg) < 0.005 else (1 if chg > 0 else -1) # 0: 보합, 1: 상승, -1: 하락
        
        # 패턴의 색상 방향과 실제 종목의 색상 방향이 다르면 필터링
        if target_color != 0 and act_c != target_color: return None 

        flow = (df_t['Close'].values * 0.7) + (df_t['Low'].values * 0.3)
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        return {'code': code, 'name': name, 'sim': (pearsonr(user_p_norm, s_res)[0]+1)*50, 'price': df_t['Close'].iloc[-1]}
    except: return None

# --- UI 상단 (캔들스틱 AI 심볼 + 글자 수평 배치) ---
st.markdown(f"""
    <div class="brand-container">
        <div class="brand-header-area">
            <img src="https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png" class="candlestick-ai-logo">
            <h1 class="brand-title">AlphaChart AI</h1>
        </div>
        <div class="brand-subtitle">PREMIUM INTELLIGENT PATTERN SCANNER</div>
    </div>
    """, unsafe_allow_html=True)

# 1단계: 검색 범위 설정
st.markdown("### 🧭 1단계: 검색 범위 설정")
search_range = st.select_slider(
    "AI가 스캔할 종목 수 (무료 버전은 200개 권장)",
    options=[100, 200, 500, 1000, 2000, 3000],
    value=200
)

# 2단계: 패턴 선택
st.markdown("### 💡 2단계: 패턴 선택")
st.markdown('<div class="example-container">', unsafe_allow_html=True)
col_ex1, col_ex2 = st.columns(2)
# 예시 이미지 파일명은 사용자의 GitHub 저장소에 있는 실제 파일명으로 가정
ex1_name = "장대양봉 중간 지키며 상승.jpg" 
ex2_name = "하락후 바닥에서 양봉.jpg"

with col_ex1:
    if os.path.exists(ex1_name):
        st.image(ex1_name, use_container_width=True)
        st.button("패턴 A 선택", on_click=select_pattern, args=(ex1_name,), use_container_width=True)
with col_ex2:
    if os.path.exists(ex2_name):
        st.image(ex2_name, use_container_width=True)
        st.button("패턴 B 선택", on_click=select_pattern, args=(ex2_name,), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("이미지 직접 업로드", type=['png', 'jpg', 'jpeg'])
if uploaded_file: st.session_state.selected_path = None

input_target = uploaded_file if uploaded_file else st.session_state.selected_path
is_path = True if (not uploaded_file and st.session_state.selected_path) else False

if input_target:
    feat = extract_features_engine(input_target, is_file_path=is_path)
    if feat:
        user_p, n_days, target_color, original_img = feat
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        
        # 패턴 분석 결과 정보 표시
        color_info = "종가 기준 상승 패턴" if target_color == -1 else ("종가 기준 하락 패턴" if target_color == 1 else "종가 기준 보합/혼합 패턴")
        st.info(f"✓ {n_days}거래일 패턴 분석 완료: {color_info}")
        
        c1, c2 = st.columns(2)
        with c1: st.image(original_img, use_container_width=True, caption="기준 패턴")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(user_p_norm, color='#00ffff', lw=8); ax.axis('off'); fig.patch.set_alpha(0) # 캔들 색상에 맞춰 청록색
            st.pyplot(fig)

        if st.button(f"🚀 AI 통합 검색 시작", type="primary", use_container_width=True):
            results = []
            prog = st.progress(0)
            with ThreadPoolExecutor(max_workers=30) as ex:
                futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm, n_days, target_color) for s in stock_list[:search_range]]
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    if res: results.append(res)
                    if i % 10 == 0: prog.progress(min((i+1)/search_range, 1.0))
            
            results.sort(key=lambda x: x['sim'], reverse=True)
            st.markdown("### 🏆 AI 매칭 결과 Top 15")
            if not results:
                st.warning("일치하는 패턴을 찾을 수 없습니다. 검색 범위를 늘리거나 다른 패턴을 시도해보세요.")
            for i, res in enumerate(results[:15]):
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:18px; font-weight:800; color:#1e3a8a;">{i+1}. {res['name']}</span>
                        <span class="match-badge">{res['sim']:.1f}% Match</span>
                    </div>
                    <div style="font-size:14px; margin-top:5px; color:#64748b;">CODE: {res['code']} | <b>{res['price']:,.0f}원</b></div>
                    <a href="https://finance.naver.com/item/main.naver?code={res['code']}" target="_blank" 
                       style="display:inline-block; margin-top:10px; color:#008080; text-decoration:none; font-weight:700;">상세 분석 →</a>
                </div>
                """, unsafe_allow_html=True)

st.sidebar.caption("© 2026 AlphaChart AI v2.6")
