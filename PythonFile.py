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
st.set_page_config(page_title="AlphaChart AI", page_icon="📈", layout="wide")

# 세션 상태 초기화
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

def select_pattern(path):
    st.session_state.selected_path = path

# 2. 프리미엄 디자인 시스템 (빛나는 효과 및 애니메이션)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800&display=swap');
    * { font-family: 'Pretendard', sans-serif; }
    
    /* 배경 및 컨테이너 */
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
        padding: 40px 15px; border-radius: 30px; color: white; margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* 빛나는 심볼 애니메이션 */
    @keyframes glow-pulse {
        0% { filter: drop-shadow(0 0 5px #60a5fa); transform: scale(1); }
        50% { filter: drop-shadow(0 0 20px #60a5fa); transform: scale(1.05); }
        100% { filter: drop-shadow(0 0 5px #60a5fa); transform: scale(1); }
    }
    
    .premium-symbol {
        font-size: 50px;
        animation: glow-pulse 3s infinite ease-in-out;
        margin-bottom: 10px;
    }

    /* 빛나는 텍스트 효과 */
    .brand-title {
        font-size: 42px; font-weight: 800; margin: 0;
        background: linear-gradient(to right, #ffffff, #60a5fa, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(96, 165, 250, 0.4);
    }
    
    .brand-subtitle { font-size: 15px; opacity: 0.7; margin-top: 10px; letter-spacing: 1px; }
    
    /* 카드 디자인 */
    .stock-card {
        padding: 20px; border-radius: 18px; background: white; margin-bottom: 15px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .match-badge {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white; padding: 5px 12px; border-radius: 10px;
        font-weight: 700; font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_krx_list():
    try:
        return fdr.StockListing('KRX')[['Code', 'Name']].values.tolist()
    except:
        return [["005930", "삼성전자"]]

stock_list = get_krx_list()

# --- 분석 로직 (기존 엔진 보존) ---
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
        mask_b = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
        mask_k = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r, mask_b), mask_k)
        height, width = combined.shape
        p_avg, p_low, colors, valid_x = [], [], [], []
        for x in range(width):
            px = np.where(combined[:, x] > 0)[0]
            if len(px) > 0:
                p_avg.append(height - np.mean(px))
                p_low.append(height - np.max(px))
                valid_x.append(x)
                rk, bk, kk = mask_r[:, x].sum(), mask_b[:, x].sum(), mask_k[:, x].sum()
                if kk > max(rk, bk): colors.append(0)
                elif rk > bk: colors.append(1)
                else: colors.append(-1)
        if not p_avg: return None
        diffs = np.diff(valid_x); candle_count = 1
        for d in diffs:
            if d > 2: candle_count += 1
        combined_p = (np.array(p_avg) * 0.7) + (np.array(p_low) * 0.3)
        res_p = np.interp(np.linspace(0, len(combined_p)-1, 50), np.arange(len(combined_p)), combined_p)
        last_c = np.mean(colors[-10:])
        target_color = 0 if abs(last_c) < 0.3 else (1 if last_c > 0 else -1)
        return res_p, candle_count, target_color, img
    except: return None

def analyze_stock(code, name, user_p_norm, n_days, target_color):
    try:
        df = fdr.DataReader(code).tail(n_days + 5)
        if len(df) < n_days: return None
        df_t = df.tail(n_days)
        chg = (df_t['Close'].iloc[-1] - df_t['Open'].iloc[-1]) / (df_t['Open'].iloc[-1] + 1e-5)
        act_c = 0 if abs(chg) < 0.005 else (1 if chg > 0 else -1)
        if target_color != 0 and act_c != target_color: return None
        flow = (df_t['Close'].values * 0.7) + (df_t['Low'].values * 0.3)
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        return {'code': code, 'name': name, 'sim': (pearsonr(user_p_norm, s_res)[0]+1)*50, 'price': df_t['Close'].iloc[-1]}
    except: return None

# --- UI 상단 ---
st.markdown("""
    <div class="brand-container">
        <div class="premium-symbol">🚀</div>
        <h1 class="brand-title">AlphaChart AI</h1>
        <div class="brand-subtitle">AI 신경망 기반 정밀 차트 패턴 분석 시스템</div>
    </div>
    """, unsafe_allow_html=True)

# 1단계: 검색 범위 설정 (100단위 조절 및 무료 버전 제한)
st.markdown("### 🧭 1단계: 검색 범위 설정")
search_range = st.select_slider(
    "AI가 스캔할 종목 수 (무료버전 상한: 200개)",
    options=[100, 200, 500, 1000, 2000, 3000],
    value=200
)

if search_range > 200:
    st.warning("⚠️ 200개 초과 검색 시 분석 시간이 길어질 수 있습니다.")

st.markdown("### 💡 2단계: 패턴 선택")
st.markdown('<div class="example-container">', unsafe_allow_html=True)
col_ex1, col_ex2 = st.columns(2)
ex1_name = "장대양봉 중간 지키며 상승.jpg"
ex2_name = "하락후 바닥에서 양봉.jpg"

with col_ex1:
    if os.path.exists(ex1_name):
        st.image(ex1_name, use_container_width=True)
        st.button("장대양봉 지지", on_click=select_pattern, args=(ex1_name,), use_container_width=True)
with col_ex2:
    if os.path.exists(ex2_name):
        st.image(ex2_name, use_container_width=True)
        st.button("바닥 반등", on_click=select_pattern, args=(ex2_name,), use_container_width=True)
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

        st.info(f"✓ {n_days}거래일 패턴 탐지 완료")
        c1, c2 = st.columns(2)
        with c1: st.image(original_img, use_container_width=True, caption="선택된 패턴")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(user_p_norm, color='#3b82f6', lw=8)
            ax.axis('off'); fig.patch.set_alpha(0)
            st.pyplot(fig)

        if st.button(f"🚀 AI 통합 검색 시작 ({search_range}개 스캔)", type="primary", use_container_width=True):
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
            for i, res in enumerate(results[:15]):
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:18px; font-weight:800; color:#1e40af;">{i+1}. {res['name']}</span>
                        <span class="match-badge">{res['sim']:.1f}% Match</span>
                    </div>
                    <div style="font-size:14px; margin-top:5px; color:#64748b;">CODE: {res['code']} | <b>{res['price']:,.0f}원</b></div>
                    <a href="https://finance.naver.com/item/main.naver?code={res['code']}" target="_blank" 
                       style="display:inline-block; margin-top:10px; color:#3b82f6; text-decoration:none; font-weight:700; font-size:13px;">차트 분석하기 →</a>
                </div>
                """, unsafe_allow_html=True)

st.sidebar.caption("© 2026 AlphaChart AI Premium v2.2")
