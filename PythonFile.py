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

# 1. 앱 설정 (반응형 레이아웃 강화)
st.set_page_config(
    page_title="AlphaChart AI", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed" # 모바일 가독성을 위해 사이드바는 접어둠
)

# 2. 세션 상태 초기화
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

def select_pattern(path):
    st.session_state.selected_path = path

# 3. 디자인 시스템 (모바일 최적화 및 로고 강조 CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800&display=swap');
    * { font-family: 'Pretendard', sans-serif; }
    
    /* 로고 및 브랜드 컨테이너 (모바일 가변 패딩) */
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        padding: 40px 15px; border-radius: 20px; color: white; margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .brand-symbol { font-size: 60px; margin-bottom: 10px; }
    .brand-title { font-size: 42px; font-weight: 800; letter-spacing: -1px; margin: 0; }
    .brand-subtitle { font-size: 16px; opacity: 0.8; margin-top: 5px; }
    
    /* 모바일 세로 대응: 카드 레이아웃 */
    .stock-card {
        padding: 20px; border-radius: 15px; background: white; margin-bottom: 15px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .match-badge {
        background: #3b82f6; color: white; padding: 4px 10px; border-radius: 8px;
        font-weight: 700; font-size: 14px;
    }
    
    /* 메인 컨테이너 여백 최적화 */
    .block-container { padding-top: 1.5rem !important; }
    
    /* 버튼 스타일 커스텀 */
    .stButton>button { width: 100%; border-radius: 10px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- 로직 섹션 (기존 코드 유지) ---

@st.cache_data
def get_krx_list():
    try:
        return fdr.StockListing('KRX')[['Code', 'Name']].values.tolist()
    except:
        return [["005930", "삼성전자"]]

stock_list = get_krx_list()

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
        diffs = np.diff(valid_x)
        candle_count = 1
        for d in diffs:
            if d > 2: candle_count += 1
        
        detected_n = candle_count
        combined_p = (np.array(p_avg) * 0.7) + (np.array(p_low) * 0.3)
        res_p = np.interp(np.linspace(0, len(combined_p)-1, 50), np.arange(len(combined_p)), combined_p)
        last_c = np.mean(colors[-10:])
        target_color = 0 if abs(last_c) < 0.3 else (1 if last_c > 0 else -1)
        return res_p, detected_n, target_color, img
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

# --- UI 레이아웃 시작 ---

# 상단 브랜드 헤더 (심볼 강조)
st.markdown("""
    <div class="brand-container">
        <div class="brand-symbol">📈</div>
        <h1 class="brand-title">AlphaChart AI</h1>
        <p class="brand-subtitle">AI 신경망 기반 정밀 차트 패턴 분석 시스템</p>
    </div>
    """, unsafe_allow_html=True)

# 예시 패턴 섹션 (가로 배치 -> 모바일 자동 대응)
st.markdown("### 💡 추천 예시 패턴")
col_ex1, col_ex2 = st.columns(2)

ex1_name = "장대양봉 중간 지키며 상승.jpg"
ex2_name = "하락후 바닥에서 양봉.jpg"

with col_ex1:
    if os.path.exists(ex1_name):
        st.image(ex1_name, caption="장대양봉 지지형", use_container_width=True)
        st.button("분석 실행 (패턴 1)", key="btn_ex1", on_click=select_pattern, args=(ex1_name,))
    else: st.info("샘플1 준비 중")

with col_ex2:
    if os.path.exists(ex2_name):
        st.image(ex2_name, caption="바닥 반등형", use_container_width=True)
        st.button("분석 실행 (패턴 2)", key="btn_ex2", on_click=select_pattern, args=(ex2_name,))
    else: st.info("샘플2 준비 중")

# 사이드바 설정
st.sidebar.header("🧭 분석 제어")
search_range = st.sidebar.slider("AI 스캔 종목 수", 100, 3000, 1000, 100)
uploaded_file = st.sidebar.file_uploader("이미지 직접 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    st.session_state.selected_path = None

input_target = uploaded_file if uploaded_file else st.session_state.selected_path
is_path = True if (not uploaded_file and st.session_state.selected_path) else False

# 분석 결과 섹션
if input_target:
    feat = extract_features_engine(input_target, is_file_path=is_path)
    if feat:
        user_p, n_days, target_color, original_img = feat
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()

        st.markdown("---")
        st.subheader("🎯 선택된 패턴 분석")
        
        # 모바일 가독성을 위해 3열 -> 모바일에서는 위아래로 쌓임
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        with c1: 
            st.image(original_img, caption="원본 이미지", use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(user_p_norm, color='#3b82f6', lw=8)
            ax.axis('off')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
        with c3:
            st.markdown(f"""
            **분석 요약**
            - **범위**: {n_days} 거래일 탐지
            - **타겟**: {'🔴 양봉 마감' if target_color==1 else '🔵 음봉 마감' if target_color==-1 else '⚫ 도지 마감'}
            - **상태**: 패턴 추출 완료
            """)
            if st.button("🚀 AI 통합 검색 시작", type="primary"):
                results = []
                prog = st.progress(0)
                with ThreadPoolExecutor(max_workers=30) as ex:
                    futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm, n_days, target_color) for s in stock_list[:search_range]]
                    for i, f in enumerate(as_completed(futures)):
                        res = f.result()
                        if res: results.append(res)
                        if i % 20 == 0: prog.progress((i+1)/search_range)
                
                results.sort(key=lambda x: x['sim'], reverse=True)
                st.session_state.search_results = results[:15]

        # 매칭 결과 표시
        if 'search_results' in st.session_state:
            st.markdown("### 🏆 AI 정밀 매칭 결과")
            for i, res in enumerate(st.session_state.search_results):
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #3b82f6;">{i+1}. {res['name']}</span>
                            <span style="font-size: 0.9rem; color: #64748b; margin-left: 10px;">{res['code']}</span>
                        </div>
                        <div class="match-badge">{res['sim']:.1f}%</div>
                    </div>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 1.1rem; font-weight: 700;">{res['price']:,.0f}원</div>
                        <a href="https://finance.naver.com/item/main.naver?code={res['code']}" target="_blank" 
                           style="color: #3b82f6; text-decoration: none; font-size: 0.9rem; font-weight: 600;">차트 상세보기 →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 하단 푸터
st.markdown("---")
st.caption("© 2026 AlphaChart AI | 신경망 패턴 분석 시스템 v2.0")
