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

# 앱 설정
st.set_page_config(page_title="AlphaChart - AI 정밀 분석", layout="wide")

# 세션 상태 초기화 및 콜백 함수
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

def select_pattern(path):
    st.session_state.selected_path = path

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800&display=swap');
    * { font-family: 'Pretendard', sans-serif; }
    
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: radial-gradient(circle at center, #1e3a8a 0%, #0f172a 100%);
        padding: 50px 20px; border-radius: 30px; color: white; margin-bottom: 40px;
    }
    
    .example-container {
        background: #f8fafc; padding: 25px; border-radius: 20px; border: 1px dashed #cbd5e1; margin-bottom: 30px;
    }
    .stock-card {
        padding: 24px; border-radius: 20px; background: white; margin-bottom: 15px;
        border: 1px solid #f1f5f9; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stock-link {
        display: inline-block; margin-top: 12px; padding: 8px 16px;
        background-color: #3b82f6; color: white; border-radius: 10px;
        font-size: 14px; font-weight: 600; text-decoration: none;
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

def extract_features_engine(img_input, is_file_path=False):
    try:
        if is_file_path:
            # 한글 경로 인식을 위한 처리
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
        # 캔들 개수 1:1 카운팅
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

# --- UI 상단 ---
st.markdown("""<div class="brand-container"><div class="brand-title">AlphaChart</div><div class="brand-subtitle">AI 신경망 기반 정밀 차트 패턴 분석 시스템</div></div>""", unsafe_allow_html=True)

st.markdown("### 💡 예시 패턴으로 바로 분석하기")
st.markdown('<div class="example-container">', unsafe_allow_html=True)
col_ex1, col_ex2 = st.columns(2)

ex1_name = "장대양봉 중간 지키며 상승.jpg"
ex2_name = "하락후 바닥에서 양봉.jpg"

with col_ex1:
    if os.path.exists(ex1_name):
        st.image(ex1_name, caption="장대양봉 중간 유지 상승", use_container_width=True)
        st.button("분석 실행 (장대양봉)", on_click=select_pattern, args=(ex1_name,))
    else: st.warning("파일 없음")

with col_ex2:
    if os.path.exists(ex2_name):
        st.image(ex2_name, caption="하락 후 바닥 양봉", use_container_width=True)
        st.button("분석 실행 (바닥 반등)", on_click=select_pattern, args=(ex2_name,))
    else: st.warning("파일 없음")
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.header("🧭 분석 설정")
search_range = st.sidebar.slider("검색 범위", 100, 3000, 1000, 100)
uploaded_file = st.sidebar.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

# 업로드 시 예시 선택 해제
if uploaded_file:
    st.session_state.selected_path = None

# 최종 분석 대상 결정
input_target = uploaded_file if uploaded_file else st.session_state.selected_path
is_path = True if (not uploaded_file and st.session_state.selected_path) else False

if input_target:
    feat = extract_features_engine(input_target, is_file_path=is_path)
    if feat:
        user_p, n_days, target_color, original_img = feat
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()

        st.info(f"✓ {n_days}거래일 패턴 분석 결과")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: st.image(original_img, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(user_p_norm, color='#3b82f6', lw=6)
            ax.axis('off')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
        with c3:
            st.success(f"**패턴 범위**: {n_days}거래일\n\n**로직**: 마지막 캔들 몸통 및 꼬리 분석\n\n**상태**: {'🔴 양봉' if target_color==1 else '🔵 음봉' if target_color==-1 else '⚫ 도지'}")

        if st.button("🚀 AI 통합 검색 시작"):
            results = []
            prog = st.progress(0)
            with ThreadPoolExecutor(max_workers=30) as ex:
                futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm, n_days, target_color) for s in stock_list[:search_range]]
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    if res: results.append(res)
                    if i % 20 == 0: prog.progress((i+1)/search_range)
            
            results.sort(key=lambda x: x['sim'], reverse=True)
            st.markdown("### 🏆 AI 매칭 결과")
            for i, res in enumerate(results[:15]):
                st.markdown(f"""
                <div class="stock-card">
                    <table style="width:100%;">
                        <tr>
                            <td style="width:10%; font-size:24px; font-weight:800; color:#3b82f6;">{i+1}</td>
                            <td style="width:40%; font-size:20px; font-weight:700;">{res['name']} <br><small>CODE: {res['code']}</small></td>
                            <td style="width:25%; font-size:19px; font-weight:700;">{res['price']:,.0f}원</td>
                            <td style="width:25%; text-align:right;"><span style="background:#3b82f6; color:white; padding:8px 12px; border-radius:10px;">{res['sim']:.1f}% Match</span></td>
                        </tr>
                    </table>
                    <a href="https://finance.naver.com/item/main.naver?code={res['code']}" target="_blank" class="stock-link">📊 상세 차트 보기</a>
                </div>
                """, unsafe_allow_html=True)