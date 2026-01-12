import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import cv2
import os
import base64
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 앱 설정
st.set_page_config(page_title="AlphaChart AI", page_icon="🦅", layout="wide", initial_sidebar_state="collapsed")

# --- 🔐 라이선스 및 세션 관리 ---
if 'is_pro' not in st.session_state:
    st.session_state.is_pro = False
if 'show_license_input' not in st.session_state:
    st.session_state.show_license_input = False
if 'detected_period' not in st.session_state:
    st.session_state.detected_period = 20

with st.sidebar:
    st.header("⚙️ Settings")
    if st.session_state.is_pro:
        st.success("✅ PRO License Active")
        if st.button("Logout / Reset", use_container_width=True):
            st.session_state.is_pro = False
            st.session_state.show_license_input = False
            st.rerun()
    else:
        st.info("현재: Free Version")
        if not st.session_state.show_license_input:
            if st.button("👑 PRO 업그레이드", use_container_width=True):
                st.session_state.show_license_input = True
                st.rerun()
        if st.session_state.show_license_input:
            with st.expander("🔑 라이선스 키 입력", expanded=True):
                license_key = st.text_input("License Key", type="password", label_visibility="collapsed")
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("확인", use_container_width=True):
                    if license_key == "alpha2026": 
                        st.session_state.is_pro = True
                        st.session_state.show_license_input = False
                        st.rerun()
                    else:
                        st.error("잘못된 키입니다.")
                if c_btn2.button("취소", use_container_width=True):
                    st.session_state.show_license_input = False
                    st.rerun()
    
    st.markdown("---")
    st.header("🕵️‍♂️ 결과 검증 (디버깅)")
    debug_code = st.text_input("종목코드 입력 (예: 272210)", placeholder="결과에 안 나올 때 확인용")
    if debug_code:
        st.warning(f"⚠️ {debug_code} 종목을 강제로 추적합니다.")

    st.markdown("---")
    st.caption("AlphaChart AI v19.1")

IS_PRO = st.session_state.is_pro

# --- 🎯 [설정] 심볼 파일명 ---
FREE_SYMBOL_URL = "https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png"
PRO_SYMBOL_FILE = "독수리 심볼.jfif"

# --- 🎯 [고정] 패턴 DB ---
PATTERN_DB = {
    "A": {"file": "장대양봉 허리 지지 상승.jpg", "name": "A. 장대양봉 허리 지지 상승", "locked": False, "type": "A"},
    "B": {"file": "급락후 바닥에서 반등.jpg", "name": "B. 급락후 바닥에서 반등", "locked": False, "type": "B"}, 
    "C": {"file": "큰하락 후 정배열, 상승 지속(컵위드핸들).jpg", "name": "C. 큰하락 후 정배열, 상승 지속 🔒", "locked": not IS_PRO, "type": "Custom"},
    "D": {"file": "쌍바닥(단기간).jpg", "name": "D. 쌍바닥(단기간) 🔒", "locked": not IS_PRO, "type": "Custom"},
    "E": {"file": "쌍바닥(상승전 시작점).jpg", "name": "E. 쌍바닥(상승전 시작점) 🔒", "locked": not IS_PRO, "type": "Custom"},
    "F": {"file": "급락후 연속 도지.jpg", "name": "F. 급락후 연속 도지 🔒", "locked": not IS_PRO, "type": "Custom"},
    "G": {"file": "횡보후 급락 및 연속도지.jpg", "name": "G. 횡보후 급락 및 연속도지 🔒", "locked": not IS_PRO, "type": "Custom"},
    "H": {"file": "하락 횡보, 급락후 양봉.jpg", "name": "H. 하락 횡보, 급락후 양봉 🔒", "locked": not IS_PRO, "type": "Custom"},
    "I": {"file": "장기횡보, 급락후 바닥확인 연속캔들.jpg", "name": "I. 장기횡보, 급락후 바닥확인 연속캔들 🔒", "locked": not IS_PRO, "type": "Custom"},
    "J": {"file": "3중바닥.jpg", "name": "J. 3중바닥 🔒", "locked": not IS_PRO, "type": "Custom"}
}

if 'selected_key' not in st.session_state:
    st.session_state.selected_key = "B"

def update_pattern(key):
    st.session_state.selected_key = key

# 2. 디자인 시스템
theme_color = "#fbbf24" if IS_PRO else "#38bdf8"
bg_gradient = "linear-gradient(135deg, #1e293b 0%, #000000 100%)" if IS_PRO else "linear-gradient(135deg, #0f172a 0%, #334155 100%)"
symbol_style = "border: 4px solid #fbbf24; border-radius: 50%; box-shadow: 0 0 25px rgba(251, 191, 36, 0.6);" if IS_PRO else "animation: floating 3s ease-in-out infinite;"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * {{ font-family: 'Pretendard', sans-serif; }}
    .stApp {{ background-color: #f8fafc; color: #1e293b; }}
    
    @keyframes floating {{
        0% {{ transform: translateY(0px); filter: drop-shadow(0 5px 15px rgba(56, 189, 248, 0.4)); }}
        50% {{ transform: translateY(-15px); filter: drop-shadow(0 20px 30px rgba(56, 189, 248, 0.6)); }}
        100% {{ transform: translateY(0px); filter: drop-shadow(0 5px 15px rgba(56, 189, 248, 0.4)); }}
    }}
    .symbol-img {{ {symbol_style} width: 160px; height: 160px; object-fit: cover; margin-bottom: 15px; background: white; }}
    
    .brand-container {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: {bg_gradient};
        padding: 60px 15px 50px 15px;
        border-radius: 24px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2); text-align: center; margin-top: -60px;
        border: {'2px solid #fbbf24' if IS_PRO else 'none'};
    }}
    
    .pro-badge {{ background: #fbbf24; color: black; font-weight: 900; padding: 2px 8px; border-radius: 4px; font-size: 14px; vertical-align: middle; margin-left: 10px; }}
    
    .upgrade-pro-btn {{
        display: inline-block; padding: 15px 50px; margin-top: 25px;
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #000 !important; font-weight: 900; font-size: 20px;
        text-decoration: none; border-radius: 50px; border: 2px solid #ffffff;
        transition: transform 0.2s;
    }}
    .upgrade-pro-btn:hover {{ transform: scale(1.05); }}
    
    .mission-box {{ background: white; padding: 25px; border-radius: 15px; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; line-height: 1.8; color: #334155; font-size: 15px; }}
    .mission-highlight {{ color: {'#b45309' if IS_PRO else '#0284c7'}; font-weight: 800; }}
    .pattern-info {{ font-size: 14px; color: #334155; line-height: 1.6; background: #f1f5f9; padding: 18px; border-radius: 10px; border-left: 5px solid {theme_color}; margin-bottom: 20px; }}
    
    .result-card {{ 
        padding: 18px; border-radius: 12px; background: white; border: 1px solid #e2e8f0; margin-bottom: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); 
    }}
    .stock-info {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
    .stock-name {{ font-weight: 900; font-size: 19px; color: #0f172a; }}
    .stock-code {{ font-size: 13px; color: #64748b; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; margin-left: 5px; }}
    .sim-score {{ font-size: 20px; font-weight: 900; color: {'#b45309' if IS_PRO else '#0284c7'}; }}
    
    /* 버튼 스타일 */
    .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .custom-btn {{ 
        display: inline-flex; align-items: center; justify-content: center;
        padding: 8px 16px; border-radius: 8px; text-decoration: none !important; 
        font-size: 13px; font-weight: bold; transition: 0.2s; border: none; cursor: pointer;
    }}
    .btn-pc {{ background: #f1f5f9; color: #475569 !important; border: 1px solid #cbd5e1; }}
    .btn-pc:hover {{ background: #e2e8f0; color: #1e293b !important; }}
    .btn-mo {{ background: #03c75a; color: white !important; border: 1px solid #03c75a; }}
    .btn-mo:hover {{ background: #02b351; color: white !important; }}
    .btn-global {{ background: {theme_color}; color: {'black' if IS_PRO else 'white'} !important; }}
    
    .locked-card {{ padding: 20px; border-radius: 12px; background: #fffbeb; border: 2px dashed #fbbf24; text-align: center; color: #b45309; font-weight: bold; margin-top: 10px; }}
    .debug-alert {{ background: #fee2e2; border-left: 5px solid #ef4444; padding: 15px; margin-bottom: 20px; border-radius: 8px; color: #b91c1c; font-size: 14px; line-height: 1.6; }}
    .debug-badge {{ background: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-right: 5px; }}
    .info-tag {{ background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; }}
    </style>
""", unsafe_allow_html=True)

# 3. 로고 및 헤더
def get_img_tag(path_or_url, is_local=False):
    if is_local and os.path.exists(path_or_url):
        with open(path_or_url, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"
    return path_or_url

if IS_PRO:
    symbol_src = get_img_tag(PRO_SYMBOL_FILE, is_local=True)
    header_html = f"""
    <div class="brand-container">
        <img src="{symbol_src}" class="symbol-img">
        <div style="font-size: 36px; font-weight: 900; color: white; letter-spacing: -1px;">AlphaChart AI <span class="pro-badge">PRO</span></div>
        <div style="font-size: 15px; color: #fbbf24; font-weight: 700; letter-spacing: 3px; margin-bottom: 10px;">MEET YOUR CHART DOPPELGANGER</div>
    </div>"""
else:
    header_html = f"""
    <div class="brand-container">
        <img src="{FREE_SYMBOL_URL}" class="symbol-img">
        <div style="font-size: 36px; font-weight: 900; color: white; letter-spacing: -1px;">AlphaChart AI</div>
        <div style="font-size: 15px; color: #38bdf8; font-weight: 700; letter-spacing: 3px; margin-bottom: 10px;">MEET YOUR CHART DOPPELGANGER</div>
        <a href="https://your-payment-link.com" target="_blank" class="upgrade-pro-btn">👑 PRO 버전 보기 / 업그레이드</a>
    </div>"""
st.markdown(header_html, unsafe_allow_html=True)

# 4. 핵심 설명문
st.markdown(f"""
    <div class="mission-box">
        오랜 주식 거래의 역사를 볼 때, 캔들의 단순한 형태보다는 수거래일 동안의 <span class="mission-highlight">추세와 마지막 몇개의 캔들 형태를 함께 보는 것</span>이 중요하다는 사실이 수많은 연구자와 투자자들로부터 검증되어 왔습니다.<br><br>
        이러한 과거의 패턴을 통한 미래의 패턴 예측, 그 중에서도 우리는 오늘까지의 차트를 보고 내일의 캔들 형태와 방향을 알고 싶습니다. 그래서 그동안 일일이 상장된 모든 종목을 찾아서 내가 원하는 차트와 유사한 종목을 찾고, 증권사 PC 프로그램을 통해 원하는 그림을 그리거나 조건을 입력해서 검색해 왔죠. 그러나 무겁고 불편하고 부정확했습니다.<br><br>
        이제 <span class="mission-highlight">AlphaChart AI</span>가 몇 분만에 도플갱어 종목들을 찾아줄 수 있습니다. 또, 그동안 차트 매매의 대가들이 정립해 놓은 검증된 패턴들을 기본 장착하여 사용자가 가져온 차트 업로드 없이도 바로 내일이나 모레 살만한 종목 후보군을 찾을 수 있게 되었습니다. 차트 매매를 주로 하시는 데이 트레이더, 기업가치와 함께 차트를 같이 보시는 단중기 트레이더 모두 AlphaChart AI를 통해 <span class="mission-highlight">불필요한 시간 투입 없이, 투자의 성공확률을 극대화</span> 하시길 기원합니다. 세계인이 함께 쓰는 글로벌 서칭 시스템으로서 과부하를 막고 양질의 결과를 도출하기 위해 무료버전은 기능을 제한하고 있습니다. 추후 서버 증설 등 투자 확대를 통해 무료 범위를 확대할 계획이니 너그러이 양해 부탁드립니다.
    </div>
""", unsafe_allow_html=True)

# --- 🌏 글로벌 증시 선택 ---
st.markdown("### 🌏 글로벌 증시 선택")
c_m1, c_m2 = st.columns([2, 1])
with c_m1:
    market_map = {"Korea (KRX)": "KRX", "USA (NASDAQ)": "NASDAQ", "USA (NYSE)": "NYSE", "Japan (TSE)": "TSE", "Hong Kong (HKEX)": "HKEX"}
    market_label = st.selectbox("시장 선택", list(market_map.keys()), label_visibility="collapsed")
    market_code = market_map[market_label]

@st.cache_data
def get_stock_list_info(market):
    try:
        df = fdr.StockListing(market)
        if market == 'KRX' and 'Marcap' in df.columns:
            df['Marcap'] = pd.to_numeric(df['Marcap'], errors='coerce')
            df = df.sort_values(by='Marcap', ascending=False)
        elif 'Market Cap' in df.columns:
            df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
            df = df.sort_values(by='Market Cap', ascending=False)
            
        code_col = 'Code' if 'Code' in df.columns else 'Symbol'
        if market == "TSE": df[code_col] = df[code_col].astype(str) + ".T"
        elif market == "HKEX": df[code_col] = df[code_col].apply(lambda x: "{:04d}.HK".format(int(x)) if str(x).isdigit() else str(x) + ".HK")
        return df[[code_col, 'Name']].values.tolist()
    except: return []

stock_data = get_stock_list_info(market_code)
total_count = len(stock_data)

with c_m2:
    if IS_PRO:
        limit_val = st.slider(f"검색 범위 제한 (전체 {total_count:,}개 중)", 10, total_count, min(1000, total_count), label_visibility="collapsed")
        st.success(f"✅ PRO 활성화: {limit_val}개 정밀 스캔")
    else:
        limit_val = st.slider(f"검색 범위 제한 (시가총액 상위 {total_count:,}개 중)", 10, total_count, 300, disabled=True, label_visibility="collapsed")
        st.caption(f"🔒 무료 버전은 시가총액 상위 300개만 스캔 가능")

# --- 🎯 상세 필터 설정 ---
with st.expander("🎯 상세 필터 설정 (눌러서 열기)"):
    c_f1, c_f2 = st.columns(2)
    only_bullish = c_f1.checkbox("✅ 양봉(상승)만 보기", value=True)
    only_doji = c_f2.checkbox("✅ 도지(십자가)만 보기", value=False)
    
    st.markdown("---")
    st.caption("📅 분석 기간 설정 (캔들 개수)")
    search_period = st.slider("비교할 과거 기간 (자동 인식됨)", 5, 120, st.session_state.detected_period)
    
    if search_period > 60:
        st.caption("💡 Tip: 기간이 길면(60일 이상) 전체 추세를, 짧으면(20일 이하) 최근 캔들 모양을 중점적으로 봅니다.")

# --- 💡 패턴 섹션 ---
st.markdown("### 💡 1. AlphaChart AI 에 기본 장착된 패턴 모델 선택 <span style='font-size:16px; color:#64748b; font-weight:normal;'>(차트매매 대가들이 사용)</span>", unsafe_allow_html=True)
st.markdown("""<div class="pattern-info">
이 패턴들은 상승 지속형 2개, 하락에서 반등형 8개이며 내일 또는 모레 매수해도 단타나 스윙으로 성공할 확률이 높은 대표적인 모델입니다. 단, 기업가치, 거래량, 뉴스, 공시 등 내재가치와 외부환경은 매매 전에 함께 고려해야 할 것입니다. 물론, 복잡한 내재와 외부를 고려하지 않고 그냥 매수해도 안전할 확률이 높은 편이지만 돌다리도 두드리고 건널 필요는 있겠지요. 자 이제, 도플갱어를 찾은 후 최종 선택은 여러분의 몫입니다.
</div>""", unsafe_allow_html=True)

keys = list(PATTERN_DB.keys())
cols1 = st.columns(5); cols2 = st.columns(5)
for i, key in enumerate(keys):
    target_col = cols1[i] if i < 5 else cols2[i-5]
    with target_col:
        p = PATTERN_DB[key]
        p_name = p['name'].replace("🔒", "") if IS_PRO else p['name']
        st.button(p_name, key=f"btn_{key}", use_container_width=True, on_click=update_pattern, args=(key,))

# --- 📷 2. 나만의 차트 업로드 ---
st.markdown("### 📷 2. 또는 나만의 차트 업로드")
uploaded_file = st.file_uploader("이미지 파일 업로드 (jpg, png)", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    target_input = uploaded_file
    is_path_mode = False
    sel_p_name = "사용자 업로드 차트"
    sel_p_type = "Custom"
    sel_p_locked = False
else:
    sel_key = st.session_state.selected_key
    sel_p = PATTERN_DB[sel_key]
    target_input = sel_p['file']
    is_path_mode = True
    sel_p_name = sel_p['name'].replace("🔒", "")
    sel_p_type = sel_p.get('type', 'Custom')
    sel_p_locked = sel_p['locked']

# --- 🧠 분석 엔진 (업그레이드) ---
def count_candles_engine(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), mask_blue)
        
        # 노이즈 제거
        kernel = np.ones((2,2), np.uint8) # 커널 크기를 줄여서 세밀하게
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return 20
        
        # 💡 [핵심] 캔들 폭(Width) 보정 로직
        widths = []
        valid_contours = []
        height, width = img.shape[:2]
        min_area = (height * width) * 0.0001 # 아주 작은 점 제외
        
        for c in contours:
            if cv2.contourArea(c) > min_area:
                x, y, w, h = cv2.boundingRect(c)
                widths.append(w)
                valid_contours.append(c)
        
        if not widths: return 20
        
        # 캔들 폭의 중간값(Median) 계산 -> 이걸 표준 1개 폭으로 간주
        median_w = np.median(widths)
        if median_w == 0: median_w = 1
        
        total_candles = 0
        for w in widths:
            # 뭉쳐있는 덩어리는 너비에 비례해서 개수 추가
            count_in_blob = max(1, round(w / median_w))
            total_candles += count_in_blob
            
        # 범위 제한
        if total_candles < 5: total_candles = 20
        if total_candles > 120: total_candles = 120
        
        return int(total_candles)
    except:
        return 20

def extract_features_engine(img_input, is_file_path=False):
    try:
        if is_file_path:
            img_array = np.fromfile(img_input, np.uint8); img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8); img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return None
        
        # 캔들 개수 추정
        candle_count = count_candles_engine(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                                cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])))
        mask_b = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        mask_g = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        mask_k = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r, mask_b), cv2.bitwise_or(mask_k, mask_g))
        height, width = combined.shape
        p_avg = []
        for x in range(width):
            px = np.where(combined[:, x] > 0)[0]
            if len(px) > 0: p_avg.append(height - np.mean(px))
        if not p_avg: return None
        res_p = np.interp(np.linspace(0, len(p_avg)-1, 50), np.arange(len(p_avg)), np.array(p_avg))
        return res_p, img, candle_count
    except: return None

def analyze_stock_legacy(code, name, user_p_norm, n_days=20, market_type="KRX", require_bullish=False, require_doji=False, pattern_type="Custom", force_include=False):
    try:
        df = fdr.DataReader(code).tail(n_days + 10)
        if len(df) < n_days: return None
        if not force_include and df['Volume'].iloc[-1] == 0: return None 
        last_open = df['Open'].iloc[-1]; last_close = df['Close'].iloc[-1]
        last_high = df['High'].iloc[-1]; last_low = df['Low'].iloc[-1]
        if not force_include and market_type != "KRX" and last_close < 1.0: return None
        
        candle_range = last_high - last_low
        body_size = abs(last_close - last_open)
        is_doji = (candle_range > 0 and (body_size / candle_range) <= 0.1)
        
        filter_status = "Pass"
        if require_bullish:
            if last_close <= last_open: filter_status = "Fail_Bearish (음봉)"
            if is_doji: filter_status = "Fail_Doji (도지)"
        if require_doji and not is_doji: filter_status = "Fail_NotDoji (도지아님)"
        
        if not force_include and filter_status != "Pass": return None

        df_t = df.tail(n_days); flow = df_t['Close'].values
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        corr_total = pearsonr(user_p_norm, s_res)[0]
        if np.isnan(corr_total): return None
        tail_len = 10; corr_tail = pearsonr(user_p_norm[-tail_len:], s_res[-tail_len:])[0]
        if np.isnan(corr_tail): corr_tail = 0
        final_sim = (corr_total * 0.7) + (corr_tail * 0.3)
        return {'code': code, 'name': name, 'sim': (final_sim + 1) * 50, 'price': last_close, 'filter_status': filter_status}
    except: return None

# --- 🖼️ 프리뷰 및 실행 ---
st.markdown("---")
c_p1, c_p2, c_p3 = st.columns([1, 10, 1])
feat_data = None
with c_p2:
    if uploaded_file:
        feat_data = extract_features_engine(target_input, is_file_path=False)
        st.image(uploaded_file, caption="업로드된 차트", width=300)
        
        if feat_data:
            _, _, detected_cnt = feat_data
            if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                st.session_state.detected_period = detected_cnt
                st.session_state.last_file = uploaded_file.name
                st.rerun()

    elif not sel_p_locked and os.path.exists(target_input):
        feat_data = extract_features_engine(target_input, is_file_path=True)
        with open(target_input, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""<div style="border:2px solid {theme_color}; border-radius:15px; overflow:hidden; text-align:center;"><img src="data:image/jpeg;base64,{b64}" style="width:100%; height:auto; max-height:250px; object-fit:contain;"></div>""", unsafe_allow_html=True)
        
    if feat_data:
        user_p, _, cnt = feat_data
        st.markdown(f"""<div style="margin-top:10px; margin-bottom:5px;"><span class="info-tag">🔍 AI 분석</span> 캔들 <b>{cnt}</b>개 인식됨 │ 패턴 추출 완료</div>""", unsafe_allow_html=True)
        
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.plot(user_p_norm, color=theme_color, lw=2.5)
        ax.set_title("AI가 인식한 차트 흐름 (Pattern Shape)", fontsize=10, color='gray')
        ax.axis('off')
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        
    elif sel_p_locked: st.warning("🔒 PRO 전용 모델입니다.")

clean_name = sel_p_name.split('. ', 1)[-1] if '. ' in sel_p_name else sel_p_name
button_label = f"🚀 [{clean_name}] 분석 시작"

if st.button(button_label, type="primary", use_container_width=True):
    if sel_p_locked and not uploaded_file:
        st.error("PRO 전용 패턴입니다. 업그레이드 후 이용해 주세요.")
    elif not feat_data:
        st.error("이미지를 분석할 수 없습니다. 파일을 확인해 주세요.")
    else:
        period_msg = f" | 📅 분석 기간: {search_period}일 (자동설정)"
        info_msg = f"({limit_val}개 정밀 스캔{period_msg})" if IS_PRO else f"(시가총액 상위 300개{period_msg})"
        st.info(f"최적의 도플갱어 종목을 스캔 중입니다... {info_msg}")
        
        progress_bar = st.progress(0)
        user_p, _, _ = feat_data
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        results = []
        
        target_stocks = stock_data[:limit_val]
        if debug_code:
            exists = False
            for s in target_stocks:
                if s[0] == debug_code: exists = True; break
            if not exists:
                found_name = "강제추가종목"
                for s in stock_data:
                    if s[0] == debug_code: found_name = s[1]; break
                target_stocks.insert(0, [debug_code, found_name])

        total_scan = len(target_stocks)
        
        with ThreadPoolExecutor(max_workers=30) as ex:
            futures = [ex.submit(analyze_stock_legacy, s[0], s[1], user_p_norm, search_period, market_code, only_bullish, only_doji, sel_p_type, (s[0] == debug_code)) for s in target_stocks]
            for idx, f in enumerate(as_completed(futures)):
                res = f.result()
                if res:
                    if res['code'] == debug_code:
                        if res['filter_status'] != "Pass":
                            st.markdown(f"""<div class="debug-alert"><span class="debug-badge">DEBUG</span> <b>{res['name']}({res['code']})</b><br>• 유사도: {res['sim']:.1f}%<br>• 필터 탈락 사유: {res['filter_status']}<br>→ 강제로 결과 리스트에 포함시킵니다.</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<div class="debug-alert" style="background:#ecfccb; border-color:#84cc16; color:#365314;"><span class="debug-badge" style="background:#84cc16;">DEBUG</span> <b>{res['name']}({res['code']})</b><br>• 유사도: {res['sim']:.1f}%<br>• 상태: 정상 통과 (결과 리스트에서 순위를 확인하세요)</div>""", unsafe_allow_html=True)
                        results.append(res)
                    else:
                        results.append(res)
                progress_bar.progress((idx + 1) / total_scan)
        
        results.sort(key=lambda x: x['sim'], reverse=True)
        
        final_display_list = []
        if IS_PRO:
            high_sim = [r for r in results if r['sim'] >= 80.0]
            if len(high_sim) < 10: final_display_list = results[:10]
            else: final_display_list = high_sim[:100]
        else:
            final_display_list = results[:5]

        st.markdown(f"### 🏆 분석 결과 (총 {len(final_display_list)}개 발견)")
        if not final_display_list: st.warning("조건에 맞는 종목을 찾지 못했습니다.")
        for i, res in enumerate(final_display_list):
            
            if market_code == "KRX":
                pc_link = f"https://finance.naver.com/item/fchart.naver?code={res['code']}"
                mo_link = f"https://m.stock.naver.com/domestic/stock/{res['code']}/chart"
                links_html = f'<div class="btn-row"><a href="{pc_link}" target="_blank" class="custom-btn btn-pc">💻 PC용 차트</a><a href="{mo_link}" target="_blank" class="custom-btn btn-mo">📱 모바일용 차트</a></div>'
            elif market_code in ["NASDAQ", "NYSE"]:
                link = f"https://www.tradingview.com/chart/?symbol={res['code']}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">📈 차트 보기</a>'
            elif market_code == "TSE":
                link = f"https://www.tradingview.com/chart/?symbol=TSE:{res['code'].replace('.T','')}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">📈 차트 보기</a>'
            elif market_code == "HKEX":
                link = f"https://www.tradingview.com/chart/?symbol=HKEX:{res['code'].replace('.HK','')}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">📈 차트 보기</a>'
            else:
                link = f"https://finance.yahoo.com/quote/{res['code']}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">📈 차트 보기</a>'

            st.markdown(f"""
            <div class="result-card">
                <div class="stock-info">
                    <div>
                        <span class="stock-name">{res['name']}</span>
                        <span class="stock-code">{res['code']}</span>
                    </div>
                    <div class="sim-score">{res['sim']:.1f}%</div>
                </div>
                {links_html}
            </div>
            """, unsafe_allow_html=True)
            
        if not IS_PRO and len(results) > 5:
            st.markdown("""<div class="locked-card">🔒 TOP 6 ~ 10 및 전종목 검색 결과는<br>PRO 버전 업그레이드 시 확인 가능합니다.</div>""", unsafe_allow_html=True)

st.caption("AlphaChart AI v19.1 | Smart Candle Counter")