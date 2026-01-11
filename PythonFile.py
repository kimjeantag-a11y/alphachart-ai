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
    page_icon="🦅", 
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

# 2. 디자인 시스템
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * { font-family: 'Pretendard', sans-serif; -webkit-tap-highlight-color: transparent; }
    .stApp { background-color: #f8fafc; }
    
    /* 상단 헤더 및 툴바 숨김 (Clean UI) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .brand-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        padding: 40px 15px; border-radius: 25px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5); text-align: center;
        border: 1px solid #475569;
        margin-top: -50px; 
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
        background: linear-gradient(to right, #ffffff, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .brand-subtitle { font-size: 14px; color: #38bdf8; letter-spacing: 3px; font-weight: 700; margin-top: 10px; text-transform: uppercase; }

    .mission-box {
        background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0;
        margin-bottom: 2rem; line-height: 1.8; color: #334155; font-size: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .mission-highlight { color: #0284c7; font-weight: 800; }

    .result-card { padding: 15px; border-radius: 18px; background: white; border: 1px solid #cbd5e1; margin-bottom: 10px; transition: transform 0.2s; }
    .result-card:hover { transform: translateY(-2px); border-color: #0ea5e9; }
    .compact-img img { max-height: 120px !important; width: auto !important; border-radius: 8px; }
    
    .link-btn {
        display: inline-block; margin-top: 8px; padding: 6px 14px;
        color: white !important; border-radius: 8px; font-size: 12px; font-weight: 700; text-decoration: none;
    }
    .naver-link { background-color: #03c75a; } 
    .yahoo-link { background-color: #720e9e; }
    .jp-link { background-color: #ff0033; }
    .hk-link { background-color: #0047bb; }
    </style>
    """, unsafe_allow_html=True)

# --- 🌏 설정 및 사이드바 ---
st.sidebar.header("🌏 검색 옵션")

# 1. 시장 선택
market_map = {
    "Korea (KRX)": "KRX",
    "USA (NASDAQ)": "NASDAQ",
    "USA (NYSE)": "NYSE",
    "Japan (TSE)": "TSE",
    "Hong Kong (HKEX)": "HKEX"
}
market_label = st.sidebar.selectbox("검색할 시장", list(market_map.keys()))
market_code = market_map[market_label]

# 2. 필터 스위치
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 캔들 필터 (Candle Filter)")
st.sidebar.caption("원하는 마지막 캔들 모양을 선택하세요.")

only_bullish = st.sidebar.checkbox("✅ 마지막 캔들 양봉(몸통 있는 상승)만", value=True, help="체크하면 십자가(도지)는 제외하고, 몸통이 확실한 상승 캔들만 보여줍니다.")
only_doji = st.sidebar.checkbox("✅ 마지막 캔들 도지(십자가)만", value=False, help="체크하면 몸통이 매우 짧은 십자가 형태(보합)의 종목만 보여줍니다.")

@st.cache_data
def get_stock_list(market):
    try:
        df = fdr.StockListing(market)
        
        if market == "KRX":
            try:
                # [Korea] 리스트 단계에서 시총/가격 필터링
                if 'Marcap' in df.columns:
                    df['Marcap'] = pd.to_numeric(df['Marcap'], errors='coerce')
                    df = df.dropna(subset=['Marcap'])
                    df = df[df['Marcap'] >= 100_000_000_000] # 1000억 이상
                if 'Close' in df.columns:
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    df = df.dropna(subset=['Close'])
                    df = df[df['Close'] >= 1000] # 동전주 제거
                if 'Name' in df.columns:
                    noise = "스팩|ETF|ETN|우$|우B|홀딩스|리츠"
                    df = df[~df['Name'].str.contains(noise, regex=True)]
            except: pass
            return df[['Code', 'Name']].values.tolist()
        else:
            # [Global] 리스트 단계에서는 이름 필터링만
            if 'Name' in df.columns:
                noise = "SPAC|ETF|ETN|Acquisition|Fund|Trust" 
                df = df[~df['Name'].str.contains(noise, case=False, regex=True)]
            
            code_col = 'Symbol' if 'Symbol' in df.columns else 'Code'
            if code_col not in df.columns: return []

            # 국가별 코드 보정
            if market == "TSE": df[code_col] = df[code_col].astype(str) + ".T"
            elif market == "HKEX": 
                def format_hk(x):
                    try: return "{:04d}.HK".format(int(x))
                    except: return str(x) + ".HK"
                df[code_col] = df[code_col].apply(format_hk)

            return df[[code_col, 'Name']].values.tolist()
        return []
    except Exception as e:
        return [[f"Error: {e}", "데이터 로드 실패"]]

stock_list = get_stock_list(market_code)

# --- 🧠 분석 엔진 ---
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
        return res_p, img
    except: return None

def analyze_stock(code, name, user_p_norm, n_days=20, market_type="KRX", require_bullish=False, require_doji=False, pattern_type="Custom"):
    try:
        df = fdr.DataReader(code).tail(n_days + 10)
        
        if len(df) < n_days: return None
        if df['Volume'].iloc[-1] == 0: return None 
        
        last_open = df['Open'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        last_high = df['High'].iloc[-1]
        last_low = df['Low'].iloc[-1]
        
        # [Global] 동전주 필터링
        if market_type != "KRX" and last_close < 1.0: return None
        
        candle_range = last_high - last_low
        body_size = abs(last_close - last_open)
        is_doji = (candle_range > 0 and (body_size / candle_range) <= 0.1)

        # 공통 필터
        if require_bullish:
            if last_close <= last_open: return None
            if is_doji: return None 
        if require_doji:
            if not is_doji: return None

        # 🎯 [패턴 A] Midpoint Support
        if pattern_type == "A":
            subset = df.tail(10)
            closes = subset['Close'].values
            opens = subset['Open'].values
            
            max_body = 0
            anchor_idx = -1
            
            for i in range(2, 7):
                body = closes[i] - opens[i]
                if body > 0 and body > max_body:
                    max_body = body
                    anchor_idx = i
            
            if anchor_idx != -1:
                midpoint = (opens[anchor_idx] + closes[anchor_idx]) / 2
                for j in range(anchor_idx + 1, 10):
                    if closes[j] < midpoint:
                        return None 
            else:
                return None 

        df_t = df.tail(n_days)
        flow = df_t['Close'].values
        s_res = np.interp(np.linspace(0, n_days-1, 50), np.arange(n_days), MinMaxScaler().fit_transform(flow.reshape(-1, 1)).flatten())
        
        corr_total = pearsonr(user_p_norm, s_res)[0]
        if np.isnan(corr_total): return None

        tail_len = 10 
        corr_tail = pearsonr(user_p_norm[-tail_len:], s_res[-tail_len:])[0]
        if np.isnan(corr_tail): corr_tail = 0
        
        final_sim = (corr_total * 0.7) + (corr_tail * 0.3)
        
        return {
            'code': code, 
            'name': name, 
            'sim': (final_sim + 1) * 50,
            'price': last_close
        }
    except: return None

# --- UI 메인 ---
st.markdown(f"""
    <div class="brand-container">
        <img src="https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png" class="candlestick-ai-logo">
        <div class="brand-title">AlphaChart AI</div>
        <div class="brand-subtitle">STOCK CHART DOPPELGANGER</div>
    </div>
    """, unsafe_allow_html=True)

# 📝 핵심 미션 텍스트
st.markdown(f"""
    <div class="mission-box">
        오랜 주식 거래의 역사를 볼 때, 캔들의 단순한 형태보다는 수거래일 동안의 <span class="mission-highlight">추세와 마지막 몇개의 캔들 형태를 함께 보는 것</span>이 중요하다는 사실이 수많은 연구자와 투자자들로부터 검증되어 왔습니다.<br><br>
        이러한 과거의 패턴을 통한 미래의 패턴 예측, 그 중에서도 우리는 오늘까지의 차트를 보고 내일의 캔들 형태와 방향을 알고 싶습니다. 일일이 상장된 모든 종목을 찾아서 내가 원하는 차트와 유사한 종목을 찾고, 증권사 PC 프로그램을 통해 원하는 그림을 그리거나 조건을 입력해서 검색해 왔죠. 그러나 무겁고 불편하고 부정확했습니다.<br><br>
        이제 <span class="mission-highlight">AlphaChart AI</span>가 몇 분만에 도플갱어 종목들을 찾아줄 수 있습니다. 또, 그동안 차트 매매의 대가들이 정립해 놓은 검증된 패턴들을 기본 장착하여 사용자가 가져온 차트 업로드 없이도 바로 내일이나 모레 살만한 종목 후보군을 찾을 수 있게 되었습니다. 차트 매매를 주로 하시는 데이 트레이더, 기업가치와 함께 차트를 같이 보시는 단중기 트레이더 모두 AlphaChart AI를 통해 <span class="mission-highlight">불필요한 시간 투입 없이, 투자의 성공확률을 극대화</span> 하시길 기원합니다.
    </div>
    """, unsafe_allow_html=True)

# 1단계
st.markdown("### 🧭 1단계: 검색 범위 설정")

if len(stock_list) > 0 and "Error" in stock_list[0][0]:
    st.error(f"❌ {stock_list[0][0]}")
    st.warning("데이터 소스 연결 실패. 잠시 후 다시 시도해주세요.")
else:
    filters = []
    if only_bullish: filters.append("진짜 양봉")
    if only_doji: filters.append("도지")
    filter_txt = " + ".join(filters) if filters else "없음 (전체)"
    
    # [안내 문구]
    if market_code == "KRX":
        target_msg = f"{len(stock_list):,}개 (ETF/SPAC/동전주/천억미만 제외)"
    else:
        target_msg = f"{len(stock_list):,}개 (실시간 필터링 적용)"

    st.info(f"현재 선택된 시장: **{market_label}** | 적용 필터: **{filter_txt}** | 대상: **{target_msg}**")
    
    # [슬라이더]
    total_count = len(stock_list)
    default_val = min(500, total_count)
    if total_count > 0:
        search_limit = st.slider("검색할 종목 수", 10, total_count, default_val, step=10)
    else:
        search_limit = 0

    st.markdown("---")

    # 2단계 제1방법
    st.markdown("### 💡 2단계 제1방법: AI 추천 패턴")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if os.path.exists(ex1_name):
            st.markdown('<div class="menu-card compact-img">', unsafe_allow_html=True)
            st.image(ex1_name, caption="패턴 A: 급등 지속 (허리 지지)"); st.button("패턴 A 선택", on_click=select_pattern, args=(ex1_name,), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with col_p2:
        if os.path.exists(ex2_name):
            st.markdown('<div class="menu-card compact-img">', unsafe_allow_html=True)
            st.image(ex2_name, caption="패턴 B: 급락 후 반등"); st.button("패턴 B 선택", on_click=select_pattern, args=(ex2_name,), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2단계 제2방법
    st.markdown("### 📷 또는 2단계 제2방법: 나만의 차트 업로드도 가능")
    uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    input_target = uploaded_file if uploaded_file else st.session_state.selected_path
    is_path = True if (not uploaded_file and st.session_state.selected_path) else False

    current_pattern_type = "Custom"
    if st.session_state.selected_path == ex1_name: current_pattern_type = "A"
    elif st.session_state.selected_path == ex2_name: current_pattern_type = "B"

    if input_target:
        feat = extract_features_engine(input_target, is_file_path=is_path)
        if feat:
            user_p, original_img = feat
            user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
            
            st.markdown("<div style='font-size:13px; font-weight:700; color:#0ea5e9; margin-top:10px;'>🎯 분석 대상</div>", unsafe_allow_html=True)
            view_c1, view_c2, view_c3 = st.columns([1, 1.2, 2])
            with view_c1:
                st.markdown('<div class="compact-img">', unsafe_allow_html=True); st.image(original_img); st.markdown('</div>', unsafe_allow_html=True)
            with view_c2:
                fig, ax = plt.subplots(figsize=(2.5, 1.2))
                ax.plot(user_p_norm, color='#00ffff', lw=5)
                ax.scatter(len(user_p_norm)-1, user_p_norm[-1], color='#ef4444', s=50, zorder=5)
                ax.axis('off'); fig.patch.set_alpha(0); st.pyplot(fig)
            with view_c3:
                st.write(""); search_btn = st.button(f"🚀 AI 검색 시작 ({market_code})", type="primary", use_container_width=True)

            if search_btn:
                results = []
                prog_bar = st.progress(0)
                status_text = st.empty()
                
                scan_targets = stock_list[:search_limit]
                workers = 50 if market_code == "KRX" else 30
                
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(analyze_stock, s[0], s[1], user_p_norm, 20, market_code, only_bullish, only_doji, current_pattern_type) for s in scan_targets]
                    for i, f in enumerate(as_completed(futures)):
                        res = f.result()
                        if res: results.append(res)
                        percent = (i + 1) / len(scan_targets)
                        prog_bar.progress(min(percent, 1.0))
                        status_text.text(f"Scanning... {i+1}/{len(scan_targets)} complete")
                
                results.sort(key=lambda x: x['sim'], reverse=True)
                status_text.empty()
                
                st.markdown(f"### 🏆 분석 결과 Top 10 ({market_label})")
                
                if not results:
                    st.warning("조건에 맞는 종목이 없습니다.")
                
                for i, res in enumerate(results[:10]):
                    if market_code == "KRX":
                        link_url = f"https://finance.naver.com/item/fchart.naver?code={res['code']}"
                        link_text = "Naver Chart ↗"
                        link_class = "naver-link"
                        currency = "KRW"
                    elif market_code == "TSE":
                        link_url = f"https://finance.yahoo.co.jp/quote/{res['code']}/chart?styl=c&term=6m"
                        link_text = "Yahoo!JP Chart ↗"
                        link_class = "jp-link"
                        currency = "JPY"
                    elif market_code == "HKEX":
                        link_url = f"https://hk.finance.yahoo.com/quote/{res['code']}/chart?range=6mo&interval=1d"
                        link_text = "Yahoo HK Chart ↗"
                        link_class = "hk-link"
                        currency = "HKD"
                    else:
                        link_url = f"https://finance.yahoo.com/quote/{res['code']}/chart?range=6mo&interval=1d"
                        link_text = "Yahoo Chart ↗"
                        link_class = "yahoo-link"
                        currency = "USD"

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <div style="font-size:16px; font-weight:800; color:#1e293b;">{i+1}. {res['name']}</div>
                                <div style="font-size:12px; color:#64748b; margin-top:2px;">Code: {res['code']} | {res['price']:,.2f} {currency}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="color:#0ea5e9; font-weight:900; font-size:20px;">{res['sim']:.1f}%</div>
                                <div style="font-size:10px; color:#94a3b8;">Match Rate</div>
                            </div>
                        </div>
                        <a href="{link_url}" target="_blank" class="link-btn {link_class}">{link_text}</a>
                    </div>
                    """, unsafe_allow_html=True)

st.caption("AlphaChart AI v8.5 | Brand Subtitle Updated")