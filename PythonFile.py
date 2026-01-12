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

# --- 🌐 언어 데이터 팩 (Language Pack) ---
TRANS = {
    "KR": {
        "sidebar_header": "⚙️ 설정",
        "license_active": "✅ PRO 라이선스 활성",
        "logout": "로그아웃 / 리셋",
        "current_free": "현재: 무료 버전 (Free)",
        "upgrade_btn": "👑 PRO 업그레이드",
        "license_input": "🔑 라이선스 키 입력",
        "confirm": "확인",
        "cancel": "취소",
        "wrong_key": "잘못된 키입니다.",
        "market_select": "시장 선택",
        "limit_search": "검색 범위 제한 (전체 {:,}개 중)",
        "limit_search_free": "검색 범위 제한 (시가총액 상위 {:,}개 중)",
        "pro_active_msg": "✅ PRO 활성화: {}개 정밀 스캔",
        "free_limit_msg": "🔒 무료 버전은 시가총액 상위 300개만 스캔 가능",
        "filter_detail": "🎯 상세 필터 설정 (눌러서 열기)",
        "filter_bullish": "마지막(최근) 캔들 양봉(상승)만 보기",
        "filter_doji": "마지막(최근) 캔들 도지(십자가)만 보기",
        "period_set_caption": "⏱️ 분석 기간 설정",
        "period_info_fmt": "💠 **[{}]** 기준: AI가 차트에서 **{}일** 치 패턴을 자동 인식하여 분석합니다.",
        "section1_title": "### 🧬 1. AlphaChart AI 에 기본 장착된 패턴 모델 선택 <span style='font-size:16px; color:#64748b; font-weight:normal;'>(차트매매 대가들이 사용)</span>",
        "pattern_desc": """<div class="pattern-info">
        이 패턴들은 상승 지속형 6개, 하락에서 반등형 8개이며 하락 경직 또는 눌림목 상태이므로 내일 또는 모레 매수해도 단타나 스윙으로 성공할 확률이 높은 대표적인 모델입니다.<br><br>
        단, 기업가치, 거래량, 뉴스, 공시 등 내재가치와 외부환경은 매매 전에 함께 고려해야 할 것입니다.<br><br>
        물론, 복잡한 내재와 외부를 고려하지 않고 그냥 매수해도 안전할 확률이 높은 편이지만 돌다리도 두드리고 건널 필요는 있겠지요.<br><br>
        자! 이제, 도플갱어를 찾은 후 최종 선택은 여러분의 몫입니다.
        </div>""",
        "section2_title": "### 🖼️ 2. 또는 나만의 차트 업로드",
        "upload_label": "이미지 파일 업로드 (jpg, png)",
        "upload_caption": "이동평균선 등을 제외하고 캔들차트만 있을수록 정확합니다.",
        "ai_analysis_badge": "🔍 AI 분석",
        "ai_pattern_shape": "AI Pattern Shape",
        "candles_detected": "캔들 <b>{}</b>개 인식됨",
        "pro_only_model": "🔒 PRO 전용 모델입니다.",
        "btn_analyze": "🚀 [{}] 분석 시작",
        "error_pro_only": "PRO 전용 패턴입니다. 업그레이드 후 이용해 주세요.",
        "error_no_file": "이미지를 분석할 수 없습니다. 파일을 확인해 주세요.",
        "scanning_msg": "최적의 도플갱어 종목을 스캔 중입니다... {}",
        "result_title": "### 🏆 분석 결과 (총 {}개 발견)",
        "no_result": "조건에 맞는 종목을 찾지 못했습니다.",
        "chart_view": "📈 차트 보기",
        "pc_chart": "💻 PC용 차트",
        "mo_chart": "📱 모바일용 차트",
        "locked_msg": "🔒 TOP 6위 이후 결과 및 전종목 정밀 스캔은<br>PRO 버전 업그레이드 시 확인 가능합니다.",
        "mission_html": """
        <div class="mission-box">
            오랜 주식 거래의 역사를 볼 때, 캔들의 단순한 형태보다는 수거래일 동안의 <span class="mission-highlight">추세와 마지막 몇개의 캔들 형태를 함께 보는 것</span>이 중요하다는 사실이 수많은 연구자와 투자자들로부터 검증되어 왔습니다.<br><br>
            이러한 과거의 패턴을 통한 미래의 패턴 예측, 그 중에서도 우리는 오늘까지의 차트를 보고 내일의 캔들 형태와 방향을 알고 싶습니다. <b>그래서</b> 일일이 상장된 모든 종목을 찾아서 내가 원하는 차트와 유사한 종목을 찾고, 증권사 PC 프로그램을 통해 원하는 그림을 그리거나 조건을 입력해서 검색해 왔죠. 그러나 무겁고 불편하고 부정확했습니다.<br><br>
            이제 <span class="mission-highlight">AlphaChart AI</span>가 몇 분만에 도플갱어 종목들을 찾아줄 수 있습니다.<br><br>
            또, 그동안 차트 매매의 대가들이 정립해 놓은 검증된 패턴들을 기본 장착하여, 사용자가 가져온 차트 업로드 없이도 바로 내일이나 모레 살만한 종목 후보군을 찾을 수 있게 되었습니다.<br><br>
            차트 매매를 주로 하시는 데이 트레이더, 기업가치와 함께 차트를 같이 보시는 단중기 트레이더 모두 AlphaChart AI를 통해, <span class="mission-highlight">불필요한 시간 투입 없이 투자의 성공확률을 극대화</span> 하시길 기원합니다.<br><br>
            세계인이 함께 쓰는 글로벌 서칭 시스템으로서 과부하를 막고 양질의 결과를 도출하기 위해, <b>부득이</b> 무료버전은 기능을 제한하고 있습니다.<br><br>
            추후 서버 증설 등 투자 확대를 통해 무료 범위를 확대할 계획이니 너그러이 양해 부탁드립니다.
        </div>"""
    },
    "EN": {
        "sidebar_header": "⚙️ Settings",
        "license_active": "✅ PRO License Active",
        "logout": "Logout / Reset",
        "current_free": "Current: Free Version",
        "upgrade_btn": "👑 Upgrade to PRO",
        "license_input": "🔑 Enter License Key",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "wrong_key": "Invalid Key.",
        "market_select": "Select Market",
        "limit_search": "Search Limit (Scanning {:,} stocks)",
        "limit_search_free": "Search Limit (Top {:,} Market Cap)",
        "pro_active_msg": "✅ PRO Active: Precision scan of {} stocks",
        "free_limit_msg": "🔒 Free version scans top 300 market cap only",
        "filter_detail": "🎯 Advanced Filters (Click to expand)",
        "filter_bullish": "Last candle must be Bullish (Green/Red)",
        "filter_doji": "Last candle must be Doji (Cross)",
        "period_set_caption": "⏱️ Analysis Period",
        "period_info_fmt": "💠 Based on **[{}]**: AI automatically detects and analyzes **{} days** pattern.",
        "section1_title": "### 🧬 1. Select AI Built-in Patterns <span style='font-size:16px; color:#64748b; font-weight:normal;'>(Used by Master Traders)</span>",
        "pattern_desc": """<div class="pattern-info">
        These patterns include 6 bullish continuation and 8 reversal-from-bottom types. They represent models with high probability of success for day or swing trading if bought tomorrow or the day after, as they are in a state of downward rigidity or pullback.<br><br>
        However, intrinsic values such as corporate value, volume, news, and disclosures, as well as external environments, should be considered before trading.<br><br>
        Of course, the probability of safety is high even if you buy without considering complex internal and external factors, but it is always better to be safe than sorry.<br><br>
        Now! After finding your chart doppelganger, the final choice is yours.
        </div>""",
        "section2_title": "### 🖼️ 2. Or Upload Your Own Chart",
        "upload_label": "Upload Image (jpg, png)",
        "upload_caption": "Accuracy improves if only candlestick charts are present (exclude Moving Averages, etc).",
        "ai_analysis_badge": "🔍 AI Analysis",
        "ai_pattern_shape": "AI Pattern Shape",
        "candles_detected": "<b>{}</b> Candles Detected",
        "pro_only_model": "🔒 PRO Version Only.",
        "btn_analyze": "🚀 Start Analysis [{}]",
        "error_pro_only": "This is a PRO pattern. Please upgrade to use.",
        "error_no_file": "Cannot analyze image. Please check the file.",
        "scanning_msg": "Scanning for optimal doppelgangers... {}",
        "result_title": "### 🏆 Analysis Results ({})",
        "no_result": "No stocks found matching criteria.",
        "chart_view": "📈 View Chart",
        "pc_chart": "💻 PC Chart",
        "mo_chart": "📱 Mobile Chart",
        "locked_msg": "🔒 Results from Top 6 onwards & Full Scan<br>available upon PRO upgrade.",
        "mission_html": """
        <div class="mission-box">
            Looking at the long history of stock trading, it has been verified by numerous researchers and investors that it is more important to look at the <b>trend over several trading days and the shape of the last few candles</b> rather than the simple shape of a single candle.<br><br>
            Through this prediction of future patterns via past patterns, we specifically want to know tomorrow's candle shape and direction based on the chart up to today. <b>Therefore</b>, we used to search for all listed stocks one by one to find stocks similar to the chart we wanted, drawing desired pictures or entering conditions through securities PC programs. However, it was heavy, inconvenient, and inaccurate.<br><br>
            Now, <span class="mission-highlight">AlphaChart AI</span> can find doppelganger stocks in just a few minutes.<br><br>
            In addition, by equipping verified patterns established by chart trading masters as standard, you can immediately find candidate stocks to buy tomorrow or the day after without uploading a user chart.<br><br>
            We hope that both day traders who mainly trade charts and short-to-medium term traders who look at charts along with corporate value will maximize their <span class="mission-highlight">success probability without unnecessary time investment</span> through AlphaChart AI.<br><br>
            As a global searching system used by people around the world, the free version inevitably limits functions to prevent overload and derive high-quality results.<br><br>
            We ask for your generous understanding as we plan to expand the free scope through future investments such as server expansion.
        </div>"""
    },
    "JP": {
        "sidebar_header": "⚙️ 設定",
        "license_active": "✅ PROライセンス有効",
        "logout": "ログアウト / リセット",
        "current_free": "現在: 無料版 (Free)",
        "upgrade_btn": "👑 PROにアップグレード",
        "license_input": "🔑 ライセンスキー入力",
        "confirm": "確認",
        "cancel": "キャンセル",
        "wrong_key": "無効なキーです。",
        "market_select": "市場選択",
        "limit_search": "検索範囲制限 (全体 {:,} 銘柄中)",
        "limit_search_free": "検索範囲制限 (時価総額上位 {:,} 銘柄)",
        "pro_active_msg": "✅ PRO有効化: {}銘柄 精密スキャン",
        "free_limit_msg": "🔒 無料版は時価総額上位300銘柄のみスキャン可能",
        "filter_detail": "🎯 詳細フィルタ設定 (クリックして展開)",
        "filter_bullish": "直近ローソク足が「陽線」のみ",
        "filter_doji": "直近ローソク足が「十字線(同時線)」のみ",
        "period_set_caption": "⏱️ 分析期間設定",
        "period_info_fmt": "💠 **[{}]** 基準: AIがチャートから **{}日分** のパターンを自動認識して分析します。",
        "section1_title": "### 🧬 1. AlphaChart AI 搭載のパターンモデルを選択 <span style='font-size:16px; color:#64748b; font-weight:normal;'>(チャート売買の大家たちが使用)</span>",
        "pattern_desc": """<div class="pattern-info">
        これらのパターンは上昇持続型6つ、下落からの反発型8つであり、下落硬直または押し目買いの状態にあるため、明日や明後日に購入してもデイトレやスイングトレードで成功する確率が高い代表的なモデルです。<br><br>
        ただし、企業価値、出来高、ニュース、開示情報などの内在価値と外部環境は、売買前に併せて考慮する必要があります。<br><br>
        もちろん、複雑な内在・外部要因を考慮せずに購入しても安全である確率は高い方ですが、石橋を叩いて渡る必要はあるでしょう。<br><br>
        さあ！ドッペルゲンガー(瓜二つのチャート)を見つけた後の最終選択は、あなたの役割です。
        </div>""",
        "section2_title": "### 🖼️ 2. または自分のチャートをアップロード",
        "upload_label": "画像ファイルアップロード (jpg, png)",
        "upload_caption": "移動平均線などを除き、ローソク足チャートのみであるほど正確です。",
        "ai_analysis_badge": "🔍 AI分析",
        "ai_pattern_shape": "AI Pattern Shape",
        "candles_detected": "ローソク足 <b>{}</b>本 認識",
        "pro_only_model": "🔒 PRO専用モデルです。",
        "btn_analyze": "🚀 [{}] 分析開始",
        "error_pro_only": "PRO専用パターンです。アップグレードしてご利用ください。",
        "error_no_file": "画像を分析できません。ファイルを確認してください。",
        "scanning_msg": "最適なドッペルゲンガー銘柄をスキャン中... {}",
        "result_title": "### 🏆 分析結果 (計 {}件 発見)",
        "no_result": "条件に合う銘柄が見つかりませんでした。",
        "chart_view": "📈 チャートを見る",
        "pc_chart": "💻 PC用チャート",
        "mo_chart": "📱 モバイル用チャート",
        "locked_msg": "🔒 6位以降の結果および全銘柄スキャンは<br>PROバージョンへのアップグレードで確認可能です。",
        "mission_html": """
        <div class="mission-box">
            長い株式取引の歴史を見ると、単一のローソク足の単純な形よりも、数取引日間の<span class="mission-highlight">トレンドと最後の数本のローソク足の形を共に見ること</span>が重要であるという事実が、数多くの研究者や投資家によって検証されてきました。<br><br>
            このような過去のパターンを通じた未来のパターン予測、その中でも私たちは今日までのチャートを見て、明日のローソク足の形と方向を知りたいのです。<b>そこで</b>、上場されたすべての銘柄を一つ一つ探して、自分が望むチャートと類似した銘柄を見つけたり、証券会社のPCプログラムを通じて希望の絵を描いたり条件を入力して検索してきました。しかし、それは重く、不便で、不正確でした。<br><br>
            今や<span class="mission-highlight">AlphaChart AI</span>が数分でドッペルゲンガー銘柄を見つけ出します。<br><br>
            また、これまでのチャート売買の大家たちが確立した検証済みのパターンを基本搭載し、ユーザーが持ってきたチャートのアップロードなしでも、すぐに明日や明後日に買うべき銘柄候補群を見つけることができるようになりました。<br><br>
            チャート売買を主とするデイトレーダー、企業価値と共にチャートも見る短期・中期トレーダーの皆様が、AlphaChart AIを通じて<span class="mission-highlight">不必要な時間の投入なしに、投資の成功確率を最大化</span>されることを祈ります。<br><br>
            世界中の人々が共に使うグローバル検索システムとして、過負荷を防ぎ良質な結果を導き出すために、<b>やむを得ず</b>無料版では機能を制限しています。<br><br>
            今後、サーバー増設などの投資拡大を通じて無料範囲を拡大する計画ですので、何卒ご了承いただけますようお願い申し上げます。
        </div>"""
    }
}

# --- 🔐 라이선스 및 세션 관리 ---
if 'is_pro' not in st.session_state:
    st.session_state.is_pro = False
if 'show_license_input' not in st.session_state:
    st.session_state.show_license_input = False
if 'detected_period' not in st.session_state:
    st.session_state.detected_period = 20
if 'lang' not in st.session_state:
    st.session_state.lang = "KR"

with st.sidebar:
    # 🌍 언어 선택 (국기 추가)
    lang_choice = st.selectbox("🌐 Language", ["🇰🇷 Korean (한국어)", "🇺🇸 English", "🇯🇵 Japanese (日本語)"])
    if "Korean" in lang_choice:
        st.session_state.lang = "KR"
    elif "English" in lang_choice:
        st.session_state.lang = "EN"
    else:
        st.session_state.lang = "JP"
    
    t = TRANS[st.session_state.lang]

    st.header(t['sidebar_header'])
    if st.session_state.is_pro:
        st.success(t['license_active'])
        if st.button(t['logout'], use_container_width=True):
            st.session_state.is_pro = False
            st.session_state.show_license_input = False
            st.rerun()
    else:
        st.info(t['current_free'])
        if not st.session_state.show_license_input:
            if st.button(t['upgrade_btn'], use_container_width=True):
                st.session_state.show_license_input = True
                st.rerun()
        if st.session_state.show_license_input:
            with st.expander(t['license_input'], expanded=True):
                license_key = st.text_input("License Key", type="password", label_visibility="collapsed")
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button(t['confirm'], use_container_width=True):
                    if license_key == "alpha2026": 
                        st.session_state.is_pro = True
                        st.session_state.show_license_input = False
                        st.rerun()
                    else:
                        st.error(t['wrong_key'])
                if c_btn2.button(t['cancel'], use_container_width=True):
                    st.session_state.show_license_input = False
                    st.rerun()
    
    st.markdown("---")
    st.caption("AlphaChart AI v21.5 Global")

IS_PRO = st.session_state.is_pro
t = TRANS[st.session_state.lang]
debug_code = None 

# --- 🎯 [설정] 심볼 파일명 ---
FREE_SYMBOL_URL = "https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png"
PRO_SYMBOL_FILE = "독수리 심볼.jfif"

# --- 🎯 [고정] 패턴 DB (언어별 이름 분기) ---
RAW_PATTERN_DB = {
    "A": {"file": "장대양봉 허리 지지 상승.jpg", "name_KR": "A. 장대양봉 허리 지지 상승", "name_EN": "A. Long Bullish Support", "name_JP": "A. 大陽線の腰押し支持上昇", "locked": False, "type": "A"},
    "B": {"file": "급락후 바닥에서 반등.jpg", "name_KR": "B. 급락후 바닥에서 반등", "name_EN": "B. Rebound after Plunge", "name_JP": "B. 急落後の底値反発", "locked": False, "type": "B"}, 
    "C": {"file": "큰하락 후 정배열 상승1파(컵위드핸들).jpg", "name_KR": "C. 큰하락 후 정배열 상승1파\n(컵위드핸들)", "name_EN": "C. 1st Wave after Drop\n(Cup w/ Handle)", "name_JP": "C. 大暴落後の整列上昇1波\n(カップ・ウィズ・ハンドル)", "locked": False, "type": "Custom"},
    "D": {"file": "큰하락 후 정배열 상승2파(컵위드핸들).jpg", "name_KR": "D. 큰하락 후 정배열 상승2파\n(컵위드핸들)", "name_EN": "D. 2nd Wave after Drop\n(Cup w/ Handle)", "name_JP": "D. 大暴落後の整列上昇2波\n(カップ・ウィズ・ハンドル)", "locked": not IS_PRO, "type": "Custom"},
    "E": {"file": "큰하락 후 정배열 상승3파(컵위드핸들).jpg", "name_KR": "E. 큰하락 후 정배열 상승3파\n(컵위드핸들)", "name_EN": "E. 3rd Wave after Drop\n(Cup w/ Handle)", "name_JP": "E. 大暴落後の整列上昇3波\n(カップ・ウィズ・ハンドル)", "locked": not IS_PRO, "type": "Custom"},
    "F": {"file": "적당한 하락 후 정배열 상승(컵위드핸들2형).jpg", "name_KR": "F. 적당한 하락 후 정배열 상승\n(컵위드핸들2형)", "name_EN": "F. Rise after Mild Drop\n(Cup w/ Handle Type 2)", "name_JP": "F. 適度な下落後の整列上昇\n(カップ・ウィズ・ハンドル2型)", "locked": not IS_PRO, "type": "Custom"},
    "G": {"file": "적당한 하락 후 정배열 상승2(컵위드핸들2형).jpg", "name_KR": "G. 적당한 하락 후 정배열 상승2\n(컵위드핸들2형)", "name_EN": "G. Rise after Mild Drop 2\n(Cup w/ Handle Type 2)", "name_JP": "G. 適度な下落後の整列上昇2\n(カップ・ウィズ・ハンドル2型)", "locked": not IS_PRO, "type": "Custom"},
    "H": {"file": "쌍바닥(단기간).jpg", "name_KR": "H. 쌍바닥(단기간)", "name_EN": "H. Double Bottom (Short-term)", "name_JP": "H. 二重底 (短期)", "locked": not IS_PRO, "type": "Custom"},
    "I": {"file": "쌍바닥(상승전 시작점).jpg", "name_KR": "I. 쌍바닥(상승전 시작점)", "name_EN": "I. Double Bottom (Start)", "name_JP": "I. 二重底 (上昇開始点)", "locked": not IS_PRO, "type": "Custom"},
    "J": {"file": "급락후 연속 도지.jpg", "name_KR": "J. 급락후 연속 도지", "name_EN": "J. Doji after Plunge", "name_JP": "J. 急落後の連続十字線", "locked": not IS_PRO, "type": "Custom"},
    "K": {"file": "횡보후 급락 및 연속도지.jpg", "name_KR": "K. 횡보후 급락 및 연속도지", "name_EN": "K. Plunge & Doji after Flat", "name_JP": "K. 横ばい後の急落・連続十字", "locked": not IS_PRO, "type": "Custom"},
    "L": {"file": "하락 횡보, 급락후 양봉.jpg", "name_KR": "L. 하락 횡보, 급락후 양봉", "name_EN": "L. Bullish after Drop & Flat", "name_JP": "L. 下落横ばい、急落後の陽線", "locked": not IS_PRO, "type": "Custom"},
    "M": {"file": "장기횡보, 급락후 바닥확인 연속캔들.jpg", "name_KR": "M. 장기횡보, 급락후\n바닥확인 연속캔들", "name_EN": "M. Bottom Check Candles\nafter Long Flat & Plunge", "name_JP": "M. 長期横ばい、急落後の\n底値確認連続ローソク足", "locked": not IS_PRO, "type": "Custom"},
    "N": {"file": "3중바닥.jpg", "name_KR": "N. 3중바닥", "name_EN": "N. Triple Bottom", "name_JP": "N. 三重底 (トリプルボトム)", "locked": not IS_PRO, "type": "Custom"}
}    
    
if 'selected_key' not in st.session_state:
    st.session_state.selected_key = "A"

def update_pattern(key):
    st.session_state.selected_key = key

# 2. 디자인 시스템
theme_color = "#fbbf24" if IS_PRO else "#38bdf8"
bg_gradient = "linear-gradient(135deg, #1e293b 0%, #000000 100%)" if IS_PRO else "linear-gradient(135deg, #0f172a 0%, #334155 100%)"

symbol_style = "border: 4px solid #fbbf24; border-radius: 50%; box-shadow: 0 0 25px rgba(251, 191, 36, 0.6); animation: dynamic-pulse 2s infinite;" if IS_PRO else "animation: dynamic-pulse 2.5s infinite;"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700;800;900&display=swap');
    * {{ font-family: 'Pretendard', sans-serif; }}
    .stApp {{ background-color: #f8fafc; color: #1e293b; }}
    
    @keyframes dynamic-pulse {{
        0% {{ transform: translateY(0px) scale(1); filter: drop-shadow(0 5px 15px rgba(56, 189, 248, 0.4)); }}
        50% {{ transform: translateY(-8px) scale(1.03); filter: drop-shadow(0 15px 25px rgba(56, 189, 248, 0.7)); }}
        100% {{ transform: translateY(0px) scale(1); filter: drop-shadow(0 5px 15px rgba(56, 189, 248, 0.4)); }}
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
    
    div.stButton > button {{
        width: 100%;
        min-height: 4.5rem;
        height: auto;
        white-space: pre-wrap !important;
        word-wrap: break-word;
        line-height: 1.4 !important;
        padding: 8px 5px !important;
        vertical-align: middle;
        font-size: 14px !important;
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
    
    /* 💡 텍스트 줄바꿈 및 정렬 개선 */
    .mission-box {{ 
        background: white; padding: 25px; border-radius: 15px; 
        border: 1px solid #e2e8f0; margin-bottom: 1.5rem; 
        line-height: 1.8; color: #334155; 
        font-size: 17px;
        word-break: keep-all; 
        overflow-wrap: break-word;
    }}
    .mission-highlight {{ color: {'#b45309' if IS_PRO else '#0284c7'}; font-weight: 800; }}
    
    .pattern-info {{ 
        font-size: 16px; color: #334155; line-height: 1.7; 
        background: #f1f5f9; padding: 20px; border-radius: 10px; 
        border-left: 5px solid {theme_color}; margin-bottom: 20px; 
        word-break: keep-all; 
        overflow-wrap: break-word;
    }}
    
    .result-card {{ 
        padding: 18px; border-radius: 12px; background: white; border: 1px solid #e2e8f0; margin-bottom: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); 
    }}
    .stock-info {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
    .stock-name {{ font-weight: 900; font-size: 19px; color: #0f172a; }}
    .stock-code {{ font-size: 13px; color: #64748b; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; margin-left: 5px; }}
    .sim-score {{ font-size: 20px; font-weight: 900; color: {'#b45309' if IS_PRO else '#0284c7'}; }}
    
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
        <a href="https://your-payment-link.com" target="_blank" class="upgrade-pro-btn">{t['upgrade_btn']}</a>
    </div>"""
st.markdown(header_html, unsafe_allow_html=True)

# 4. 핵심 설명문 (언어별)
st.markdown(t['mission_html'], unsafe_allow_html=True)

# --- 🌏 글로벌 증시 선택 ---
st.markdown(f"### 🏛️ {t['market_select']}") 
c_m1, c_m2 = st.columns([2, 1])
with c_m1:
    market_map = {"🇰🇷 Korea (KRX)": "KRX", "🇺🇸 USA (NASDAQ)": "NASDAQ", "🇺🇸 USA (NYSE)": "NYSE", "🇯🇵 Japan (TSE)": "TSE", "🇭🇰 Hong Kong (HKEX)": "HKEX"}
    market_label = st.selectbox("Market", list(market_map.keys()), label_visibility="collapsed")
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
        limit_val = st.slider(t['limit_search'].format(total_count), 10, total_count, min(1000, total_count), label_visibility="collapsed")
        st.success(t['pro_active_msg'].format(limit_val))
    else:
        limit_val = st.slider(t['limit_search_free'].format(total_count), 10, total_count, 300, disabled=True, label_visibility="collapsed")
        st.caption(t['free_limit_msg'])

# --- 🎯 상세 필터 설정 ---
with st.expander(t['filter_detail']):
    c_f1, c_f2 = st.columns(2)
    only_bullish = c_f1.checkbox(t['filter_bullish'], value=False)
    only_doji = c_f2.checkbox(t['filter_doji'], value=False)
    
    st.markdown("---")
    st.caption(t['period_set_caption'])
    
    cur_key = st.session_state.selected_key
    # 언어에 맞는 이름 가져오기
    name_key = 'name_' + st.session_state.lang
    cur_name = RAW_PATTERN_DB[cur_key][name_key].replace('\n', ' ')
    if not IS_PRO and RAW_PATTERN_DB[cur_key]['locked']:
        cur_name = "🔒 " + cur_name

    st.info(t['period_info_fmt'].format(cur_name, st.session_state.detected_period))
    search_period = st.session_state.detected_period

# --- 💡 패턴 섹션 ---
st.markdown(t['section1_title'], unsafe_allow_html=True)
st.markdown(t['pattern_desc'], unsafe_allow_html=True)

# 패턴 버튼 그리기 (5개씩 3줄)
keys = list(RAW_PATTERN_DB.keys())
cols1 = st.columns(5)
cols2 = st.columns(5)
cols3 = st.columns(5)

for i, key in enumerate(keys):
    if i < 5:
        target_col = cols1[i]
    elif i < 10:
        target_col = cols2[i-5]
    else:
        target_col = cols3[i-10] # 11~15번째 패턴 (K~O)
    
    with target_col:
        p = RAW_PATTERN_DB[key]
        # 언어별 이름
        display_name = p['name_' + st.session_state.lang]
        if not IS_PRO and p['locked']:
            display_name = "🔒 " + display_name
        
        btn_type = "primary" if st.session_state.selected_key == key else "secondary"
        
        st.button(display_name, key=f"btn_{key}", use_container_width=True, type=btn_type, on_click=update_pattern, args=(key,))

# --- 📷 2. 나만의 차트 업로드 ---
st.markdown(t['section2_title']) 
uploaded_file = st.file_uploader(t['upload_label'], type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
st.caption(t['upload_caption'])

if uploaded_file:
    target_input = uploaded_file
    is_path_mode = False
    sel_p_name = "User Upload"
    sel_p_type = "Custom"
    sel_p_locked = False
else:
    sel_key = st.session_state.selected_key
    sel_p = RAW_PATTERN_DB[sel_key]
    target_input = sel_p['file']
    is_path_mode = True
    sel_p_name = sel_p['name_' + st.session_state.lang].replace("\n", "") 
    sel_p_type = sel_p.get('type', 'Custom')
    sel_p_locked = sel_p['locked']

# --- 🧠 분석 엔진 ---
def count_candles_engine(img):
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        if np.mean(gray) > 127: # 밝은 배경
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else: # 어두운 배경
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)) 
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return 20

        valid_widths = []
        height, width = img.shape[:2]
        min_h = height * 0.02 
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > min_h: 
                valid_widths.append(w)
                
        if not valid_widths: return 20

        median_w = np.median(valid_widths)
        if median_w == 0: median_w = 1
        
        total_cnt = 0
        for w in valid_widths:
            cnt = max(1, round(w / median_w))
            total_cnt += cnt
            
        if total_cnt < 5: return 20
        if total_cnt > 120: return 60 
        
        return int(total_cnt)

    except Exception as e:
        return 20

def extract_features_engine(img_input, is_file_path=False):
    try:
        if is_file_path:
            img_array = np.fromfile(img_input, np.uint8); img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8); img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return None
        
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
        st.image(uploaded_file, caption=t['section2_title'], width=300)
        
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
            _, _, detected_cnt = feat_data
            if st.session_state.detected_period != detected_cnt:
                st.session_state.detected_period = detected_cnt
                st.rerun()

    if feat_data:
        user_p, _, cnt = feat_data
        st.markdown(f"""<div style="margin-top:10px; margin-bottom:5px;"><span class="info-tag">{t['ai_analysis_badge']}</span> {t['candles_detected'].format(cnt)} │ <b>{t['ai_pattern_shape']}</b></div>""", unsafe_allow_html=True)
        
        user_p_norm = MinMaxScaler().fit_transform(user_p.reshape(-1, 1)).flatten()
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.plot(user_p_norm, color=theme_color, lw=2.5)
        ax.axis('off')
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        
    elif sel_p_locked: st.warning(t['pro_only_model'])

clean_name = sel_p_name.replace('\n', ' ') 
button_label = t['btn_analyze'].format(clean_name)

if st.button(button_label, type="primary", use_container_width=True):
    if sel_p_locked and not uploaded_file:
        st.error(t['error_pro_only'])
    elif not feat_data:
        st.error(t['error_no_file'])
    else:
        period_msg = f" | {t['period_set_caption']}: {search_period}"
        info_msg = f"({limit_val}{period_msg})"
        st.info(t['scanning_msg'].format(info_msg))
        
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
                found_name = "Target"
                for s in stock_data:
                    if s[0] == debug_code: found_name = s[1]; break
                target_stocks.insert(0, [debug_code, found_name])

        total_scan = len(target_stocks)
        
        with ThreadPoolExecutor(max_workers=30) as ex:
            futures = [ex.submit(analyze_stock_legacy, s[0], s[1], user_p_norm, search_period, market_code, only_bullish, only_doji, sel_p_type, (s[0] == debug_code)) for s in target_stocks]
            for idx, f in enumerate(as_completed(futures)):
                res = f.result()
                if res:
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

        st.markdown(t['result_title'].format(len(final_display_list)))
        if not final_display_list: st.warning(t['no_result'])
        for i, res in enumerate(final_display_list):
            
            if market_code == "KRX":
                pc_link = f"https://finance.naver.com/item/fchart.naver?code={res['code']}"
                mo_link = f"https://m.stock.naver.com/domestic/stock/{res['code']}/chart"
                links_html = f'<div class="btn-row"><a href="{pc_link}" target="_blank" class="custom-btn btn-pc">{t["pc_chart"]}</a><a href="{mo_link}" target="_blank" class="custom-btn btn-mo">{t["mo_chart"]}</a></div>'
            elif market_code in ["NASDAQ", "NYSE"]:
                link = f"https://www.tradingview.com/chart/?symbol={res['code']}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">{t["chart_view"]}</a>'
            elif market_code == "TSE":
                link = f"https://www.tradingview.com/chart/?symbol=TSE:{res['code'].replace('.T','')}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">{t["chart_view"]}</a>'
            elif market_code == "HKEX":
                link = f"https://www.tradingview.com/chart/?symbol=HKEX:{res['code'].replace('.HK','')}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">{t["chart_view"]}</a>'
            else:
                link = f"https://finance.yahoo.com/quote/{res['code']}"
                links_html = f'<a href="{link}" target="_blank" class="custom-btn btn-global">{t["chart_view"]}</a>'

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
            st.markdown(f"""<div class="locked-card">{t['locked_msg']}</div>""", unsafe_allow_html=True)

st.caption("AlphaChart AI v21.5 Global")
