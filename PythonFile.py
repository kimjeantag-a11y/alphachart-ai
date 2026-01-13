import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import cv2
import os
import base64
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection 
import json

# --- 🔐 [인증 및 시크릿 설정] ---
try:
    # secrets.toml 파일에서 gsheets 섹션을 읽어옵니다.
    gs_info = st.secrets["gsheets"]
    
    # 사장님의 시트 주소를 가져옵니다.
    spreadsheet_url = gs_info["spreadsheet"]
    
except Exception as e:
    st.error(f"초기 설정(secrets.toml) 읽기 오류: {e}")
    st.info("프로젝트 폴더/.streamlit/secrets.toml 파일이 있는지 확인해주세요.")
    st.stop()

# --- 🔐 [라이선스 확인 함수] ---
def check_license_from_sheet(input_key):
    try:
        # 1. secrets.toml의 [gsheets] 설정을 읽어 연결을 만듭니다.
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # 2. '어떤 시트'를 읽을지 주소를 명확하게 지정합니다.
        df = conn.read(
            spreadsheet=spreadsheet_url, 
            ttl=0
        )
        
        # 3. 데이터가 잘 왔는지 확인
        if df is None or df.empty:
            return False, "시트 데이터를 가져오지 못했습니다.", None

        # 4. 라이선스 키 검색 및 인증 (사장님 시트 컬럼명: license_key)
        df['license_key'] = df['license_key'].astype(str).str.strip()
        user_row = df[df['license_key'] == str(input_key).strip()]
        
        if not user_row.empty:
            if str(user_row.iloc[0]['status']).lower() == 'active':
                expiry_date = user_row.iloc[0]['expiry_date']
                return True, "인증 성공", expiry_date
            return False, "비활성화된 라이선스입니다.", None
        return False, "유효하지 않은 라이선스 키입니다.", None

    except Exception as e:
        st.error(f"서버 연결 최종 시도 실패: {e}")
        return False, f"연결 실패: {e}", None

# --- 🌐 언어 데이터 팩 (Language Pack) ---
TRANS = {
    "KR": {
        "sidebar_header": "⚙️ 설정",
        "license_active": "✅ PRO 라이선스 활성",
        "license_info": "남은 기간: {}",
        "logout": "로그아웃 / 리셋",
        "current_free": "현재: 무료 버전 (Free)",
        "upgrade_btn": "👑 PRO 업그레이드 (구매하기)",
        "license_input": "🔑 라이선스 키 입력",
        "confirm": "인증 확인",
        "cancel": "취소",
        "checking": "라이선스 확인 중...",
        "market_select": "시장 선택",
        "limit_search": "검색 범위 제한 (전체 {:,}개 중)",
        "limit_search_free": "검색 범위 제한 (시가총액 상위 {:,}개 중)",
        "pro_active_msg": "✅ PRO 활성화: {}개 정밀 스캔 가능",
        "free_limit_msg": "🔒 무료 버전은 시가총액 상위 300개만 스캔 가능",
        "filter_detail": "🎯 상세 필터 설정 (눌러서 열기)",
        "filter_bullish": "마지막(최근) 캔들 양봉(상승)만 보기",
        "filter_doji": "마지막(최근) 캔들 도지(십자가)만 보기",
        "filter_hammer": "마지막 캔들 양봉/도지이면서 아래꼬리 아주 긴 것(망치형)",
        "period_set_caption": "⏱️ 분석 기간 설정",
        "period_info_fmt": "💠 **[{}]** 기준: AI가 차트에서 **오늘부터 과거 {}일** 치 패턴을 자동 인식하여 분석합니다.",
        "section1_title": "### 🧬 1. AlphaChart AI 에 기본 장착된 패턴 모델 선택 <span style='font-size:16px; color:#64748b; font-weight:normal;'>(차트매매 대가들이 사용)</span>",
        "guide_html": """
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; line-height: 1.6; color: #333;">
            이 패턴들은 상승 지속형 6개, 하락에서 반등형 9개이며 하락 경직 또는 눌림목 상태이므로,<br>
            내일 또는 모레 매수해도 단타나 스윙으로 성공할 확률이 높은 대표적인 모델입니다.<br><br>
            단, 기업가치, 거래량, 뉴스, 공시 등 내재가치와 외부환경은 매매 전에 함께 고려해야 할 것입니다.<br><br>
            물론, 복잡한 내재와 외부를 고려하지 않고 그냥 매수해도 안전할 확률이 높은 편이지만,<br> 
            돌다리도 두드리고 건널 필요는 있겠지요.<br><br>    
            자! 이제, <span class="mission-highlight">도플갱어를 찾은 후 최종 선택</span>은 여러분의 몫입니다.
        </div>
        """,
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
        "result_title": "### 🏆 분석 결과 (80% 이상 일치: {}개)",
        "no_result": "80% 이상 일치하는 종목을 찾지 못했습니다.",
        "chart_view": "📈 차트 보기",
        "pc_chart": "💻 PC용 차트",
        "mo_chart": "📱 모바일용 차트",
        "locked_msg": "🔒 TOP 6위 이후 결과 및 전종목 정밀 스캔은<br>PRO 버전에서 확인 가능합니다.",
        "mission_html": """
        <div class="mission-box">
            먼저, <span class="mission-highlight">AlphaChart의 미션</span>은 급등주, 대박주를 찾아 투자하도록 돕는 것이 아닙니다. 그런 차트들은 통계적으로 유의하지 않기 때문입니다.<br>
            즉, 그동안의 급등주, 대박주들의 패턴이 일정하지 않아 성공률이 낮습니다.<br>
            AlphaChart의 미션은 내일이나 모레 몇프로라도 상승할 확률이 높고 안전한 종목을 찾는 것입니다.<br>
            안전하다는 것은, 상승 패턴에서는 <span class="mission-highlight">상승이 유지되는 상태</span>, 하락 패턴에서는 드디어 <span class="mission-highlight">상승으로 전환하기 직전</span>의 상태를 말합니다.<br><br>
            오랜 주식 거래의 경험과 반복을 통해, 캔들의 단순한 형태보다는 수거래일 동안의 <span class="mission-highlight">추세와 마지막 몇개의 캔들 형태를 함께 보는 것</span>이 중요하다는 사실이 수많은 연구자와 투자자들로부터 검증되어 왔습니다.<br><br>
            이러한 과거의 패턴을 통한 단기의 패턴 예측, 그 중에서도 오늘까지의 차트를 보고 내일의 캔들 형태와 방향을 알고 싶습니다. 그래서, 일일이 상장된 모든 종목을 찾아서 내가 원하는 차트와 유사한 종목을 찾고, 증권사의 프로그램을 통해 원하는 그림을 그리거나 조건을 입력해서 검색해 왔죠. 그러나 무겁고 불편하고 부정확했습니다.<br><br>
            이제 <span class="mission-highlight">AlphaChart AI</span>가 몇 분만에 도플갱어 종목들을 찾아줄 수 있습니다.<br><br>
            또, 많은 시행착오를 통해 차트 매매의 대가들이 정립해 놓은 검증된 패턴들을 기본 장착하여, 사용자가 일일이 관심 차트를 찾아 업로드 하지 않아도, 내일이나 모레 매수 할 만한 종목 후보군을 찾을 수 있게 되었습니다.<br><br>
            차트 매매를 주로 하시는 데이 트레이더, 기업가치와 함께 차트를 같이 보시는 단중기 트레이더 모두 AlphaChart AI를 통해 불필요한 시간 낭비 없이, <span class="mission-highlight">투자의 성공확률을 극대화</span> 하시길 기원합니다.<br><br>
            세계인이 함께 쓰는 글로벌 서칭 시스템으로서 과부하를 막고 양질의 결과를 도출하기 위해, 부득이 무료버전은 기능을 제한하고 있습니다.<br>
            추후 서버 증설 등 투자 확대를 통해 무료 범위를 확대할 계획이니 너그러이 양해 부탁드립니다. <br><br>
            원하는 차트의 도플갱어가 어쩌면 매일 나오지는 않을 수도 있습니다. 하지만 성공률을 높이는 것이 중요하니 또 내일을 기다리면 됩니다.<br><br>
            앞으로, 위에서 매도해야 하는 패턴도 제공할 계획이며, 계속해서<span class="mission-highlight"> 혁신적인 인사이트</span>를 제시하겠습니다. 감사합니다.
        </div>"""
    },
    "EN": {
        "sidebar_header": "⚙️ Settings",
        "license_active": "✅ PRO License Active",
        "license_info": "Remaining: {}",
        "logout": "Logout / Reset",
        "current_free": "Current: Free Version",
        "upgrade_btn": "👑 Upgrade to PRO (Buy Now)",
        "license_input": "🔑 Enter License Key",
        "confirm": "Verify",
        "cancel": "Cancel",
        "checking": "Verifying...",
        "market_select": "Select Market",
        "limit_search": "Search Limit (Scanning {:,} stocks)",
        "limit_search_free": "Search Limit (Top {:,} Market Cap)",
        "pro_active_msg": "✅ PRO Active: Precision scan of {} stocks",
        "free_limit_msg": "🔒 Free version scans top 300 market cap only",
        "filter_detail": "🎯 Advanced Filters (Click to expand)",
        "filter_bullish": "Last candle must be Bullish (Green/Red)",
        "filter_doji": "Last candle must be Doji (Cross)",
        "filter_hammer": "Last candle Bullish/Doji with Very Long Lower Shadow (Hammer)",
        "period_set_caption": "⏱️ Analysis Period",
        "period_info_fmt": "💠 Based on **[{}]**: AI automatically detects and analyzes the pattern of **past {} days from today**.",
        "section1_title": "### 🧬 1. Select AI Built-in Patterns <span style='font-size:16px; color:#64748b; font-weight:normal;'>(Used by Master Traders)</span>",
        "guide_html": """
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; line-height: 1.6; color: #333;">
            These patterns consist of 6 bullish continuation types and 9 reversal-from-bottom types. As they represent a state of consolidated decline or pullback,<br>
            they are representative models with a high probability of success for day or swing trading, even if bought tomorrow or the day after.<br><br>
            However, intrinsic values and external environments such as corporate value, trading volume, news, and disclosures should be considered together before trading.<br><br>
            Of course, the probability of safety is high even if you buy without considering complex internal and external factors,<br>
            but it is necessary to look before you leap.<br><br>
            Now! After finding the doppelganger, <span class="mission-highlight">the final choice</span> is yours.
        </div>
        """,
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
        "result_title": "### 🏆 Analysis Results (Found {} stocks > 80%)",
        "no_result": "No stocks found with > 80% similarity.",
        "chart_view": "📈 View Chart",
        "pc_chart": "💻 PC Chart",
        "mo_chart": "📱 Mobile Chart",
        "locked_msg": "🔒 Results from Top 6 onwards & Full Scan<br>available in PRO Version.",
        "mission_html": """
        <div class="mission-box">
            First of all, <span class="mission-highlight">AlphaChart's mission</span> is not to help you find skyrocketing or jackpot stocks. This is because such charts are not statistically significant.<br>
            In other words, the patterns of past jackpot stocks are inconsistent, resulting in a low success rate.<br>
            AlphaChart's mission is to find safe stocks with a high probability of rising even a few percent tomorrow or the day after.<br>
            Being safe means a state where <span class="mission-highlight">the rise is maintained</span> in an upward pattern, or a state <span class="mission-highlight">just before turning into a rise</span> in a downward pattern.<br><br>
            Through extensive experience and repetition in stock trading, it has been verified by numerous researchers and investors that it is more important to look at the <span class="mission-highlight">trend over several trading days combined with the shape of the last few candles</span> rather than the simple shape of a single candle.<br><br>
            Through predicting short-term patterns using these past patterns, we specifically want to know tomorrow's candle shape and direction based on the chart up to today. Previously, we had to manually search through all listed stocks to find similar charts, or use heavy and inaccurate PC programs to draw patterns or input conditions. However, it was heavy, inconvenient, and inaccurate.<br><br>
            Now, <span class="mission-highlight">AlphaChart AI</span> can find doppelganger stocks in just a few minutes.<br><br>
            Also, by equipping verified patterns established by chart trading masters through many trials and errors, you can now find candidate stocks to buy tomorrow or the day after without uploading your own chart.<br><br>
            We hope that both day traders and short-to-medium term traders who look at charts alongside corporate value will <span class="mission-highlight">maximize their success probability</span> without wasting unnecessary time through AlphaChart AI.<br><br>
            As a global searching system used by people around the world, we inevitably limit the features of the free version to prevent server overload and ensure high-quality results.<br>
            We plan to expand the free scope through future investments such as server expansion, so we ask for your generous understanding.<br><br>
            A doppelganger of the chart you want may not appear every day. However, it is important to increase the success rate, so you can just wait for tomorrow.<br><br>
            We plan to provide patterns for selling at the top in the future, and we will continue to present <span class="mission-highlight">innovative insights</span>. Thank you.
        </div>"""
    },
    "JP": {
        "sidebar_header": "⚙️ 設定",
        "license_active": "✅ PROライセンス有効",
        "license_info": "残り期間: {}",
        "logout": "ログアウト / リセット",
        "current_free": "現在: 無料版 (Free)",
        "upgrade_btn": "👑 PROにアップグレード (購入)",
        "license_input": "🔑 ライセンスキー入力",
        "confirm": "確認",
        "cancel": "キャンセル",
        "checking": "確認中...",
        "market_select": "市場選択",
        "limit_search": "検索範囲制限 (全体 {:,} 銘柄中)",
        "limit_search_free": "検索範囲制限 (時価総額上位 {:,} 銘柄)",
        "pro_active_msg": "✅ PRO有効化: {}銘柄 精密スキャン",
        "free_limit_msg": "🔒 無料版は時価総額上位300銘柄のみスキャン可能",
        "filter_detail": "🎯 詳細フィルタ設定 (クリックして展開)",
        "filter_bullish": "直近ローソク足が「陽線」のみ",
        "filter_doji": "直近ローソク足が「十字線(同時線)」のみ",
        "filter_hammer": "直近ローソク足が陽線/十字で下ヒゲが非常に長いもの (ハンマー)",
        "period_set_caption": "⏱️ 分析期間設定",
        "period_info_fmt": "💠 **[{}]** 基準: AIがチャートから **今日から過去{}日分** のパターンを自動認識して分析します。",
        "section1_title": "### 🧬 1. AlphaChart AI 搭載のパターンモデルを選択 <span style='font-size:16px; color:#64748b; font-weight:normal;'>(チャート売買の大家たちが使用)</span>",
        "guide_html": """
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; line-height: 1.6; color: #333;">
            これらのパターンは上昇持続型6つ、下落からの反発型9つであり、下落硬直または押し目買いの状態にあるため、<br>
            明日や明後日に購入してもデイトレやスイングトレードで成功する確率が高い代表的なモデルです。<br><br>
            ただし、企業価値、出来高、ニュース、開示情報などの本質的価値と外部環境は、売買前に併せて考慮する必要があります。<br><br>
            もちろん、複雑な内外要因を考慮せずに購入しても安全である確率は高い方ですが、<br>
            石橋を叩いて渡る必要はあるでしょう。<br><br>
            さあ！ドッペルゲンガーを見つけた後の<span class="mission-highlight">最終選択</span>は、あなたの役割です。
        </div>
        """,
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
        "result_title": "### 🏆 分析結果 (80%以上一致: {}件)",
        "no_result": "80%以上一致する銘柄が見つかりませんでした。",
        "chart_view": "📈 チャートを見る",
        "pc_chart": "💻 PC用チャート",
        "mo_chart": "📱 モバイル用チャート",
        "locked_msg": "🔒 6位以降の結果および全銘柄スキャンは<br>PROバージョンで確認可能です。",
        "mission_html": """
        <div class="mission-box">
            まず、<span class="mission-highlight">AlphaChartのミッション</span>は、急騰株や大化け株を探して投資を助けることではありません。そのようなチャートは統計的に有意ではないためです。<br>
            つまり、これまでの急騰株や大化け株のパターンは一定しておらず、成功率が低いのです。<br>
            AlphaChartのミッションは、明日や明後日に数パーセントでも上昇する確率が高く、安全な銘柄を見つけることです。<br>
            安全とは、上昇パターンでは<span class="mission-highlight">上昇が維持されている状態</span>、下落パターンではついに<span class="mission-highlight">上昇に転じる直前</span>の状態を指します。<br><br>
            長年の株式取引の経験と反復を通じて、単なるローソク足の形よりも、数取引日間の<span class="mission-highlight">トレンドと最後の数本のローソク足の形を共に見ること</span>が重要であるという事実が、数多くの研究者や投資家によって検証されてきました。<br><br>
            このような過去のパターンを通じた短期パターン予測、その中でも私たちは今日までのチャートを見て、明日のローソク足の形と方向を知りたいのです。これまでは、上場されている全銘柄から自分が望むチャートと類似した銘柄を手作業で探したり、証券会社の重いプログラムを使って絵を描いたり条件を入力して検索してきました。しかし、それは重くて不便で、不正確でした。<br><br>
            今や<span class="mission-highlight">AlphaChart AI</span>が数分でドッペルゲンガー銘柄を見つけ出します。<br><br>
            また、多くの試行錯誤を経てチャート売買の大家たちが確立した検証済みのパターンを基本搭載しており、ユーザーがわざわざ関心のあるチャートを探してアップロードしなくても、明日や明後日に購入すべき銘柄候補群を見つけることができます。<br><br>
            チャート売買を主とするデイトレーダーの方も、企業価値と共にチャートを見るスイングトレーダーの方も、AlphaChart AIを通じて不必要な時間の浪費なく、<span class="mission-highlight">投資の成功確率を最大化</span>されることを祈念いたします。<br><br>
            世界中の人々が共に使用するグローバルサーチングシステムとして、過負荷を防ぎ良質な結果を導き出すために、やむを得ず無料版は機能を制限しております。<br>
            今後、サーバー増設などの投資拡大を通じて無料範囲を拡大する計画ですので、何卒寛大なご理解をお願い申し上げます。<br><br>
            ご希望のチャートのドッペルゲンガーは、毎日現れるわけではないかもしれません。しかし、成功率を高めることが重要ですので、また明日を待てばよいのです。<br><br>
            今後、高値で売却すべきパターンも提供する計画であり、引き続き<span class="mission-highlight">革新的なインサイト</span>を提示してまいります。ありがとうございます。
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
if 'license_expiry_msg' not in st.session_state:
    st.session_state.license_expiry_msg = ""

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
    
    # 🔐 라이선스 상태 표시 및 관리
    if st.session_state.is_pro:
        st.success(t['license_active'])
        if st.session_state.license_expiry_msg:
            st.caption(t['license_info'].format(st.session_state.license_expiry_msg))
            
        if st.button(t['logout'], use_container_width=True):
            st.session_state.is_pro = False
            st.session_state.show_license_input = False
            st.session_state.license_expiry_msg = ""
            st.rerun()
    else:
        st.info(t['current_free'])
        if not st.session_state.show_license_input:
            # 🔗 Gumroad 구매 페이지 링크 (여기에 실제 Gumroad 주소를 넣으세요)
            gumroad_link = "https://gumroad.com/l/YOUR_PRODUCT_LINK" 
            st.markdown(f'<a href="{gumroad_link}" target="_blank" class="upgrade-pro-btn" style="text-align:center; margin-bottom:10px;">{t["upgrade_btn"]}</a>', unsafe_allow_html=True)
            
            if st.button(t['license_input'], use_container_width=True):
                st.session_state.show_license_input = True
                st.rerun()
                
        if st.session_state.show_license_input:
            with st.expander(t['license_input'], expanded=True):
                input_key = st.text_input("License Key", type="password", label_visibility="collapsed")
                c_btn1, c_btn2 = st.columns(2)
                
                if c_btn1.button(t['confirm'], use_container_width=True):
                    with st.spinner(t['checking']):
                        # 구글 시트 확인 로직 호출
                        is_valid, msg, expiry_info = check_license_from_sheet(input_key)
                        
                        if is_valid:
                            st.session_state.is_pro = True
                            st.session_state.show_license_input = False
                            st.session_state.license_expiry_msg = expiry_info
                            st.success(f"Welcome! ({expiry_info})")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(msg)
                            
                if c_btn2.button(t['cancel'], use_container_width=True):
                    st.session_state.show_license_input = False
                    st.rerun()
    
    st.markdown("---")
    st.caption("AlphaChart AI v21.5 Global (Paid)")

IS_PRO = st.session_state.is_pro
t = TRANS[st.session_state.lang]
debug_code = None 

# --- 🎯 [설정] 심볼 파일명 ---
FREE_SYMBOL_URL = "https://raw.githubusercontent.com/kimjeantag-a11y/alphachart-ai/main/candlestick_ai_symbol.png"
PRO_SYMBOL_FILE = "독수리 심볼.jfif"

# --- 🎯 [고정] 패턴 DB (H, L 수정 반영) ---
RAW_PATTERN_DB = {
    "A": {"file": "장대양봉 허리 지지 상승.jpg", "name_KR": "A. 장대양봉 허리 지지 상승", "name_EN": "A. Long Bullish Support", "name_JP": "A. 大陽線の腰押し支持上昇", "locked": False, "type": "A"},
    "B": {"file": "급락후 바닥에서 반등.jpg", "name_KR": "B. 급락후\n 바닥에서 반등", "name_EN": "B. Rebound after Plunge", "name_JP": "B. 急落後の底値反発", "locked": False, "type": "B"}, 
    "C": {"file": "큰하락 후 정배열 상승1파(컵위드핸들).jpg", "name_KR": "C. 큰하락 후\n 정배열 상승1파\n(컵위드핸들)", "name_EN": "C. 1st Wave after Drop\n(Cup w/ Handle)", "name_JP": "C. 大暴落後の整列上昇1波\n(カップ・ウィズ・ハンドル)", "locked": False, "type": "Custom"},
    "D": {"file": "큰하락 후 정배열 상승2파(컵위드핸들).jpg", "name_KR": "D. 큰하락 후\n 정배열 상승2파\n(컵위드핸들)", "name_EN": "D. 2nd Wave after Drop\n(Cup w/ Handle)", "name_JP": "D. 大暴落後の整列上昇2波\n(カップ・ウィズ・ハンドル)", "locked": not IS_PRO, "type": "Custom"},
    "E": {"file": "큰하락 후 정배열 상승3파(컵위드핸들).jpg", "name_KR": "E. 큰하락 후\n 정배열 상승3파\n(컵위드핸들)", "name_EN": "E. 3rd Wave after Drop\n(Cup w/ Handle)", "name_JP": "E. 大暴落後の整列上昇3波\n(カップ・ウィズ・ハンドル)", "locked": not IS_PRO, "type": "Custom"},
    "F": {"file": "적당한 하락 후 정배열 상승(컵위드핸들2형).jpg", "name_KR": "F. 적당한 하락 후 정배열 상승\n(컵위드핸들2형)", "name_EN": "F. Rise after Mild Drop\n(Cup w/ Handle Type 2)", "name_JP": "F. 適度な下落後の整列上昇\n(カップ・ウィズ・ハンドル2型)", "locked": not IS_PRO, "type": "Custom"},
    "G": {"file": "적당한 하락 후 정배열 상승2(컵위드핸들2형).jpg", "name_KR": "G. 적당한 하락 후 정배열 상승2\n(컵위드핸들2형)", "name_EN": "G. Rise after Mild Drop 2\n(Cup w/ Handle Type 2)", "name_JP": "G. 適度な下落後の整列上昇2\n(カップ・ウィズ・ハンドル2型)", "locked": not IS_PRO, "type": "Custom"},
    "H": {"file": "쌍바닥(완만).jpg", "name_KR": "H. 쌍바닥(완만)", "name_EN": "H. Double Bottom (Gentle)", "name_JP": "H. 二重底 (緩やか)", "locked": not IS_PRO, "type": "Custom"},
    
    "I": {"file": "쌍바닥(급경사).jpg", "name_KR": "I. 쌍바닥(급경사)", "name_EN": "I. Double Bottom (Steep)", "name_JP": "I. 二重底 (急勾配)", "locked": not IS_PRO, "type": "Custom"},
    
    "J": {"file": "쌍바닥(상승전 시작점).jpg", "name_KR": "J. 쌍바닥\n(상승전 시작점)", "name_EN": "J. Double Bottom (Start of Rise)", "name_JP": "J. 二重底 (上昇開始点)", "locked": not IS_PRO, "type": "Custom"},
    
    "K": {"file": "급락후 연속 도지.jpg", "name_KR": "K. 급락후\n 연속 도지", "name_EN": "K. Doji after Plunge", "name_JP": "K. 急落後の連続十字線", "locked": not IS_PRO, "type": "Custom"},
    
    "L": {"file": "횡보, 급락후 바닥확인.jpg", "name_KR": "L. 횡보, 급락후 바닥확인", "name_EN": "L. Bottom Check after Flat & Plunge", "name_JP": "L. 横ばい・急落後の底値確認", "locked": not IS_PRO, "type": "Custom", "fixed_period": 13},
    
    "M": {"file": "하락 횡보, 급락후 반등.jpg", "name_KR": "M. 하락 횡보,\n 급락후 반등", "name_EN": "M. Bullish after Drop & Flat", "name_JP": "M. 下落横ばい、急落後の陽線", "locked": not IS_PRO, "type": "Custom", "fixed_period": 29},
    
    "N": {"file": "장기횡보, 급락후 바닥확인 연속캔들.jpg", "name_KR": "N. 장기횡보,\n 급락후 바닥확인\n 연속캔들", "name_EN": "N. Bottom Check Candles\nafter Long Flat & Plunge", "name_JP": "N. 長期横ばい、急落後の\n底値確認連続ローソク足", "locked": not IS_PRO, "type": "Custom"},
    "O": {"file": "3중바닥.jpg", "name_KR": "O. 3중바닥", "name_EN": "O. Triple Bottom", "name_JP": "O. 三重底 (トリプルボトム)", "locked": not IS_PRO, "type": "Custom"}
}     
    
if 'selected_key' not in st.session_state:
    st.session_state.selected_key = "A"

def update_pattern(key):
    st.session_state.selected_key = key

# 2. 디자인 시스템 (모바일 토글 강조 + 헤더 Umlaut)
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
    
    /* 사이드바 열기 버튼 강력 강조 */
    [data-testid="stSidebarCollapsedControl"] {{
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        color: black !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.6) !important;
        animation: attention-pulse 2s infinite !important;
        width: 3rem !important;
        height: 3rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-top: 5px !important;
        margin-left: 5px !important;
        z-index: 999999 !important; /* 최상단 보장 */
    }}
    
    /* 화살표 아이콘 색상 강제 변경 */
    [data-testid="stSidebarCollapsedControl"] svg {{
        fill: black !important;
        stroke: black !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }}

    @keyframes attention-pulse {{
        0% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.7); }}
        70% {{ transform: scale(1.15); box-shadow: 0 0 0 15px rgba(251, 191, 36, 0); }}
        100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0); }}
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .upgrade-pro-btn:hover {{ transform: scale(1.05); box-shadow: 0 6px 8px rgba(0,0,0,0.2); }}
    
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
        <div style="font-size: 15px; color: #fbbf24; font-weight: 700; letter-spacing: 3px; margin-bottom: 10px;">MEET YOUR CHART DOPPELGÄNGER</div>
    </div>"""
else:
    header_html = f"""
    <div class="brand-container">
        <img src="{FREE_SYMBOL_URL}" class="symbol-img">
        <div style="font-size: 36px; font-weight: 900; color: white; letter-spacing: -1px;">AlphaChart AI</div>
        <div style="font-size: 15px; color: #38bdf8; font-weight: 700; letter-spacing: 3px; margin-bottom: 10px;">MEET YOUR CHART DOPPELGÄNGER</div>
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
        # [수정] step=100 설정
        limit_val = st.slider(t['limit_search'].format(total_count), 100, total_count, min(1000, total_count), step=100, label_visibility="collapsed")
        st.success(t['pro_active_msg'].format(limit_val))
    else:
        limit_val = st.slider(t['limit_search_free'].format(total_count), 10, total_count, 300, disabled=True, label_visibility="collapsed")
        st.caption(t['free_limit_msg'])

# --- 🎯 상세 필터 설정 ---
with st.expander(t['filter_detail']):
    c_f1, c_f2 = st.columns(2)
    only_bullish = c_f1.checkbox(t['filter_bullish'], value=False)
    only_doji = c_f2.checkbox(t['filter_doji'], value=False)
    only_hammer = st.checkbox(t['filter_hammer'], value=False)
    
    st.markdown("---")
    st.caption(t['period_set_caption'])
    
    cur_key = st.session_state.selected_key
    name_key = 'name_' + st.session_state.lang
    cur_name = RAW_PATTERN_DB[cur_key][name_key].replace('\n', ' ')
    if not IS_PRO and RAW_PATTERN_DB[cur_key]['locked']:
        cur_name = "🔒 " + cur_name

    st.info(t['period_info_fmt'].format(cur_name, st.session_state.detected_period))
    search_period = st.session_state.detected_period

# --- 💡 패턴 섹션 ---
st.markdown(t['section1_title'], unsafe_allow_html=True)
st.markdown(t['guide_html'], unsafe_allow_html=True)

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
    
    # [K, L, M 패턴 고정 로직]
    if 'fixed_period' in sel_p:
        st.session_state.detected_period = sel_p['fixed_period']

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

# [수정됨] require_hammer 인자 추가 및 로직 구현 (Strict Version)
def analyze_stock_legacy(code, name, user_p_norm, n_days=20, market_type="KRX", require_bullish=False, require_doji=False, require_hammer=False, pattern_type="Custom", force_include=False):
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
        
        # [신규 로직] 망치형 필터 (양봉/도지 + 긴 아래꼬리) - Strict Version
        if require_hammer:
            is_bullish = last_close >= last_open
            upper_shadow = last_high - max(last_open, last_close)
            lower_shadow = min(last_open, last_close) - last_low
            total_range = last_high - last_low

            if not (is_bullish or is_doji):
                filter_status = "Fail_Hammer_Shape"
            # 1. 위꼬리가 거의 없어야 함 (아래꼬리가 위꼬리의 3배 이상)
            elif lower_shadow < upper_shadow * 3.0:
                filter_status = "Fail_Upper_Shadow_Too_Long"
            else:
                tail_condition = False
                # 2. 몸통이 있는 경우: 아래꼬리가 몸통의 3배 이상 (아주 길어야 함)
                if body_size > 0:
                    if lower_shadow >= body_size * 3.0: tail_condition = True
                # 3. 도지인 경우: 아래꼬리가 전체 길이의 70% 이상 (대부분이 아래꼬리여야 함)
                elif total_range > 0:
                    if lower_shadow >= total_range * 0.7: tail_condition = True
                
                # [추가] 몸통이 너무 크면 안됨 (장대봉 방지) - 전체 길이의 30% 이하
                if total_range > 0 and (body_size / total_range) > 0.3:
                    tail_condition = False

                # [추가] 아래꼬리의 절대 비중이 60% 이상이어야 함 (안전장치)
                if total_range > 0 and (lower_shadow / total_range) < 0.6:
                    tail_condition = False
                
                if not tail_condition: filter_status = "Fail_Hammer_Tail_Length"

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
            # [수정] fixed_period가 있으면 그것을 우선 사용
            if 'fixed_period' in sel_p:
                detected_cnt = sel_p['fixed_period']
                
            if st.session_state.detected_period != detected_cnt:
                st.session_state.detected_period = detected_cnt
                st.rerun()

    if feat_data:
        user_p, _, cnt = feat_data
        st.markdown(f"""<div style="margin-top:10px; margin-bottom:5px;"><span class="info-tag">{t['ai_analysis_badge']}</span> {t['candles_detected'].format(st.session_state.detected_period)} │ <b>{t['ai_pattern_shape']}</b></div>""", unsafe_allow_html=True)
        
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
            # [수정됨] only_hammer 인자 전달
            futures = [ex.submit(analyze_stock_legacy, s[0], s[1], user_p_norm, search_period, market_code, only_bullish, only_doji, only_hammer, sel_p_type, (s[0] == debug_code)) for s in target_stocks]
            for idx, f in enumerate(as_completed(futures)):
                res = f.result()
                if res:
                    results.append(res)
                progress_bar.progress((idx + 1) / total_scan)
        
        # [수정] 80% 이상만 필터링
        results = [r for r in results if r['sim'] >= 80.0]
        results.sort(key=lambda x: x['sim'], reverse=True)
        
        final_display_list = []
        if IS_PRO:
            if len(results) < 10: final_display_list = results
            else: final_display_list = results[:100]
        else:
            final_display_list = results[:5]

        st.markdown(t['result_title'].format(len(results)))
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
