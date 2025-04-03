from flask import Flask, jsonify, render_template, request
from btc_analyzer import CryptoAnalyzer
import pandas as pd
import numpy as np
import json
import logging
import traceback
from datetime import datetime
import sys
import os

# 設置控制台輸出編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 全局變量存儲閾值設置
signal_settings = {
    'threshold': 3.0,
    'ratio': 1.2
}

# 自定義JSON編碼器處理NaN值
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            if pd.isna(obj) or np.isinf(obj):
                return None
            return float(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

SUPPORTED_SYMBOLS = {
    'BTC/USDT': '比特幣',
    'ETH/USDT': '以太坊',
    'DOGE/USDT': '狗狗幣'
}

# 全局錯誤處理
@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"發生未處理的錯誤: {error}")
    logging.error(traceback.format_exc())
    return jsonify({
        'error': '服務器內部錯誤，請稍後再試',
        'details': str(error)
    }), 500

@app.route('/')
def index():
    try:
        return render_template('index.html', symbols=SUPPORTED_SYMBOLS)
    except Exception as e:
        logging.error(f"渲染首頁時發生錯誤: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': '載入頁面時發生錯誤，請稍後再試'}), 500

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    try:
        data = request.json
        signal_settings['threshold'] = float(data['signal_threshold'])
        signal_settings['ratio'] = float(data['signal_ratio'])
        logging.info(f"閾值已更新: {signal_settings}")
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"更新閾值時發生錯誤: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze')
def analyze():
    symbol = request.args.get('symbol', 'BTC/USDT')
    logging.info(f"收到分析請求: {symbol}")
    try:
        if not symbol:
            logging.warning("請求中未提供交易對參數")
            return jsonify({'error': '未提供交易對參數'}), 400
            
        if symbol not in SUPPORTED_SYMBOLS:
            logging.warning(f"請求了不支援的交易對: {symbol}")
            return jsonify({'error': '不支援的交易對'}), 400
            
        logging.info(f"初始化 CryptoAnalyzer for {symbol}")
        analyzer = CryptoAnalyzer(symbol=symbol)
        
        logging.info(f"正在獲取 {symbol} 的歷史數據 (limit=48h, interval=1h)")
        df = analyzer.get_historical_data(limit='48h', interval='1h')
        
        if df is None or df.empty:
            logging.error(f"無法獲取 {symbol} 的歷史數據")
            return jsonify({'error': f'無法獲取 {symbol} 的數據，交易所可能無響應或數據不足'}), 500
        logging.info(f"成功獲取 {symbol} 的歷史數據，形狀: {df.shape}")
        logging.debug(f"數據預覽 (前5行):\n{df.head().to_string()}")
        logging.debug(f"數據預覽 (後5行):\n{df.tail().to_string()}")
            
        logging.info(f"正在分析 {symbol} 的趨勢")
        df = analyzer.analyze_trend(df)
        logging.info(f"完成 {symbol} 趨勢分析，數據框形狀: {df.shape}")
        logging.debug(f"分析後數據預覽 (後5行):\n{df.tail().to_string()}")
        
        logging.info(f"正在生成 {symbol} 的交易信號，閾值: {signal_settings['threshold']}, 比例: {signal_settings['ratio']}")
        signals = analyzer.get_trading_signal(df, 
                                           signal_threshold=signal_settings['threshold'],
                                           signal_ratio=signal_settings['ratio'])
        logging.info(f"成功生成 {symbol} 的交易信號: {signals}")
        
        # 準備圖表數據前檢查數據有效性
        logging.info(f"正在準備 {symbol} 的圖表和響應數據")
        if df.empty:
            logging.error(f"{symbol} 的數據框在分析後變為空")
            return jsonify({'error': '數據分析過程中發生錯誤，數據為空'}), 500

        # 確保索引是 DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             logging.warning(f"{symbol} 的 DataFrame 索引不是 DatetimeIndex，嘗試轉換。")
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as idx_e:
                 logging.error(f"無法將 {symbol} 的索引轉換為 DatetimeIndex: {idx_e}")
                 return jsonify({'error': '數據索引格式錯誤'}), 500

        latest_timestamp = df.index[-1]
        if pd.isna(latest_timestamp):
            logging.error(f"{symbol} 的最新時間戳為 NaT")
            return jsonify({'error': '數據時間戳錯誤'}), 500

        # 函數用於安全地獲取和轉換值
        def get_safe_float(series, index=-1):
            if series.empty or len(series) <= abs(index):
                logging.warning(f"嘗試從空的或過短的 Series 獲取值: {series.name}")
                return 0.0
            value = series.iloc[index]
            if pd.isna(value):
                logging.warning(f"在 Series '{series.name}' 的索引 {index} 處發現 NaN 值，將其設為 0")
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError) as e:
                logging.error(f"無法將 Series '{series.name}' 的值 {value} 轉換為 float: {e}")
                return 0.0
        
        # 安全地準備圖表數據
        def get_safe_history(series):
             if series.empty:
                 logging.warning(f"嘗試從空的 Series 獲取歷史數據: {series.name}")
                 return []
             # 替換 NaN 和 inf
             cleaned_series = series.replace([np.inf, -np.inf], np.nan).fillna(0)
             try:
                 return [float(x) for x in cleaned_series.values]
             except (ValueError, TypeError) as e:
                 logging.error(f"無法將 Series '{series.name}' 的歷史數據轉換為 float 列表: {e}")
                 return [0.0] * len(series)

        chart_data = {
            'timestamps': [ts.isoformat() for ts in df.index],
            'rsi_history': get_safe_history(df['RSI']),
            'macd_history': get_safe_history(df['MACD']),
            'signal_history': get_safe_history(df['MACD_Signal']),
            'k_history': get_safe_history(df['K']),
            'd_history': get_safe_history(df['D'])
        }
        
        # 合併所有數據
        response_data = {
            'price': get_safe_float(df['close']),
            'ma20': get_safe_float(df['MA20']),
            'ma50': get_safe_float(df['MA50']),
            'bb_upper': get_safe_float(df['BB_upper']),
            'bb_lower': get_safe_float(df['BB_lower']),
            'rsi': get_safe_float(df['RSI']),
            'signals': signals,
            'timestamp': latest_timestamp.isoformat(),
            **chart_data
        }
        
        logging.info(f"成功準備 {symbol} 的響應數據")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"分析 {symbol} 時發生未預期的錯誤: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': f'分析 {symbol} 時發生內部錯誤，請稍後再試',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # 本地開發時使用
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))