import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv
import time
from typing import Optional, Tuple, Dict
import requests
from requests.exceptions import RequestException
import numpy as np
import traceback

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 設定中文字形
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 載入環境變數
load_dotenv()

class CryptoAnalyzer:
    def __init__(self, symbol: str = 'BTC/USDT', retry_count: int = 3, retry_delay: int = 5):
        self.symbol = symbol
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self) -> ccxt.Exchange:
        """初始化交易所連接"""
        for attempt in range(self.retry_count):
            try:
                exchange = ccxt.okx({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
                # 測試連接
                exchange.load_markets()
                logging.info(f"交易所連接成功，交易對: {self.symbol}")
                return exchange
            except Exception as e:
                if attempt < self.retry_count - 1:
                    logging.warning(f"交易所初始化失敗，{self.retry_delay}秒後重試: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"交易所初始化最終失敗: {e}")
                    raise

    def _check_internet_connection(self) -> bool:
        """檢查網路連接"""
        try:
            requests.get('https://www.google.com', timeout=5)
            return True
        except RequestException:
            return False

    def get_historical_data(self, limit='48h', interval='1h'):
        try:
            # 解析時間限制
            time_unit = limit[-1]  # 獲取最後一個字符 (h/d/w)
            time_value = int(limit[:-1])  # 獲取數字部分
            
            # 將時間轉換為毫秒
            current_time = int(time.time() * 1000)
            if time_unit == 'h':
                time_diff = time_value * 60 * 60 * 1000
            elif time_unit == 'd':
                time_diff = time_value * 24 * 60 * 60 * 1000
            elif time_unit == 'w':
                time_diff = time_value * 7 * 24 * 60 * 60 * 1000
            else:
                raise ValueError(f"不支持的時間單位: {time_unit}")
                
            since = current_time - time_diff
            
            # 解析時間間隔
            timeframe_map = {
                '1m': '1m',
                '3m': '3m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1w',
                '1M': '1M'
            }
            
            if interval not in timeframe_map:
                raise ValueError(f"不支持的時間間隔: {interval}")
                
            # 獲取K線數據
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe_map[interval],
                since=since,
                limit=500  # 獲取足夠的數據點
            )
            
            # 轉換為DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"獲取歷史數據時發生錯誤: {e}")
            logging.error(traceback.format_exc())
            return None

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """計算MACD指標"""
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_kd(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 9, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """計算KD指標"""
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        return k, d

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """計算ATR指標"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def analyze_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析趨勢"""
        try:
            # 計算移動平均線
            df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['MA50'] = df['close'].rolling(window=50, min_periods=1).mean()
            df['MA200'] = df['close'].rolling(window=200, min_periods=1).mean()
            
            # 計算RSI
            df['RSI'] = self.calculate_rsi(df['close'])
            
            # 計算布林通道
            df['BB_middle'] = df['MA20']
            std = df['close'].rolling(window=20, min_periods=1).std()
            df['BB_upper'] = df['BB_middle'] + 2 * std
            df['BB_lower'] = df['BB_middle'] - 2 * std
            
            # 計算MACD
            macd, signal, hist = self.calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            
            # 計算KD指標
            df['K'], df['D'] = self.calculate_kd(df['high'], df['low'], df['close'])
            
            # 計算成交量指標
            df['Volume_MA20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            
            # 計算ATR
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'])
            
            # 處理可能的 NaN 值
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 將無限值替換為 None
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
        except Exception as e:
            logging.error(f"趨勢分析失敗: {e}")
            raise

    def plot_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """繪製分析圖表"""
        try:
            if save_path is None:
                save_path = f'{self.symbol.replace("/", "_")}_analysis.png'
                
            plt.figure(figsize=(15, 12))
            
            # 價格和移動平均線
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['close'], label='收盤價', color='blue')
            plt.plot(df.index, df['MA20'], label='20日移動平均線', color='orange')
            plt.plot(df.index, df['MA50'], label='50日移動平均線', color='red')
            plt.plot(df.index, df['BB_upper'], label='布林上軌', color='gray', linestyle='--')
            plt.plot(df.index, df['BB_lower'], label='布林下軌', color='gray', linestyle='--')
            plt.fill_between(df.index, df['BB_upper'], df['BB_lower'], color='gray', alpha=0.1)
            plt.title(f'{self.symbol} 價格走勢')
            plt.legend()
            plt.grid(True)
            
            # RSI
            plt.subplot(3, 1, 2)
            plt.plot(df.index, df['RSI'], label='RSI', color='purple')
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title('RSI 指標')
            plt.legend()
            plt.grid(True)
            
            # 成交量
            plt.subplot(3, 1, 3)
            plt.bar(df.index, df['volume'], color='blue', alpha=0.5)
            plt.title('成交量')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"{self.symbol}圖表已保存至 {save_path}")
        except Exception as e:
            logging.error(f"繪製{self.symbol}圖表失敗: {e}")
            raise

    def get_trading_signal(self, df: pd.DataFrame, signal_threshold: float = 3.0, signal_ratio: float = 1.2) -> dict:
        """獲取交易信號"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 初始化信號分數詳情
        signal_details = {
            'buy': {'total': 0, 'details': {}},
            'sell': {'total': 0, 'details': {}}
        }
        
        # 趨勢判斷
        trend = "上漲" if latest['MA20'] > latest['MA50'] else "下跌"
        if trend == "上漲":
            signal_details['buy']['details']['趨勢'] = 1
            signal_details['buy']['total'] += 1
        else:
            signal_details['sell']['details']['趨勢'] = 1
            signal_details['sell']['total'] += 1
        
        # RSI 信號
        rsi = latest['RSI']
        rsi_signal = "超買" if rsi > 70 else "超賣" if rsi < 30 else "中性"
        if rsi_signal == "超賣" and trend == "上漲":
            signal_details['buy']['details']['RSI'] = 2
            signal_details['buy']['total'] += 2
        elif rsi_signal == "超買" and trend == "下跌":
            signal_details['sell']['details']['RSI'] = 2
            signal_details['sell']['total'] += 2
        
        # MACD 信號
        macd_cross = (prev['MACD'] < prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal'])
        death_cross = (prev['MACD'] > prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal'])
        macd_signal = "金叉" if macd_cross else "死叉" if death_cross else "無信號"
        if macd_signal == "金叉":
            signal_details['buy']['details']['MACD'] = 1.5
            signal_details['buy']['total'] += 1.5
        elif macd_signal == "死叉":
            signal_details['sell']['details']['MACD'] = 1.5
            signal_details['sell']['total'] += 1.5
        
        # KD 信號
        k, d = latest['K'], latest['D']
        kd_cross = (prev['K'] < prev['D'] and k > d)
        kd_death = (prev['K'] > prev['D'] and k < d)
        kd_signal = "金叉" if kd_cross else "死叉" if kd_death else "無信號"
        if kd_signal == "金叉" and rsi < 70:
            signal_details['buy']['details']['KD'] = 1.5
            signal_details['buy']['total'] += 1.5
        elif kd_signal == "死叉" and rsi > 30:
            signal_details['sell']['details']['KD'] = 1.5
            signal_details['sell']['total'] += 1.5
        
        # 成交量信號
        volume_trend = "放量" if latest['volume'] > latest['Volume_MA20'] * 1.5 else \
                      "縮量" if latest['volume'] < latest['Volume_MA20'] * 0.5 else "正常"
        if volume_trend == "放量":
            if trend == "上漲":
                signal_details['buy']['details']['成交量'] = 1
                signal_details['buy']['total'] += 1
            else:
                signal_details['sell']['details']['成交量'] = 1
                signal_details['sell']['total'] += 1
        
        # 布林帶位置
        price = latest['close']
        bb_position = (price - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
        
        if bb_position < 0.2:  # 接近下軌
            signal_details['buy']['details']['布林帶'] = 1
            signal_details['buy']['total'] += 1
        elif bb_position > 0.8:  # 接近上軌
            signal_details['sell']['details']['布林帶'] = 1
            signal_details['sell']['total'] += 1
        
        # 決定交易方向和建議
        trade_direction = "觀望"
        entry_price = None
        stop_loss = None
        take_profit = None
        risk_reward_ratio = None
        
        buy_signals = signal_details['buy']['total']
        sell_signals = signal_details['sell']['total']
        
        # 使用可調整的閾值
        if buy_signals >= signal_threshold and buy_signals > sell_signals * signal_ratio:
            trade_direction = "做多"
            entry_price = price
            stop_loss = price * 0.98  # 2%止損
            take_profit = price * 1.06  # 6%獲利
            risk_reward_ratio = 3.0
        elif sell_signals >= signal_threshold and sell_signals > buy_signals * signal_ratio:
            trade_direction = "做空"
            entry_price = price
            stop_loss = price * 1.02  # 2%止損
            take_profit = price * 0.94  # 6%獲利
            risk_reward_ratio = 3.0
        
        return {
            'trend': trend,
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'kd_signal': kd_signal,
            'volume_signal': volume_trend,
            'trade_direction': trade_direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'signal_strength': signal_details,
            'threshold_info': {
                'signal_threshold': signal_threshold,
                'signal_ratio': signal_ratio,
                'buy_signals_total': buy_signals,
                'sell_signals_total': sell_signals
            }
        }

def main():
    try:
        analyzer = CryptoAnalyzer()
        
        # 獲取歷史數據
        df = analyzer.get_historical_data()
        if df is not None:
            # 分析趨勢
            df = analyzer.analyze_trend(df)
            
            # 繪製圖表
            analyzer.plot_analysis(df)
            
            # 獲取交易信號
            signals = analyzer.get_trading_signal(df)
            
            # 輸出分析結果
            latest = df.iloc[-1]
            print("\n=== BTC/USDT 分析報告 ===")
            print(f"當前價格: {latest['close']:.2f} USDT")
            print(f"20日移動平均線: {latest['MA20']:.2f} USDT")
            print(f"50日移動平均線: {latest['MA50']:.2f} USDT")
            print(f"布林上軌: {latest['BB_upper']:.2f} USDT")
            print(f"布林下軌: {latest['BB_lower']:.2f} USDT")
            print(f"RSI: {latest['RSI']:.2f}")
            print(f"\n趨勢分析: {signals['trend']}")
            print(f"RSI信號: {signals['rsi_signal']}")
            print(f"MACD信號: {signals['macd_signal']}")
            print(f"KD信號: {signals['kd_signal']}")
            print(f"成交量信號: {signals['volume_signal']}")
            
            # 給出交易建議
            print("\n=== 交易建議 ===")
            if signals['trend'] == "上漲" and signals['rsi_signal'] != "超買":
                print("建議: 可以考慮逢低買入")
            elif signals['trend'] == "下跌" and signals['rsi_signal'] != "超賣":
                print("建議: 可以考慮逢高賣出")
            else:
                print("建議: 觀望為主")
                
    except Exception as e:
        logging.error(f"程式執行失敗: {e}")
        raise

if __name__ == "__main__":
    main() 