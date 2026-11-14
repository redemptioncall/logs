from __future__ import annotations

"""NEXUS configurable reward-based crypto trading dashboard.

Offline historical backtesting only. The policy observes completed candles and
orders are filled at the next candle's open when available. Positions are
long-only; each symbol has HOLD, BUY, and SELL actions.

The dashboard offers ten learning methods and ten selectable neural-network
architectures. Model dimensions and depth are configurable in the UI. This remains
offline historical backtesting, not live order execution.

Set NEXUS_DATA_DIR to the folder containing proccesedValues_<SYMBOL>.csv files
(or keep the default c:/torch path on Windows).
"""

import copy
import math
import logging
import logging.handlers
import traceback
from collections import deque
import os
import pathlib
import queue
import random
import sys
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
from torch import nn

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import inspect
import json
class Properties():
    with_indicators = True
    input_size = 256
    input_factor = 10
    output_size = 64
    time_frames = 15
    sentence_per_frame = 21
    nNum = 128

    LR = 0.001
    num_symbols = 5

    attributes = ["o","t","c","v","l","h","number_of_trades"]

    features = [
            "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
            "ema9_21", "ema21_50","ma7","ma25","ma99",
            "rsi14", "macd_hist",
            "atr14"
        ]
    all_att = ["o","c","v","l","h","number_of_trades",
            #"ema9_21", "ema21_50","ma7","ma25","ma99",
            #"rsi14", "macd_hist","atr14"
            ]
    num_futures = all_att.__len__()
    def set_all(fro):
        we1 = json.loads(fro)
        for name,value in we1.items():
            setattr(Properties,name,value) 
    @classmethod
    def get_all(cls):
        return json.dumps({
            k: v for k, v in vars(cls).items()
            if not k.startswith("__")
            and not inspect.isroutine(v)
        })


# ---------------------------------------------------------------------------
# Paths, defaults, and reproducibility
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent
ROOT_C = pathlib.Path(os.environ.get("NEXUS_DATA_DIR", r"c:/torch"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 36
DEFAULT_LOOKBACK = 30
DEFAULT_EPISODE_CANDLES = 30
DEFAULT_MAX_POSITIONS = 5
DEFAULT_FEE_BPS = 10.0
INITIAL_CAPITAL = 10_000.0

# File logging is process-safe enough for this single-worker application. The
# GUI receives concise structured events through the existing multiprocessing queue.
LOG_DIR = pathlib.Path(os.environ.get("NEXUS_LOG_DIR", str(ROOT_C)))
LOG_FILE = LOG_DIR / "nexus.log"
LOG_LEVEL = os.environ.get("NEXUS_LOG_LEVEL", "INFO").upper()


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nexus")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(processName)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


LOGGER = configure_logging()

# Double-DQN training controls. Batched episodes reduce inference overhead.
DEFAULT_ROLLOUT_EPISODES = 8
DEFAULT_VALIDATION_EPISODES = 12
DEFAULT_VALIDATION_INTERVAL = 5
DEFAULT_MIN_VALIDATION_TRADES = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_UPDATES = 12
DEFAULT_REPLAY_CAPACITY = 4096
DEFAULT_REPLAY_MEMORY_MB = 192
DEFAULT_MIN_REPLAY_SIZE = 256
DEFAULT_GAMMA = 0.985
DEFAULT_TARGET_TAU = 0.01
DEFAULT_EPSILON_START = 0.90
DEFAULT_EPSILON_END = 0.05
DEFAULT_EPSILON_DECAY_STEPS = 25_000

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "NEOUSDT", "LTCUSDT", "ADAUSDT",
    "XRPUSDT", "IOTAUSDT", "XLMUSDT", "TRXUSDT", "ETCUSDT", "VETUSDT",
    "LINKUSDT", "ZILUSDT", "FETUSDT", "ZECUSDT", "DASHUSDT", "THETAUSDT",
    "ENJUSDT", "ATOMUSDT", "ALGOUSDT", "DOGEUSDT", "ANKRUSDT", "CHZUSDT",
    "XTZUSDT", "HBARUSDT", "STXUSDT", "KAVAUSDT", "BCHUSDT", "SOLUSDT",
    "COMPUSDT", "SNXUSDT", "MANAUSDT", "CRVUSDT", "SANDUSDT", "DOTUSDT",
    "SUSHIUSDT", "EGLDUSDT", "RUNEUSDT", "UNIUSDT", "AVAXUSDT", "AAVEUSDT",
    "NEARUSDT", "FILUSDT", "INJUSDT", "AXSUSDT", "ROSEUSDT", "GRTUSDT",
    "1INCHUSDT", "CELOUSDT", "CFXUSDT", "ICPUSDT", "ARUSDT", "MASKUSDT",
    "QNTUSDT", "FLOWUSDT", "MINAUSDT", "DYDXUSDT", "GALAUSDT", "ENSUSDT",
    "IMXUSDT", "WOOUSDT", "GMTUSDT", "APEUSDT", "LDOUSDT", "OPUSDT",
    "APTUSDT", "ARBUSDT", "SUIUSDT", "PEPEUSDT", "FLOKIUSDT", "PENDLEUSDT",
    "ARKMUSDT", "WLDUSDT", "SEIUSDT", "TIAUSDT", "ORDIUSDT",
]

# Small-batch inference is often slower with a large CPU thread pool. Cap the
# worker's intra-op threads; override with NEXUS_TORCH_THREADS when desired.
try:
    torch.set_num_threads(max(1, int(os.environ.get("NEXUS_TORCH_THREADS", "4"))))
except (TypeError, ValueError):
    torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_int(value, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


# ---------------------------------------------------------------------------
# Historical market data
# ---------------------------------------------------------------------------

class MarketData:
    """Aligned market data: features[T,S,F], close/open prices[T,S]."""

    def __init__(
        self,
        features,
        prices,
        symbols,
        attributes,
        train_end,
        val_end,
        feature_mean,
        feature_std,
        open_prices=None,
    ):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.prices = np.ascontiguousarray(prices, dtype=np.float32)
        self.open_prices = np.ascontiguousarray(
            prices if open_prices is None else open_prices, dtype=np.float32
        )
        self.symbols = list(symbols)
        self.attributes = list(attributes)
        self.train_end = int(train_end)
        self.val_end = int(val_end)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)

        if self.features.ndim != 3:
            raise ValueError("features must have shape [time, symbols, features]")
        if self.prices.shape != self.features.shape[:2]:
            raise ValueError("close-price matrix does not match feature dimensions")
        if self.open_prices.shape != self.prices.shape:
            raise ValueError("open-price matrix does not match close-price dimensions")
        if not np.isfinite(self.prices).all() or np.any(self.prices <= 0):
            raise ValueError("close prices must be finite and strictly positive")
        if not np.isfinite(self.open_prices).all() or np.any(self.open_prices <= 0):
            raise ValueError("open prices must be finite and strictly positive")
        if self.features.shape[0] < 100:
            raise ValueError("At least 100 aligned candles are required")


def _find_timestamp_column(frame: pd.DataFrame):
    candidates = ("timestamp", "open_time", "openTime", "time", "date", "datetime")
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_market_data(config) -> MarketData:
    attributes = list(Properties.all_att)
    if not attributes:
        raise RuntimeError("Properties.all_att is empty")

    close_matches = [
        i for i, name in enumerate(attributes)
        if str(name).strip().lower() in {"c", "close", "closing", "close_price"}
    ]
    if not close_matches:
        raise RuntimeError(
            "Properties.all_att must contain a close column named 'c' or 'close'"
        )
    close_index = close_matches[0]
    open_matches = [
        i for i, name in enumerate(attributes)
        if str(name).strip().lower() in {"o", "open", "open_price"}
    ]
    open_index = open_matches[0] if open_matches else None

    num_symbols = _positive_int(config["num_symbols"], "num_symbols")
    if num_symbols > len(SYMBOLS):
        raise ValueError(f"At most {len(SYMBOLS)} symbols are configured")
    selected_symbols = [symbol[:-4] for symbol in SYMBOLS[:num_symbols]]

    frames = []
    for symbol in selected_symbols:
        path = ROOT_C / f"proccesedValues_{symbol}.csv"
        if not path.exists():
            # Keep compatibility with installations that used the old .json
            # suffix even though the file contents were read as CSV.
            legacy_path = ROOT_C / f"proccesedValues_{symbol}.json"
            if legacy_path.exists():
                path = legacy_path
            else:
                raise FileNotFoundError(f"Missing processed market file: {path}")
        frames.append(pd.read_csv(path))

    row_counts = [len(frame) for frame in frames]
    if len(set(row_counts)) != 1:
        raise ValueError(
            "Symbol files have different row counts. Cross-symbol trading needs "
            f"aligned candles; found row counts {row_counts}"
        )

    first_time_col = _find_timestamp_column(frames[0])
    if first_time_col is not None:
        reference_times = frames[0][first_time_col].astype(str).to_numpy()
        for symbol, frame in zip(selected_symbols[1:], frames[1:]):
            time_col = _find_timestamp_column(frame)
            if time_col is None:
                raise ValueError(
                    f"{symbol} has no timestamp column while other files do"
                )
            current_times = frame[time_col].astype(str).to_numpy()
            if not np.array_equal(reference_times, current_times):
                raise ValueError(f"Timestamp order in {symbol} is not aligned")

    total_rows = row_counts[0]
    raw = np.empty((total_rows, num_symbols, len(attributes)), dtype=np.float32)
    prices = np.empty((total_rows, num_symbols), dtype=np.float32)
    open_prices = np.empty((total_rows, num_symbols), dtype=np.float32)

    for symbol_index, (symbol, frame) in enumerate(zip(selected_symbols, frames)):
        for feature_index, attribute in enumerate(attributes):
            column = f"{symbol}_{attribute}"
            if column not in frame.columns:
                raise KeyError(f"Column {column!r} is missing from {symbol}'s file")
            series = pd.to_numeric(frame[column], errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan).ffill()
            if series.isna().any():
                raise ValueError(
                    f"Column {column!r} contains missing values at the start of the "
                    "series (or invalid values after forward-fill). Clean the "
                    "source CSV; backward-fill is intentionally disabled to "
                    "prevent future-data leakage."
                )
            raw[:, symbol_index, feature_index] = series.to_numpy(dtype=np.float32)

        close = raw[:, symbol_index, close_index]
        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError(f"{symbol} close prices must be finite and positive")
        prices[:, symbol_index] = close

        if open_index is None:
            # Close-only files are supported, but next-close fills are less
            # realistic than next-open fills.
            open_prices[:, symbol_index] = close
        else:
            open_values = raw[:, symbol_index, open_index]
            if not np.isfinite(open_values).all() or np.any(open_values <= 0):
                raise ValueError(f"{symbol} open prices must be finite and positive")
            open_prices[:, symbol_index] = open_values

    del frames

    train_end = int(total_rows * 0.90)
    val_end = int(total_rows * 0.99)
    if train_end < 50 or val_end - train_end < 5:
        raise ValueError("Not enough rows for chronological train/validation split")

    # Fit normalization strictly on the training period.
    mean = raw[:train_end].mean(axis=0, keepdims=True)
    std = raw[:train_end].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = np.clip((raw - mean) / std, -10.0, 10.0)

    return MarketData(
        normalized,
        prices,
        selected_symbols,
        attributes,
        train_end,
        val_end,
        mean.squeeze(0),
        std.squeeze(0),
        open_prices,
    )


# ---------------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------------

class TradingEnvironment:
    """Long-only, multi-asset simulator with next-open order execution."""

    def __init__(self, market: MarketData, config, data_end: int):
        self.market = market
        self.features = market.features
        self.prices = market.prices
        self.num_symbols = self.prices.shape[1]
        self.num_features = self.features.shape[2]
        self.lookback = _positive_int(config["lookback_candles"], "lookback_candles")
        self.episode_candles = _positive_int(
            config["episode_candles"], "episode_candles"
        )
        self.max_positions = _positive_int(config["max_positions"], "max_positions")
        self.cost_rate = max(0.0, float(config["fee_bps"])) / 10_000.0
        self.data_end = int(data_end)
        if self.lookback >= self.data_end:
            raise ValueError("Lookback window is longer than the available data")
        if self.max_positions > self.num_symbols:
            raise ValueError("max_positions cannot exceed num_symbols")
        self.reset(self.lookback - 1, INITIAL_CAPITAL)

    def reset(
        self,
        start_t: int,
        initial_cash: float = INITIAL_CAPITAL,
        episode_candles: int | None = None,
    ):
        start_t = int(start_t)
        if start_t < self.lookback - 1:
            raise ValueError("Episode start does not have enough lookback history")
        if start_t >= self.data_end - 1:
            raise ValueError("Episode start must leave at least one future candle")
        self.current_t = start_t
        requested = self.episode_candles if episode_candles is None else int(episode_candles)
        self.target_steps = max(1, min(requested, self.data_end - 1 - start_t))
        self.steps = 0
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: list[dict | None] = [None] * self.num_symbols
        self.trade_pnls: list[float] = []
        self.symbol_trade_pnls: list[list[float]] = [[] for _ in range(self.num_symbols)]
        self.equity_curve = [float(initial_cash)]
        self.last_executed_actions = np.zeros(self.num_symbols, dtype=np.int64)
        self.done = False
        return self.observe()

    def _equity(self, prices=None) -> float:
        if prices is None:
            prices = self.prices[self.current_t]
        value = self.cash
        for i, position in enumerate(self.positions):
            if position is not None:
                value += position["qty"] * float(prices[i])
        return float(value)

    def _open_count(self) -> int:
        return sum(position is not None for position in self.positions)

    def _portfolio_features(self) -> np.ndarray:
        current_prices = self.prices[self.current_t]
        equity = max(self._equity(current_prices), 1e-8)
        cash_fraction = self.cash / equity
        state = np.zeros((self.num_symbols, 5), dtype=np.float32)
        for i, position in enumerate(self.positions):
            if position is None:
                state[i, 4] = cash_fraction
                continue
            price = float(current_prices[i])
            entry = float(position["entry_price"])
            unrealized = price / max(entry, 1e-12) - 1.0
            age = self.current_t - int(position["entry_t"])
            weight = position["qty"] * price / equity
            state[i] = [
                1.0,
                float(np.clip(unrealized, -1.0, 1.0)),
                float(min(age / max(self.episode_candles, 1), 1.0)),
                float(weight),
                float(cash_fraction),
            ]
        return state

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_symbols, 3), dtype=np.bool_)
        mask[:, 0] = True
        can_open = self._open_count() < self.max_positions and self.cash > 1e-6
        for i, position in enumerate(self.positions):
            if position is None:
                mask[i, 1] = can_open
            else:
                mask[i, 2] = True
        return mask

    def observe(self):
        start = self.current_t - self.lookback + 1
        # [T,S,F] -> [S,F,L]
        window = self.features[start:self.current_t + 1].transpose(1, 2, 0)
        return (
            np.ascontiguousarray(window, dtype=np.float32),
            self._portfolio_features(),
            self.action_mask(),
        )

    def _close_position(self, symbol_index: int, price: float) -> None:
        position = self.positions[symbol_index]
        if position is None:
            return
        gross = position["qty"] * float(price)
        proceeds = gross * (1.0 - self.cost_rate)
        self.cash += proceeds
        realized_pnl = float(proceeds - position["cost_basis"])
        self.trade_pnls.append(realized_pnl)
        self.symbol_trade_pnls[symbol_index].append(realized_pnl)
        self.positions[symbol_index] = None

    def step(self, actions, buy_scores=None):
        if self.done:
            raise RuntimeError("This episode is already finished")
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_symbols)
        if np.any((actions < 0) | (actions > 2)):
            raise ValueError("Actions must be 0=HOLD, 1=BUY, or 2=SELL")
        if buy_scores is None:
            buy_scores = np.zeros(self.num_symbols, dtype=np.float32)
        buy_scores = np.asarray(buy_scores, dtype=np.float32).reshape(self.num_symbols)

        old_equity = max(self._equity(), 1e-8)
        execution_prices = self.market.open_prices[self.current_t + 1]
        executed_actions = np.zeros(self.num_symbols, dtype=np.int64)

        # Execute exits first so their proceeds may fund new entries.
        for i, action in enumerate(actions):
            if action == 2 and self.positions[i] is not None:
                self._close_position(i, float(execution_prices[i]))
                executed_actions[i] = 2

        # Rank simultaneous entries by policy BUY probability. Entries that
        # exceed free slots are not executed.
        candidates = [
            i for i, action in enumerate(actions)
            if action == 1 and self.positions[i] is None
        ]
        candidates.sort(key=lambda i: float(buy_scores[i]), reverse=True)
        for i in candidates:
            if self._open_count() >= self.max_positions:
                break
            equity_now = max(self._equity(execution_prices), 1e-8)
            slot_budget = equity_now / self.max_positions
            notional = min(slot_budget, self.cash / (1.0 + self.cost_rate))
            if notional <= 1e-8:
                break
            price = float(execution_prices[i])
            quantity = notional / price
            total_cost = notional * (1.0 + self.cost_rate)
            self.cash -= total_cost
            self.positions[i] = {
                "qty": quantity,
                "entry_price": price,
                "entry_t": self.current_t,
                "cost_basis": total_cost,
            }
            executed_actions[i] = 1

        self.last_executed_actions = executed_actions
        self.current_t += 1
        self.steps += 1
        is_end = (
            self.steps >= self.target_steps
            or self.current_t >= self.data_end - 1
        )
        if is_end:
            final_prices = self.prices[self.current_t]
            for i in range(self.num_symbols):
                if self.positions[i] is not None:
                    self._close_position(i, float(final_prices[i]))

        new_equity = max(self._equity(), 1e-8)
        reward = 100.0 * math.log(new_equity / old_equity)
        self.equity_curve.append(new_equity)
        self.done = bool(is_end)
        return self.observe(), float(reward), self.done

    @property
    def final_equity(self):
        return self._equity()

    @property
    def trade_count(self):
        return len(self.trade_pnls)

    @property
    def win_rate(self):
        if not self.trade_pnls:
            return 0.0
        return 100.0 * sum(pnl > 0 for pnl in self.trade_pnls) / len(self.trade_pnls)


# ---------------------------------------------------------------------------
# Factorized dueling Q-network: temporal CNN + shared cross-symbol context
# ---------------------------------------------------------------------------

LEARNING_METHODS = [
    "DQN",
    "Double DQN",
    "Dueling DQN",
    "Prioritized Double DQN",
    "N-Step Double DQN",
    "QR-DQN",
    "PPO",
    "A2C",
    "REINFORCE",
    "Discrete SAC",
]

NETWORK_ARCHITECTURES = [
    "MLP",
    "CNN1D",
    "TCN",
    "GRU",
    "LSTM",
    "Transformer",
    "CNN-GRU",
    "CNN-Transformer",
    "ResNet1D",
    "MLP-Mixer",
]


class _Residual1DBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        groups = math.gcd(8, channels)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class _MarketEncoder(nn.Module):
    """One of ten configurable per-symbol temporal feature encoders."""

    def __init__(self, num_features: int, lookback: int, config):
        super().__init__()
        self.architecture = config["architecture"]
        self.hidden_dim = int(config["hidden_dim"])
        self.hidden_layers = int(config["hidden_layers"])
        self.conv_channels = int(config["conv_channels"])
        self.recurrent_layers = int(config["recurrent_layers"])
        self.attention_heads = int(config["attention_heads"])
        self.dropout = float(config["dropout"])
        h, c, depth, drop = (
            self.hidden_dim, self.conv_channels, self.hidden_layers, self.dropout
        )
        groups = math.gcd(8, c)
        self.output_norm = nn.LayerNorm(h)

        if self.architecture == "MLP":
            layers = []
            input_size = num_features * lookback
            for _ in range(depth):
                layers.extend((nn.Linear(input_size, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop)))
                input_size = h
            self.encoder = nn.Sequential(*layers)
            self.mode = "flat"

        elif self.architecture in {"CNN1D", "TCN", "ResNet1D"}:
            modules = [
                nn.Conv1d(num_features, c, kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(groups, c),
                nn.GELU(),
            ]
            if self.architecture == "CNN1D":
                for _ in range(depth):
                    modules.extend((
                        nn.Conv1d(c, c, 3, padding=1, bias=False),
                        nn.GroupNorm(groups, c), nn.GELU(), nn.Dropout(drop),
                    ))
            elif self.architecture == "TCN":
                modules.extend(_Residual1DBlock(c, 2 ** (i % 4), drop) for i in range(depth))
            else:
                modules.extend(_Residual1DBlock(c, 1, drop) for _ in range(depth))
            self.encoder = nn.Sequential(*modules)
            self.projection = nn.Sequential(nn.Linear(c * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "conv_pool"

        elif self.architecture in {"GRU", "LSTM"}:
            self.input_projection = nn.Sequential(nn.Linear(num_features, h), nn.LayerNorm(h), nn.GELU())
            rnn_cls = nn.GRU if self.architecture == "GRU" else nn.LSTM
            self.rnn = rnn_cls(
                h, h, num_layers=self.recurrent_layers, batch_first=True,
                dropout=drop if self.recurrent_layers > 1 else 0.0,
            )
            self.projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "rnn"

        elif self.architecture == "Transformer":
            if h % self.attention_heads:
                raise ValueError("hidden_dim must be divisible by attention_heads for Transformer")
            self.input_projection = nn.Linear(num_features, h)
            self.position_embedding = nn.Parameter(torch.zeros(1, max(512, lookback), h))
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=self.attention_heads, dim_feedforward=h * 2,
                dropout=drop, batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "transformer"

        elif self.architecture == "CNN-GRU":
            self.conv = nn.Sequential(
                nn.Conv1d(num_features, c, 5, padding=2, bias=False),
                nn.GroupNorm(groups, c), nn.GELU(),
                *[_Residual1DBlock(c, 1, drop) for _ in range(max(1, depth - 1))],
            )
            self.rnn = nn.GRU(
                c, h, num_layers=self.recurrent_layers, batch_first=True,
                dropout=drop if self.recurrent_layers > 1 else 0.0,
            )
            self.projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "cnn_rnn"

        elif self.architecture == "CNN-Transformer":
            if h % self.attention_heads:
                raise ValueError("hidden_dim must be divisible by attention_heads for CNN-Transformer")
            self.conv = nn.Sequential(
                nn.Conv1d(num_features, c, 5, padding=2, bias=False),
                nn.GroupNorm(groups, c), nn.GELU(),
                *[_Residual1DBlock(c, 1, drop) for _ in range(max(1, depth - 1))],
            )
            self.input_projection = nn.Linear(c, h)
            self.position_embedding = nn.Parameter(torch.zeros(1, max(512, lookback), h))
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=self.attention_heads, dim_feedforward=h * 2,
                dropout=drop, batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "cnn_transformer"

        elif self.architecture == "MLP-Mixer":
            self.input_projection = nn.Linear(num_features, h)
            self.mixer_blocks = nn.ModuleList()
            for _ in range(depth):
                self.mixer_blocks.append(nn.ModuleDict({
                    "token_norm": nn.LayerNorm(h),
                    "token_in": nn.Linear(lookback, lookback),
                    "token_out": nn.Linear(lookback, lookback),
                    "channel_norm": nn.LayerNorm(h),
                    "channel_mlp": nn.Sequential(
                        nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(drop),
                        nn.Linear(h * 2, h), nn.Dropout(drop),
                    ),
                }))
            self.projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
            self.mode = "mixer"
        else:
            raise ValueError(f"Unknown network architecture: {self.architecture}")

    def forward(self, x):
        # x: [batch * symbols, features, lookback]
        if self.mode == "flat":
            result = self.encoder(x.flatten(start_dim=1))
        elif self.mode == "conv_pool":
            encoded = self.encoder(x)
            summary = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
            result = self.projection(summary)
        elif self.mode == "rnn":
            seq = self.input_projection(x.transpose(1, 2))
            seq, _ = self.rnn(seq)
            result = self.projection(torch.cat((seq[:, -1], seq.mean(dim=1)), dim=-1))
        elif self.mode == "transformer":
            seq = self.input_projection(x.transpose(1, 2))
            seq = seq + self.position_embedding[:, :seq.shape[1]]
            seq = self.encoder(seq)
            result = self.projection(torch.cat((seq[:, -1], seq.mean(dim=1)), dim=-1))
        elif self.mode == "cnn_rnn":
            seq = self.conv(x).transpose(1, 2)
            seq, _ = self.rnn(seq)
            result = self.projection(torch.cat((seq[:, -1], seq.mean(dim=1)), dim=-1))
        elif self.mode == "cnn_transformer":
            seq = self.input_projection(self.conv(x).transpose(1, 2))
            seq = seq + self.position_embedding[:, :seq.shape[1]]
            seq = self.encoder(seq)
            result = self.projection(torch.cat((seq[:, -1], seq.mean(dim=1)), dim=-1))
        else:  # MLP-Mixer
            seq = self.input_projection(x.transpose(1, 2))
            for block in self.mixer_blocks:
                token = block["token_norm"](seq).transpose(1, 2)
                token = block["token_out"](torch.nn.functional.gelu(block["token_in"](token)))
                seq = seq + token.transpose(1, 2)
                seq = seq + block["channel_mlp"](block["channel_norm"](seq))
            result = self.projection(torch.cat((seq[:, -1], seq.mean(dim=1)), dim=-1))
        return self.output_norm(result)


class TradingQNetwork(nn.Module):
    """Configurable factorized actor/critic/Q model for 10 learning methods."""

    def __init__(self, num_symbols: int, num_features: int, config):
        super().__init__()
        self.num_symbols = int(num_symbols)
        self.hidden_dim = int(config["hidden_dim"])
        self.dueling = bool(config.get("dueling", False))
        self.num_quantiles = int(config.get("quantiles", 16))
        self.encoder = _MarketEncoder(num_features, config["lookback_candles"], config)
        h = self.hidden_dim
        self.position_encoder = nn.Sequential(nn.Linear(5, h), nn.LayerNorm(h), nn.GELU())
        self.symbol_embedding = nn.Embedding(self.num_symbols, h)
        self.context_projection = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
        self.policy_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(config["dropout"]),
            nn.Linear(h, 3),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1),
        )
        if self.dueling:
            self.q1_value = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
            self.q1_advantage = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
            self.q2_value = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
            self.q2_advantage = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        else:
            self.q1_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
            self.q2_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
        self.quantile_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 3 * self.num_quantiles),
        )

    def forward(self, market, portfolio):
        if market.ndim != 4:
            raise ValueError(f"Expected market [B,S,F,L], got {tuple(market.shape)}")
        batch, symbols, features, length = market.shape
        if symbols != self.num_symbols:
            raise ValueError(f"Expected {self.num_symbols} symbols, got {symbols}")
        encoded = self.encoder(market.reshape(batch * symbols, features, length))
        tokens = encoded.reshape(batch, symbols, -1)
        ids = torch.arange(symbols, device=market.device)
        tokens = tokens + self.position_encoder(portfolio) + self.symbol_embedding(ids).unsqueeze(0)
        context = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        tokens = tokens + self.context_projection(context).unsqueeze(1)

        if self.dueling:
            advantage1 = self.q1_advantage(tokens)
            advantage2 = self.q2_advantage(tokens)
            q1 = self.q1_value(tokens) + advantage1 - advantage1.mean(dim=-1, keepdim=True)
            q2 = self.q2_value(tokens) + advantage2 - advantage2.mean(dim=-1, keepdim=True)
        else:
            q1 = self.q1_head(tokens)
            q2 = self.q2_head(tokens)
        logits = self.policy_head(tokens)
        value = self.value_head(tokens.mean(dim=1)).squeeze(-1)
        quantiles = self.quantile_head(tokens).view(
            batch, symbols, 3, self.num_quantiles
        )
        return {
            "q1": q1,
            "q2": q2,
            "logits": logits,
            "value": value,
            "quantiles": quantiles,
        }


class ReplayBuffer:
    """Bounded replay buffer with optional proportional prioritization."""

    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.size = 0
        self.write_index = 0
        self.max_priority = 1.0
        self.arrays = None

    def _allocate(self, observation):
        market, portfolio, mask = observation
        c = self.capacity
        self.arrays = {
            "market": np.empty((c, *market.shape), dtype=np.float16),
            "next_market": np.empty((c, *market.shape), dtype=np.float16),
            "portfolio": np.empty((c, *portfolio.shape), dtype=np.float16),
            "next_portfolio": np.empty((c, *portfolio.shape), dtype=np.float16),
            "mask": np.empty((c, *mask.shape), dtype=np.bool_),
            "next_mask": np.empty((c, *mask.shape), dtype=np.bool_),
            "actions": np.empty((c, mask.shape[0]), dtype=np.uint8),
            "rewards": np.empty(c, dtype=np.float32),
            "discounts": np.empty(c, dtype=np.float32),
            "dones": np.empty(c, dtype=np.float32),
            "priorities": np.ones(c, dtype=np.float32),
        }

    def add(self, observation, actions, reward, next_observation, done, discount=0.985):
        if self.arrays is None:
            self._allocate(observation)
        i = self.write_index
        market, portfolio, mask = observation
        next_market, next_portfolio, next_mask = next_observation
        a = self.arrays
        a["market"][i] = market
        a["portfolio"][i] = portfolio
        a["mask"][i] = mask
        a["actions"][i] = actions
        a["rewards"][i] = float(reward)
        a["discounts"][i] = float(discount)
        a["next_market"][i] = next_market
        a["next_portfolio"][i] = next_portfolio
        a["next_mask"][i] = next_mask
        a["dones"][i] = float(done)
        a["priorities"][i] = self.max_priority
        self.write_index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device, beta: float = 0.4):
        if self.size < batch_size:
            raise ValueError("Not enough transitions to sample this batch")
        if self.prioritized:
            priorities = np.maximum(self.arrays["priorities"][:self.size], 1e-6)
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()
            indices = np.random.choice(self.size, size=int(batch_size), replace=True, p=probabilities)
            weights = (self.size * probabilities[indices]) ** (-float(beta))
            weights /= max(float(weights.max()), 1e-8)
        else:
            indices = np.random.randint(0, self.size, size=int(batch_size))
            weights = np.ones(int(batch_size), dtype=np.float32)
        a = self.arrays

        def tensor(name, dtype=None):
            value = torch.from_numpy(a[name][indices])
            return value.to(device=device, dtype=dtype, non_blocking=True) if dtype else value.to(device=device, non_blocking=True)

        return {
            "market": tensor("market", torch.float32),
            "portfolio": tensor("portfolio", torch.float32),
            "mask": tensor("mask", torch.bool),
            "actions": tensor("actions", torch.long),
            "rewards": tensor("rewards", torch.float32),
            "discounts": tensor("discounts", torch.float32),
            "next_market": tensor("next_market", torch.float32),
            "next_portfolio": tensor("next_portfolio", torch.float32),
            "next_mask": tensor("next_mask", torch.bool),
            "dones": tensor("dones", torch.float32),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        if not self.prioritized:
            return
        values = np.asarray(priorities, dtype=np.float32).reshape(-1)
        values = np.maximum(np.abs(values), 1e-5)
        self.arrays["priorities"][np.asarray(indices, dtype=np.int64)] = values
        if values.size:
            self.max_priority = max(self.max_priority, float(values.max()))


# ---------------------------------------------------------------------------
# Configurable off-policy replay and on-policy actor-critic training
# ---------------------------------------------------------------------------

class RewardTrainer:
    def __init__(self, input_queue: Queue, output_queue: Queue):
        self.input_queue = input_queue
        self.queue = output_queue
        self.market: MarketData | None = None
        self.data_ready = False
        self.model_ready = False
        self.stop_requested = False
        self.shutdown_requested = False
        self.save_requested = False
        self.config = {}
        self.best_val_return = -float("inf")
        self.best_val_excess = -float("inf")
        self.last_val_stats = None
        self.environment_steps = 0
        self.epsilon = DEFAULT_EPSILON_START
        self.replay: ReplayBuffer | None = None

    def log(self, level, message, *args):
        """Write to the rotating log file/console and forward the same event to the GUI."""
        try:
            rendered = message % args if args else str(message)
        except Exception:
            rendered = " ".join([str(message), *(str(arg) for arg in args)])
        LOGGER.log(level, rendered)
        try:
            self.queue.put(("log", {
                "level": logging.getLevelName(level),
                "message": rendered,
                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
            }))
        except Exception:
            # Logging must never crash training if the GUI queue is unavailable.
            pass

    @staticmethod
    def default_config():
        return {
            "num_symbols": int(getattr(Properties, "num_symbols", 5)),
            "lookback_candles": DEFAULT_LOOKBACK,
            "episode_candles": DEFAULT_EPISODE_CANDLES,
            "max_positions": DEFAULT_MAX_POSITIONS,
            "fee_bps": DEFAULT_FEE_BPS,
            "lr": float(getattr(Properties, "LR", 3e-4)),
            "learning_method": "Double DQN",
            "architecture": "CNN-GRU",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "conv_channels": 64,
            "attention_heads": 4,
            "recurrent_layers": 1,
            "dropout": 0.10,
            "quantiles": 16,
            "rollout_episodes": DEFAULT_ROLLOUT_EPISODES,
            "validation_episodes": DEFAULT_VALIDATION_EPISODES,
            "validation_interval": DEFAULT_VALIDATION_INTERVAL,
            "min_validation_trades": DEFAULT_MIN_VALIDATION_TRADES,
            "batch_size": DEFAULT_BATCH_SIZE,
            "train_updates": DEFAULT_TRAIN_UPDATES,
            "replay_capacity": DEFAULT_REPLAY_CAPACITY,
            "min_replay_size": DEFAULT_MIN_REPLAY_SIZE,
            "gamma": DEFAULT_GAMMA,
            "target_tau": DEFAULT_TARGET_TAU,
            "epsilon_start": DEFAULT_EPSILON_START,
            "epsilon_end": DEFAULT_EPSILON_END,
            "epsilon_decay_steps": DEFAULT_EPSILON_DECAY_STEPS,
            "n_step": 3,
            "sac_alpha": 0.10,
            "ppo_clip": 0.20,
            "entropy_coef": 0.01,
            "value_coef": 0.50,
        }

    def load_CSV(self):
        self.market = load_market_data(self.config or self.default_config())
        self.data_ready = False
        self.log(
            logging.INFO,
            "Market data loaded | symbols=%d | candles=%d | features=%d | device=%s",
            len(self.market.symbols), self.market.features.shape[0],
            self.market.features.shape[2], DEVICE,
        )

    def load_data(self):
        if self.market is None:
            raise RuntimeError("Load CSV must complete before Load Data")
        self.data_ready = True

    @staticmethod
    def _is_on_policy(method):
        return method in {"PPO", "A2C", "REINFORCE"}

    @staticmethod
    def _is_qr(method):
        return method == "QR-DQN"

    def _effective_replay_capacity(self, requested: int) -> int:
        assert self.market is not None
        market_bytes = int(np.prod(self.market.features.shape[1:])) * 2
        portfolio_bytes = self.market.features.shape[1] * 5 * 2
        mask_bytes = self.market.features.shape[1] * 3
        action_bytes = self.market.features.shape[1]
        bytes_per_transition = 2 * (market_bytes + portfolio_bytes + mask_bytes) + action_bytes + 16
        budget = DEFAULT_REPLAY_MEMORY_MB * 1024 * 1024
        memory_limited = max(64, budget // max(bytes_per_transition, 1))
        return max(64, min(int(requested), int(memory_limited)))

    def build_model(self, config=None):
        if not self.data_ready or self.market is None:
            raise RuntimeError("Load CSV and Load Data before building the model")
        merged = self.default_config()
        merged.update(self.config)
        if config:
            merged.update(config)

        int_fields = (
            "num_symbols", "lookback_candles", "episode_candles", "max_positions",
            "hidden_dim", "hidden_layers", "conv_channels", "attention_heads",
            "recurrent_layers", "quantiles", "rollout_episodes", "validation_episodes",
            "validation_interval", "min_validation_trades", "batch_size", "train_updates", "replay_capacity",
            "min_replay_size", "epsilon_decay_steps", "n_step",
        )
        for name in int_fields:
            merged[name] = _positive_int(merged[name], name)
        float_fields = (
            "fee_bps", "lr", "gamma", "target_tau", "epsilon_start", "epsilon_end",
            "dropout", "sac_alpha", "ppo_clip", "entropy_coef", "value_coef",
        )
        for name in float_fields:
            merged[name] = float(merged[name])
            if not np.isfinite(merged[name]):
                raise ValueError(f"{name} must be finite")
        if merged["learning_method"] not in LEARNING_METHODS:
            raise ValueError(f"Unknown learning method: {merged['learning_method']}")
        if merged["architecture"] not in NETWORK_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {merged['architecture']}")
        if merged["fee_bps"] < 0 or merged["lr"] <= 0:
            raise ValueError("fee_bps must be non-negative and learning rate positive")
        if not 0.0 < merged["gamma"] <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < merged["target_tau"] <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if not 0.0 <= merged["epsilon_end"] <= merged["epsilon_start"] <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 <= merged["dropout"] < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if merged["sac_alpha"] < 0 or merged["entropy_coef"] < 0:
            raise ValueError("entropy coefficients cannot be negative")
        if not 0.0 < merged["ppo_clip"] < 1.0:
            raise ValueError("ppo_clip must be in (0, 1)")
        if merged["num_symbols"] != len(self.market.symbols):
            raise ValueError("num_symbols changed after CSV loading. Load CSV again.")
        if merged["max_positions"] > merged["num_symbols"]:
            merged["max_positions"] = merged["num_symbols"]
        if self.market.train_end <= merged["lookback_candles"] + merged["episode_candles"]:
            raise ValueError("Lookback + episode length is too large for training data")
        if self.market.val_end - self.market.train_end < 2:
            raise ValueError("Validation split is too short")
        if merged["architecture"] in {"Transformer", "CNN-Transformer"}:
            if merged["hidden_dim"] % merged["attention_heads"]:
                raise ValueError("For Transformer architectures, hidden_dim must be divisible by attention_heads")

        merged["replay_capacity"] = self._effective_replay_capacity(merged["replay_capacity"])
        merged["batch_size"] = min(merged["batch_size"], merged["replay_capacity"])
        merged["min_replay_size"] = min(
            max(merged["batch_size"], merged["min_replay_size"]),
            merged["replay_capacity"],
        )
        merged["dueling"] = merged["learning_method"] == "Dueling DQN"
        self.config = merged
        seed_everything(SEED)
        self.net = TradingQNetwork(
            len(self.market.symbols), self.market.features.shape[2], merged
        ).to(DEVICE)
        self.target_net = copy.deepcopy(self.net).to(DEVICE)
        self.target_net.eval()
        for parameter in self.target_net.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=merged["lr"], eps=1e-5, weight_decay=1e-5
        )
        prioritized = merged["learning_method"] == "Prioritized Double DQN"
        self.replay = ReplayBuffer(merged["replay_capacity"], prioritized=prioritized)
        self.environment_steps = 0
        self.epsilon = (0.0 if self._is_on_policy(merged["learning_method"]) else merged["epsilon_start"])
        self.loss_history = []
        self.model_ready = True
        self.best_val_return = -float("inf")
        self.best_val_excess = -float("inf")
        self.last_val_stats = None
        parameters = sum(p.numel() for p in self.net.parameters())
        self.queue.put(("WeightsNum", parameters))
        self.log(
            logging.INFO,
            "Policy initialized | method=%s | architecture=%s | hidden=%d x %d | "
            "parameters=%s | device=%s | replay_capacity=%d",
            merged["learning_method"], merged["architecture"], merged["hidden_dim"],
            merged["hidden_layers"], f"{parameters:,}", DEVICE, merged["replay_capacity"],
        )

    def set_lr(self, value):
        value = float(value)
        if value <= 0 or not np.isfinite(value):
            raise ValueError("Learning rate must be finite and positive")
        if self.model_ready:
            for group in self.optimizer.param_groups:
                group["lr"] = value
        self.config["lr"] = value

    def _epsilon_at_step(self):
        start = self.config["epsilon_start"]
        end = self.config["epsilon_end"]
        decay = self.config["epsilon_decay_steps"]
        fraction = min(self.environment_steps / max(decay, 1), 1.0)
        self.epsilon = start + fraction * (end - start)
        return float(self.epsilon)

    def _batch_observations(self, observations):
        markets = np.stack([obs[0] for obs in observations]).astype(np.float32, copy=False)
        portfolios = np.stack([obs[1] for obs in observations]).astype(np.float32, copy=False)
        masks = np.stack([obs[2] for obs in observations]).astype(np.bool_, copy=False)
        return torch.from_numpy(markets).to(DEVICE), torch.from_numpy(portfolios).to(DEVICE), torch.from_numpy(masks).to(DEVICE)

    @staticmethod
    def _masked_logits(logits, mask):
        return logits.masked_fill(~mask, -1e9)

    def _cap_simultaneous_buys(self, actions, buy_scores, observations):
        """Limit new entries to the portfolio's genuinely available position slots."""
        actions = np.asarray(actions, dtype=np.int64).copy()
        scores = np.asarray(buy_scores, dtype=np.float32)
        max_positions = int(self.config["max_positions"])
        for batch_i, observation in enumerate(observations):
            open_positions = int(np.count_nonzero(observation[1][:, 0] > 0.5))
            free_slots = max(0, max_positions - open_positions)
            candidates = np.flatnonzero(actions[batch_i] == 1)
            if candidates.size > free_slots:
                ranked = candidates[np.argsort(scores[batch_i, candidates])[::-1]]
                actions[batch_i, ranked[free_slots:]] = 0
        return actions

    def _choose_actions(self, observations, epsilon=0.0, stochastic_policy=False):
        market_t, portfolio_t, mask_t = self._batch_observations(observations)
        with torch.inference_mode():
            output = self.net(market_t, portfolio_t)
            if stochastic_policy:
                dist = torch.distributions.Categorical(
                    logits=self._masked_logits(output["logits"], mask_t)
                )
                actions = dist.sample().cpu().numpy()
                scores = dist.probs[:, :, 1].cpu().numpy()
            else:
                q = output["q1"]
                legal_q = q.masked_fill(~mask_t, -1e9)
                actions = legal_q.argmax(dim=-1).cpu().numpy()
                scores = (q[:, :, 1] - q[:, :, 0]).cpu().numpy()
                if epsilon > 0:
                    mask_np = mask_t.cpu().numpy()
                    explore = np.random.random(actions.shape) < epsilon
                    for env_i, symbol_i in zip(*np.where(explore)):
                        legal = np.flatnonzero(mask_np[env_i, symbol_i])
                        if legal.size:
                            actions[env_i, symbol_i] = int(np.random.choice(legal))
        actions = self._cap_simultaneous_buys(actions, scores, observations)
        return actions.astype(np.int64, copy=False), scores

    def _emit_n_step(self, pending, max_steps):
        gamma = self.config["gamma"]
        entries = list(pending)[:max_steps]
        if not entries:
            return
        total_reward = 0.0
        done = False
        next_observation = entries[-1][3]
        for index, (_, _, reward, next_obs, is_done) in enumerate(entries):
            total_reward += (gamma ** index) * float(reward)
            next_observation = next_obs
            done = bool(is_done)
            if done:
                entries = entries[:index + 1]
                break
        first_observation, first_action = entries[0][0], entries[0][1]
        discount = gamma ** len(entries)
        self.replay.add(
            first_observation, first_action, total_reward, next_observation,
            done, discount=discount,
        )
        pending.popleft()

    def _collect_offpolicy_episodes(self):
        lookback = self.config["lookback_candles"]
        horizon = self.config["episode_candles"]
        last_start = self.market.train_end - horizon - 1
        if last_start < lookback - 1:
            raise RuntimeError("Not enough training rows for the configured episode length")
        envs, observations, rewards = [], [], []
        pending = []
        for _ in range(self.config["rollout_episodes"]):
            start = random.randint(lookback - 1, last_start)
            env = TradingEnvironment(self.market, self.config, self.market.train_end)
            envs.append(env)
            observations.append(env.reset(start))
            rewards.append(0.0)
            pending.append(deque())

        active = list(range(len(envs)))
        stochastic = self.config["learning_method"] == "Discrete SAC"
        self.net.eval()
        while active:
            batch_observations = [observations[i] for i in active]
            epsilon = self._epsilon_at_step() if not stochastic else 0.0
            action_batch, scores_batch = self._choose_actions(
                batch_observations, epsilon, stochastic_policy=stochastic
            )
            next_active = []
            for local_i, env_i in enumerate(active):
                env = envs[env_i]
                next_observation, reward, done = env.step(
                    action_batch[local_i], scores_batch[local_i]
                )
                # Keep the policy decision in replay. The environment may reject
                # an otherwise legal BUY when simultaneous entries exceed free slots;
                # recording that rejected BUY as HOLD would mislabel the Q-learning action.
                selected_actions = action_batch[local_i].copy()
                if self.config["learning_method"] == "N-Step Double DQN":
                    pending[env_i].append((
                        observations[env_i], selected_actions, float(reward), next_observation, done
                    ))
                    if len(pending[env_i]) >= self.config["n_step"]:
                        self._emit_n_step(pending[env_i], self.config["n_step"])
                    if done:
                        while pending[env_i]:
                            self._emit_n_step(pending[env_i], len(pending[env_i]))
                else:
                    self.replay.add(
                        observations[env_i], selected_actions, reward, next_observation,
                        done, discount=self.config["gamma"],
                    )
                observations[env_i] = next_observation
                rewards[env_i] += float(reward)
                if not done:
                    next_active.append(env_i)
            self.environment_steps += len(active)
            active = next_active
        self.net.train()
        return rewards

    def _collect_onpolicy_episodes(self):
        """Collect complete factorized categorical-policy trajectories."""
        lookback = self.config["lookback_candles"]
        horizon = self.config["episode_candles"]
        last_start = self.market.train_end - horizon - 1
        if last_start < lookback - 1:
            raise RuntimeError("Not enough training rows for the configured episode length")

        states, portfolios, masks = [], [], []
        actions, old_log_probs, values = [], [], []
        returns, advantages, episode_rewards = [], [], []
        self.net.eval()
        for _ in range(self.config["rollout_episodes"]):
            start = random.randint(lookback - 1, last_start)
            env = TradingEnvironment(self.market, self.config, self.market.train_end)
            observation = env.reset(start)
            ep_rewards, ep_values = [], []
            ep_states, ep_portfolios, ep_masks = [], [], []
            ep_actions, ep_log_probs = [], []
            total_reward = 0.0
            while True:
                market_t, portfolio_t, mask_t = self._batch_observations([observation])
                with torch.inference_mode():
                    output = self.net(market_t, portfolio_t)
                    # Sample symbols sequentially, masking BUY once all portfolio
                    # slots have been allocated. This avoids training on rejected buys.
                    logits = output["logits"][0]
                    free_slots = max(0, self.config["max_positions"] - env._open_count())
                    order = torch.randperm(self.market.features.shape[1], device=DEVICE)
                    action_t = torch.zeros(
                        self.market.features.shape[1], dtype=torch.long, device=DEVICE
                    )
                    policy_mask = mask_t[0].clone()
                    log_prob_terms = []
                    for symbol_i in order:
                        symbol_mask = policy_mask[symbol_i].clone()
                        if free_slots <= 0:
                            symbol_mask[1] = False
                            policy_mask[symbol_i, 1] = False
                        distribution = torch.distributions.Categorical(
                            logits=self._masked_logits(
                                logits[symbol_i].unsqueeze(0), symbol_mask.unsqueeze(0)
                            )
                        )
                        selected = distribution.sample()
                        action_t[symbol_i] = selected[0]
                        log_prob_terms.append(distribution.log_prob(selected)[0])
                        if int(selected.item()) == 1:
                            free_slots -= 1
                    log_prob = torch.stack(log_prob_terms).sum().reshape(1)
                    action_np = action_t.cpu().numpy().astype(np.int64)
                    scores = torch.softmax(
                        self._masked_logits(output["logits"], mask_t), dim=-1
                    )[0, :, 1].cpu().numpy()
                next_observation, reward, done = env.step(action_np, scores)
                ep_states.append(observation[0])
                ep_portfolios.append(observation[1])
                ep_masks.append(policy_mask.cpu().numpy().astype(np.bool_))
                ep_actions.append(action_np)
                ep_log_probs.append(float(log_prob.item()))
                ep_values.append(float(output["value"].item()))
                ep_rewards.append(float(reward))
                total_reward += float(reward)
                observation = next_observation
                if done:
                    break
            self.environment_steps += len(ep_rewards)

            gamma = self.config["gamma"]
            method = self.config["learning_method"]
            if method == "REINFORCE":
                running = 0.0
                ep_returns = [0.0] * len(ep_rewards)
                for t in reversed(range(len(ep_rewards))):
                    running = ep_rewards[t] + gamma * running
                    ep_returns[t] = running
                ep_adv = [ret - val for ret, val in zip(ep_returns, ep_values)]
            else:
                gae = 0.0
                next_value = 0.0
                ep_adv = [0.0] * len(ep_rewards)
                ep_returns = [0.0] * len(ep_rewards)
                lam = 0.95
                for t in reversed(range(len(ep_rewards))):
                    delta = ep_rewards[t] + gamma * next_value - ep_values[t]
                    gae = delta + gamma * lam * gae
                    ep_adv[t] = gae
                    ep_returns[t] = gae + ep_values[t]
                    next_value = ep_values[t]

            states.extend(ep_states)
            portfolios.extend(ep_portfolios)
            masks.extend(ep_masks)
            actions.extend(ep_actions)
            old_log_probs.extend(ep_log_probs)
            values.extend(ep_values)
            returns.extend(ep_returns)
            advantages.extend(ep_adv)
            episode_rewards.append(total_reward)

        rollout = {
            "market": torch.from_numpy(np.stack(states)).to(DEVICE),
            "portfolio": torch.from_numpy(np.stack(portfolios)).to(DEVICE),
            "masks": torch.from_numpy(np.stack(masks)).to(DEVICE),
            "actions": torch.from_numpy(np.stack(actions)).long().to(DEVICE),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE),
            "values": torch.tensor(values, dtype=torch.float32, device=DEVICE),
            "returns": torch.tensor(returns, dtype=torch.float32, device=DEVICE),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=DEVICE),
            "episode_rewards": episode_rewards,
        }
        if method != "REINFORCE":
            rollout["advantages"] = (
                rollout["advantages"] - rollout["advantages"].mean()
            ) / (rollout["advantages"].std(unbiased=False) + 1e-8)
        else:
            rollout["advantages"] = rollout["advantages"] - rollout["advantages"].mean()
        return rollout

    def _soft_update_target(self):
        tau = self.config["target_tau"]
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_net.parameters(), self.net.parameters()
            ):
                target_parameter.lerp_(online_parameter, tau)

    def _train_q_learning(self):
        if self.replay.size < self.config["min_replay_size"]:
            return 0.0
        method = self.config["learning_method"]
        gamma = self.config["gamma"]
        losses = []
        for update in range(self.config["train_updates"]):
            beta = min(1.0, 0.4 + 0.6 * (self.environment_steps / max(self.config["epsilon_decay_steps"], 1)))
            batch = self.replay.sample(self.config["batch_size"], DEVICE, beta=beta)
            current = self.net(batch["market"], batch["portfolio"])
            actions = batch["actions"]
            weights = batch["weights"]

            if method == "QR-DQN":
                quantiles = current["quantiles"]
                action_index = actions[:, :, None, None].expand(-1, -1, 1, self.net.num_quantiles)
                chosen = quantiles.gather(2, action_index).squeeze(2)
                with torch.no_grad():
                    next_online = self.net(batch["next_market"], batch["next_portfolio"])
                    next_action = next_online["quantiles"].mean(dim=-1).masked_fill(
                        ~batch["next_mask"], -1e9
                    ).argmax(dim=-1)
                    next_target = self.target_net(batch["next_market"], batch["next_portfolio"])
                    next_idx = next_action[:, :, None, None].expand(-1, -1, 1, self.net.num_quantiles)
                    next_quantiles = next_target["quantiles"].gather(2, next_idx).squeeze(2)
                    target = batch["rewards"][:, None, None] + (
                        batch["discounts"] * (1.0 - batch["dones"])
                    )[:, None, None] * next_quantiles
                td = target.unsqueeze(-2) - chosen.unsqueeze(-1)
                abs_td = td.abs()
                huber = torch.where(abs_td <= 1.0, 0.5 * td.square(), abs_td - 0.5)
                n = self.net.num_quantiles
                taus = ((torch.arange(n, device=DEVICE, dtype=torch.float32) + 0.5) / n).view(1, 1, n, 1)
                quantile_weight = torch.abs(taus - (td.detach() < 0).float())
                per_symbol = (quantile_weight * huber).mean(dim=(-1, -2))
                per_sample = per_symbol.mean(dim=1)
                td_priority = td.detach().abs().mean(dim=(-1, -2, -3))
                loss = (weights * per_sample).mean()
            elif method == "Discrete SAC":
                selected_q1 = current["q1"].gather(-1, actions.unsqueeze(-1)).squeeze(-1).mean(dim=1)
                selected_q2 = current["q2"].gather(-1, actions.unsqueeze(-1)).squeeze(-1).mean(dim=1)
                with torch.no_grad():
                    next_online = self.net(batch["next_market"], batch["next_portfolio"])
                    next_target = self.target_net(batch["next_market"], batch["next_portfolio"])
                    next_logits = self._masked_logits(next_online["logits"], batch["next_mask"])
                    next_log_prob = torch.log_softmax(next_logits, dim=-1)
                    next_prob = next_log_prob.exp()
                    min_target_q = torch.minimum(next_target["q1"], next_target["q2"])
                    soft_value = (
                        next_prob * (min_target_q - self.config["sac_alpha"] * next_log_prob)
                    ).sum(dim=-1).mean(dim=1)
                    target = batch["rewards"] + batch["discounts"] * (
                        1.0 - batch["dones"]
                    ) * soft_value
                td1, td2 = selected_q1 - target, selected_q2 - target
                critic_per_sample = 0.5 * (td1.square() + td2.square())
                critic_loss = (weights * critic_per_sample).mean()
                logits = self._masked_logits(current["logits"], batch["mask"])
                log_prob = torch.log_softmax(logits, dim=-1)
                probs = log_prob.exp()
                min_q = torch.minimum(current["q1"].detach(), current["q2"].detach())
                actor_per_sample = (
                    probs * (self.config["sac_alpha"] * log_prob - min_q)
                ).sum(dim=-1).mean(dim=1)
                actor_loss = actor_per_sample.mean()
                loss = critic_loss + actor_loss
                td_priority = 0.5 * (td1.detach().abs() + td2.detach().abs())
            else:
                q = current["q1"]
                chosen_q = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1).mean(dim=1)
                with torch.no_grad():
                    target_output = self.target_net(batch["next_market"], batch["next_portfolio"])
                    target_q = target_output["q1"]
                    if method in {"Double DQN", "Prioritized Double DQN", "N-Step Double DQN"}:
                        online_next = self.net(batch["next_market"], batch["next_portfolio"])
                        next_actions = online_next["q1"].masked_fill(
                            ~batch["next_mask"], -1e9
                        ).argmax(dim=-1)
                        next_q = target_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1).mean(dim=1)
                    else:  # Vanilla DQN and Dueling DQN
                        next_q = target_q.masked_fill(~batch["next_mask"], -1e9).amax(dim=-1).mean(dim=1)
                    target = batch["rewards"] + batch["discounts"] * (
                        1.0 - batch["dones"]
                    ) * next_q
                td_error = chosen_q - target
                per_sample = nn.functional.smooth_l1_loss(
                    chosen_q, target, reduction="none"
                )
                loss = (weights * per_sample).mean()
                td_priority = td_error.detach().abs()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimizer.step()
            self._soft_update_target()
            self.replay.update_priorities(batch["indices"], td_priority.cpu().numpy())
            losses.append(float(loss.detach().item()))
        return float(np.mean(losses)) if losses else 0.0

    def _train_actor_critic(self, rollout):
        self.net.train()
        method = self.config["learning_method"]
        count = rollout["actions"].shape[0]
        batch_size = min(128, count)
        epochs = 4 if method == "PPO" else 1
        losses = []
        for _ in range(epochs):
            permutation = torch.randperm(count, device=DEVICE)
            for start in range(0, count, batch_size):
                idx = permutation[start:start + batch_size]
                output = self.net(rollout["market"][idx], rollout["portfolio"][idx])
                dist = torch.distributions.Categorical(
                    logits=self._masked_logits(output["logits"], rollout["masks"][idx])
                )
                log_prob = dist.log_prob(rollout["actions"][idx]).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()
                advantage = rollout["advantages"][idx].detach()
                if method == "PPO":
                    ratio = torch.exp(log_prob - rollout["old_log_probs"][idx])
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio, 1.0 - self.config["ppo_clip"],
                        1.0 + self.config["ppo_clip"],
                    ) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                else:  # A2C and Monte-Carlo REINFORCE with learned baseline
                    policy_loss = -(log_prob * advantage).mean()
                value_loss = nn.functional.smooth_l1_loss(
                    output["value"], rollout["returns"][idx]
                )
                loss = (
                    policy_loss
                    + self.config["value_coef"] * value_loss
                    - self.config["entropy_coef"] * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses)) if losses else 0.0

    def _evaluate(self):
        """Deterministic evaluation over evenly spaced held-out windows."""
        self.net.eval()
        val_start = max(self.config["lookback_candles"] - 1, self.market.train_end - 1)
        val_end = self.market.val_end
        horizon = self.config["episode_candles"]
        last_start = max(val_start, val_end - 1 - horizon)
        starts = np.unique(np.linspace(
            val_start, last_start,
            num=self.config["validation_episodes"],
        ).round().astype(int))
        envs, observations, episode_rewards = [], [], []
        for start in starts:
            steps = min(horizon, val_end - 1 - int(start))
            if steps <= 0:
                continue
            env = TradingEnvironment(self.market, self.config, val_end)
            envs.append(env)
            observations.append(env.reset(int(start), INITIAL_CAPITAL, steps))
            episode_rewards.append(0.0)

        active = list(range(len(envs)))
        executed_action_counts = np.zeros(3, dtype=np.int64)
        proposed_action_counts = np.zeros(3, dtype=np.int64)
        with torch.inference_mode():
            while active:
                obs_batch = [observations[i] for i in active]
                market_t, portfolio_t, mask_t = self._batch_observations(obs_batch)
                output = self.net(market_t, portfolio_t)
                if self.config["learning_method"] in {"PPO", "A2C", "REINFORCE", "Discrete SAC"}:
                    actions = self._masked_logits(output["logits"], mask_t).argmax(dim=-1).cpu().numpy()
                    scores = torch.softmax(self._masked_logits(output["logits"], mask_t), dim=-1)[:, :, 1].cpu().numpy()
                else:
                    q = output["q1"]
                    actions = q.masked_fill(~mask_t, -1e9).argmax(dim=-1).cpu().numpy()
                    scores = (q[:, :, 1] - q[:, :, 0]).cpu().numpy()
                actions = self._cap_simultaneous_buys(actions, scores, obs_batch)
                proposed_action_counts += np.bincount(
                    np.asarray(actions, dtype=np.int64).ravel(), minlength=3
                )[:3]
                next_active = []
                for local_i, env_i in enumerate(active):
                    nxt, reward, done = envs[env_i].step(actions[local_i], scores[local_i])
                    executed_action_counts += np.bincount(
                        envs[env_i].last_executed_actions, minlength=3
                    )[:3]
                    observations[env_i] = nxt
                    episode_rewards[env_i] += float(reward)
                    if not done:
                        next_active.append(env_i)
                active = next_active

        if not envs:
            raise RuntimeError("Validation could not create any episodes")
        episode_returns = [
            (env.final_equity / INITIAL_CAPITAL - 1.0) * 100.0 for env in envs
        ]
        curves = [list(env.equity_curve) for env in envs]
        max_len = max(map(len, curves))
        padded = [curve + [curve[-1]] * (max_len - len(curve)) for curve in curves]
        mean_curve = np.mean(np.asarray(padded, dtype=np.float64), axis=0).tolist()
        pnls = [pnl for env in envs for pnl in env.trade_pnls]
        trades = len(pnls)
        wins = sum(pnl > 0 for pnl in pnls)
        gross_profit = float(sum(pnl for pnl in pnls if pnl > 0))
        gross_loss = float(-sum(pnl for pnl in pnls if pnl < 0))
        win_rate = 100.0 * wins / trades if trades else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0)
        mean_return = float(np.mean(episode_returns))
        symbol_stats = {}
        for symbol_index, symbol in enumerate(self.market.symbols):
            symbol_pnls = [
                pnl for env in envs for pnl in env.symbol_trade_pnls[symbol_index]
            ]
            symbol_stats[symbol] = {
                "trades": int(len(symbol_pnls)),
                "realized_pnl": float(sum(symbol_pnls)),
                "wins": int(sum(pnl > 0 for pnl in symbol_pnls)),
                "win_rate": float(100.0 * sum(pnl > 0 for pnl in symbol_pnls) / len(symbol_pnls)) if symbol_pnls else 0.0,
            }
        executed_total = int(executed_action_counts.sum())
        proposed_total = int(proposed_action_counts.sum())
        executed_action_pct = (executed_action_counts / executed_total * 100.0).tolist() if executed_total else [0.0, 0.0, 0.0]
        proposed_action_pct = (proposed_action_counts / proposed_total * 100.0).tolist() if proposed_total else [0.0, 0.0, 0.0]

        # Equal-weight buy-and-hold reference over exactly the same validation
        # windows. It is a diagnostic benchmark, not a constrained strategy.
        benchmark_returns = []
        fee = max(0.0, float(self.config["fee_bps"])) / 10_000.0
        for start, env in zip(starts, envs):
            end = env.current_t
            entry = self.market.open_prices[int(start) + 1]
            exit_prices = self.market.prices[end]
            allocation = INITIAL_CAPITAL / self.market.prices.shape[1]
            quantities = allocation / (entry * (1.0 + fee))
            benchmark_equity = float(np.sum(quantities * exit_prices * (1.0 - fee)))
            benchmark_returns.append((benchmark_equity / INITIAL_CAPITAL - 1.0) * 100.0)

        curve_array = np.asarray(mean_curve, dtype=np.float64)
        running_peak = np.maximum.accumulate(curve_array)
        drawdowns = (curve_array / np.maximum(running_peak, 1e-12) - 1.0) * 100.0
        max_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0
        return {
            "return_pct": mean_return,
            "benchmark_return_pct": float(np.mean(benchmark_returns)) if benchmark_returns else 0.0,
            "excess_return_pct": mean_return - (float(np.mean(benchmark_returns)) if benchmark_returns else 0.0),
            "max_drawdown_pct": max_drawdown,
            "profit_factor": float(profit_factor),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "reward": float(np.mean(episode_rewards)),
            "equity": mean_curve,
            "trades": trades,
            "win_rate": float(win_rate),
            "no_trade_episodes": int(sum(env.trade_count == 0 for env in envs)),
            "final_equity": float(INITIAL_CAPITAL * (1.0 + mean_return / 100.0)),
            "validation_samples": int(len(episode_returns)),
            "executed_action_pct": executed_action_pct,
            "proposed_action_pct": proposed_action_pct,
            "symbol_stats": symbol_stats,
            "validation_windows": [
                {"start": int(start), "end": int(env.current_t),
                 "return_pct": float((env.final_equity / INITIAL_CAPITAL - 1.0) * 100.0),
                 "trades": int(env.trade_count)}
                for start, env in zip(starts, envs)
            ],
        }

    def _save_checkpoint(self, epoch: int, val_stats=None):
        ROOT_C.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.net.state_dict(),
            "target_model": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": dict(self.config),
            "symbols": self.market.symbols,
            "attributes": self.market.attributes,
            "feature_mean": self.market.feature_mean,
            "feature_std": self.market.feature_std,
            "epoch": int(epoch),
            "environment_steps": int(self.environment_steps),
            "epsilon": float(self.epsilon),
            "seed": SEED,
            "validation": val_stats or {},
            "checkpoint_selection": "excess_return_pct subject to min_validation_trades",
            "min_validation_trades": int(self.config.get("min_validation_trades", DEFAULT_MIN_VALIDATION_TRADES)),
            "architecture": self.config["architecture"],
            "learning_method": self.config["learning_method"],
            "format": "nexus_multi_algorithm_v1",
        }
        target = ROOT_C / "nexus_multi_method.pth"
        temporary = ROOT_C / "nexus_multi_method.pth.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, target)

    def run(self):
        if not self.model_ready:
            raise RuntimeError("Build the policy before training")
        self.stop_requested = False
        self.shutdown_requested = False
        for epoch in range(1, 10001):
            try:
                command, value = self.input_queue.get_nowait()
            except queue.Empty:
                command, value = None, None
            if command == "lr":
                self.set_lr(value)
            elif command == "save":
                self.save_requested = True
            elif command == "stop":
                self.stop_requested = True
            elif command == "shutdown":
                self.stop_requested = True
                self.shutdown_requested = True
            elif command == "properties":
                Properties.set_all(value)
                self.stop_requested = True
                self.queue.put(("error", "Properties changed during training. Reload data and rebuild the policy."))
            if self.stop_requested:
                break

            method = self.config["learning_method"]
            if self._is_on_policy(method):
                rollout = self._collect_onpolicy_episodes()
                loss = self._train_actor_critic(rollout)
                episode_rewards = rollout["episode_rewards"]
            else:
                episode_rewards = self._collect_offpolicy_episodes()
                loss = self._train_q_learning()
            train_reward = float(np.mean(episode_rewards))
            train_return_pct = (math.exp(train_reward / 100.0) - 1.0) * 100.0

            validation_fresh = (
                self.last_val_stats is None
                or epoch % self.config["validation_interval"] == 0
            )
            if validation_fresh:
                self.last_val_stats = self._evaluate()
                val_stats = self.last_val_stats
                min_trades = int(self.config.get("min_validation_trades", DEFAULT_MIN_VALIDATION_TRADES))
                checkpoint_eligible = val_stats["trades"] >= min_trades
                if checkpoint_eligible and val_stats["excess_return_pct"] > self.best_val_excess:
                    self.best_val_excess = val_stats["excess_return_pct"]
                    self.best_val_return = val_stats["return_pct"]
                    self._save_checkpoint(epoch, val_stats)
                self.queue.put(("validationPlot", val_stats["equity"]))
            else:
                val_stats = self.last_val_stats
            if self.save_requested:
                self._save_checkpoint(epoch, val_stats)
                self.save_requested = False

            self.queue.put((
                "onMetricsChanged",
                [epoch, train_return_pct, val_stats["return_pct"],
                 self.best_val_return, val_stats["trades"], val_stats["win_rate"], loss],
            ))
            marker = " [validation]" if validation_fresh else ""
            self.log(
                logging.INFO,
                "Update %05d | method=%s | architecture=%s | train=%+.3f%% | "
                "validation=%+.3f%% | best_validation=%+.3f%% | epsilon=%.3f | "
                "loss=%.6f | trades=%d | win_rate=%.1f%% | episodes=%d | "
                "benchmark=%+.3f%% | excess=%+.3f%% | max_dd=%.3f%% | "
                "profit_factor=%.3f | no_trade_episodes=%d | min_trade_gate=%d/%d | "
                "actions_exec(H/B/S)=%.1f/%.1f/%.1f%% | actions_policy(H/B/S)=%.1f/%.1f/%.1f%% | "
                "best_excess=%+.3f%%%s",
                epoch, method, self.config["architecture"], train_return_pct,
                val_stats["return_pct"], self.best_val_return, self.epsilon, loss,
                val_stats["trades"], val_stats["win_rate"],
                val_stats.get("validation_samples", 0),
                val_stats.get("benchmark_return_pct", 0.0),
                val_stats.get("excess_return_pct", 0.0),
                val_stats.get("max_drawdown_pct", 0.0),
                val_stats.get("profit_factor", 0.0),
                val_stats.get("no_trade_episodes", 0),
                val_stats.get("trades", 0), self.config.get("min_validation_trades", DEFAULT_MIN_VALIDATION_TRADES),
                *(val_stats.get("executed_action_pct", [0.0, 0.0, 0.0])),
                *(val_stats.get("proposed_action_pct", [0.0, 0.0, 0.0])),
                self.best_val_excess, marker,
            )
            if validation_fresh:
                symbol_summary = "; ".join(
                    f"{symbol}:trades={stats['trades']},pnl={stats['realized_pnl']:+.2f},win={stats['win_rate']:.1f}%"
                    for symbol, stats in val_stats.get("symbol_stats", {}).items()
                )
                self.log(logging.INFO, "Validation per-symbol | %s", symbol_summary)
                self.log(
                    logging.INFO,
                    "Validation windows | %s",
                    "; ".join(
                        f"[{w['start']}:{w['end']}] return={w['return_pct']:+.3f}% trades={w['trades']}"
                        for w in val_stats.get("validation_windows", [])
                    ),
                )
                if val_stats.get("trades", 0) < self.config.get("min_validation_trades", DEFAULT_MIN_VALIDATION_TRADES):
                    self.log(
                        logging.WARNING,
                        "Checkpoint not eligible: validation trades %d below minimum %d",
                        val_stats.get("trades", 0),
                        self.config.get("min_validation_trades", DEFAULT_MIN_VALIDATION_TRADES),
                    )
        self.net.eval()


# ---------------------------------------------------------------------------
# Multiprocessing worker bridge
# ---------------------------------------------------------------------------

def training_process(input_queue: Queue, output_queue: Queue, initial_properties=None):
    if initial_properties is not None:
        Properties.set_all(initial_properties)
    trainer = RewardTrainer(input_queue, output_queue)
    trainer.log(logging.INFO, "Worker process started")

    while True:
        command, value = input_queue.get()
        try:
            LOGGER.info("Worker command received: %s", command)
            if command == "load_CSV":
                trainer.config = dict(value or trainer.default_config())
                trainer.load_CSV()
                output_queue.put(("CSVLoaded", []))
            elif command == "load_data":
                trainer.load_data()
                output_queue.put(("dataLoaded", []))
            elif command == "build_model":
                trainer.build_model(value or {})
                output_queue.put(("modelInitialized", []))
            elif command == "properties":
                Properties.set_all(value)
            elif command == "run":
                if not trainer.model_ready:
                    output_queue.put(("error", "Load data and build the policy before training."))
                else:
                    trainer.run()
                    output_queue.put(("trainingStopped", []))
                    if trainer.shutdown_requested:
                        break
            elif command == "stop":
                trainer.stop_requested = True
                output_queue.put(("trainingStopped", []))
            elif command == "lr":
                trainer.set_lr(float(value))
            elif command == "save":
                if trainer.model_ready:
                    trainer._save_checkpoint(-1, trainer.last_val_stats)
                else:
                    trainer.save_requested = True
            elif command == "shutdown":
                trainer.stop_requested = True
                trainer.log(logging.INFO, "Worker shutdown requested")
                break
        except Exception as exc:
            details = traceback.format_exc()
            LOGGER.exception("Worker command failed: %s", command)
            output_queue.put(("log", {
                "level": "ERROR",
                "message": f"{command} failed: {type(exc).__name__}: {exc}\\n{details}",
                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
            }))
            output_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class ThreadModel(QObject):
    metrics = Signal(list)
    validationPlot = Signal(object)
    processChanged = Signal(str)
    error = Signal(str)
    logMessage = Signal(dict)
    WeightsNum = Signal(int)

    def __init__(self):
        super().__init__()
        self.output_queue = Queue()
        self.input_queue = Queue()
        self.proc: Process | None = None
        self.properties_snapshot = copy.deepcopy(Properties.get_all())

    def start_worker(self, properties_snapshot=None):
        if self.proc is not None and self.proc.is_alive():
            return
        if properties_snapshot is None:
            properties_snapshot = Properties.get_all()
        self.properties_snapshot = copy.deepcopy(properties_snapshot)
        self.proc = Process(
            target=training_process,
            args=(self.input_queue, self.output_queue, self.properties_snapshot),
            daemon=True,
        )
        self.proc.start()

    def send(self, command, value=None):
        if self.proc is None or not self.proc.is_alive():
            self.start_worker(Properties.get_all())
        self.input_queue.put((command, value))

    def check_query(self):
        while True:
            try:
                message, value = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if message in {"modelInitialized", "dataLoaded", "trainingStopped", "CSVLoaded"}:
                self.processChanged.emit(message)
            elif message == "onMetricsChanged":
                self.metrics.emit(value)
            elif message == "validationPlot":
                self.validationPlot.emit(value)
            elif message == "error":
                self.error.emit(str(value))
            elif message == "log":
                self.logMessage.emit(value)
            elif message == "WeightsNum":
                self.WeightsNum.emit(int(value))

    def shutdown(self):
        if self.proc is not None and self.proc.is_alive():
            self.input_queue.put(("shutdown", None))
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=1)


# ---------------------------------------------------------------------------
# Dashboard backend
# ---------------------------------------------------------------------------

class Backend(QObject):
    statusChanged = Signal()
    metricsChanged = Signal()
    validationPlotChanged = Signal(object)
    weightsChanged = Signal(int)
    errorChanged = Signal(str)
    logAdded = Signal(dict)

    def __init__(self):
        super().__init__()
        self.nnThreadModel: ThreadModel | None = None
        self.qtimer: QTimer | None = None
        self.status = "SYSTEM READY"
        self.error = ""
        defaults = RewardTrainer.default_config()
        self.config = {
            key: defaults[key]
            for key in (
                "num_symbols", "lookback_candles", "episode_candles", "max_positions",
                "fee_bps", "lr", "learning_method", "architecture", "hidden_dim",
                "hidden_layers", "conv_channels", "attention_heads", "recurrent_layers",
                "dropout", "quantiles",
            )
        }
        self.metrics = []
        self.equity_curve = []
        self.latest = {
            "epoch": 0,
            "train_return": 0.0,
            "val_return": 0.0,
            "best_val_return": None,
            "trades": 0,
            "win_rate": 0.0,
            "dqn_loss": 0.0,
        }
        self.weights_num = 0
        self.log_records = deque(maxlen=2000)

    def add_log(self, level, message, timestamp=None):
        record = {
            "level": str(level).upper(),
            "message": str(message),
            "time": timestamp or pd.Timestamp.now().strftime("%H:%M:%S"),
        }
        self.log_records.append(record)
        self.logAdded.emit(record)

    @Slot(dict)
    def _receive_log(self, record):
        self.add_log(record.get("level", "INFO"), record.get("message", ""),
                     record.get("time"))

    def initClass(self):
        if self.nnThreadModel is not None:
            return
        self.nnThreadModel = ThreadModel()
        self.nnThreadModel.processChanged.connect(self._thread_event)
        self.nnThreadModel.metrics.connect(self._receive_metrics)
        self.nnThreadModel.validationPlot.connect(self._receive_plot)
        self.nnThreadModel.error.connect(self._set_error)
        self.nnThreadModel.logMessage.connect(self._receive_log)
        self.nnThreadModel.WeightsNum.connect(self._receive_weights)
        self.nnThreadModel.start_worker(Properties.get_all())
        self.qtimer = QTimer(self)
        self.qtimer.timeout.connect(self.nnThreadModel.check_query)
        self.qtimer.start(50)
        self.status = "WORKER READY"
        self.add_log("INFO", "Worker process initialized")
        self.statusChanged.emit()

    @Slot(str)
    def _thread_event(self, event):
        mapping = {
            "CSVLoaded": "CSV LOADED — READY TO LOAD DATA",
            "dataLoaded": "DATA READY — BUILD THE POLICY",
            "modelInitialized": "POLICY READY — START TRAINING",
            "trainingStopped": "TRAINING STOPPED",
        }
        self.status = mapping.get(event, str(event))
        self.add_log("INFO", self.status)
        self.statusChanged.emit()

    @Slot(list)
    def _receive_metrics(self, data):
        if len(data) < 7:
            return
        epoch, train_ret, val_ret, best_ret, trades, win_rate, dqn_loss = data[:7]
        self.latest = {
            "epoch": int(epoch),
            "train_return": float(train_ret),
            "val_return": float(val_ret),
            "best_val_return": None if best_ret is None else float(best_ret),
            "trades": int(trades),
            "win_rate": float(win_rate),
            "dqn_loss": float(dqn_loss),
        }
        self.metrics.append(dict(self.latest))
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
        self.metricsChanged.emit()

    @Slot(object)
    def _receive_plot(self, equity):
        self.equity_curve = list(equity or [])
        self.validationPlotChanged.emit(self.equity_curve)

    @Slot(int)
    def _receive_weights(self, value):
        self.weights_num = int(value)
        self.weightsChanged.emit(self.weights_num)

    @Slot(str)
    def _set_error(self, message):
        self.error = str(message)
        self.status = f"ERROR: {self.error}"
        self.add_log("ERROR", self.error)
        self.errorChanged.emit(self.error)
        self.statusChanged.emit()

    def update_config(self, name, raw_value):
        integer_fields = {
            "num_symbols", "lookback_candles", "episode_candles", "max_positions",
            "hidden_dim", "hidden_layers", "conv_channels", "attention_heads",
            "recurrent_layers", "quantiles",
        }
        float_fields = {"fee_bps", "lr", "dropout"}
        try:
            if name in integer_fields:
                value = int(raw_value)
                if value <= 0:
                    raise ValueError
                if name == "num_symbols" and value > len(SYMBOLS):
                    raise ValueError
            elif name in float_fields:
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError
                if name == "fee_bps" and value < 0:
                    raise ValueError
                if name == "lr" and value <= 0:
                    raise ValueError
                if name == "dropout" and not 0 <= value < 1:
                    raise ValueError
            elif name == "learning_method":
                value = str(raw_value)
                if value not in LEARNING_METHODS:
                    raise ValueError
            elif name == "architecture":
                value = str(raw_value)
                if value not in NETWORK_ARCHITECTURES:
                    raise ValueError
            else:
                return
        except (TypeError, ValueError):
            self.status = f"Invalid {name} value"
            self.statusChanged.emit()
            return

        self.config[name] = value
        if name == "num_symbols":
            Properties.num_symbols = value
            self.status = "SYMBOL COUNT CHANGED — RELOAD CSV, DATA, AND POLICY"
        elif name == "lr":
            Properties.LR = value
            if self.nnThreadModel is not None:
                proc = self.nnThreadModel.proc
                if proc is not None and proc.is_alive():
                    self.nnThreadModel.send("lr", value)
            self.status = "LEARNING RATE UPDATED"
        else:
            self.status = "MODEL SETTINGS CHANGED — REBUILD POLICY TO APPLY"
        self.statusChanged.emit()

    def _send(self, command, value=None):
        self.initClass()
        self.nnThreadModel.send(command, value)

    @Slot()
    def loadCSV(self):
        self._send("load_CSV", dict(self.config))
        self.add_log("INFO", "Loading aligned market CSV files")
        self.status = "LOADING ALIGNED CSV DATA..."
        self.statusChanged.emit()

    @Slot()
    def loadData(self):
        self._send("load_data")
        self.add_log("INFO", "Validating loaded market data")
        self.status = "VALIDATING MARKET DATA..."
        self.statusChanged.emit()

    @Slot()
    def buildModel(self):
        self._send("build_model", dict(self.config))
        self.add_log("INFO", "Building selected policy and network architecture")
        self.status = "BUILDING SELECTED POLICY + ARCHITECTURE..."
        self.statusChanged.emit()

    @Slot()
    def startTraining(self):
        self._send("run")
        self.add_log("INFO", "Training run requested")
        self.status = "MODEL TRAINING RUNNING"
        self.statusChanged.emit()

    @Slot()
    def stopTraining(self):
        if self.nnThreadModel is not None:
            self.nnThreadModel.send("stop")
            self.add_log("WARNING", "Training stop requested")
            self.status = "STOP REQUESTED — FINISHING CURRENT TRAINING BLOCK"
            self.statusChanged.emit()

    @Slot()
    def saveModel(self):
        if self.nnThreadModel is not None:
            self.nnThreadModel.send("save")
            self.add_log("INFO", "Checkpoint save requested")
            self.status = "CHECKPOINT SAVE REQUESTED"
            self.statusChanged.emit()

    @Slot()
    def terminate(self):
        if self.qtimer is not None:
            self.qtimer.stop()
            self.qtimer = None
        if self.nnThreadModel is not None:
            self.nnThreadModel.shutdown()
            self.nnThreadModel = None
        self.status = "WORKER STOPPED"
        self.statusChanged.emit()

    def clearHistory(self):
        self.log_records.clear()
        self.metrics.clear()
        self.equity_curve.clear()
        self.latest.update({
            "epoch": 0,
            "train_return": 0.0,
            "val_return": 0.0,
            "best_val_return": None,
            "trades": 0,
            "win_rate": 0.0,
            "dqn_loss": 0.0,
        })
        self.metricsChanged.emit()
        self.validationPlotChanged.emit([])


# ---------------------------------------------------------------------------
# Qt dashboard styling and widgets
# ---------------------------------------------------------------------------

BG = "#050b14"
PANEL = "#0b1826"
INPUT_BG = "#06111e"
BORDER = "#16465c"
NEON = "#00eaff"
GREEN = "#00ff9d"
MUTED = "#64869e"
TEXT = "#d9faff"
CYAN_BUTTON = "#00a9c4"

APP_STYLE = f"""
QMainWindow, QWidget {{
    background: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif;
}}
QFrame#panel {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
}}
QFrame#terminalPanel {{
    background: #06111a; border: 1px solid {BORDER}; border-radius: 8px;
}}
QLineEdit {{
    background: {INPUT_BG}; color: #e4faff; border: 1px solid {BORDER};
    border-radius: 5px; padding: 7px;
}}
QLineEdit:focus {{ border: 1px solid {NEON}; }}
QPushButton {{
    background: {CYAN_BUTTON}; color: #00121a; border: none;
    border-radius: 5px; padding: 8px 10px; font-weight: bold; min-height: 21px;
}}
QPushButton:hover {{ background: #26cce3; }}
QPushButton#clearButton {{
    background: #30202c; color: #ff829d; border: 1px solid #a64d68;
}}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{
    background: #0c1b29; border: 1px solid #183247; border-radius: 4px;
    padding: 7px; margin: 2px 0; color: {TEXT};
}}
"""


class MetricCard(QFrame):
    def __init__(self, label, value="—", accent=NEON):
        super().__init__()
        self.setObjectName("panel")
        self.setMinimumHeight(75)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 9, 7, 9)
        layout.setSpacing(5)
        title = QLabel(label)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {MUTED}; font-size: 9px; font-weight: bold;")
        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet(
            f"color: {accent}; font-size: 16px; font-weight: bold;"
        )
        layout.addWidget(title)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class NexusWindow(QMainWindow):
    def __init__(self, backend: Backend):
        super().__init__()
        self.backend = backend
        self.parameter_edits = {}
        self.setWindowTitle("NEXUS // REWARD-BASED TRADING AGENT")
        self.resize(1350, 900)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(APP_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 15, 18, 15)
        root.setSpacing(11)
        root.addWidget(self._build_header())
        root.addWidget(self._build_config_panel())
        root.addLayout(self._build_metrics())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_plot_panel())
        splitter.addWidget(self._build_terminal_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([930, 320])
        root.addWidget(splitter, 1)
        root.addWidget(self._build_footer())

        self._connect_backend()
        for record in self.backend.log_records:
            self._render_log_record(record)
        self.refresh()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(500)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start()

    def _build_header(self):
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        title_col = QVBoxLayout()
        title = QLabel("N E X U S")
        title.setStyleSheet(f"color: {NEON}; font-size: 27px; font-weight: bold;")
        subtitle = QLabel("MULTI-ALGORITHM AGENT // CONFIGURABLE POLICY + ENCODER")
        subtitle.setStyleSheet(f"color: {MUTED}; font-size: 10px; letter-spacing: 2px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        row.addLayout(title_col)
        row.addStretch()
        self.online_label = QLabel("● OFFLINE TRAINING")
        self.online_label.setStyleSheet(f"color: {GREEN}; font-size: 11px; font-weight: bold;")
        row.addWidget(self.online_label)
        return wrapper

    def _build_config_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)

        # Data/trading settings.
        config_row = QHBoxLayout()
        config_row.setSpacing(8)
        fields = [
            ("num_symbols", "Symbols"),
            ("lookback_candles", "Lookback"),
            ("episode_candles", "Trade horizon"),
            ("max_positions", "Max positions"),
            ("fee_bps", "Cost / side (bps)"),
            ("lr", "Learning rate"),
        ]
        for key, label_text in fields:
            col = QVBoxLayout()
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {NEON}; font-size: 9px;")
            edit = QLineEdit(str(self.backend.config[key]))
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setMinimumWidth(76)
            edit.editingFinished.connect(
                lambda name=key, widget=edit: self._apply_config(name, widget.text())
            )
            col.addWidget(label)
            col.addWidget(edit)
            holder = QWidget()
            holder.setLayout(col)
            config_row.addWidget(holder, 1)
            self.parameter_edits[key] = edit
        layout.addLayout(config_row)

        # Independent learner and architecture selectors.
        selector_row = QHBoxLayout()
        selector_row.setSpacing(8)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(LEARNING_METHODS)
        self.algorithm_combo.setCurrentText(self.backend.config["learning_method"])
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems(NETWORK_ARCHITECTURES)
        self.architecture_combo.setCurrentText(self.backend.config["architecture"])
        for label_text, combo in (("LEARNING METHOD", self.algorithm_combo), ("NETWORK ARCHITECTURE", self.architecture_combo)):
            holder = QWidget()
            col = QVBoxLayout(holder)
            col.setContentsMargins(0, 0, 0, 0)
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {NEON}; font-size: 9px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(label)
            col.addWidget(combo)
            selector_row.addWidget(holder, 1)
        self.algorithm_combo.currentTextChanged.connect(
            lambda value: self._apply_config("learning_method", value)
        )
        self.architecture_combo.currentTextChanged.connect(
            lambda value: self._apply_config("architecture", value)
        )
        layout.addLayout(selector_row)

        # Editable neural-network dimensions. The selected architecture uses
        # the relevant controls; the same config is stored in the checkpoint.
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_fields = [
            ("hidden_dim", "Hidden dimension"),
            ("hidden_layers", "Hidden layers"),
            ("conv_channels", "CNN channels"),
            ("attention_heads", "Attention heads"),
            ("recurrent_layers", "RNN layers"),
            ("dropout", "Dropout"),
            ("quantiles", "QR quantiles"),
        ]
        for key, label_text in model_fields:
            col = QVBoxLayout()
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {NEON}; font-size: 9px;")
            edit = QLineEdit(str(self.backend.config[key]))
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setMinimumWidth(64)
            edit.editingFinished.connect(
                lambda name=key, widget=edit: self._apply_config(name, widget.text())
            )
            col.addWidget(label)
            col.addWidget(edit)
            holder = QWidget()
            holder.setLayout(col)
            model_row.addWidget(holder, 1)
            self.parameter_edits[key] = edit
        layout.addLayout(model_row)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        buttons = [
            ("Init Worker", self.backend.initClass),
            ("Load CSV", self.backend.loadCSV),
            ("Load Data", self.backend.loadData),
            ("Build Model", self.backend.buildModel),
            ("Start Training", self.backend.startTraining),
            ("Save", self.backend.saveModel),
            ("STOP", self.backend.stopTraining),
            ("Terminate", self.backend.terminate),
        ]
        for label, callback in buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        clear = QPushButton("CLEAR")
        clear.setObjectName("clearButton")
        clear.clicked.connect(self._clear_dashboard)
        actions.addWidget(clear)
        layout.addLayout(actions)
        return panel

    def _apply_config(self, name, value):
        self.backend.update_config(name, value)
        if name in self.parameter_edits:
            self.parameter_edits[name].setText(str(self.backend.config[name]))
        self.refresh()

    def _clear_dashboard(self):
        self.backend.clearHistory()
        if hasattr(self, "log_list"):
            self.log_list.clear()

    def _build_metrics(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        self.metric_epoch = MetricCard("MODEL UPDATES", "0", NEON)
        self.metric_train = MetricCard("TRAIN EPISODE RETURN", "—", GREEN)
        self.metric_val = MetricCard("VALIDATION RETURN", "—", GREEN)
        self.metric_best = MetricCard("BEST VALIDATION", "—", GREEN)
        self.metric_trades = MetricCard("VAL TRADES", "0", NEON)
        self.metric_win = MetricCard("WIN RATE", "—", NEON)
        self.metric_weights = MetricCard("POLICY PARAMETERS", "—", NEON)
        for card in (
            self.metric_epoch, self.metric_train, self.metric_val,
            self.metric_best, self.metric_trades, self.metric_win,
            self.metric_weights,
        ):
            row.addWidget(card, 1)
        return row

    def _section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet(f"color: {NEON}; font-size: 10px; font-weight: bold;")
        return label

    def _style_plot(self, plot, ylabel, xlabel):
        plot.setBackground("#071522")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setLabel("left", ylabel, color="#a9d6e5")
        plot.setLabel("bottom", xlabel, color="#a9d6e5")
        plot.getAxis("left").setTextPen("#a9d6e5")
        plot.getAxis("bottom").setTextPen("#a9d6e5")
        plot.getAxis("left").setPen(BORDER)
        plot.getAxis("bottom").setPen(BORDER)

    def _build_plot_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.addWidget(self._section_label("02 // POLICY REWARD + BACKTEST"))
        self.tabs = QTabWidget()

        reward_tab = QWidget()
        reward_layout = QVBoxLayout(reward_tab)
        self.reward_plot = pg.PlotWidget()
        self._style_plot(self.reward_plot, "Return (%)", "DQN update")
        self.reward_plot.setTitle("Training vs sampled validation return", color=NEON, size="11pt")
        self.reward_plot.addLegend(offset=(10, 10))
        self.train_curve = self.reward_plot.plot(
            [], [], name="Train episode return", pen=pg.mkPen(NEON, width=2),
            symbol="o", symbolSize=4,
        )
        self.val_curve = self.reward_plot.plot(
            [], [], name="Sampled validation return", pen=pg.mkPen(GREEN, width=2),
            symbol="t", symbolSize=5,
        )
        reward_layout.addWidget(self.reward_plot)
        self.tabs.addTab(reward_tab, "REWARD HISTORY")

        equity_tab = QWidget()
        equity_layout = QVBoxLayout(equity_tab)
        self.equity_plot = pg.PlotWidget()
        self._style_plot(self.equity_plot, "Portfolio equity", "Validation candle")
        self.equity_plot.setTitle("Mean sampled validation equity", color=NEON, size="11pt")
        self.equity_curve = self.equity_plot.plot([], [], pen=pg.mkPen(GREEN, width=2.5))
        equity_layout.addWidget(self.equity_plot)
        self.tabs.addTab(equity_tab, "VALIDATION EQUITY")

        layout.addWidget(self.tabs, 1)
        return panel

    def _build_terminal_panel(self):
        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel.setMinimumWidth(230)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(11, 9, 11, 9)
        heading = QHBoxLayout()
        heading.addWidget(self._section_label("03 // SYSTEM + TRAINING LOG"))
        heading.addStretch()
        live = QLabel("OFFLINE")
        live.setStyleSheet(f"color: {GREEN}; font-size: 9px;")
        heading.addWidget(live)
        layout.addLayout(heading)
        self.log_list = QListWidget()
        self.log_list.setSpacing(4)
        self.log_list.setUniformItemSizes(False)
        self.log_list.setWordWrap(True)
        self.log_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.log_list, 1)
        return panel

    def _build_footer(self):
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        left = QLabel("NEXUS // PYSIDE6 + PYQTGRAPH + PYTORCH RL")
        left.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        row.addWidget(left)
        row.addStretch()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {GREEN}; font-size: 9px;")
        row.addWidget(self.status_label)
        return wrapper

    def _connect_backend(self):
        self.backend.metricsChanged.connect(self.refresh)
        self.backend.statusChanged.connect(self.refresh)
        self.backend.validationPlotChanged.connect(self._show_equity)
        self.backend.weightsChanged.connect(
            lambda value: self.metric_weights.set_value(f"{value:,}")
        )
        self.backend.errorChanged.connect(self._show_error)
        self.backend.logAdded.connect(self._append_log)

    @Slot(dict)
    def _append_log(self, record):
        self._render_log_record(record)

    def _render_log_record(self, record):
        level = record.get("level", "INFO").upper()
        message = record.get("message", "")
        timestamp = record.get("time", "")
        item = QListWidgetItem(f"{timestamp}  [{level}]  {message}")
        colors = {
            "DEBUG": MUTED, "INFO": TEXT, "WARNING": "#ffc857",
            "ERROR": "#ff647c", "CRITICAL": "#ff3b5c",
        }
        item.setForeground(QColor(colors.get(level, TEXT)))
        item.setToolTip(message)
        self.log_list.addItem(item)
        while self.log_list.count() > 2000:
            self.log_list.takeItem(0)
        self.log_list.scrollToBottom()

    @Slot(object)
    def _show_equity(self, values):
        values = list(values or [])
        self.equity_curve.setData(np.arange(len(values)), values)
        if values:
            self.equity_plot.enableAutoRange(axis="y", enable=True)

    @Slot(str)
    def _show_error(self, message):
        self.status_label.setText(f"ERROR: {message}")

    def refresh(self):
        latest = self.backend.latest
        self.metric_epoch.set_value(latest["epoch"])
        self.metric_train.set_value(f"{latest['train_return']:+.3f}%")
        self.metric_val.set_value(f"{latest['val_return']:+.3f}%")
        best = latest["best_val_return"]
        self.metric_best.set_value("—" if best is None else f"{best:+.3f}%")
        self.metric_trades.set_value(latest["trades"])
        self.metric_win.set_value(f"{latest['win_rate']:.1f}%")
        self.status_label.setText(self.backend.status)

        records = self.backend.metrics
        if records:
            x = [record["epoch"] for record in records]
            train = [record["train_return"] for record in records]
            val = [record["val_return"] for record in records]
            self.train_curve.setData(x, train)
            self.val_curve.setData(x, val)
            last_epoch = records[-1]["epoch"]
            if (
                not self.log_list.count()
                or self.log_list.item(self.log_list.count() - 1).data(Qt.ItemDataRole.UserRole)
                != last_epoch
            ):
                r = records[-1]
                item = QListWidgetItem(
                    f"UPDATE {r['epoch']:04d} | TRAIN {r['train_return']:+.3f}%\n"
                    f"VAL {r['val_return']:+.3f}% | TRADES {r['trades']} "
                    f"| WIN {r['win_rate']:.1f}%"
                )
                item.setData(Qt.ItemDataRole.UserRole, last_epoch)
                self.log_list.addItem(item)
                if r["best_val_return"] is not None and r["val_return"] >= r["best_val_return"]:
                    item.setForeground(QColor(GREEN))
                self.log_list.scrollToBottom()

    def closeEvent(self, event):
        if hasattr(self, "refresh_timer"):
            self.refresh_timer.stop()
        self.backend.terminate()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    backend = Backend()
    window = NexusWindow(backend)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
