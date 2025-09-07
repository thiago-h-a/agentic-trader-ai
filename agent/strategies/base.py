from __future__ import annotations

"""
# ─────────────────────────────────────────────────────────────────────────────
# agent/strategies/base.py
# Small Strategy base types with a clear, serializable TradingSignal.
# ─────────────────────────────────────────────────────────────────────────────
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import pandas as pd


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class StrategyConfig:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """
    A single trading action suggestion produced by a strategy.
    Designed to be trivially JSON-serializable (via .to_dict()).
    """
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float
    quantity: int = 0
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.name,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class TradingStrategy(ABC):
    """
    Implementations must produce a list of TradingSignal objects from a price dataframe.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.name = config.name
        self.description = config.description
        self.parameters = dict(config.parameters or {})

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        ...

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
