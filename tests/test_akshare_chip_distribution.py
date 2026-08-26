# -*- coding: utf-8 -*-
"""Tests for AkshareFetcher.get_chip_distribution."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.realtime_types import ChipDistribution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fetcher():
    """创建一个休眠为 0 的 AkshareFetcher，加速测试。"""
    return AkshareFetcher(sleep_min=0, sleep_max=0)


def _make_chip_df(rows: int = 5) -> pd.DataFrame:
    """构造 ak.stock_cyq_em 返回的模拟 DataFrame。"""
    data = {
        "日期": [f"2026-08-{20 + i}" for i in range(rows)],
        "获利比例": [0.35 + i * 0.05 for i in range(rows)],
        "平均成本": [1800.0 + i * 10 for i in range(rows)],
        "90成本-低": [1700.0 + i * 5 for i in range(rows)],
        "90成本-高": [1900.0 + i * 15 for i in range(rows)],
        "90集中度": [0.12 - i * 0.005 for i in range(rows)],
        "70成本-低": [1750.0 + i * 5 for i in range(rows)],
        "70成本-高": [1850.0 + i * 10 for i in range(rows)],
        "70集中度": [0.08 - i * 0.003 for i in range(rows)],
    }
    return pd.DataFrame(data)


def _install_fake_akshare(monkeypatch, stock_cyq_em_func):
    """将 akshare 模块替换为 fake，只暴露 stock_cyq_em。"""
    fake_akshare = SimpleNamespace(stock_cyq_em=stock_cyq_em_func)
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------

class TestGetChipDistributionSuccess:
    """正常获取筹码分布数据的场景。"""

    def test_returns_chip_distribution_for_a_stock(self, fetcher, monkeypatch):
        """A 股代码应返回 ChipDistribution 对象，取最新一天数据。"""
        df = _make_chip_df(5)
        _install_fake_akshare(monkeypatch, lambda symbol: df)

        result = fetcher.get_chip_distribution("600519")

        assert result is not None
        assert isinstance(result, ChipDistribution)
        assert result.code == "600519"
        # 应取最后一行（最新一天）
        assert result.date == "2026-08-24"
        assert result.profit_ratio == pytest.approx(0.55)
        assert result.avg_cost == pytest.approx(1840.0)
        assert result.concentration_90 == pytest.approx(0.10)
        assert result.concentration_70 == pytest.approx(0.068)

    def test_single_row_dataframe(self, fetcher, monkeypatch):
        """只有一行数据时也应正常返回。"""
        df = _make_chip_df(1)
        _install_fake_akshare(monkeypatch, lambda symbol: df)

        result = fetcher.get_chip_distribution("000001")

        assert result is not None
        assert result.code == "000001"
        assert result.date == "2026-08-20"

    def test_all_fields_populated(self, fetcher, monkeypatch):
        """验证 ChipDistribution 所有字段都被正确填充。"""
        df = _make_chip_df(3)
        _install_fake_akshare(monkeypatch, lambda symbol: df)

        result = fetcher.get_chip_distribution("600519")

        assert result is not None
        # 90 成本区间
        assert result.cost_90_low == pytest.approx(1710.0)
        assert result.cost_90_high == pytest.approx(1930.0)
        # 70 成本区间
        assert result.cost_70_low == pytest.approx(1760.0)
        assert result.cost_70_high == pytest.approx(1870.0)
        # source 默认值
        assert result.source == "akshare"


# ---------------------------------------------------------------------------
# 不支持的代码类型 → 直接返回 None
# ---------------------------------------------------------------------------

class TestGetChipDistributionSkipped:
    """不支持的代码类型应直接返回 None，不调用 API。"""

    def test_us_stock_returns_none(self, fetcher, monkeypatch):
        """美股代码应返回 None。"""
        called = False

        def fake_cyq(symbol):
            nonlocal called
            called = True
            return _make_chip_df()

        _install_fake_akshare(monkeypatch, fake_cyq)

        result = fetcher.get_chip_distribution("AAPL")
        assert result is None
        assert not called, "美股不应调用 ak.stock_cyq_em"

    def test_hk_stock_returns_none(self, fetcher, monkeypatch):
        """港股代码应返回 None。"""
        called = False

        def fake_cyq(symbol):
            nonlocal called
            called = True
            return _make_chip_df()

        _install_fake_akshare(monkeypatch, fake_cyq)

        result = fetcher.get_chip_distribution("00700")
        assert result is None
        assert not called

    def test_hk_stock_with_prefix_returns_none(self, fetcher, monkeypatch):
        """带 HK 前缀的港股代码也应返回 None。"""
        result = fetcher.get_chip_distribution("HK00700")
        assert result is None

    def test_etf_returns_none(self, fetcher, monkeypatch):
        """ETF 代码应返回 None。"""
        called = False

        def fake_cyq(symbol):
            nonlocal called
            called = True
            return _make_chip_df()

        _install_fake_akshare(monkeypatch, fake_cyq)

        result = fetcher.get_chip_distribution("512400")
        assert result is None
        assert not called

    def test_shenzhen_etf_returns_none(self, fetcher, monkeypatch):
        """深交所 ETF 代码也应返回 None。"""
        result = fetcher.get_chip_distribution("159995")
        assert result is None


# ---------------------------------------------------------------------------
# 异常与边界情况
# ---------------------------------------------------------------------------

class TestGetChipDistributionEdgeCases:
    """API 异常和边界数据处理。"""

    def test_empty_dataframe_returns_none(self, fetcher, monkeypatch):
        """ak.stock_cyq_em 返回空 DataFrame 时应返回 None。"""
        _install_fake_akshare(monkeypatch, lambda symbol: pd.DataFrame())

        result = fetcher.get_chip_distribution("600519")
        assert result is None

    def test_api_exception_returns_none(self, fetcher, monkeypatch):
        """API 调用抛异常时应返回 None，不向外抛出。"""

        def fake_cyq(symbol):
            raise ConnectionError("网络不通")

        _install_fake_akshare(monkeypatch, fake_cyq)

        result = fetcher.get_chip_distribution("600519")
        assert result is None

    def test_api_timeout_returns_none(self, fetcher, monkeypatch):
        """API 超时异常也应返回 None。"""

        def fake_cyq(symbol):
            raise TimeoutError("请求超时")

        _install_fake_akshare(monkeypatch, fake_cyq)

        result = fetcher.get_chip_distribution("600519")
        assert result is None

    def test_nan_values_cause_graceful_failure(self, fetcher, monkeypatch):
        """全 NaN 数据时，日志格式化 None 会触发异常，方法应优雅返回 None。

        这反映了实际行为：get_chip_distribution 内部的 logger.info 使用
        f"{chip.profit_ratio:.1%}" 格式化，当 profit_ratio 为 None 时会
        抛 TypeError，被外层 except 捕获后返回 None。
        """
        import numpy as np

        df = pd.DataFrame({
            "日期": ["2026-08-25"],
            "获利比例": [np.nan],
            "平均成本": [np.nan],
            "90成本-低": [np.nan],
            "90成本-高": [np.nan],
            "90集中度": [np.nan],
            "70成本-低": [np.nan],
            "70成本-高": [np.nan],
            "70集中度": [np.nan],
        })
        _install_fake_akshare(monkeypatch, lambda symbol: df)

        result = fetcher.get_chip_distribution("600519")
        # 当前行为：日志格式化 None 导致异常，返回 None
        assert result is None

    def test_correct_symbol_passed_to_api(self, fetcher, monkeypatch):
        """验证传给 ak.stock_cyq_em 的 symbol 参数正确。"""
        captured_symbols = []

        def fake_cyq(symbol):
            captured_symbols.append(symbol)
            return _make_chip_df(1)

        _install_fake_akshare(monkeypatch, fake_cyq)

        fetcher.get_chip_distribution("000001")
        assert captured_symbols == ["000001"]

        fetcher.get_chip_distribution("300750")
        assert captured_symbols == ["000001", "300750"]


# ---------------------------------------------------------------------------
# 集成测试（需要网络，标记为 network）
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestGetChipDistributionLive:
    """真实网络调用集成测试，默认不执行。"""

    def test_live_chip_distribution(self):
        """真实获取 600519 筹码分布，验证返回结构完整。"""
        fetcher = AkshareFetcher(sleep_min=0.5, sleep_max=1.0)
        result = fetcher.get_chip_distribution("600519")

        if result is None:
            pytest.skip("非交易时间或 API 不可用，筹码数据可能为空")

        assert isinstance(result, ChipDistribution)
        assert result.code == "600519"
        assert result.date != ""
        # 至少平均成本应大于 0
        if result.avg_cost is not None:
            assert result.avg_cost > 0
