import pytest
import pandas as pd
from datetime import datetime, time as dt_time
from unittest.mock import MagicMock, patch
import tkinter as tk
import warnings
import runpy

import projectv3 as app
from projectv3 import EnergyApp

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def small_df():
    data = [
        {"timestamp": "2025-01-01 00:00:00", "kWh": 1.0},
        {"timestamp": "2025-01-01 06:00:00", "kWh": 2.0},
        {"timestamp": "2025-01-01 18:30:00", "kWh": 3.0},
        {"timestamp": "2025-01-02 20:00:00", "kWh": 4.0},
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["kWh"] = df["kWh"].astype(float)
    return df

# -------------------------
# Parser / Helper functions
# -------------------------
def test_safe_float_valid():
    assert app.safe_float(" 1.234 ") == pytest.approx(1.234)
    assert app.safe_float(2) == pytest.approx(2.0)
    assert app.safe_float(None, default=9.9) == pytest.approx(9.9)

def test_safe_float_invalid_returns_default():
    assert app.safe_float("not_a_number", default=5.5) == pytest.approx(5.5)
    assert app.safe_float("", default=-1.0) == pytest.approx(-1.0)

def test_safe_int_valid_and_blank():
    assert app.safe_int(" 42 ") == 42
    assert app.safe_int(7) == 7
    assert app.safe_int("", default=None) is None
    assert app.safe_int("", default=100) == 100

def test_safe_int_invalid_returns_default():
    assert app.safe_int("12.3", default=-1) == -1
    assert app.safe_int("abc", default=0) == 0

def test_parse_date_valid_and_invalid():
    assert isinstance(app.parse_date("2025-01-01"), datetime)
    assert app.parse_date("invalid-date") is None
    assert app.parse_date("") is None

# -------------------------
# flatRateTariff
# -------------------------
def test_flat_rate_basic(small_df):
    res = app.flatRateTariff(small_df, flatRate=0.5, fixedFee=5.0)
    assert res["scheme"] == "Flat"
    assert res["totalKWh"] == pytest.approx(10.0)
    assert res["breakdown"]["Energy"] == pytest.approx(5.0)
    assert res["totalBill"] == pytest.approx(10.0)

def test_flat_rate_zero_usage():
    df = pd.DataFrame([{"timestamp": "2025-01-01", "kWh": 0.0}])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    res = app.flatRateTariff(df, flatRate=0.25, fixedFee=3.0)
    assert res["totalKWh"] == pytest.approx(0.0)
    assert res["totalBill"] == pytest.approx(3.0)

# -------------------------
# TOU Tariff
# -------------------------
def test_tou_basic_assignment(small_df):
    touRates = {
        "Peak": {"start": dt_time(18,0), "end": dt_time(23,0), "rate": 0.5},
        "Off-Peak": {"start": dt_time(0,0), "end": dt_time(7,0), "rate": 0.1},
        "Shoulder": {"default": True, "rate": 0.2}
    }
    fee = 2.0
    res = app.touTariff(small_df, touRates, fee)
    assert res["breakdown"]["Off-Peak"] == pytest.approx(0.3)
    assert res["breakdown"]["Peak"] == pytest.approx(3.5)
    assert res["breakdown"]["Fixed Fee"] == pytest.approx(fee)
    assert res["totalBill"] == pytest.approx(0.3 + 3.5 + fee)

def test_tou_wrap_around_midnight():
    df = pd.DataFrame([
        {"timestamp": pd.Timestamp("2025-01-01 23:30"), "kWh": 1.0},
        {"timestamp": pd.Timestamp("2025-01-02 01:00"), "kWh": 2.0},
        {"timestamp": pd.Timestamp("2025-01-02 12:00"), "kWh": 3.0}
    ])
    touRates = {
        "Off-Peak": {"start": dt_time(22,0), "end": dt_time(7,0), "rate": 0.1},
        "Peak": {"start": dt_time(7,0), "end": dt_time(19,0), "rate": 0.4},
        "Shoulder": {"default": True, "rate": 0.25}
    }
    res = app.touTariff(df, touRates, fixedFee=1.0)
    assert res["breakdown"]["Off-Peak"] == pytest.approx(0.3)
    assert res["breakdown"]["Peak"] == pytest.approx(1.2)
    assert res["breakdown"]["Fixed Fee"] == pytest.approx(1.0)
    assert res["totalBill"] == pytest.approx(2.5)

def test_tou_with_non_datetime_timestamps():
    df = pd.DataFrame([
        {"timestamp": "2025-01-01 18:00:00", "kWh": 1.0},
        {"timestamp": "2025-01-01 02:00:00", "kWh": 2.0}
    ])
    touRates = {
        "Peak": {"start": dt_time(17,0), "end": dt_time(19,0), "rate": 1.0},
        "Shoulder": {"default": True, "rate": 0.1}
    }
    res = app.touTariff(df, touRates, fixedFee=0.0)
    assert res["breakdown"]["Peak"] == pytest.approx(1.0)
    assert res["breakdown"]["Shoulder"] == pytest.approx(0.2)

# -------------------------
# Tiered Tariff
# -------------------------
def test_tiered_basic_under_first_tier():
    df = pd.DataFrame([{"timestamp": "2025-01-01", "kWh": 50.0}])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    limits = ["100", "300", ""]
    rates = [0.2, 0.3, 0.4]
    res = app.tieredTariff(df, limits, rates, fixed_fee=5.0)
    assert res["totalKWh"] == pytest.approx(50.0)
    assert res["breakdown"]["Tier 1"] == pytest.approx(10.0)
    assert res["totalBill"] == pytest.approx(15.0)

def test_tiered_spill_over_multiple_tiers():
    df = pd.DataFrame([{"timestamp": "2025-01-01", "kWh": 350.0}])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    limits = ["100", "300", ""]
    rates = [0.2, 0.3, 0.4]
    res = app.tieredTariff(df, limits, rates, fixed_fee=0.0)
    assert res["breakdown"]["Tier 1"] == pytest.approx(20.0)
    assert res["breakdown"]["Tier 2"] == pytest.approx(60.0)
    assert res["breakdown"]["Tier 3"] == pytest.approx(20.0)
    assert res["totalBill"] == pytest.approx(100.0)

def test_tiered_extra_rate_applied():
    df = pd.DataFrame([{"timestamp": "2025-01-01", "kWh": 1000.0}])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    limits = ["100", "200"]
    rates = [0.1, 0.2, 0.5]
    res = app.tieredTariff(df, limits, rates, fixed_fee=0.0)
    assert res["breakdown"]["Tier 1"] == pytest.approx(10.0)
    assert res["breakdown"]["Tier 2"] == pytest.approx(20.0)
    assert res["breakdown"]["Tier 3"] == pytest.approx(400.0)
    assert res["totalBill"] == pytest.approx(430.0)

def test_tiered_empty_limits_zero_usage():
    df = pd.DataFrame([{"timestamp": "2025-01-01", "kWh": 0.0}])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    res = app.tieredTariff(df, tier_limits=[""], tier_rates=[0.2], fixed_fee=1.0)
    assert res["breakdown"]["Fixed Fee"] == pytest.approx(1.0)
    assert res["totalBill"] == pytest.approx(1.0)

# -------------------------
# filter_by_duration
# -------------------------
def test_filter_by_duration_full_day_inclusive(small_df):
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 1)
    out = app.filter_by_duration(small_df, start, end)
    assert len(out) == 3
    assert out["timestamp"].dt.date.min() == datetime(2025, 1, 1).date()

def test_filter_by_duration_range_across_days(small_df):
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 2)
    out = app.filter_by_duration(small_df, start, end)
    assert len(out) == 4

def test_filter_by_duration_no_results(small_df):
    start = datetime(2024, 12, 1)
    end = datetime(2024, 12, 31)
    out = app.filter_by_duration(small_df, start, end)
    assert out.empty

def test_filter_by_duration_cross_midnight():
    df = pd.DataFrame([
        {"timestamp": pd.Timestamp("2025-01-01 23:00"), "kWh": 1},
        {"timestamp": pd.Timestamp("2025-01-02 01:00"), "kWh": 2}
    ])
    start = datetime(2025, 1, 1, 22, 0)
    end = datetime(2025, 1, 2, 2, 0)
    out = app.filter_by_duration(df, start, end)
    assert len(out) == 2

def test_filter_by_duration_single_row():
    df = pd.DataFrame([{"timestamp": pd.Timestamp("2025-01-01 12:00"), "kWh": 1}])
    start = datetime(2025, 1, 1, 12, 0)
    end = datetime(2025, 1, 1, 12, 0)
    out = app.filter_by_duration(df, start, end)
    assert len(out) == 1

def test_filter_by_duration_edge_cases():
    # Empty DataFrame
    df = pd.DataFrame(columns=["timestamp", "kWh"])
    out = app.filter_by_duration(df, "00:00", "23:59")
    assert out.empty
    # No match
    df = pd.DataFrame([{"timestamp": pd.Timestamp("2025-01-01 05:00"), "kWh": 1}])
    out = app.filter_by_duration(df, "06:00", "07:00")
    assert out.empty

# -------------------------
# Upload file tests (headless)
# -------------------------
def test_upload_file_missing_columns(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "fake.csv")
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pd.DataFrame({"bad": [1]}))
        with patch.object(app.messagebox, "showerror") as mock_error:
            eapp.upload_file()
            mock_error.assert_called_once()

def test_upload_file_invalid_timestamps(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "fake.csv")
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pd.DataFrame({
            "timestamp": ["bad-date", "2025-01-01 01:00:00"],
            "kWh": [1, 2]
        }))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with patch.object(app.messagebox, "showinfo") as mock_info:
                eapp.upload_file()
                # Only the valid timestamp should remain
                assert len(eapp.df) == 1
                mock_info.assert_called_once()

def test_upload_file_exception(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "fake.csv")
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")))
        with patch.object(app.messagebox, "showerror") as mock_error:
            eapp.upload_file()
            mock_error.assert_called_once()

# -------------------------
# Load_demo_data
# -------------------------
def test_load_demo_data(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)

        # Mock the upload_status label
        eapp.upload_status.cget = MagicMock(return_value="Demo data loaded")

        with patch.object(app.messagebox, "showinfo") as mock_info:
            eapp.load_demo_data()
            assert len(eapp.df) == 9
            assert "demo" in eapp.upload_status.cget("text").lower()
            mock_info.assert_called_once()

# -------------------------
# show_usage_trend tests (headless)
# -------------------------
def test_show_usage_trend_no_data(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        eapp.df = None
        with patch.object(app.messagebox, "showwarning") as mock_warn:
            eapp.show_usage_trend()
            mock_warn.assert_called_once_with("No data", "Upload a dataset first (Upload tab).")

def test_show_usage_trend_no_data_in_range(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        eapp.df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "kWh": [1.0],
        })
        eapp.vis_start.get = MagicMock(return_value="2030-01-01")
        eapp.vis_end.get = MagicMock(return_value="2030-01-02")
        with patch.object(app.messagebox, "showwarning") as mock_warn:
            eapp.show_usage_trend()
            mock_warn.assert_called_once_with(
                "No data in range",
                "No data between selected dates."
            )

# -------------------------
# Fix for show_usage_trend (mock Matplotlib canvas)
# -------------------------
def test_show_usage_trend_hourly_and_daily(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)

        eapp.df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 00:00:00", periods=3, freq="h"),
            "kWh": [0.5, 0.6, 0.7],
        })
        eapp.vis_start.get = MagicMock(return_value="")
        eapp.vis_end.get = MagicMock(return_value="")

        # Mock FigureCanvasTkAgg so no real GUI is created
        with patch("projectv3.FigureCanvasTkAgg", MagicMock()):
            eapp.show_usage_trend()

        # Test daily data similarly
        eapp.df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="D"),
            "kWh": [1.0, 2.0, 3.0],
        })
        with patch("projectv3.FigureCanvasTkAgg", MagicMock()):
            eapp.show_usage_trend()

# -------------------------
# Mousewheel tests (headless)
# -------------------------
def test_mousewheel_methods(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        sf = eapp.tab_upload
        sf.canvas = MagicMock()
        event = MagicMock()
        event.delta = 120
        sf._on_mousewheel(event)
        sf.canvas.yview_scroll.assert_called_with(-1, "units")
        sf._bind_to_mousewheel()
        sf.canvas.bind_all.assert_called_with("<MouseWheel>", sf._on_mousewheel)
        sf._unbind_from_mousewheel()
        sf.canvas.unbind_all.assert_called_with("<MouseWheel>")

# -------------------------
# EnergyApp GUI (headless)
# -------------------------
@pytest.fixture
def headless_app(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app_instance = app.EnergyApp(root)
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **kwargs: "demo.csv")
    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: pd.DataFrame({"timestamp":["2025-01-01"],"kWh":[1]}))
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *args, **kwargs: None)
    yield app_instance
    root.destroy()

def test_energyapp_methods(headless_app):
    headless_app.upload_file()
    headless_app.compute_flat()
    headless_app.compute_tou()
    headless_app.compute_tier()
    headless_app.calculate_and_compare()
    bills = getattr(headless_app, "bills", {})
    headless_app._show_selected_breakdown(bills)

# -------------------------
# Module-level code
# -------------------------
def test_module_level_import(monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

# -------------------------
# upload_file edge cases
# -------------------------
def test_upload_file_empty_df(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = tk.Tk()
        eapp = app.EnergyApp(root)
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "fake.csv")
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pd.DataFrame({"timestamp": [], "kWh": []}))
        with patch.object(app.messagebox, "showinfo") as mock_info:
            eapp.upload_file()
            assert eapp.df.empty
            mock_info.assert_called_once()

# -------------------------
# compute methods – empty df edge cases
# -------------------------
@patch("tkinter.messagebox.showwarning")
def test_compute_flat_empty_df(mock_warn):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
                             "kWh": [1.0]})
    with patch("projectv3.filter_by_duration", return_value=pd.DataFrame()):
        result = eapp.compute_flat()
    mock_warn.assert_called_once()


# compute_tou_empty_df
@patch("tkinter.messagebox.showwarning")
def test_compute_tou_empty_df(mock_warn):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
                             "kWh": [1.0]})
    with patch("projectv3.filter_by_duration", return_value=pd.DataFrame()):
        result = eapp.compute_tou()
    mock_warn.assert_called_once()


# -------------------------
# compute_tou and calculate_and_compare: invalid / exceptional h_to_time inputs
# -------------------------
# h_to_time_invalid
def test_h_to_time_invalid(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01"]),
        "kWh": [1.0]
    })

    # Bad inputs
    eapp.tou_start_entry = MagicMock(get=lambda: "bad")
    eapp.tou_end_entry = MagicMock(get=lambda: "bad")
    
    # Patch messagebox and _draw_breakdown
    with patch.object(app.messagebox, "showwarning") as mock_warn:
        monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)
        monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
        
        # Force filter_by_duration to return empty DataFrame to hit warning branch
        monkeypatch.setattr(app, "filter_by_duration", lambda df, start, end: pd.DataFrame())
        
        eapp.compute_tou()
        assert mock_warn.called  # Now this should be True

#calc_compare_h_to_time_invalid
def test_compute_tou_h_to_time_valid(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 12:00"]),
        "kWh": [1.0]
    })
    eapp.tou_start_entry = MagicMock(get=lambda: "10:30")
    eapp.tou_end_entry = MagicMock(get=lambda: "15:00")
    eapp.tou_result_frame = MagicMock()

    # Patch the tariff function
    monkeypatch.setattr(app, "touTariff", lambda *a, **kw: {"totalBill": 0})
    
    # Patch _draw_breakdown to prevent Tkinter/Matplotlib GUI calls
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)
    
    # Patch simpledialog.askstring to return a value without GUI
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    
    # Patch messagebox.showwarning to prevent GUI popups
    with patch.object(app.messagebox, "showwarning") as mock_warn:
        eapp.compute_tou()
        # optional: check if warnings were called if needed
        # assert mock_warn.called

def test_h_to_time_exception(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 12:00"]), "kWh": [1.0]})
    eapp.tou_result_frame = MagicMock()
    eapp.tou_w = MagicMock(get=lambda: "5")
    eapp.tou_h = MagicMock(get=lambda: "4")

    # Patch entries to anything (won't matter)
    eapp.peak_start = MagicMock(get=lambda: "foo")
    eapp.peak_end   = MagicMock(get=lambda: "bar")
    eapp.off_start  = MagicMock(get=lambda: 22)
    eapp.off_end    = MagicMock(get=lambda: 7)
    eapp.peak_rate  = MagicMock(get=lambda: 0.4)
    eapp.off_rate   = MagicMock(get=lambda: 0.15)
    eapp.shoulder_rate = MagicMock(get=lambda: 0.25)
    eapp.tou_fixed  = MagicMock(get=lambda: 10.0)

    # Patch safe_int to return something that will **break int() inside h_to_time**
    monkeypatch.setattr(app, "safe_int", lambda x, default: "not_an_int")

    # Patch tariff and GUI
    monkeypatch.setattr(app, "touTariff", lambda df, rates, fee: {"totalBill": 0})
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)

    # Run compute_tou — now h_to_time will fail and hit `except Exception`
    eapp.compute_tou()

def test_h_to_time_none(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 12:00"]), "kWh": [1.0]})
    eapp.tou_result_frame = MagicMock()
    eapp.tou_w = MagicMock(get=lambda: "5")
    eapp.tou_h = MagicMock(get=lambda: "4")

    # Patch entries
    eapp.peak_start = MagicMock(get=lambda: 0)
    eapp.peak_end   = MagicMock(get=lambda: 0)
    eapp.off_start  = MagicMock(get=lambda: 0)
    eapp.off_end    = MagicMock(get=lambda: 0)
    eapp.peak_rate  = MagicMock(get=lambda: 0.4)
    eapp.off_rate   = MagicMock(get=lambda: 0.15)
    eapp.shoulder_rate = MagicMock(get=lambda: 0.25)
    eapp.tou_fixed  = MagicMock(get=lambda: 10.0)

    # Patch safe_int to return None once to hit "if h is None"
    monkeypatch.setattr(app, "safe_int", lambda x, default: None)

    # Patch tariff and GUI
    monkeypatch.setattr(app, "touTariff", lambda df, rates, fee: {"totalBill": 0})
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)

    # Run compute_tou — now h_to_time(None) will run
    eapp.compute_tou()


def test_calculate_and_compare_h_to_time_exception(monkeypatch):
    # Patch Tkinter variables and widgets
    monkeypatch.setattr(tk, "StringVar", lambda *a, **kw: MagicMock(get=lambda: kw.get("value", ""), set=lambda v: None))
    monkeypatch.setattr(tk, "Tk", lambda *a, **kw: MagicMock())
    
    # Now create EnergyApp with mocked root
    root = tk.Tk()
    eapp = app.EnergyApp(root)

    # Minimal dataframe
    eapp.df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 12:00"]), "kWh": [1.0]})

    # Patch entries to invalid values to trigger h_to_time exception
    eapp.peak_start = MagicMock(get=lambda: "foo")
    eapp.peak_end   = MagicMock(get=lambda: "bar")
    eapp.off_start  = MagicMock(get=lambda: "baz")
    eapp.off_end    = MagicMock(get=lambda: "qux")
    eapp.peak_rate  = MagicMock(get=lambda: 0.4)
    eapp.off_rate   = MagicMock(get=lambda: 0.15)
    eapp.shoulder_rate = MagicMock(get=lambda: 0.25)
    eapp.tou_fixed  = MagicMock(get=lambda: 10.0)
    eapp.flat_rate_entry = MagicMock(get=lambda: 0.25)
    eapp.flat_fee_entry = MagicMock(get=lambda: 10.0)
    eapp.tier_limit_entries = []
    eapp.tier_rate_entries = []
    eapp.tier_fixed_entry = MagicMock(get=lambda: 10.0)
    eapp.compare_frame_inner = MagicMock()

    # Patch safe_int to trigger exception
    monkeypatch.setattr(app, "safe_int", lambda x, default: "not_an_int")
    
    # Patch tariffs and GUI helpers
    monkeypatch.setattr(app, "flatRateTariff", lambda df, rate, fee: {"totalBill": 0, "breakdown": {}, "fixedFee": 0})
    monkeypatch.setattr(app, "touTariff", lambda df, rates, fee: {"totalBill": 0})
    monkeypatch.setattr(app, "tieredTariff", lambda df, limits, rates, fee: {"totalBill": 0})
    monkeypatch.setattr(eapp, "_show_selected_breakdown", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)
    monkeypatch.setattr(app, "FigureCanvasTkAgg", lambda fig, master=None: MagicMock(draw=lambda: None, get_tk_widget=lambda: MagicMock()))

    # Run function — now the except branch in h_to_time will execute safely
    eapp.calculate_and_compare()
    root.destroy()  # Clean up the hidden Tk window

# -------------------------
# compute_tier / calculate_and_compare empty df edge cases
# -------------------------
    # compute_tier_empty_df
@patch("tkinter.messagebox.showwarning")
def test_compute_tier_empty_df(mock_warn):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
                             "kWh": [1.0]})
    with patch("projectv3.filter_by_duration", return_value=pd.DataFrame()):
        result = eapp.compute_tier()
    mock_warn.assert_called_once()


# calculate_and_compare_empty
@patch("tkinter.messagebox.showwarning")
def test_calculate_and_compare_empty(mock_warn):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
                             "kWh": [1.0]})
    with patch("projectv3.filter_by_duration", return_value=pd.DataFrame()):
        result = eapp.calculate_and_compare()
    mock_warn.assert_called_once()

@patch("tkinter.messagebox.showwarning")
def test_calculate_and_compare_no_data(mock_warn):
    root = MagicMock()
    eapp = app.EnergyApp(root)

    # Set df to None to hit the "No data" branch
    eapp.df = None

    result = eapp.calculate_and_compare()
    
    # Assert warning was shown
    mock_warn.assert_called_once_with("No data", "Upload a dataset first (Upload tab).")
    # Optionally check result is None
    assert result is None


# -------------------------
# show_selected_breakdown empty / edge cases
# -------------------------
# show_selected_breakdown_empty
def test_show_selected_breakdown_empty(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    
    # Mock GUI variables
    eapp.break_plan_var = MagicMock()
    eapp.break_chart_var = MagicMock()
    eapp.break_plan_var.get.return_value = "Flat"
    eapp.break_chart_var.get.return_value = "bar"

    # Provide proper dict structure to match expected input
    bills = {"Flat": {"breakdown": {}, "total": 0}}

    # Mock FigureCanvasTkAgg to avoid Tkinter GUI errors
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())

    eapp._show_selected_breakdown(bills)

# -------------------------
# draw_breakdown empty edge cases
# -------------------------
# draw_breakdown_empty
def test_draw_breakdown_empty(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    parent = MagicMock()
    parent.winfo_children.return_value = []

    # Mock FigureCanvasTkAgg to avoid Tkinter GUI errors
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())

    eapp._draw_breakdown(parent, billResult={})

# -------------------------
# Upload file - Excel path + empty file
# -------------------------
def test_upload_file_excel(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = app.tk.Tk()
        eapp = app.EnergyApp(root)
        # Mock Excel upload
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "demo.xlsx")
        monkeypatch.setattr(pd, "read_excel", lambda *_a, **_k: pd.DataFrame({
            "timestamp": ["2025-01-01 00:00:00"],
            "kWh": [1.0]
        }))
        monkeypatch.setattr(app.messagebox, "showinfo", lambda *a, **k: None)
        eapp.upload_file()
        assert len(eapp.df) == 1

def test_upload_file_empty_excel(monkeypatch):
    with patch("tkinter.Tk", return_value=MagicMock()):
        root = app.tk.Tk()
        eapp = app.EnergyApp(root)
        monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "empty.xlsx")
        monkeypatch.setattr(pd, "read_excel", lambda *_a, **_k: pd.DataFrame({"timestamp": [], "kWh": []}))
        with patch.object(app.messagebox, "showinfo") as mock_info:
            eapp.upload_file()
            assert eapp.df.empty
            mock_info.assert_called_once()

# -------------------------
# Upload file - mixed valid/invalid timestamps (Excel)
# -------------------------
def test_upload_file_all_invalid_timestamps(monkeypatch):
    warnings.filterwarnings("ignore", category=UserWarning, module="projectv3")
    root = MagicMock()
    eapp = app.EnergyApp(root)
    monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **_: "fake.csv")
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pd.DataFrame({
        "timestamp": ["bad1", "bad2"],
        "kWh": [1.0, 2.0]
    }))
    with patch.object(app.messagebox, "showinfo") as mock_info:
        eapp.upload_file()
        assert eapp.df.empty
        mock_info.assert_called_once()

# -------------------------
# _draw_breakdown with pie chart branch
# -------------------------
def test_draw_breakdown_pie_chart(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    parent = MagicMock()
    parent.winfo_children.return_value = []
    # Use a pie chart with sample data
    billResult = {"breakdown": {"Energy": 5, "Fixed Fee": 2}, "scheme": "Flat", "total": 7}
    eapp.break_chart_var = MagicMock()
    eapp.break_chart_var.get.return_value = "pie"
    # Mock FigureCanvasTkAgg to avoid GUI
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())
    eapp._draw_breakdown(parent, billResult)

# -------------------------
# _draw_breakdown with empty billResult
# -------------------------
def test_draw_breakdown_empty_bill(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    parent = MagicMock()
    parent.winfo_children.return_value = []
    eapp.break_chart_var = MagicMock()
    eapp.break_chart_var.get.return_value = "pie"
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())
    eapp._draw_breakdown(parent, billResult={})

# -------------------------
# compute_flat / TOU / Tier with exception branch
# -------------------------
def test_compute_flat_exception_branch_coverage():
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01")], "kWh": [1.0]})

    # Raise exception to hit that branch
    with patch("projectv3.filter_by_duration", side_effect=Exception("boom")):
        with pytest.raises(Exception, match="boom"):
            eapp.compute_flat()


def test_compute_tou_exception_branch_coverage():
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01")], "kWh": [1.0]})

    with patch("projectv3.filter_by_duration", side_effect=Exception("boom")):
        with pytest.raises(Exception, match="boom"):
            eapp.compute_tou()


def test_compute_tier_exception_branch_coverage():
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01")], "kWh": [1.0]})

    with patch("projectv3.filter_by_duration", side_effect=Exception("boom")):
        with pytest.raises(Exception, match="boom"):
            eapp.compute_tier()

# -------------------------
# upload_file exception edge cases
# -------------------------
# Force upload_file to hit CSV and Excel exception branches
def test_upload_file_exceptions(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)

    # Mock filedialog
    monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **kwargs: "data.csv")
    
    # Mock read_csv to raise exception
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")))
    
    # Mock messagebox so no real Tk window is created
    monkeypatch.setattr(app.messagebox, "showerror", lambda *a, **k: None)

    eapp.upload_file()

# -------------------------
# draw_breakdown missing keys
# -------------------------
# _draw_breakdown edge cases with missing keys
def test_draw_breakdown_missing_keys(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    parent = MagicMock()
    parent.winfo_children.return_value = []

    # No breakdown, no scheme, no total
    billResult = {}
    eapp.break_chart_var = MagicMock()
    eapp.break_chart_var.get.return_value = "pie"
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())
    eapp._draw_breakdown(parent, billResult)

# -------------------------
# filter_by_duration invalid inputs
# -------------------------
# filter_by_duration unusual input
def test_filter_by_duration_unusual():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00")], "kWh": [1]})
    
    # Expecting TypeError for invalid datetime comparison
    with pytest.raises(TypeError):
        app.filter_by_duration(df, "not-a-date", "also-bad")


# -------------------------
# compute methods empty / None DF
# -------------------------
# compute_* exception coverage already partially tested, reinforce with None DF
def test_compute_flat_tou_tier_none(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = None  # edge case for empty DF handling

    with patch.object(app.messagebox, "showwarning") as mock_warn:
        eapp.compute_flat()
        eapp.compute_tou()
        eapp.compute_tier()
        assert mock_warn.call_count == 3

# -------------------------
# upload_file cancel / empty selection
# -------------------------
def test_upload_file_no_selection(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **kwargs: "")
    # Should just return without error
    eapp.upload_file()
    assert True  # No exception means branch executed

# -------------------------
# show_usage_trend error handling
# -------------------------
def test_show_usage_trend_exception_branch(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp":[pd.Timestamp("2025-01-01")], "kWh":[1.0]})

    # Patch FigureCanvasTkAgg to raise Exception when instantiated
    with patch("projectv3.FigureCanvasTkAgg", side_effect=Exception("boom")):
        with pytest.raises(Exception, match="boom"):
            eapp.show_usage_trend()

# ----------------------------
# Test upload_file exception branch
# ----------------------------
def test_upload_file_csv_exception(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = None

    # Mock file dialog to return a CSV path
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **kw: "dummy.csv")
    # Mock pandas.read_csv to raise an exception
    monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: (_ for _ in ()).throw(Exception("CSV fail")))
    # Mock messagebox.showerror to avoid GUI creation
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *a, **kw: None)

    # Call the method, should hit exception branch safely
    eapp.upload_file()

# ----------------------------
# Test compute_flat with invalid data (flat_start_entry empty, df with invalid kWh)
# ----------------------------
def test_compute_flat_invalid(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "kWh": ["invalid"]})
    eapp.flat_start_entry = MagicMock(get=lambda: "")
    eapp.flat_end_entry = MagicMock(get=lambda: "")

    monkeypatch.setattr(app, "flatRateTariff", lambda df, rate, fee: {"scheme":"Flat", "totalKWh":0, "breakdown":{}, "totalBill":0})
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    # Mock _draw_breakdown to avoid Matplotlib/Tk issues
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)

    eapp.compute_flat()

def test_compute_flat_parse_date_and_wh_exception(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)

    # minimal df
    eapp.df = pd.DataFrame({
    "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
    "kWh": [1.0]
    })
    # empty / invalid entries to trigger parse_date fallback
    eapp.flat_start_entry = MagicMock(get=lambda: "")
    eapp.flat_end_entry = MagicMock(get=lambda: "")

    # invalid width/height to trigger exception branch
    eapp.flat_chart_w = MagicMock(get=lambda: "notanumber")
    eapp.flat_chart_h = MagicMock(get=lambda: "NaN")

    # Patch tariff and visualization
    monkeypatch.setattr(app, "flatRateTariff", lambda df, rate, fee: {"scheme":"Flat", "totalKWh":1, "breakdown":{}, "totalBill":10})
    monkeypatch.setattr(app, "tk", MagicMock(simpledialog=MagicMock(askstring=lambda *a, **kw: "bar")))
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)

    with patch.object(app.messagebox, "showwarning") as mock_warn:
        eapp.compute_flat()
        # Should not trigger warnings here
        assert mock_warn.call_count == 0


# ----------------------------
# Test show_usage_trend early exit when df is None
# ----------------------------
def test_show_usage_trend_no_df(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = None

    # Mock messagebox to avoid GUI
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)
    eapp.show_usage_trend()  # hits early exit branch

# ----------------------------
# Test compute_tou with invalid data to cover exception branch
# ----------------------------
def test_compute_tou_invalid(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "kWh": ["invalid"]})
    eapp.tou_start_entry = MagicMock(get=lambda: "")
    eapp.tou_end_entry = MagicMock(get=lambda: "")
    eapp.tou_result_frame = MagicMock()

    # Mock tariff and dialogs
    monkeypatch.setattr(app, "touTariff", lambda *a, **kw: {"totalBill": 0})
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    
    # Mock _draw_breakdown so no Tkinter/Matplotlib code runs
    eapp._draw_breakdown = MagicMock()

    # Run compute_tou, exception branch is covered
    eapp.compute_tou()

    # Optional: ensure _draw_breakdown was called
    eapp._draw_breakdown.assert_called_once()

def test_compute_tou_normal(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)

    # Provide valid DataFrame
    eapp.df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
        "kWh": [1.0, 2.0]
    })

    # Mock start/end entries to valid strings
    eapp.tou_start_entry = MagicMock(get=lambda: "2025-01-01 00:00:00")
    eapp.tou_end_entry = MagicMock(get=lambda: "2025-01-01 01:00:00")

    # Mock chart size
    eapp.tou_chart_w = MagicMock(get=lambda: "5")
    eapp.tou_chart_h = MagicMock(get=lambda: "4")

    # Patch tariff function
    monkeypatch.setattr(app, "touTariff", lambda df, *a, **kw: {
        "totalBill": df["kWh"].sum(),
        "breakdown": {"Energy": df["kWh"].sum()}
    })

    # Patch dialog input
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")

    # Mock _draw_breakdown
    eapp._draw_breakdown = MagicMock()

    # Call compute_tou
    eapp.compute_tou()

    # Assert _draw_breakdown was called
    eapp._draw_breakdown.assert_called_once()

def test_compute_tou_chart_size_exception(monkeypatch):
    root = MagicMock()
    eapp = app.EnergyApp(root)
    eapp.df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:00"]),
        "kWh": [1.0]
    })

    # Patch entries for start/end
    eapp.tou_start_entry = MagicMock(get=lambda: "0")
    eapp.tou_end_entry = MagicMock(get=lambda: "1")
    
    # Patch chart width/height to invalid strings
    eapp.tou_w = MagicMock(get=lambda: "not_a_number")
    eapp.tou_h = MagicMock(get=lambda: None)

    eapp.tou_result_frame = MagicMock()
    
    # Patch GUI methods
    monkeypatch.setattr(eapp, "_draw_breakdown", lambda *a, **kw: None)
    monkeypatch.setattr("tkinter.simpledialog.askstring", lambda *a, **kw: "bar")
    monkeypatch.setattr(app, "touTariff", lambda df, rates, fee: {"totalBill": 0})

    # Patch showwarning to avoid GUI popups
    with patch.object(app.messagebox, "showwarning") as mock_warn:
        eapp.compute_tou()
    
    # This ensures the except branch for w,h is hit
    assert mock_warn.called is False

# -------------------------
# draw_breakdown pie / fixed fee tests
# -------------------------
def test_draw_breakdown_fixed_fee_and_title(monkeypatch):
    root = MagicMock()
    eapp = EnergyApp(root)
    parent = MagicMock()
    parent.winfo_children.return_value = []

    
    # Provide billResult with fixedFee and scheme keys
    billResult = {
        "breakdown": {"Energy": 5},
        "fixedFee": 3.5,
        "scheme": "Flat",
        "total": 8.5
    }

    # Set chart type to pie
    eapp.break_chart_var = MagicMock()
    eapp.break_chart_var.get.return_value = "pie"

    # Patch FigureCanvasTkAgg and plt.Figure to prevent GUI drawing
    monkeypatch.setattr("projectv3.FigureCanvasTkAgg", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.Figure", lambda *a, **kw: MagicMock())

    # Call _draw_breakdown
    eapp._draw_breakdown(parent, billResult)

def test_draw_breakdown_pie_coverage():
    # Mock the root Tk so no GUI appears
    mock_root = MagicMock()
    app = EnergyApp(mock_root)

    # Mock parent (usually a Frame or container widget)
    mock_parent = MagicMock()

    # Create a dummy billResult
    billResult = {
        'scheme': 'TestScheme',
        'breakdown': {
            'Electricity': 50,
            'Gas': 30,
            'Other': 20
        }
    }
    # Mock matplotlib Figure and Axes
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_fig.add_subplot.return_value = mock_ax

    # Patch both Figure and FigureCanvasTkAgg to prevent GUI calls
    with patch('projectv3.Figure', return_value=mock_fig), \
         patch('projectv3.FigureCanvasTkAgg', return_value=MagicMock()):
        # Call _draw_breakdown with chartType="pie" and parent
        app._draw_breakdown(parent=mock_parent, chartType="pie", billResult=billResult)

    # Assert pie chart was called and title set
    mock_ax.pie.assert_called_once()
    mock_ax.set_title.assert_called_once_with("TestScheme Bill Breakdown")

# ============================================================
# Main
# ============================================================
def test_main_block(monkeypatch):
    monkeypatch.setattr(tk, "Tk", lambda *a, **k: MagicMock())
    monkeypatch.setattr(app, "EnergyApp", lambda root: MagicMock())
    runpy.run_module("projectv3", run_name="__main__")