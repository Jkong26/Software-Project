"""
XPower - Household Tariff Analysis (Prototype)
Run: python this_file.py
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------
# Helper functions
# ---------------------------
def safe_float(text, default=0.0):
    try:
        return float(str(text).strip())
    except Exception:
        return default

def safe_int(text, default=None):
    t = str(text).strip()
    if t == "":
        return default
    try:
        return int(t)
    except Exception:
        return default

def parse_date(text):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d")
    except Exception:
        return None

# ---------------------------
# Tariff calculation routines
# ---------------------------

# Function to test 1
def flatRateTariff(df, flatRate, fixedFee): #returns a dictionary 
    """Return breakdown dict and total for flat rate"""
    totalKWh = df["kWh"].sum() #sums the kWh column
    energyCost = totalKWh * flatRate #calculates the energy cost
    breakdown = {"Energy": float(energyCost)} #creates a breakdown dictionary
    return {"scheme": "Flat", "totalBill": float(energyCost + fixedFee), "breakdown": breakdown, "fixedFee": float(fixedFee), "totalKWh": float(totalKWh)}

#Function to test 2
def touTariff(df, touRates, fixedFee): #returns a dictionary
    # Ensure timestamp column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    breakdown_costs = {} # dictionary to hold costs per period
    for _, row in df.iterrows(): # iterate over rows
        timestamp = row["timestamp"].time() 
        kWh = float(row["kWh"])
        applied = False
        for period, info in touRates.items():
            if info.get("default"):
                continue
            s = info["start"]
            e = info["end"]
            # normal interval
            if s < e:
                if s <= timestamp < e: # check if timestamp falls within the period
                    breakdown_costs[period] = breakdown_costs.get(period, 0.0) + kWh * info["rate"] #calculate cost
                    applied = True
                    break
            else:
                # wraps midnight
                if timestamp >= s or timestamp < e:
                    breakdown_costs[period] = breakdown_costs.get(period, 0.0) + kWh * info["rate"]
                    applied = True
                    break
        if not applied:
            # assign to default (shoulder)
            for period, info in touRates.items():
                if info.get("default"):
                    breakdown_costs[period] = breakdown_costs.get(period, 0.0) + kWh * info["rate"]
    # add fixed fee as a separate entry
    breakdown_costs["Fixed Fee"] = float(fixedFee)
    total = sum(breakdown_costs.values())
    return {"scheme": "TOU", "totalBill": float(total), "breakdown": {k: float(v) for k, v in breakdown_costs.items()}, "fixedFee": float(fixedFee)}

# Function to test 3
def tieredTariff(df, tier_limits, tier_rates, fixed_fee):
    total_usage = float(df["kWh"].sum())  # Total consumption
    remaining_usage = total_usage         # Usage left to assign to tiers
    breakdown = {}                        # Store costs per tier
    prev_limit = 0.0                      # Track lower bound of tier

    for i, limit in enumerate(tier_limits):
        rate = float(tier_rates[i])

        # If tier has no limit -> all remaining usage is in this tier
        if not limit:  
            used = remaining_usage
        else:
            cap = float(limit) - prev_limit  # Maximum this tier can take
            used = min(remaining_usage, cap)

        # Cost for this tier
        breakdown[f"Tier {i+1}"] = used * rate

        # Update trackers
        remaining_usage -= used
        prev_limit = float(limit) if limit else prev_limit + used

        # Stop if no usage left
        if remaining_usage <= 0:
            break

    # If still usage left (beyond defined tiers), apply last rate
    if remaining_usage > 0 and len(tier_rates) > len(tier_limits):
        breakdown[f"Tier {len(tier_limits)+1}"] = remaining_usage * float(tier_rates[-1])
        remaining_usage = 0

    # Add fixed fee
    breakdown["Fixed Fee"] = float(fixed_fee)

    # Calculate total bill
    total_bill = sum(breakdown.values())

    return {
        "scheme": "Tiered",
        "totalBill": total_bill,
        "breakdown": breakdown,
        "fixedFee": float(fixed_fee),
        "totalKWh": total_usage
    }
# ---------------------------
# Data filtering for duration
# ---------------------------

def filter_by_duration(df, start_dt, end_dt):
    """Return rows with timestamp between start_dt (00:00:00) and end_dt (23:59:59) inclusive."""
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_ts = datetime.combine(start_dt.date(), dt_time.min) if isinstance(start_dt, datetime) else start_dt
    end_ts = datetime.combine(end_dt.date(), dt_time.max) if isinstance(end_dt, datetime) else end_dt
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)

# ---------------------------
# GUI helpers: scrollable frame
# ---------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, height=400, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        # Allow mousewheel scrolling on inner frame
        self.inner.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.inner.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _bind_to_mousewheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_from_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")

# ---------------------------
# Main application (Tkinter)
# ---------------------------
class EnergyApp:
    def __init__(self, root):
        self.root = root
        root.title("XPower - Household Tariff Analysis (Prototype)")
        root.geometry("1000x700")

        self.df = None  # uploaded DataFrame

        # Notebook tabs
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        # Create tabs inside scrollable frames
        self.tab_upload = ScrollableFrame(self.nb, height=200)
        self.tab_flat = ScrollableFrame(self.nb, height=500)
        self.tab_tou = ScrollableFrame(self.nb, height=600)
        self.tab_tier = ScrollableFrame(self.nb, height=600)
        self.tab_compare = ScrollableFrame(self.nb, height=700)
        self.tab_visual = ScrollableFrame(self.nb, height=700)
        self.tab_exit = ScrollableFrame(self.nb, height=200)

        self.nb.add(self.tab_upload, text="Upload Data")
        self.nb.add(self.tab_flat, text="Flat Rate")
        self.nb.add(self.tab_tou, text="Time-of-Use (TOU)")
        self.nb.add(self.tab_tier, text="Tiered (Block)")
        self.nb.add(self.tab_compare, text="Bill Calc & Compare")
        self.nb.add(self.tab_visual, text="Visualization & Reporting")
        self.nb.add(self.tab_exit, text="Exit")

        self.build_upload_tab()
        self.build_flat_tab()
        self.build_tou_tab()
        self.build_tier_tab()
        self.build_compare_tab()
        self.build_visual_tab()
        self.build_exit_tab()

    # ---------------
    # Upload tab
    # ---------------
    def build_upload_tab(self):
        f = self.tab_upload.inner
        ttk.Label(f, text="Upload CSV or Excel with columns: timestamp (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS), kWh", font=("Arial", 11)).pack(pady=8)
        brow = ttk.Button(f, text="Browse and Upload File (CSV or XLSX)", command=self.upload_file)
        brow.pack(pady=6)

        # Provide small example & quick load button for demo data (if user wants)
        ttk.Label(f, text="(Demo) Load sample monthly hourly dataset included in app:").pack(pady=(10,0))
        demo_btn = ttk.Button(f, text="Load small demo dataset", command=self.load_demo_data)
        demo_btn.pack(pady=4)

        self.upload_status = ttk.Label(f, text="No file loaded", foreground="blue")
        self.upload_status.pack(pady=8)

    def upload_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            # Basic validation
            if "timestamp" not in df.columns or "kWh" not in df.columns:
                messagebox.showerror("Invalid file", "File must contain 'timestamp' and 'kWh' columns.")
                return
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].isna().any():
                # Some parsing failed — show warning but keep rows with parsed timestamps
                df = df.dropna(subset=["timestamp"])
            df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce").fillna(0.0)
            self.df = df.reset_index(drop=True)
            self.upload_status.config(text=f"Loaded: {path}   Rows: {len(self.df)}")
            messagebox.showinfo("Upload", f"File loaded ({len(self.df)} rows).")
        except Exception as e:
            messagebox.showerror("Upload error", str(e))

    def load_demo_data(self):
        # small hourly demo-built in (hardcoded few rows) to help users try app quickly
        demo = [
            {"timestamp": "2025-01-01 00:00:00", "kWh": 0.25},
            {"timestamp": "2025-01-01 01:00:00", "kWh": 0.42},
            {"timestamp": "2025-01-01 02:00:00", "kWh": 0.48},
            {"timestamp": "2025-01-01 03:00:00", "kWh": 0.40},
            {"timestamp": "2025-01-01 04:00:00", "kWh": 0.21},
            {"timestamp": "2025-01-01 05:00:00", "kWh": 0.30},
            {"timestamp": "2025-01-01 18:00:00", "kWh": 1.5},
            {"timestamp": "2025-01-01 19:00:00", "kWh": 2.0},
            {"timestamp": "2025-01-02 20:00:00", "kWh": 1.1},
        ]
        df = pd.DataFrame(demo)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["kWh"] = df["kWh"].astype(float)
        self.df = df
        self.upload_status.config(text="Loaded demo dataset (9 rows)")
        messagebox.showinfo("Demo data", "Demo dataset loaded (9 rows).")

    # ---------------
    # Flat tab
    # ---------------
    def build_flat_tab(self):
        f = self.tab_flat.inner
        ttk.Label(f, text="Flat Rate Tariff", font=("Arial", 12, "bold")).pack(pady=6)
        row = ttk.Frame(f); row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Flat rate ($/kWh):").pack(side="left")
        self.flat_rate_entry = ttk.Entry(row, width=10); self.flat_rate_entry.pack(side="left", padx=6)
        self.flat_rate_entry.insert(0, "0.25")

        ttk.Label(row, text="Fixed fee ($):").pack(side="left", padx=(20,0))
        self.flat_fee_entry = ttk.Entry(row, width=10); self.flat_fee_entry.pack(side="left", padx=6)
        self.flat_fee_entry.insert(0, "10")

        # Duration selection
        dur_row = ttk.Frame(f); dur_row.pack(fill="x", padx=6, pady=6)
        ttk.Label(dur_row, text="Start date (YYYY-MM-DD):").pack(side="left")
        self.flat_start_entry = ttk.Entry(dur_row, width=12); self.flat_start_entry.pack(side="left", padx=6)
        ttk.Label(dur_row, text="End date (YYYY-MM-DD):").pack(side="left", padx=(10,0))
        self.flat_end_entry = ttk.Entry(dur_row, width=12); self.flat_end_entry.pack(side="left", padx=6)

        # Chart size control
        size_row = ttk.Frame(f); size_row.pack(fill="x", padx=6, pady=4)
        ttk.Label(size_row, text="Chart width:").pack(side="left")
        self.flat_chart_w = ttk.Entry(size_row, width=6); self.flat_chart_w.pack(side="left", padx=4); self.flat_chart_w.insert(0,"6")
        ttk.Label(size_row, text="height:").pack(side="left")
        self.flat_chart_h = ttk.Entry(size_row, width=6); self.flat_chart_h.pack(side="left", padx=4); self.flat_chart_h.insert(0,"4")

        btn = ttk.Button(f, text="Compute & Visualize (bar/pie)", command=self.compute_flat)
        btn.pack(pady=8)

        self.flat_result_frame = ttk.Frame(f)
        self.flat_result_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def compute_flat(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload a dataset first (Upload tab).")
            return
        start = parse_date(self.flat_start_entry.get()) or self.df["timestamp"].min()
        end = parse_date(self.flat_end_entry.get()) or self.df["timestamp"].max()
        df_sel = filter_by_duration(self.df, start, end)
        if df_sel.empty:
            messagebox.showwarning("No data in range", "No data between selected dates.")
            return
        rate = safe_float(self.flat_rate_entry.get(), 0.25)
        fee = safe_float(self.flat_fee_entry.get(), 10.0)
        result = flatRateTariff(df_sel, rate, fee)
        # prepare breakdown dict format expected by visualize function
        # breakdown in returned result already contains "Energy"; ensure Fixed Fee present for visualization
        result["breakdown"]["Fixed Fee"] = float(result.get("fixedFee", fee))
        # clear result frame
        for w in self.flat_result_frame.winfo_children(): w.destroy()
        # Chart type prompt
        choice = tk.simpledialog.askstring("Chart type", "Bar or Pie? (enter 'bar' or 'pie')", initialvalue="bar")
        ctype = (choice or "bar").strip().lower()
        try:
            w = max(4, int(self.flat_chart_w.get()))
            h = max(3, int(self.flat_chart_h.get()))
        except Exception:
            w, h = 6, 4
        self._draw_breakdown(self.flat_result_frame, result, chartType=ctype, figsize=(w,h))
        # show summary text
        ttk.Label(self.flat_result_frame, text=f"Total kWh: {result.get('totalKWh',0):.2f} — Total bill: ${result['totalBill']:.2f}", foreground="green").pack()

    # ---------------
    # TOU tab
    # ---------------
    def build_tou_tab(self):
        f = self.tab_tou.inner
        ttk.Label(f, text="Time-of-Use Tariff (hours 0-23)", font=("Arial", 12, "bold")).pack(pady=6)
        # Peak
        peak_row = ttk.Frame(f); peak_row.pack(fill="x", padx=6, pady=3)
        ttk.Label(peak_row, text="Peak start hour:").pack(side="left")
        self.peak_start = ttk.Entry(peak_row, width=6); self.peak_start.pack(side="left", padx=4); self.peak_start.insert(0,"18")
        ttk.Label(peak_row, text="end hour:").pack(side="left", padx=(8,0))
        self.peak_end = ttk.Entry(peak_row, width=6); self.peak_end.pack(side="left", padx=4); self.peak_end.insert(0,"22")
        ttk.Label(peak_row, text="rate ($/kWh):").pack(side="left", padx=(8,0))
        self.peak_rate = ttk.Entry(peak_row, width=8); self.peak_rate.pack(side="left", padx=4); self.peak_rate.insert(0,"0.40")

        # Off-peak
        off_row = ttk.Frame(f); off_row.pack(fill="x", padx=6, pady=3)
        ttk.Label(off_row, text="Off-Peak start hour:").pack(side="left")
        self.off_start = ttk.Entry(off_row, width=6); self.off_start.pack(side="left", padx=4); self.off_start.insert(0,"22")
        ttk.Label(off_row, text="end hour:").pack(side="left", padx=(8,0))
        self.off_end = ttk.Entry(off_row, width=6); self.off_end.pack(side="left", padx=4); self.off_end.insert(0,"7")
        ttk.Label(off_row, text="rate ($/kWh):").pack(side="left", padx=(8,0))
        self.off_rate = ttk.Entry(off_row, width=8); self.off_rate.pack(side="left", padx=4); self.off_rate.insert(0,"0.15")

        # Shoulder + fixed
        sh_row = ttk.Frame(f); sh_row.pack(fill="x", padx=6, pady=3)
        ttk.Label(sh_row, text="Shoulder rate ($/kWh):").pack(side="left")
        self.shoulder_rate = ttk.Entry(sh_row, width=8); self.shoulder_rate.pack(side="left", padx=4); self.shoulder_rate.insert(0,"0.25")
        ttk.Label(sh_row, text="Fixed fee ($):").pack(side="left", padx=(12,0))
        self.tou_fixed = ttk.Entry(sh_row, width=8); self.tou_fixed.pack(side="left", padx=4); self.tou_fixed.insert(0,"10")

        # duration selection
        dur_row = ttk.Frame(f); dur_row.pack(fill="x", padx=6, pady=6)
        ttk.Label(dur_row, text="Start date (YYYY-MM-DD):").pack(side="left")
        self.tou_start_entry = ttk.Entry(dur_row, width=12); self.tou_start_entry.pack(side="left", padx=6)
        ttk.Label(dur_row, text="End date (YYYY-MM-DD):").pack(side="left", padx=(10,0))
        self.tou_end_entry = ttk.Entry(dur_row, width=12); self.tou_end_entry.pack(side="left", padx=6)

        # chart size
        size_row = ttk.Frame(f); size_row.pack(fill="x", padx=6, pady=4)
        ttk.Label(size_row, text="Chart width:"); ttk.Label(size_row).pack_forget()
        self.tou_w = ttk.Entry(size_row, width=6); self.tou_w.pack(side="left", padx=4); self.tou_w.insert(0,"6")
        ttk.Label(size_row, text="height:").pack(side="left")
        self.tou_h = ttk.Entry(size_row, width=6); self.tou_h.pack(side="left", padx=4); self.tou_h.insert(0,"4")

        ttk.Button(f, text="Compute & Visualize (bar/pie)", command=self.compute_tou).pack(pady=6)
        self.tou_result_frame = ttk.Frame(f); self.tou_result_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def compute_tou(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload a dataset first (Upload tab).")
            return
        start = parse_date(self.tou_start_entry.get()) or self.df["timestamp"].min()
        end = parse_date(self.tou_end_entry.get()) or self.df["timestamp"].max()
        df_sel = filter_by_duration(self.df, start, end)
        if df_sel.empty:
            messagebox.showwarning("No data in range", "No data between selected dates.")
            return
        # parse rates and hours
        ps = safe_int(self.peak_start.get(), 18); pe = safe_int(self.peak_end.get(), 22)
        os = safe_int(self.off_start.get(), 22); oe = safe_int(self.off_end.get(), 7)
        pr = safe_float(self.peak_rate.get(), 0.40); orr = safe_float(self.off_rate.get(), 0.15); sr = safe_float(self.shoulder_rate.get(), 0.25)
        fee = safe_float(self.tou_fixed.get(), 10.0)
        # build touRates with time objects
        def h_to_time(h):
            if h is None: return dt_time(0,0)
            try:
                return dt_time(int(h)%24, 0)
            except Exception:
                return dt_time(0,0)
        touRates = {
            "Peak": {"start": h_to_time(ps), "end": h_to_time(pe), "rate": pr},
            "Off-Peak": {"start": h_to_time(os), "end": h_to_time(oe), "rate": orr},
            "Shoulder": {"default": True, "rate": sr}
        }
        result = touTariff(df_sel, touRates, fee)
        # clear frame
        for w in self.tou_result_frame.winfo_children(): w.destroy()
        # ask chart type
        choice = tk.simpledialog.askstring("Chart type", "Bar or Pie? (enter 'bar' or 'pie')", initialvalue="bar")
        ctype = (choice or "bar").strip().lower()
        try:
            w = max(4, int(self.tou_w.get()))
            h = max(3, int(self.tou_h.get()))
        except Exception:
            w, h = 6, 4
        self._draw_breakdown(self.tou_result_frame, result, chartType=ctype, figsize=(w,h))
        ttk.Label(self.tou_result_frame, text=f"Total bill: ${result['totalBill']:.2f}", foreground="green").pack()

    # ---------------
    # Tier tab
    # ---------------
    def build_tier_tab(self):
        f = self.tab_tier.inner
        ttk.Label(f, text="Tiered Tariff (Block rates)", font=("Arial", 12, "bold")).pack(pady=6)
        self.tier_limit_entries = []
        self.tier_rate_entries = []
        defaults_limits = ["100", "300", ""]  # "" = unlimited
        defaults_rates = ["0.20", "0.30", "0.40"]
        for i in range(3):
            row = ttk.Frame(f); row.pack(fill="x", padx=6, pady=4)
            ttk.Label(row, text=f"Tier {i+1} upper limit (kWh, blank = unlimited):").pack(side="left")
            e_lim = ttk.Entry(row, width=8); e_lim.pack(side="left", padx=4); e_lim.insert(0, defaults_limits[i])
            ttk.Label(row, text="rate ($/kWh):").pack(side="left", padx=(8,0))
            e_rate = ttk.Entry(row, width=8); e_rate.pack(side="left", padx=4); e_rate.insert(0, defaults_rates[i])
            self.tier_limit_entries.append(e_lim); self.tier_rate_entries.append(e_rate)
        fee_row = ttk.Frame(f); fee_row.pack(fill="x", padx=6, pady=6)
        ttk.Label(fee_row, text="Fixed fee ($):").pack(side="left")
        self.tier_fixed_entry = ttk.Entry(fee_row, width=8); self.tier_fixed_entry.pack(side="left", padx=4); self.tier_fixed_entry.insert(0,"10")
        # duration
        dur_row = ttk.Frame(f); dur_row.pack(fill="x", padx=6, pady=6)
        ttk.Label(dur_row, text="Start date (YYYY-MM-DD):").pack(side="left")
        self.tier_start_entry = ttk.Entry(dur_row, width=12); self.tier_start_entry.pack(side="left", padx=6)
        ttk.Label(dur_row, text="End date (YYYY-MM-DD):").pack(side="left", padx=(10,0))
        self.tier_end_entry = ttk.Entry(dur_row, width=12); self.tier_end_entry.pack(side="left", padx=6)

        ttk.Button(f, text="Compute & Visualize (bar/pie)", command=self.compute_tier).pack(pady=8)
        self.tier_result_frame = ttk.Frame(f); self.tier_result_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def compute_tier(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload a dataset first (Upload tab).")
            return
        start = parse_date(self.tier_start_entry.get()) or self.df["timestamp"].min()
        end = parse_date(self.tier_end_entry.get()) or self.df["timestamp"].max()
        df_sel = filter_by_duration(self.df, start, end)
        if df_sel.empty:
            messagebox.showwarning("No data in range", "No data between selected dates.")
            return
        tier_limits = [ (e.get().strip() if e.get().strip() != "" else "") for e in self.tier_limit_entries ]
        tier_rates = [ safe_float(e.get(), 0.2) for e in self.tier_rate_entries ]
        fee = safe_float(self.tier_fixed_entry.get(), 10.0)
        result = tieredTariff(df_sel, tier_limits, tier_rates, fee)
        # clear frame
        for w in self.tier_result_frame.winfo_children(): w.destroy()
        choice = tk.simpledialog.askstring("Chart type", "Bar or Pie? (enter 'bar' or 'pie')", initialvalue="bar")
        ctype = (choice or "bar").strip().lower()
        self._draw_breakdown(self.tier_result_frame, result, chartType=ctype, figsize=(6,4))
        ttk.Label(self.tier_result_frame, text=f"Total kWh: {result.get('totalKWh',0):.2f} — Total bill: ${result['totalBill']:.2f}", foreground="green").pack()

    # ---------------
    # Compare tab (all tariffs together) - GIVEN DURATION
    # ---------------
    def build_compare_tab(self):
        f = self.tab_compare.inner
        ttk.Label(f, text="Bill Calculation & Comparison", font=("Arial", 12, "bold")).pack(pady=6)
        dr = ttk.Frame(f); dr.pack(pady=4)
        ttk.Label(dr, text="Start date (YYYY-MM-DD):").pack(side="left")
        self.comp_start = ttk.Entry(dr, width=12); self.comp_start.pack(side="left", padx=6)
        ttk.Label(dr, text="End date (YYYY-MM-DD):").pack(side="left", padx=(10,0))
        self.comp_end = ttk.Entry(dr, width=12); self.comp_end.pack(side="left", padx=6)
        ttk.Button(f, text="Calculate & Compare All Tariffs", command=self.calculate_and_compare).pack(pady=8)
        self.compare_frame_inner = ttk.Frame(f); self.compare_frame_inner.pack(fill="both", expand=True, padx=6, pady=6)

    def calculate_and_compare(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload a dataset first (Upload tab).")
            return
        start = parse_date(self.comp_start.get()) or self.df["timestamp"].min()
        end = parse_date(self.comp_end.get()) or self.df["timestamp"].max()
        df_sel = filter_by_duration(self.df, start, end)
        if df_sel.empty:
            messagebox.showwarning("No data in range", "No data between selected dates.")
            return
        # Build tariffs using values from other tabs (they have defaults)
        # Flat
        flat_rate = safe_float(self.flat_rate_entry.get(), 0.25)
        flat_fee = safe_float(self.flat_fee_entry.get(), 10.0)
        flat_res = flatRateTariff(df_sel, flat_rate, flat_fee)
        flat_res["breakdown"]["Fixed Fee"] = float(flat_res.get("fixedFee", flat_fee))
        # TOU
        ps = safe_int(self.peak_start.get(), 18); pe = safe_int(self.peak_end.get(), 22)
        os = safe_int(self.off_start.get(), 22); oe = safe_int(self.off_end.get(), 7)
        pr = safe_float(self.peak_rate.get(), 0.40); orr = safe_float(self.off_rate.get(), 0.15); sr = safe_float(self.shoulder_rate.get(), 0.25)
        fee_tou = safe_float(self.tou_fixed.get(), 10.0)
        def h_to_time(h):
            try:
                return dt_time(int(h)%24, 0)
            except Exception:
                return dt_time(0,0)
        touRates = {
            "Peak": {"start": h_to_time(ps), "end": h_to_time(pe), "rate": pr},
            "Off-Peak": {"start": h_to_time(os), "end": h_to_time(oe), "rate": orr},
            "Shoulder": {"default": True, "rate": sr}
        }
        tou_res = touTariff(df_sel, touRates, fee_tou)
        # Tiered
        tier_limits = [ (e.get().strip() if e.get().strip() != "" else "") for e in self.tier_limit_entries ]
        tier_rates = [ safe_float(e.get(), 0.2) for e in self.tier_rate_entries ]
        fee_tier = safe_float(self.tier_fixed_entry.get(), 10.0)
        tier_res = tieredTariff(df_sel, tier_limits, tier_rates, fee_tier)
        # Aggregate
        bills = {"Flat": flat_res, "TOU": tou_res, "Tiered": tier_res}
        # Find cheapest
        cheapest = min(bills.keys(), key=lambda k: bills[k]["totalBill"])
        # Clear compare frame and plot summary bar
        for w in self.compare_frame_inner.winfo_children(): w.destroy()
        # summary label
        summary_text = "\n".join([f"{k}: ${v['totalBill']:.2f}{'  <-- cheapest' if k==cheapest else ''}" for k, v in bills.items()])
        ttk.Label(self.compare_frame_inner, text=summary_text, foreground="green").pack(pady=6)
        # plot totals
        fig = Figure(figsize=(6,4), dpi=100)
        ax = fig.add_subplot(111)
        names = list(bills.keys())
        vals = [bills[n]["totalBill"] for n in names]
        bars = ax.bar(names, vals)
        ax.set_title("Total Bill Comparison")
        ax.set_ylabel("Total ($)")
        # highlight cheapest
        for i, name in enumerate(names):
            if name == cheapest:
                bars[i].set_edgecolor("green")
                bars[i].set_linewidth(3)
        canvas = FigureCanvasTkAgg(fig, master=self.compare_frame_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        # Provide controls to visualize breakdown of a selected plan
        ctrl_frame = ttk.Frame(self.compare_frame_inner); ctrl_frame.pack(pady=8)
        ttk.Label(ctrl_frame, text="Select plan to view breakdown:").pack(side="left")
        self.break_plan_var = tk.StringVar(value="Flat")
        plan_opt = ttk.Combobox(ctrl_frame, textvariable=self.break_plan_var, values=list(bills.keys()), state="readonly", width=10)
        plan_opt.pack(side="left", padx=6)
        ttk.Label(ctrl_frame, text="Chart (bar/pie):").pack(side="left", padx=(8,0))
        self.break_chart_var = tk.StringVar(value="bar")
        chart_opt = ttk.Combobox(ctrl_frame, textvariable=self.break_chart_var, values=["bar","pie"], state="readonly", width=6)
        chart_opt.pack(side="left", padx=6)
        ttk.Button(ctrl_frame, text="Show breakdown", command=lambda: self._show_selected_breakdown(bills)).pack(side="left", padx=8)

    def _show_selected_breakdown(self, bills):
        plan = self.break_plan_var.get()
        ctype = self.break_chart_var.get()
        if plan not in bills:
            messagebox.showwarning("No plan", "Select a valid plan.")
            return
        # clear area below controls (pack everything below controls into new frame)
        bf = ttk.Frame(self.compare_frame_inner); bf.pack(fill="both", expand=True, pady=6)
        self._draw_breakdown(bf, bills[plan], chartType=ctype, figsize=(6,4))

    # ---------------
    # Visualization & Reporting tab
    # ---------------
    def build_visual_tab(self):
        f = self.tab_visual.inner
        ttk.Label(f, text="Visualization & Reporting", font=("Arial", 12, "bold")).pack(pady=6)
        # duration selection for usage trend
        dur = ttk.Frame(f); dur.pack(pady=6)
        ttk.Label(dur, text="Start date (YYYY-MM-DD):").pack(side="left")
        self.vis_start = ttk.Entry(dur, width=12); self.vis_start.pack(side="left", padx=6)
        ttk.Label(dur, text="End date (YYYY-MM-DD):").pack(side="left", padx=(8,0))
        self.vis_end = ttk.Entry(dur, width=12); self.vis_end.pack(side="left", padx=6)
        ttk.Button(f, text="Show Usage Trend (line)", command=self.show_usage_trend).pack(pady=8)
        self.vis_frame = ttk.Frame(f); self.vis_frame.pack(fill="both", expand=True, padx=6, pady=6)

    #function to test 4
    def show_usage_trend(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload a dataset first (Upload tab).")
            return
        start = parse_date(self.vis_start.get()) or self.df["timestamp"].min() #use dates or default to full range
        end = parse_date(self.vis_end.get()) or self.df["timestamp"].max()
        df_sel = filter_by_duration(self.df, start, end)
        if df_sel.empty:
            messagebox.showwarning("No data in range", "No data between selected dates.")
            return
        # Decide grouping: if original data has hourly timestamps, show hourly trend; if daily, group by day
        df_temp = df_sel.copy()
        if any(df_temp["timestamp"].dt.hour != 0):  # presence of hours suggests hourly data
            # show hourly series aggregated by timestamp (if many points downsample? Keep as is)
            series = df_temp.set_index("timestamp")["kWh"].resample("h").sum().fillna(0)
        else:
            series = df_temp.set_index("timestamp")["kWh"].resample("D").sum().fillna(0)
        # clear frame
        for w in self.vis_frame.winfo_children(): w.destroy()
        fig = Figure(figsize=(8,4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(series.index, series.values, marker="o")
        ax.set_title("Electricity Usage Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("kWh")
        fig.autofmt_xdate()
        canvas = FigureCanvasTkAgg(fig, master=self.vis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------
    # Exit tab
    # ---------------
    def build_exit_tab(self):
        f = self.tab_exit.inner
        ttk.Label(f, text="Exit application safely", font=("Arial", 12)).pack(pady=8)
        ttk.Button(f, text="Exit", command=self.root.quit).pack(pady=8)

    # ---------------
    # Small drawing helper for breakdown charts
    # ---------------
    def _draw_breakdown(self, parent, billResult, chartType="bar", figsize=(6,4)):
        # Ensure breakdown is a dict and Fixed Fee present
        breakdown = billResult.get("breakdown", {})
        if "Fixed Fee" not in breakdown and "fixedFee" in billResult:
            breakdown["Fixed Fee"] = float(billResult.get("fixedFee", 0.0))
        # clear parent
        for w in parent.winfo_children(): w.destroy()
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        if chartType == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{billResult.get('scheme','Tariff')} Bill Breakdown")
        else:
            ax.bar(labels, values)
            ax.set_ylabel("Cost ($)")
            ax.set_title(f"{billResult.get('scheme','Tariff')} Bill Breakdown")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyApp(root)
    root.mainloop()
