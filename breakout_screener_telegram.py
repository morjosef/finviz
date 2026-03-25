"""
Breakout Stock Screener → Telegram
Only shows stocks that have ALREADY broken out upward in the last 1–5 days,
confirmed with elevated volume.
"""

import os
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from finviz.screener import Screener
from datetime import date

# ========== Config ==========
UNIVERSE_FILTERS = [
    "cap_midover",       # Cap > $2B
    "sh_avgvol_o1000",   # Avg Vol > 1M
    "ta_sma50_pa",       # Price above SMA50 (uptrend)
    "ta_sma200_pa",      # Price above SMA200 (major uptrend)
]

PERIOD = "6mo"
LOOKBACK_DAYS = 60       # days to search for prior resistance
BREAKOUT_WINDOW = 5      # breakout must occur within last N trading days
VOLUME_MULTIPLIER = 1.5  # volume spike minimum threshold
MIN_SCORE = 3            # minimum score (out of 6) to pass
MAX_TICKERS = 300        # universe cap from Finviz
MAX_CHARTS = 30          # max charts to send
PAGE_SIZE = 12           # charts per grid image
COLS = 3
# ============================

BOT_TOKEN = os.environ["BREAKOUT_BOT_TOKEN"].strip()
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"].strip()


# ─────────────────────────── Telegram ───────────────────────────

def send_message(text: str):
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=30,
    )


def send_photo(image_bytes: bytes, caption: str = ""):
    resp = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("chart.png", image_bytes, "image/png")},
        timeout=60,
    )
    if not resp.ok:
        print(f"  ⚠️  Telegram error: {resp.text}")


# ─────────────────────────── Breakout Logic ───────────────────────────

def find_resistance_levels(series_high: pd.Series, order: int = 5) -> list[float]:
    """Find local swing-high prices and merge nearby ones (within 1.5%)."""
    values = series_high.values
    from scipy.signal import argrelextrema
    idx = argrelextrema(values, np.greater, order=order)[0]
    levels = sorted(values[idx], reverse=True)

    merged: list[float] = []
    for lvl in levels:
        if not merged or abs(lvl - merged[-1]) / merged[-1] > 0.015:
            merged.append(lvl)
    return merged


def detect_breakout(df: pd.DataFrame) -> dict | None:
    """
    Return breakout info if the stock broke above a resistance level
    within the last BREAKOUT_WINDOW days on elevated volume.
    Returns None if no confirmed breakout found.
    """
    if len(df) < 30:
        return None

    avg_vol = df["Volume"].rolling(20).mean()

    # Use price history BEFORE the breakout window to find resistance
    lookback_end = len(df) - BREAKOUT_WINDOW
    if lookback_end < 20:
        return None

    history_high = df["High"].iloc[max(0, lookback_end - LOOKBACK_DAYS):lookback_end]
    resistances  = find_resistance_levels(history_high)
    if not resistances:
        return None

    best: dict | None = None

    for i in range(BREAKOUT_WINDOW):
        day_idx = len(df) - BREAKOUT_WINDOW + i
        row         = df.iloc[day_idx]
        prev_close  = df["Close"].iloc[day_idx - 1]
        vol_avg     = avg_vol.iloc[day_idx - 1]
        if vol_avg <= 0 or np.isnan(vol_avg):
            continue
        volume_ratio = row["Volume"] / vol_avg

        for level in resistances:
            # Confirmed breakout: previous close was below, this close is clearly above
            if prev_close < level * 1.002 and row["Close"] > level * 1.005:
                if volume_ratio >= VOLUME_MULTIPLIER:
                    days_ago = BREAKOUT_WINDOW - i - 1  # 0 = today
                    if best is None or volume_ratio > best["volume_ratio"]:
                        best = {
                            "level":          float(level),
                            "breakout_date":  df.index[day_idx],
                            "days_ago":       days_ago,
                            "volume_ratio":   float(volume_ratio),
                            "breakout_close": float(row["Close"]),
                            "current_close":  float(df["Close"].iloc[-1]),
                        }

    # Reject false breakouts: price fell back below the level
    if best and df["Close"].iloc[-1] < best["level"] * 0.99:
        return None

    return best


def score_breakout(df: pd.DataFrame, breakout: dict) -> tuple[int, list[str]]:
    """
    Score the breakout quality (0–6).
    Returns (score, [reason strings]).
    """
    score   = 0
    reasons: list[str] = []

    close   = df["Close"]
    volume  = df["Volume"]
    current = float(close.iloc[-1])

    ma20  = close.rolling(20).mean().iloc[-1]
    ma150 = close.rolling(150).mean().iloc[-1]

    # 1. Volume strength (1 or 2 points)
    vr = breakout["volume_ratio"]
    if vr >= 3.0:
        score += 2
        reasons.append(f"🔥 ווליום חזק מאוד ({vr:.1f}x)")
    elif vr >= 1.5:
        score += 1
        reasons.append(f"📈 ווליום מאשר ({vr:.1f}x)")

    # 2. Price above MA20
    if not pd.isna(ma20) and current > ma20:
        score += 1
        reasons.append("✅ מעל MA20")

    # 3. Price above MA150
    if not pd.isna(ma150) and current > ma150:
        score += 1
        reasons.append("✅ מעל MA150")

    # 4. Fresh breakout (≤ 2 days ago)
    if breakout["days_ago"] <= 2:
        score += 1
        reasons.append(f"⚡ פריצה טריה ({breakout['days_ago']} ימים)")

    # 5. Price still near breakout level (0.5%–6% above — not over-extended)
    extension = (current - breakout["level"]) / breakout["level"] * 100
    if 0.5 <= extension <= 6.0:
        score += 1
        reasons.append(f"📍 {extension:.1f}% מעל הרמה — לא מרוחק")

    return score, reasons


# ─────────────────────────── Chart ───────────────────────────

def make_style():
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit", volume="inherit",
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridcolor="#2a2e39",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            "axes.labelcolor": "#d1d4dc",
            "xtick.color":     "#787b86",
            "ytick.color":     "#787b86",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlecolor": "#d1d4dc",
            "axes.titlesize":  9,
        },
    )


def render_chart(ticker: str, breakout: dict, score: int, style) -> bytes | None:
    """Render candlestick chart with the broken resistance line marked."""
    try:
        data = yf.download(ticker, period=PERIOD, interval="1d",
                           progress=False, auto_adjust=True)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.index = pd.DatetimeIndex(data.index)

        close = data["Close"]
        plots = []

        if close.rolling(20).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(20).mean(),
                                          color="#2962ff", width=1.0))
        if close.rolling(150).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(150).mean(),
                                          color="#ff6d00", width=1.0))

        resistance_line = pd.Series(breakout["level"], index=data.index)
        plots.append(mpf.make_addplot(resistance_line,
                                      color="#ffcc00", width=1.2,
                                      linestyle="--", alpha=0.8))

        label = (f"\n{ticker}  {score}/6  "
                 f"Vol {breakout['volume_ratio']:.1f}x  "
                 f"{'Today' if breakout['days_ago']==0 else str(breakout['days_ago'])+'d ago'}")

        fig, _ = mpf.plot(data, type="candle", style=style, volume=True,
                          title=label, figsize=(10, 6),
                          tight_layout=True, returnfig=True, addplot=plots)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#131722")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


def send_document(file_bytes: bytes, filename: str, caption: str = ""):
    resp = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
        data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"document": (filename, file_bytes, "text/html")},
        timeout=60,
    )
    if not resp.ok:
        print(f"  ⚠️  Telegram doc error: {resp.text}")


def send_stock(r: dict, style):
    """Send a single stock as: text message + PNG chart."""
    ticker  = r["ticker"]
    score   = r["score"]
    bo      = r["breakout"]
    reasons = r["reasons"]
    url     = f"https://www.tradingview.com/chart/?symbol={ticker}"

    days_str = "היום" if bo["days_ago"] == 0 else f"{bo['days_ago']} ימים"

    text = (
        f'<a href="{url}">{ticker}</a> — ${bo["current_close"]:.2f}\n'
        f'ציון מיכה {score}/6\n\n'
        f'פריצה: {days_str} | Vol {bo["volume_ratio"]:.1f}x\n'
        f'רמת פריצה: ${bo["level"]:.2f}\n\n'
    )
    for reason in reasons:
        text += f'{reason}\n'
    text += f'\n<a href="{url}">פתח ב-TradingView</a>'

    send_message(text)

    img = render_chart(ticker, bo, score, style)
    if img:
        send_photo(img, caption=ticker)


# ─────────────────────────── Summary message ───────────────────────────

def send_summary(results: list[dict], today: str):
    lines = [
        f"🚀 <b>Breakout Screener — {today}</b>\n"
        f"✅ <b>{len(results)}</b> מניות פרצו כלפי מעלה\n"
        f"Cap&gt;2B · Vol&gt;1M · SMA50 · SMA200 · ווליום פריצה ≥1.5x\n"
        f"🔵 MA20  |  🟠 MA150  |  🟡 רמת פריצה\n"
    ]
    full_text = "\n".join(lines)
    send_message(full_text)


# ─────────────────────────── Main ───────────────────────────

def main():
    today = date.today().strftime("%d/%m/%Y")
    print(f"📅 {today} — מריץ Breakout Screener...")

    # 1. Finviz universe
    screener = Screener(filters=UNIVERSE_FILTERS, table="Technical", order="ticker")
    all_rows = list(screener)
    tickers  = [r["Ticker"] for r in all_rows[:MAX_TICKERS]]
    print(f"✅ {len(tickers)} מניות ביקום | בודק פריצות...")

    if not tickers:
        send_message(f"🚀 <b>Breakout Screener — {today}</b>\n\nלא נמצאו מניות ביקום.")
        return

    # 2. Breakout detection
    results: list[dict] = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=PERIOD, interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

            breakout = detect_breakout(df)
            if breakout is None:
                continue

            score, reasons = score_breakout(df, breakout)
            if score < MIN_SCORE:
                continue

            results.append({
                "ticker":   ticker,
                "score":    score,
                "breakout": breakout,
                "reasons":  reasons,
                "df":       df,
            })
            print(f"  💥 {ticker}  ציון {score}/6  Vol {breakout['volume_ratio']:.1f}x")

        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    print(f"\n✅ {len(results)} מניות עם פריצה מאושרת")

    if not results:
        send_message(f"🚀 <b>Breakout Screener — {today}</b>\n\n"
                     "לא נמצאו פריצות מאושרות היום.")
        return

    # Sort by score descending, then volume ratio
    results.sort(key=lambda x: (x["score"], x["breakout"]["volume_ratio"]), reverse=True)

    # 3. Send summary header
    send_summary(results, today)

    # 4. Send each stock individually
    style = make_style()
    for r in results[:MAX_CHARTS]:
        print(f"  📤 {r['ticker']}...")
        try:
            send_stock(r, style)
        except Exception as e:
            print(f"  ❌ {r['ticker']}: {e}")

    print("✅ הושלם!")


if __name__ == "__main__":
    main()
