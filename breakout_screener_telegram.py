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

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID   = os.environ["BREAKOUT_CHAT_ID"].strip()


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


def render_chart(ticker: str, breakout: dict, score: int, style) -> tuple[str, bytes] | None:
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

        close  = data["Close"]
        plots  = []

        if close.rolling(20).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(20).mean(),
                                          color="#2962ff", width=1.0))
        if close.rolling(150).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(150).mean(),
                                          color="#ff6d00", width=1.0))

        # Resistance line (horizontal)
        resistance_line = pd.Series(breakout["level"], index=data.index)
        plots.append(mpf.make_addplot(resistance_line,
                                      color="#ffcc00", width=1.2,
                                      linestyle="--", alpha=0.8))

        label = (f"\n{ticker}  ⭐{score}/6  "
                 f"Vol {breakout['volume_ratio']:.1f}x  "
                 f"{'Today' if breakout['days_ago']==0 else str(breakout['days_ago'])+'d ago'}")

        kwargs = dict(
            type="candle", style=style, volume=True,
            title=label, figsize=(5, 3.2),
            tight_layout=True, returnfig=True,
            addplot=plots,
        )

        fig, _ = mpf.plot(data, **kwargs)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90,
                    bbox_inches="tight", facecolor="#131722")
        plt.close(fig)
        buf.seek(0)
        return ticker, buf.read()

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


def build_grid(batch: list[tuple[str, bytes]]) -> bytes:
    n    = len(batch)
    rows = max(1, (n + COLS - 1) // COLS)

    fig, axes = plt.subplots(rows, COLS,
                             figsize=(COLS * 5.5, rows * 3.5),
                             facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    if rows == 1 and COLS == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif COLS == 1:
        axes = [[ax] for ax in axes]

    for i, (_, img_bytes) in enumerate(batch):
        r, c = i // COLS, i % COLS
        img_arr = plt.imread(io.BytesIO(img_bytes))
        axes[r][c].imshow(img_arr)
        axes[r][c].axis("off")

    for i in range(n, rows * COLS):
        r, c = i // COLS, i % COLS
        axes[r][c].set_visible(False)

    plt.subplots_adjust(hspace=0.06, wspace=0.04)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────── Summary message ───────────────────────────

def send_summary(results: list[dict], today: str):
    lines = [
        f"🚀 <b>Breakout Screener — {today}</b>\n"
        f"✅ <b>{len(results)}</b> מניות פרצו כלפי מעלה\n"
        f"פילטרים: Cap&gt;2B · Vol&gt;1M · SMA50↑ · SMA200↑ · ווליום פריצה ≥1.5x\n"
        f"🟡 קו צהוב = רמת התנגדות שנפרצה  |  🔵 MA20  |  🟠 MA150\n\n"
        f"📋 <b>מניות (ממוינות לפי ציון):</b>\n"
    ]

    for r in results:
        ticker = r["ticker"]
        score  = r["score"]
        bo     = r["breakout"]
        url    = f"https://www.tradingview.com/chart/?symbol={ticker}"
        stars  = "⭐" * score
        lines.append(
            f'• <a href="{url}">{ticker}</a>  '
            f'${bo["current_close"]:.2f}  '
            f'ציון {score}/6 {stars}  '
            f'Vol {bo["volume_ratio"]:.1f}x  '
            f'{"היום" if bo["days_ago"]==0 else str(bo["days_ago"])+" ימים"}'
        )
        if r["reasons"]:
            lines.append(f'  └ {" · ".join(r["reasons"][:3])}')

    full_text = "\n".join(lines)
    for i in range(0, len(full_text), 4000):
        send_message(full_text[i:i + 4000])


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

    # 3. Send summary message
    send_summary(results, today)

    # 4. Render and send charts
    style  = make_style()
    charts = []

    for r in results[:MAX_CHARTS]:
        print(f"  📊 {r['ticker']}...")
        result = render_chart(r["ticker"], r["breakout"], r["score"], style)
        if result:
            charts.append(result)

    pages = [charts[i:i + PAGE_SIZE] for i in range(0, len(charts), PAGE_SIZE)]
    for idx, page in enumerate(pages, 1):
        print(f"  📤 שולח עמוד {idx}/{len(pages)}...")
        grid_bytes = build_grid(page)
        caption    = f"פריצות — עמוד {idx}/{len(pages)}  |  {', '.join(t for t, _ in page)}"
        send_photo(grid_bytes, caption)

    print("✅ הושלם!")


if __name__ == "__main__":
    main()
