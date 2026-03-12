"""
Daily Stock Screener → Telegram
Runs automatically via GitHub Actions every weekday.
"""

import os
import io
import json
import requests
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from finviz.screener import Screener
from datetime import date

# ========== טעינת הגדרות ==========
def load_settings() -> dict:
    try:
        with open("settings.json") as f:
            return json.load(f)
    except Exception:
        return {"rsi_threshold": 50, "sma50": "above", "sma200": "above"}

def build_filters(s: dict) -> list:
    filters = ["cap_midover", "sh_avgvol_o1000"]
    rsi = s.get("rsi_threshold", 50)
    filters.append(f"ta_rsi_u{rsi}")
    sma50 = s.get("sma50", "above")
    if sma50 == "above":
        filters.append("ta_sma50_pa")
    elif sma50 == "below":
        filters.append("ta_sma50_pb")
    sma200 = s.get("sma200", "above")
    if sma200 == "above":
        filters.append("ta_sma200_pa")
    elif sma200 == "below":
        filters.append("ta_sma200_pb")
    return filters

SETTINGS   = load_settings()
FILTERS    = build_filters(SETTINGS)
PERIOD     = "6mo"
MAX_CHARTS = 100
PAGE_SIZE  = 12
COLS       = 3
# ==============================

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]


# ---------- Telegram helpers ----------

def send_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text,
                              "parse_mode": "HTML"}, timeout=30)


def send_photo(image_bytes: bytes, caption: str = ""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    resp = requests.post(
        url,
        data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", image_bytes, "image/png")},
        timeout=60,
    )
    if not resp.ok:
        print(f"  ⚠️  Telegram error: {resp.text}")


# ---------- Chart helpers ----------

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
            "xtick.color": "#787b86",
            "ytick.color": "#787b86",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.titlecolor": "#d1d4dc",
            "axes.titlesize": 9,
        },
    )


def render_chart(ticker: str, style) -> tuple[str, bytes] | None:
    """Returns (ticker, image_bytes) or None on failure."""
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
        if close.rolling(50).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(50).mean(),
                                          color="#2962ff", width=1.0))
        if close.rolling(200).mean().notna().any():
            plots.append(mpf.make_addplot(close.rolling(200).mean(),
                                          color="#ff6d00", width=1.0))

        kwargs = dict(type="candle", style=style, volume=True,
                      title=f"\n{ticker}", figsize=(5, 3.2),
                      tight_layout=True, returnfig=True)
        if plots:
            kwargs["addplot"] = plots

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
    """Combine multiple chart images into one grid image."""
    n    = len(batch)
    rows = max(1, (n + COLS - 1) // COLS)

    fig, axes = plt.subplots(rows, COLS,
                             figsize=(COLS * 5.5, rows * 3.5),
                             facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # נרמול axes
    if rows == 1 and COLS == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif COLS == 1:
        axes = [[ax] for ax in axes]

    for i, (ticker, img_bytes) in enumerate(batch):
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


# ---------- Caption builder ----------

def build_caption(page: list[tuple[str, bytes]], rows_map: dict,
                  idx: int, total_pages: int, total: int, shown: int, today: str) -> str:
    """Build an HTML caption with TradingView links for each ticker in the page."""
    ticker_links = []
    for ticker, _ in page:
        url = f"https://www.tradingview.com/chart/?symbol={ticker}"
        row = rows_map.get(ticker, {})
        price = row.get("Price", "—")
        rsi   = row.get("RSI", "—")
        ticker_links.append(f'<a href="{url}">{ticker}</a> ${price} RSI:{rsi}')

    header = ""
    if idx == 1:
        header = (f"📊 <b>Stock Screener — {today}</b>\n"
                  f"✅ {total} מניות | מוצגות {shown} | 🔵SMA50 🟠SMA200\n\n")

    page_line = f"עמוד {idx}/{total_pages}\n"
    return header + page_line + " · ".join(ticker_links)


# ---------- Main ----------

def main():
    today = date.today().strftime("%d/%m/%Y")
    print(f"📅 {today} — מריץ סקרינר...")

    # 1. Finviz screener (table=Technical for RSI + price data)
    screener     = Screener(filters=FILTERS, table="Technical", order="ticker")
    all_rows     = list(screener)
    total        = len(all_rows)
    rows         = all_rows[:MAX_CHARTS]
    tickers      = [r["Ticker"] for r in rows]

    print(f"✅ נמצאו {total} מניות | מציג {len(tickers)}")

    if not tickers:
        send_message(f"📊 <b>Stock Screener — {today}</b>\n\nלא נמצאו מניות לפי הפילטרים.")
        return

    rows_map = {r["Ticker"]: r for r in rows}

    # 2. Render charts
    style = make_style()
    charts = []
    for ticker in tickers:
        print(f"  📊 {ticker}...")
        result = render_chart(ticker, style)
        if result:
            charts.append(result)

    # 3. Send grid pages with linked captions
    pages = [charts[i:i + PAGE_SIZE] for i in range(0, len(charts), PAGE_SIZE)]
    for idx, page in enumerate(pages, 1):
        print(f"  📤 שולח עמוד {idx}/{len(pages)}...")
        grid_bytes = build_grid(page)
        caption    = build_caption(page, rows_map, idx, len(pages),
                                   total, len(tickers), today)
        send_photo(grid_bytes, caption)

    print("✅ הושלם!")


if __name__ == "__main__":
    main()
