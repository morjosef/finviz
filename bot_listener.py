"""
Telegram Bot Listener — שינוי הגדרות סורק
רץ דרך GitHub Actions כל 5 דקות.
"""

import os
import json
import subprocess
import requests

BOT_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID      = os.environ["TELEGRAM_CHAT_ID"]
SETTINGS_FILE = "settings.json"
BASE_URL     = f"https://api.telegram.org/bot{BOT_TOKEN}"

RSI_OPTIONS  = [30, 40, 50, 60, 70]
SMA_OPTIONS  = ["above", "below", "off"]
SMA_LABEL    = {"above": "מעל", "below": "מתחת", "off": "כבוי"}


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"rsi_threshold": 50, "sma50": "above", "sma200": "above", "last_update_id": 0}


def save_settings(s: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def commit_and_push():
    subprocess.run(["git", "config", "user.email", "bot@github.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Screener Bot"], check=True)
    subprocess.run(["git", "add", SETTINGS_FILE], check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        subprocess.run(["git", "commit", "-m", "bot: update screener settings"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "HEAD"], check=True)
        print("✅ הגדרות נשמרו ונדחפו")
    else:
        print("ℹ️ אין שינויים לשמור")


def build_keyboard(s: dict) -> dict:
    rsi = s["rsi_threshold"]
    sma50 = s["sma50"]
    sma200 = s["sma200"]

    rsi_row = [
        {"text": f"{'✅ ' if v == rsi else ''}{v}", "callback_data": f"rsi_{v}"}
        for v in RSI_OPTIONS
    ]
    sma50_row = [
        {"text": f"{'✅ ' if v == sma50 else ''}SMA50 {SMA_LABEL[v]}", "callback_data": f"sma50_{v}"}
        for v in SMA_OPTIONS
    ]
    sma200_row = [
        {"text": f"{'✅ ' if v == sma200 else ''}SMA200 {SMA_LABEL[v]}", "callback_data": f"sma200_{v}"}
        for v in SMA_OPTIONS
    ]
    return {"inline_keyboard": [rsi_row, sma50_row, sma200_row]}


def settings_text(s: dict) -> str:
    return (
        f"⚙️ <b>הגדרות סורק</b>\n\n"
        f"📊 RSI מקסימום: <b>{s['rsi_threshold']}</b>\n"
        f"📈 SMA50: <b>{SMA_LABEL[s['sma50']]}</b>\n"
        f"📉 SMA200: <b>{SMA_LABEL[s['sma200']]}</b>"
    )


def send_menu(s: dict, chat_id: int):
    requests.post(f"{BASE_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": settings_text(s),
        "parse_mode": "HTML",
        "reply_markup": build_keyboard(s),
    }, timeout=30)


def edit_menu(s: dict, chat_id: int, message_id: int):
    requests.post(f"{BASE_URL}/editMessageText", json={
        "chat_id": chat_id,
        "message_id": message_id,
        "text": settings_text(s),
        "parse_mode": "HTML",
        "reply_markup": build_keyboard(s),
    }, timeout=30)


def answer_callback(callback_id: str):
    requests.post(f"{BASE_URL}/answerCallbackQuery", json={
        "callback_query_id": callback_id,
        "text": "✅ עודכן",
    }, timeout=10)


def get_updates(offset: int) -> list:
    resp = requests.get(f"{BASE_URL}/getUpdates", params={
        "offset": offset,
        "timeout": 0,
        "allowed_updates": ["message", "callback_query"],
    }, timeout=15)
    return resp.json().get("result", [])


def main():
    settings = load_settings()
    offset = settings.get("last_update_id", 0)
    if offset:
        offset += 1

    updates = get_updates(offset)
    changed = False

    for update in updates:
        uid = update["update_id"]
        settings["last_update_id"] = max(settings.get("last_update_id", 0), uid)

        if "message" in update:
            msg = update["message"]
            if msg.get("text", "").strip() in ("/settings", "/start"):
                send_menu(settings, msg["chat"]["id"])

        elif "callback_query" in update:
            cb   = update["callback_query"]
            data = cb["data"]
            chat_id    = cb["message"]["chat"]["id"]
            message_id = cb["message"]["message_id"]

            if data.startswith("rsi_"):
                settings["rsi_threshold"] = int(data.split("_")[1])
                changed = True
            elif data.startswith("sma50_"):
                settings["sma50"] = data.split("_", 1)[1]
                changed = True
            elif data.startswith("sma200_"):
                settings["sma200"] = data.split("_", 1)[1]
                changed = True

            answer_callback(cb["id"])
            edit_menu(settings, chat_id, message_id)

    save_settings(settings)
    if updates:
        commit_and_push()


if __name__ == "__main__":
    main()
