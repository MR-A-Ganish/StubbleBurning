# alerts.py

import pywhatkit
from plyer import notification
import datetime
import time

# 🔒 YOUR NUMBER
PHONE_NUMBER = "+917200298871"

# ⏱️ COOLDOWN SETTINGS (10 minutes)
last_sent_time = 0
COOLDOWN = 600


# --------------------------------------------------
# 📱 WHATSAPP ALERT
# --------------------------------------------------
def send_whatsapp_alert(message):

    try:
        now = datetime.datetime.now()

        hour = now.hour
        minute = now.minute + 2  # must be 1–2 mins ahead

        pywhatkit.sendwhatmsg(
            PHONE_NUMBER,
            message,
            hour,
            minute
        )

        return True

    except Exception as e:
        print("WhatsApp Error:", e)
        return False


# --------------------------------------------------
# 💻 DESKTOP ALERT
# --------------------------------------------------
def send_desktop_alert(message):

    try:
        notification.notify(
            title="🔥 Fire Alert",
            message=message,
            timeout=10
        )
    except Exception as e:
        print("Desktop Alert Error:", e)


# --------------------------------------------------
# 🚨 MAIN ALERT FUNCTION
# --------------------------------------------------
def trigger_alert(top_data):

    global last_sent_time

    current_time = time.time()

    # ⛔ Prevent spam
    if current_time - last_sent_time < COOLDOWN:
        return "⏳ Cooldown active (wait before next alert)"

    message = "🔥 FIRE ALERT 🚨\n\n"

    # SAFELY HANDLE DATA
    for _, row in top_data.iterrows():

        district = row.get("district", "Unknown")
        state = row.get("state", "Unknown")
        fires = row.get("fires", 0)

        message += f"{district} ({state}) - {fires} fires\n"

    # SEND ALERTS
    send_desktop_alert(message)
    send_whatsapp_alert(message)

    last_sent_time = current_time

    return message