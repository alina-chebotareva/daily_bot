import io
import logging
import os
import sqlite3
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "daily_bot.sqlite3"

PROFILE_WEIGHT, PROFILE_HEIGHT, PROFILE_AGE, PROFILE_ACTIVITY, PROFILE_CITY = range(5)
FOOD_GRAMS = 10


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                weight REAL NOT NULL,
                height REAL NOT NULL,
                age INTEGER NOT NULL,
                activity INTEGER NOT NULL,
                city TEXT NOT NULL,
                water_goal INTEGER NOT NULL,
                calorie_goal INTEGER NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS water_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                ml INTEGER NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                product TEXT NOT NULL,
                calories REAL NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                workout_type TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                burned_kcal INTEGER NOT NULL
            )
            """
        )

        conn.commit()


def upsert_profile(user_id, weight, height, age, activity, city, water_goal, calorie_goal):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO profiles (user_id, weight, height, age, activity, city, water_goal, calorie_goal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                weight=excluded.weight,
                height=excluded.height,
                age=excluded.age,
                activity=excluded.activity,
                city=excluded.city,
                water_goal=excluded.water_goal,
                calorie_goal=excluded.calorie_goal
            """,
            (user_id, weight, height, age, activity, city, water_goal, calorie_goal),
        )
        conn.commit()


def get_profile(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT weight, height, age, activity, city, water_goal, calorie_goal
            FROM profiles
            WHERE user_id=?
            """,
            (user_id,),
        )
        return cur.fetchone()


def add_water_log(user_id, ml, day_str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO water_logs (user_id, day, ml) VALUES (?, ?, ?)",
            (user_id, day_str, ml),
        )
        conn.commit()


def get_water_sum(user_id, day_str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(SUM(ml), 0) FROM water_logs WHERE user_id=? AND day=?",
            (user_id, day_str),
        )
        (total,) = cur.fetchone()
        return int(total)


def add_food_log(user_id, day_str, product, calories):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO food_logs (user_id, day, product, calories) VALUES (?, ?, ?, ?)",
            (user_id, day_str, product, calories),
        )
        conn.commit()


def get_food_calories_sum(user_id, day_str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(SUM(calories), 0) FROM food_logs WHERE user_id=? AND day=?",
            (user_id, day_str),
        )
        (total,) = cur.fetchone()
        return float(total)


def add_workout_log(user_id, day_str, workout_type, minutes, burned_kcal):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO workout_logs (user_id, day, workout_type, minutes, burned_kcal)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, day_str, workout_type, minutes, burned_kcal),
        )
        conn.commit()


def get_workout_burned_sum(user_id, day_str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(SUM(burned_kcal), 0) FROM workout_logs WHERE user_id=? AND day=?",
            (user_id, day_str),
        )
        (total,) = cur.fetchone()
        return int(total)


def get_workout_minutes_sum(user_id, day_str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(SUM(minutes), 0) FROM workout_logs WHERE user_id=? AND day=?",
            (user_id, day_str),
        )
        (total,) = cur.fetchone()
        return int(total)


def get_current_temp_c(city):
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["main"]["temp"])
    except (requests.RequestException, KeyError, TypeError, ValueError):
        logger.exception("Weather request failed")
        return None


def get_food_kcal_100g(product_name):
    try:
        r = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={
                "search_terms": product_name,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 1,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        products = data.get("products") or []
        if not products:
            return None

        nutriments = products[0].get("nutriments") or {}
        kcal = nutriments.get("energy-kcal_100g")
        if kcal is None:
            return None

        return float(kcal)
    except (requests.RequestException, KeyError, TypeError, ValueError):
        logger.exception("OpenFoodFacts request failed")
        return None


def calc_water_goal_ml(weight_kg, activity_min, temp_c):
    base = weight_kg * 30
    activity_bonus = int((activity_min / 30) * 500)

    heat_bonus = 0
    if temp_c is not None:
        if temp_c > 30:
            heat_bonus = 1000
        elif temp_c > 25:
            heat_bonus = 750

    return int(base + activity_bonus + heat_bonus)


def calc_calorie_goal_kcal(weight_kg, height_cm, age, activity_min):
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age

    if activity_min >= 60:
        activity_bonus = 400
    elif activity_min >= 30:
        activity_bonus = 300
    elif activity_min > 0:
        activity_bonus = 200
    else:
        activity_bonus = 0

    return int(base + activity_bonus)


def to_float(text):
    try:
        return float(text.replace(",", ".").strip())
    except ValueError:
        return None


def to_int(text):
    try:
        return int(text.strip())
    except ValueError:
        return None


def days_list(days):
    days = max(1, min(days, 90))
    start_dt = date.today() - timedelta(days=days - 1)
    return [(start_dt + timedelta(days=i)).isoformat() for i in range(days)]


def daily_sum_int(table, col, user_id, start_day):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT day, COALESCE(SUM({col}), 0)
            FROM {table}
            WHERE user_id=? AND day>=?
            GROUP BY day
            """,
            (user_id, start_day),
        )
        return {d: int(v) for d, v in cur.fetchall()}


def daily_sum_float(table, col, user_id, start_day):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT day, COALESCE(SUM({col}), 0)
            FROM {table}
            WHERE user_id=? AND day>=?
            GROUP BY day
            """,
            (user_id, start_day),
        )
        return {d: float(v) for d, v in cur.fetchall()}


async def send_line_plot(update, title, y_label, labels, values, goals=None, caption=""):
    plt.figure(figsize=(9, 4))

    x = list(range(len(labels)))
    plt.plot(x, values, marker="o", linewidth=1)
    if goals is not None:
        plt.plot(x, goals, linewidth=1)

    plt.title(title)
    plt.ylabel(y_label)

    step = max(1, len(labels) // 7)
    ticks = list(range(0, len(labels), step))
    plt.xticks(ticks, [labels[i] for i in ticks], rotation=45, ha="right")

    plt.grid(True, linewidth=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    plt.close()
    buf.seek(0)

    await update.message.reply_photo(photo=buf, caption=caption or "")


async def start(update, _context):
    await update.message.reply_text(
        "Привет! Я бот для воды и калорий.\n"
        "Сначала настрой профиль: /set_profile\n"
        "Подсказка: /help"
    )


async def help_cmd(update, _context):
    await update.message.reply_text(
        "Команды:\n"
        "/start\n"
        "/help\n"
        "/set_profile\n"
        "/log_water <мл>\n"
        "/log_food <продукт>\n"
        "/log_workout <тип> <минуты>\n"
        "/check_progress\n"
        "/plot_progress [days]\n"
        "/cancel"
    )


async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "Упс, произошла ошибка. Попробуйте ещё раз или начните заново командой /cancel."
        )


async def log_water(update, context):
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text("Формат: /log_water <мл>\nНапример: /log_water 250")
        return

    ml = to_int(context.args[0])
    if ml is None or ml <= 0 or ml > 5000:
        await update.message.reply_text("Введи количество воды в мл (например 250).")
        return

    day_str = date.today().isoformat()
    add_water_log(user_id, ml, day_str)

    total = get_water_sum(user_id, day_str)
    await update.message.reply_text(f"Записал: {ml} мл.\nСегодня всего: {total} мл.")


async def log_workout(update, context):
    user_id = update.effective_user.id
    day_str = date.today().isoformat()

    if len(context.args) < 2:
        await update.message.reply_text(
            "Формат: /log_workout <тип> <минуты>\nНапример: /log_workout running 45"
        )
        return

    workout_type = " ".join(context.args[:-1]).strip()
    minutes = to_int(context.args[-1])

    if not workout_type:
        await update.message.reply_text("Тип тренировки не должен быть пустым.")
        return

    if minutes is None or minutes <= 0 or minutes > 1000:
        await update.message.reply_text("Минуты — целое число (например 45).")
        return

    burned = minutes * 10
    add_workout_log(user_id, day_str, workout_type, minutes, burned)

    await update.message.reply_text(
        f"Записал тренировку: {workout_type}, {minutes} мин.\nСожжено: {burned} ккал."
    )


async def check_progress(update, _context):
    user_id = update.effective_user.id
    day_str = date.today().isoformat()

    prof = get_profile(user_id)
    if not prof:
        await update.message.reply_text("Профиль не найден. Сначала сделай /set_profile")
        return

    water_goal = int(prof[5])
    calorie_goal = int(prof[6])

    water_total = get_water_sum(user_id, day_str)
    food_total = get_food_calories_sum(user_id, day_str)
    burned_total = get_workout_burned_sum(user_id, day_str)

    workout_minutes = get_workout_minutes_sum(user_id, day_str)
    extra_water = int((workout_minutes / 30) * 200)
    water_goal_adj = water_goal + extra_water

    net_cal = max(food_total - burned_total, 0.0)

    left_water = max(water_goal_adj - water_total, 0)
    left_kcal = max(calorie_goal - net_cal, 0)

    await update.message.reply_text(
        "Прогресс за сегодня:\n"
        f"Вода: {water_total}/{water_goal_adj} мл\n"
        f"Осталось воды: {left_water} мл\n"
        f"\nКалории (еда - тренировки): {net_cal:.0f}/{calorie_goal} ккал\n"
        f"Съедено: {food_total:.1f} ккал, сожжено: {burned_total} ккал\n"
        f"Осталось калорий: {left_kcal:.0f} ккал"
    )


async def plot_progress(update, context):
    user_id = update.effective_user.id

    prof = get_profile(user_id)
    if not prof:
        await update.message.reply_text("Профиль не найден. Сначала сделай /set_profile")
        return

    days = 14
    if context.args:
        d = to_int(context.args[0])
        if d is not None:
            days = d

    labels = days_list(days)
    start_day = labels[0]

    water_goal_base = int(prof[5])
    calorie_goal = int(prof[6])

    water_map = daily_sum_int("water_logs", "ml", user_id, start_day)
    food_map = daily_sum_float("food_logs", "calories", user_id, start_day)
    workout_min_map = daily_sum_int("workout_logs", "minutes", user_id, start_day)
    burned_map = daily_sum_int("workout_logs", "burned_kcal", user_id, start_day)

    water_values = []
    water_goals = []
    net_cal_values = []
    cal_goals = []

    for d in labels:
        w = water_map.get(d, 0)
        mins = workout_min_map.get(d, 0)
        extra_water = int((mins / 30) * 200)
        w_goal = water_goal_base + extra_water

        food = food_map.get(d, 0.0)
        burned = burned_map.get(d, 0)
        net = max(food - burned, 0.0)

        water_values.append(float(w))
        water_goals.append(float(w_goal))
        net_cal_values.append(float(net))
        cal_goals.append(float(calorie_goal))

    if sum(water_values) == 0 and sum(net_cal_values) == 0 and sum(burned_map.values()) == 0:
        await update.message.reply_text(
            "Пока нет данных за выбранный период. Заполни /log_water, /log_food, /log_workout."
        )
        return

    last_water = water_values[-1]
    last_water_goal = water_goals[-1]
    water_pct = (last_water / last_water_goal * 100) if last_water_goal > 0 else 0
    avg_water = sum(water_values) / len(water_values)

    last_cal = net_cal_values[-1]
    last_cal_goal = cal_goals[-1]
    cal_pct = (last_cal / last_cal_goal * 100) if last_cal_goal > 0 else 0
    avg_cal = sum(net_cal_values) / len(net_cal_values)

    await send_line_plot(
        update,
        title=f"Вода за последние {len(labels)} дней",
        y_label="мл",
        labels=labels,
        values=water_values,
        goals=water_goals,
        caption=(
            "Линия 1: выпито (мл). Линия 2: цель воды (с учётом тренировок).\n"
            f"Сегодня: {int(last_water)}/{int(last_water_goal)} мл ({water_pct:.0f}%). "
            f"Среднее за период: {avg_water:.0f} мл/день."
        ),
    )

    await send_line_plot(
        update,
        title=f"Калории за последние {len(labels)} дней",
        y_label="ккал",
        labels=labels,
        values=net_cal_values,
        goals=cal_goals,
        caption=(
            "Линия 1: (еда − тренировки), не ниже 0. Линия 2: цель калорий.\n"
            f"Сегодня: {last_cal:.0f}/{last_cal_goal:.0f} ккал ({cal_pct:.0f}%). "
            f"Среднее за период: {avg_cal:.0f} ккал/день."
        ),
    )


async def set_profile_start(update, _context):
    await update.message.reply_text("Введи вес в кг (например: 70)")
    return PROFILE_WEIGHT


async def set_weight(update, context):
    w = to_float(update.message.text)
    if w is None or w <= 0 or w > 500:
        await update.message.reply_text("Не похоже на вес. Введи число, например: 70")
        return PROFILE_WEIGHT

    context.user_data["weight"] = w
    await update.message.reply_text("Теперь рост в см (например: 175)")
    return PROFILE_HEIGHT


async def set_height(update, context):
    h = to_float(update.message.text)
    if h is None or h <= 0 or h > 300:
        await update.message.reply_text("Не похоже на рост. Введи число, например: 175")
        return PROFILE_HEIGHT

    context.user_data["height"] = h
    await update.message.reply_text("Теперь возраст (например: 25)")
    return PROFILE_AGE


async def set_age(update, context):
    age = to_int(update.message.text)
    if age is None or age <= 0 or age > 120:
        await update.message.reply_text("Не похоже на возраст. Введи целое число, например: 25")
        return PROFILE_AGE

    context.user_data["age"] = age
    await update.message.reply_text("Сколько минут активности в день? (например: 30)")
    return PROFILE_ACTIVITY


async def set_activity(update, context):
    act = to_int(update.message.text)
    if act is None or act < 0 or act > 1000:
        await update.message.reply_text("Не похоже на минуты. Введи целое число, например: 30")
        return PROFILE_ACTIVITY

    context.user_data["activity"] = act
    await update.message.reply_text("Город (например: Moscow)")
    return PROFILE_CITY


async def set_city_finish(update, context):
    city = update.message.text.strip()
    if not city:
        await update.message.reply_text("Город не должен быть пустым. Введи, например: Moscow")
        return PROFILE_CITY

    user_id = update.effective_user.id
    weight = float(context.user_data["weight"])
    height = float(context.user_data["height"])
    age = int(context.user_data["age"])
    activity = int(context.user_data["activity"])

    temp_c = get_current_temp_c(city)
    water_goal = calc_water_goal_ml(weight, activity, temp_c)
    calorie_goal = calc_calorie_goal_kcal(weight, height, age, activity)

    upsert_profile(user_id, weight, height, age, activity, city, water_goal, calorie_goal)

    if temp_c is None:
        temp_part = "Погоду не учитывал (ключ OpenWeatherMap пока не задан)."
    else:
        temp_part = f"Учёл погоду: сейчас примерно {temp_c:.1f}°C."

    await update.message.reply_text(
        "Профиль сохранён.\n"
        f"Вес: {weight} кг, Рост: {height} см, Возраст: {age}, Активность: {activity} мин/день, Город: {city}\n"
        f"Норма воды: {water_goal} мл/день\n"
        f"Норма калорий: {calorie_goal} ккал/день\n"
        f"{temp_part}"
    )

    context.user_data.clear()
    return ConversationHandler.END


async def log_food(update, context):
    if not context.args:
        await update.message.reply_text("Формат: /log_food <продукт>\nНапример: /log_food banana")
        return ConversationHandler.END

    product = " ".join(context.args).strip()
    kcal_100g = get_food_kcal_100g(product)
    if kcal_100g is None:
        await update.message.reply_text(
            "Не смог найти калорийность в OpenFoodFacts.\nПопробуй другое название (часто помогает английский вариант)."
        )
        return ConversationHandler.END

    context.user_data["food_product"] = product
    context.user_data["food_kcal_100g"] = float(kcal_100g)

    await update.message.reply_text(f"{product} — {kcal_100g:.0f} ккал на 100 г. Сколько грамм вы съели?")
    return FOOD_GRAMS


async def log_food_grams(update, context):
    user_id = update.effective_user.id
    day_str = date.today().isoformat()

    grams = to_float(update.message.text)
    if grams is None or grams <= 0 or grams > 5000:
        await update.message.reply_text("Введи граммы числом, например: 150")
        return FOOD_GRAMS

    product = context.user_data.get("food_product")
    kcal_100g = context.user_data.get("food_kcal_100g")

    if not product or kcal_100g is None:
        await update.message.reply_text("Давай начнём заново: /log_food <продукт>")
        return ConversationHandler.END

    calories = (float(kcal_100g) * float(grams)) / 100.0
    add_food_log(user_id, day_str, product, calories)

    total = get_food_calories_sum(user_id, day_str)

    context.user_data.pop("food_product", None)
    context.user_data.pop("food_kcal_100g", None)

    await update.message.reply_text(
        f"Записано: {calories:.1f} ккал.\nСегодня всего: {total:.1f} ккал."
    )
    return ConversationHandler.END


async def cancel(update, context):
    context.user_data.clear()
    await update.message.reply_text("Ок, отменил.")
    return ConversationHandler.END


def run_polling(app):
    app.run_polling(drop_pending_updates=True)


def run_webhook(app):
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse, Response
    from starlette.routing import Route
    from telegram import Update

    public_url = os.getenv("PUBLIC_URL", "").rstrip("/")
    port = int(os.getenv("PORT", "8000"))
    webhook_path = os.getenv("WEBHOOK_PATH", "telegram").strip("/")

    webhook_url = f"{public_url}/{webhook_path}"

    async def health(_: Request):
        return PlainTextResponse("ok")

    async def telegram_hook(request: Request):
        data = await request.json()
        await app.update_queue.put(Update.de_json(data=data, bot=app.bot))
        return Response()

    web_app = Starlette(
        routes=[
            Route("/", health, methods=["GET"]),
            Route(f"/{webhook_path}", telegram_hook, methods=["POST"]),
        ]
    )

    @web_app.on_event("startup")
    async def _startup():
        await app.initialize()
        await app.start()
        await app.bot.set_webhook(url=webhook_url, allowed_updates=Update.ALL_TYPES)

    @web_app.on_event("shutdown")
    async def _shutdown():
        await app.stop()
        await app.shutdown()

    uvicorn.run(web_app, host="0.0.0.0", port=port)


def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Не найден BOT_TOKEN. Проверь файл .env")

    init_db()

    app = Application.builder().token(token).build()
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("log_water", log_water))
    app.add_handler(CommandHandler("log_workout", log_workout))
    app.add_handler(CommandHandler("check_progress", check_progress))
    app.add_handler(CommandHandler("plot_progress", plot_progress))

    profile_conv = ConversationHandler(
        entry_points=[CommandHandler("set_profile", set_profile_start)],
        states={
            PROFILE_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_weight)],
            PROFILE_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_height)],
            PROFILE_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_age)],
            PROFILE_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_activity)],
            PROFILE_CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_city_finish)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(profile_conv)

    food_conv = ConversationHandler(
        entry_points=[CommandHandler("log_food", log_food)],
        states={
            FOOD_GRAMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, log_food_grams)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(food_conv)

    logger.info("Bot started")
    if os.getenv("PUBLIC_URL"):
        run_webhook(app)
    else:
        run_polling(app)


if __name__ == "__main__":
    main()
