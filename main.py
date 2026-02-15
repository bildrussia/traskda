"""
Telegram-бот с интеграцией ИИ через OpenRouter
Использует aiogram 3.x для работы с Telegram API
"""

import asyncio
import os
from collections import defaultdict, deque
from typing import Dict, Deque

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()

# Инициализация бота и диспетчера
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в .env файле")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализация клиента OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не найден в .env файле")

# URL для OpenRouter API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Инициализируем клиент OpenAI для работы с OpenRouter
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

# Модель по умолчанию (можно изменить в .env)
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-pro-1.5")

# Хранилище истории сообщений для каждого пользователя
# Формат: {user_id: deque([{"role": "user", "content": "..."}, ...], maxlen=10)}
user_messages: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=10))


def format_messages_for_api(user_id: int, new_message: str) -> list:
    """
    Форматирует историю сообщений для отправки в API.
    Сохраняет последние 10 сообщений (5 пар вопрос-ответ).
    
    Args:
        user_id: ID пользователя Telegram
        new_message: Новое сообщение от пользователя
    
    Returns:
        Список сообщений в формате для OpenAI API
    """
    # Добавляем новое сообщение пользователя в историю
    user_messages[user_id].append({"role": "user", "content": new_message})
    
    # Формируем список сообщений для API
    messages = list(user_messages[user_id])
    
    # Добавляем системное сообщение в начало (если нужно)
    # Можно настроить поведение бота через системное сообщение
    system_message = {
        "role": "system",
        "content": "Ты полезный ассистент. Отвечай на русском языке."
    }
    
    return [system_message] + messages


async def get_ai_response(user_id: int, user_message: str) -> str:
    """
    Получает ответ от ИИ через OpenRouter API.
    
    Args:
        user_id: ID пользователя Telegram
        user_message: Сообщение от пользователя
    
    Returns:
        Ответ от ИИ
    """
    try:
        # Форматируем сообщения для API
        messages = format_messages_for_api(user_id, user_message)
        
        # Отправляем запрос в OpenRouter
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.7,
        )
        
        # Извлекаем ответ
        ai_response = response.choices[0].message.content
        
        # Сохраняем ответ ассистента в историю
        user_messages[user_id].append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    except Exception as e:
        error_message = f"Ошибка при обращении к ИИ: {str(e)}"
        print(error_message)  # Логируем ошибку
        return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже."


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    Приветствует пользователя и очищает историю сообщений
    """
    user_id = message.from_user.id
    
    # Очищаем историю сообщений при старте
    if user_id in user_messages:
        user_messages[user_id].clear()
    
    welcome_text = (
        "👋 Привет! Я бот с интеграцией ИИ.\n\n"
        "Просто отправь мне текстовое сообщение, и я отвечу с помощью искусственного интеллекта.\n\n"
        "Используй /start для начала нового диалога (очистки истории)."
    )
    
    await message.answer(welcome_text)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """
    Обработчик команды /help
    Показывает справку по использованию бота
    """
    help_text = (
        "📖 Справка по использованию бота:\n\n"
        "/start - Начать новый диалог (очистить историю)\n"
        "/help - Показать эту справку\n\n"
        "Просто отправь текстовое сообщение, и я отвечу с помощью ИИ.\n"
        "Бот помнит последние 10 сообщений в нашем диалоге."
    )
    
    await message.answer(help_text)


@dp.message()
async def handle_text_message(message: Message):
    """
    Обработчик всех текстовых сообщений от пользователей.
    Отправляет сообщение в ИИ и возвращает ответ.
    """
    # Игнорируем сообщения без текста
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return
    
    user_id = message.from_user.id
    user_message = message.text
    
    # Отправляем индикатор "печатает..."
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # Получаем ответ от ИИ
    ai_response = await get_ai_response(user_id, user_message)
    
    # Отправляем ответ пользователю
    await message.answer(ai_response)


async def main():
    """
    Главная функция для запуска бота
    """
    print("🤖 Бот запускается...")
    print(f"📝 Используется модель: {DEFAULT_MODEL}")
    
    # Запускаем бота
    try:
        await dp.start_polling(bot)
    except Exception as e:
        print(f"❌ Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    # Запускаем бота
    asyncio.run(main())
