"""
Обработчики событий Telegram бота
Реализует логику обработки сообщений, команд и кнопок
"""
from datetime import datetime
from typing import Optional
import structlog

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes

from ..config import config, PersonaType
from ..models import Message, MessageRole, ChatState
from ..database import db_manager
from ..services.ai_service import ai_service, AIServiceError
from ..services.vector_service import vector_service

logger = structlog.get_logger()


class TelegramHandlers:
    """Класс с обработчиками событий Telegram бота"""
    
    @staticmethod
    def get_persona_keyboard() -> ReplyKeyboardMarkup:
        """Создает клавиатуру с кнопками выбора персоны в один ряд"""
        buttons = config.get_persona_buttons()
        # Все кнопки в одном ряду
        keyboard = [[KeyboardButton(button) for button in buttons]]
        
        return ReplyKeyboardMarkup(
            keyboard,
            resize_keyboard=True,
            is_persistent=False,  # Клавиатуру можно скрыть
            one_time_keyboard=True  # Скрывается после нажатия
        )
    
    @staticmethod
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /start
        Очищает историю диалога и показывает приветствие через AI
        """
        if not update.effective_chat or not update.effective_user:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        try:
            # Очищаем историю диалога
            deleted_count = db_manager.clear_chat_history(chat_id)
            
            # Получаем текущее состояние чата (создается новое если не существует)
            chat_state = db_manager.get_chat_state(chat_id)
            
            # Генерируем приветственное сообщение через AI с новой архитектурой
            ai_response = await ai_service.generate_response(
                chat_id=chat_id,
                current_message="",  # Пустое сообщение для приветствия
                persona=chat_state.current_persona
            )
            
            # Сохраняем ответ AI в базу данных
            ai_message = Message(
                id=None,
                chat_id=chat_id,
                user_id=None,  # Сообщение от бота
                message_id=None,  # Будет установлен после отправки
                role=MessageRole.ASSISTANT,
                content=ai_response,
                persona_type=chat_state.current_persona,
                created_at=datetime.now()
            )
            
            db_manager.save_message(ai_message)
            
            # Отправляем сообщение с клавиатурой
            keyboard = TelegramHandlers.get_persona_keyboard()
            sent_message = await update.message.reply_text(
                ai_response,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            # Обновляем message_id в базе данных
            ai_message.message_id = sent_message.message_id
            
            logger.info("Start command processed",
                       chat_id=chat_id,
                       user_id=user_id,
                       deleted_messages=deleted_count,
                       persona=chat_state.current_persona.value)
            
        except AIServiceError as e:
            logger.error("AI service error in start command", 
                        chat_id=chat_id, error=str(e))
            await TelegramHandlers._send_error_message(update, context)
        except Exception as e:
            logger.error("Unexpected error in start command", 
                        chat_id=chat_id, error=str(e))
            await TelegramHandlers._send_error_message(update, context)
    
    @staticmethod
    async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик текстовых сообщений
        Определяет тип сообщения (выбор персоны или обычный вопрос) и обрабатывает соответственно
        """
        if not update.effective_chat or not update.effective_user or not update.message:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        message_text = update.message.text
        message_id = update.message.message_id
        
        try:
            # Проверяем, является ли сообщение выбором персоны
            persona_buttons = config.get_persona_buttons()
            if message_text in persona_buttons:
                await TelegramHandlers._handle_persona_selection(
                    update, context, chat_id, user_id, message_text
                )
            else:
                await TelegramHandlers._handle_user_question(
                    update, context, chat_id, user_id, message_text, message_id
                )
                
        except Exception as e:
            logger.error("Error in message handler", 
                        chat_id=chat_id, error=str(e))
            await TelegramHandlers._send_error_message(update, context)
    
    @staticmethod
    async def _handle_persona_selection(
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        user_id: int,
        button_text: str
    ) -> None:
        """Обрабатывает выбор персоны пользователем"""
        try:
            # Определяем новую персону
            new_persona = config.get_persona_type(button_text)
            
            # Обновляем персону в базе данных
            db_manager.update_chat_persona(chat_id, new_persona)
            
            # Простое подтверждение смены режима (без вызова DeepSeek API)
            confirmation_messages = {
                PersonaType.BUSINESS: "✅ Переключен на деловой режим. Готов предоставить профессиональную техническую поддержку.",
                PersonaType.BELLA: "✅ Привет! Теперь с тобой общается Белла 😊 Готова помочь с любыми вопросами!",
                PersonaType.BEN: "✅ Йо! Бэн на связи 👋 Давай решать твои вопросы вместе!"
            }
            
            confirmation_text = confirmation_messages.get(new_persona, confirmation_messages[PersonaType.BUSINESS])
            
            # Отправляем подтверждение
            keyboard = TelegramHandlers.get_persona_keyboard()
            await update.message.reply_text(confirmation_text, reply_markup=keyboard, parse_mode='Markdown')
            
            logger.info("Persona changed",
                       chat_id=chat_id,
                       user_id=user_id,
                       new_persona=new_persona.value)
            
        except Exception as e:
            logger.error("Error in persona selection", 
                        chat_id=chat_id, error=str(e))
            await TelegramHandlers._send_error_message(update, context)
    
    @staticmethod
    async def _handle_user_question(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE, 
        chat_id: int,
        user_id: int,
        message_text: str,
        message_id: int
    ) -> None:
        """Обрабатывает обычный вопрос пользователя"""
        try:
            # Получаем состояние чата
            chat_state = db_manager.get_chat_state(chat_id)
            
            # Сохраняем сообщение пользователя
            user_message = Message(
                id=None,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                role=MessageRole.USER,
                content=message_text,
                persona_type=chat_state.current_persona,
                created_at=datetime.now()
            )
            
            db_manager.save_message(user_message)
            
            # Генерируем ответ через AI с новой архитектурой
            ai_response = await ai_service.generate_response(
                chat_id=chat_id,
                current_message=message_text,
                persona=chat_state.current_persona
            )
            
            # Сохраняем ответ AI
            ai_message = Message(
                id=None,
                chat_id=chat_id,
                user_id=None,
                message_id=None,
                role=MessageRole.ASSISTANT,
                content=ai_response,
                persona_type=chat_state.current_persona,
                created_at=datetime.now()
            )
            
            db_manager.save_message(ai_message)
            
            # Отправляем ответ
            keyboard = TelegramHandlers.get_persona_keyboard()
            await update.message.reply_text(ai_response, reply_markup=keyboard, parse_mode='Markdown')
            
            logger.info("User question processed",
                       chat_id=chat_id,
                       user_id=user_id,
                       persona=chat_state.current_persona.value,
                       question_length=len(message_text))
            
        except AIServiceError as e:
            logger.error("AI service error in question handling", 
                        chat_id=chat_id, error=str(e))
            await TelegramHandlers._send_error_message(update, context)
    

    
    @staticmethod
    async def _send_error_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Отправляет сообщение об ошибке пользователю"""
        try:
            keyboard = TelegramHandlers.get_persona_keyboard()
            await update.message.reply_text(
                config.ERROR_MESSAGE,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error("Failed to send error message", error=str(e))
    
    @staticmethod
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Глобальный обработчик ошибок"""
        logger.error("Telegram bot error", error=str(context.error))
        
        # Если есть update с сообщением, отправляем сообщение об ошибке
        if isinstance(update, Update) and update.effective_message:
            try:
                keyboard = TelegramHandlers.get_persona_keyboard()
                await update.effective_message.reply_text(
                    config.ERROR_MESSAGE,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error("Failed to send error message in error handler", error=str(e))
