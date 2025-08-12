"""
Сервис для работы с системными промтами
Загружает промты из файлов и подставляет динамические значения
"""
import os
from typing import Optional, Dict
import structlog

from ..config import config, PersonaType

logger = structlog.get_logger()


class PromptService:
    """Сервис для загрузки и обработки системных промтов"""
    
    def __init__(self):
        self.prompts_dir = config.PROMPTS_DIR
        self._prompt_cache: Dict[PersonaType, str] = {}
    
    def _get_prompt_filename(self, persona: PersonaType) -> str:
        """Возвращает имя файла промта для персоны"""
        filename_mapping = {
            PersonaType.BUSINESS: "system_business.txt",
            PersonaType.BELLA: "system_bella.txt",
            PersonaType.BEN: "system_ben.txt"
        }
        return filename_mapping.get(persona, "system_business.txt")
    
    def _load_prompt_from_file(self, persona: PersonaType) -> Optional[str]:
        """Загружает промт из файла"""
        filename = self._get_prompt_filename(persona)
        file_path = os.path.join(self.prompts_dir, filename)
        
        try:
            if not os.path.exists(file_path):
                logger.warning("Prompt file not found", 
                             persona=persona.value, 
                             file_path=file_path)
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
            
            if not prompt_content:
                logger.warning("Prompt file is empty", 
                             persona=persona.value, 
                             file_path=file_path)
                return None
            
            logger.debug("Prompt loaded from file", 
                        persona=persona.value, 
                        content_length=len(prompt_content))
            
            return prompt_content
            
        except Exception as e:
            logger.error("Failed to load prompt file", 
                        persona=persona.value, 
                        file_path=file_path, 
                        error=str(e))
            return None
    
    def _get_default_prompt(self, persona: PersonaType) -> str:
        """Возвращает дефолтный промт если файл не найден"""
        default_prompts = {
            PersonaType.BUSINESS: (
                "Ты - профессиональный консультант технической поддержки онлайн казино. "
                "Отвечай в деловом стиле, будь вежливым и информативным. "
                "НЕ выдавай себя за бота или искусственный интеллект."
            ),
            PersonaType.BELLA: (
                "Привет! Я Белла, твоя помощница в казино! 😊 "
                "Общаюсь дружелюбно с легким флиртом, но остаюсь профессиональной. "
                "НЕ выдаю себя за бота или искусственный интеллект."
            ),
            PersonaType.BEN: (
                "Привет! Я Бэн, твой надежный партнер в казино! 🎯 "
                "Общаюсь уверенно и по-дружески, помогаю решать любые вопросы. "
                "НЕ выдаю себя за бота или искусственный интеллект."
            )
        }
        return default_prompts.get(persona, default_prompts[PersonaType.BUSINESS])
    
    def _load_promotions_text(self) -> Optional[str]:
        """Загружает текст промоакций из файла"""
        try:
            file_path = config.PROMOTIONS_FILE
            if not file_path or not os.path.exists(file_path):
                logger.warning("Promotions file not found", file_path=file_path)
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content if content else None
        except Exception as e:
            logger.warning("Failed to load promotions file", error=str(e))
            return None

    def get_system_prompt(
        self, 
        persona: PersonaType, 
        promotions_info: Optional[str] = None,
        context_info: Optional[str] = None
    ) -> str:
        """
        Получает системный промт для персоны с подстановкой динамических значений
        
        Args:
            persona: Тип персоны бота
            promotions_info: Информация о промоакциях для подстановки в {promotions}
            context_info: Контекст из векторного поиска для подстановки в {context}
            
        Returns:
            Готовый системный промт с подставленными значениями
        """
        # Загружаем промт из файла (каждый раз заново для динамических изменений)
        prompt_template = self._load_prompt_from_file(persona)
        
        if not prompt_template:
            logger.warning("Using default prompt", persona=persona.value)
            prompt_template = self._get_default_prompt(persona)
        
        # Подставляем динамические значения
        final_prompt = prompt_template
        
        # Подстановка информации о промоакциях
        if "{promotions}" in final_prompt:
            if promotions_info is None:
                promotions_info = self._load_promotions_text()
            promotions_text = promotions_info or "Актуальные промоакции временно недоступны."
            final_prompt = final_prompt.replace("{promotions}", promotions_text)
        
        # Подстановка контекста из векторного поиска
        if "{context}" in final_prompt:
            context_text = context_info or ""
            if context_text:
                context_section = f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:\n{context_text}\n"
            else:
                context_section = ""
            final_prompt = final_prompt.replace("{context}", context_section)
        
        logger.debug("System prompt prepared", 
                    persona=persona.value,
                    has_promotions=bool(promotions_info),
                    has_context=bool(context_info),
                    final_length=len(final_prompt))
        
        return final_prompt


class PromptServiceError(Exception):
    """Исключение для ошибок сервиса промтов"""
    pass


# Глобальный экземпляр сервиса промтов
prompt_service = PromptService()
