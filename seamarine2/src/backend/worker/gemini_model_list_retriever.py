from PySide6.QtCore import Signal, QThread
import asyncio
from backend.core import TranslateCore

class GeminiModelListRetriever(QThread):
    retrieved = Signal(list)

    def __init__(self, core: TranslateCore, keys: list[str]):
        super().__init__()
        self._core = core
        self._keys = keys
        
    def run(self):
        asyncio.run(self._async_execute())

    async def _async_execute(self):
        for key in self._keys:
            if self._core.register_keys([key]):
                model_list = self._core.get_model_list()
                if model_list:
                    self._core.register_keys(self._keys)
                    self.retrieved.emit(model_list)
                    self.finished.emit()
                    return
        self.retrieved.emit([])
        self.finished.emit()
