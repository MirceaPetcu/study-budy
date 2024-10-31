from abc import ABC, abstractmethod


class GeneratorBase(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

