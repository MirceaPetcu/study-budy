from abc import ABC, abstractmethod


class IndexBase(ABC):
    @staticmethod
    def load_documents(document_dir_path: str, extensions: list):
        pass

    @abstractmethod
    def insert_documents(self, document_dir_path: str, extensions: list):
        pass

    @abstractmethod
    def delete_documents(self, document_ids: list):
        pass
