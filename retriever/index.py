import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from index_base import IndexBase
import mlflow
import time
from class_with_logger import BaseClassWithLogging
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class Index(IndexBase, BaseClassWithLogging):
    def __init__(self,
                 extensions: list = [".pdf"],
                 embed_model: str = "BAAI/bge-small-en-v1.5",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 output_dir: str = "./chroma_db",):
        super().__init__(class_name="Index")
        self.start_logging()

        # Log key parameters
        mlflow.log_param("embed_model", embed_model)
        mlflow.log_param("extensions", extensions)
        mlflow.log_param("device", device)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_text("RAG_INDEX_TYPE", "Vector")

        start_time = time.time()

        self.doc_registry = {}

        embedding_model = HuggingFaceEmbedding(model_name=embed_model, device=device)

        db = chromadb.PersistentClient(path=output_dir)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._vector_store_index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context,
            show_progress=True,
            embed_model=embedding_model
        )
        self.num_docs = 0
        # self._vector_store_index.storage_context.persist(persist_dir=output_dir)

        mlflow.log_metric("indexing_time", time.time() - start_time)
        mlflow.log_artifact(output_dir)

        mlflow.end_run()

    @property
    def index(self):
        return self._vector_store_index

    @staticmethod
    def load_documents(document_dir_path: str, extensions: list):
        return SimpleDirectoryReader(
            input_dir=document_dir_path,
            recursive=True,
            required_exts=extensions,
        ).load_data(show_progress=True)

    def insert_documents(self, document_dir_path: str, extensions: list):
        mlflow.start_run()
        new_documents = Index.load_documents(document_dir_path, extensions=extensions)

        mlflow.log_metric("num_documents_inserted", len(new_documents))

        for document in new_documents:
            self._vector_store_index.insert(document)
            self.doc_registry[document.metadata['file_name']] = document.doc_id
            self.num_docs += 1
            mlflow.log_text(f'inserted_doc_{document.doc_id}', document.metadata['file_name'])
        mlflow.end_run()

    def delete_documents(self, document_names: list):
        mlflow.start_run()
        document_ids = [self.doc_registry[doc_name] for doc_name in document_names]
        mlflow.log_param("document_names_deleted", document_names)

        for doc_id in document_ids:
            self._vector_store_index.delete_ref_doc(doc_id)
            mlflow.log_text(f'deleted_doc_{doc_id}', document_names[document_ids.index(doc_id)])
        mlflow.end_run()

