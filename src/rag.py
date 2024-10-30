
import os
import time
import torch
from typing import List
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

from transformers import BertTokenizer, BertModel
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from sentence_transformers import SentenceTransformer
from typing import Tuple
from langchain_community.retrievers import BM25Retriever

from src.prompts import *
from src.utils import Utils

class STEmbedding(Embeddings):
    def __init__(self, model):
        self.mpnet_embedding_model = SentenceTransformer(model)
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.mpnet_embedding_model.encode(text)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        #print("Text to embed:\n",text)
        return self.embed_documents([text])[0]

class Retrieve:
    @staticmethod
    def create_vectorstore(directory_path: str, 
                           vector_store_name: str, 
                           override: bool=False
                           )-> Chroma:
        """
        Create a vector store.
        Args:
            directory_path (str): The path where the vector store should be created.
            vector_store_name (str): The vector store name.
            override (bool): Whether to override an existing vectorstore.
        Returns:
            Chroma: Chroma vectorstore.
        """
        
        # optimum chunk size 
        if override == False and os.path.exists(vector_store_name):
            return Retrieve.load_vector_store(vector_store_name)
        
        documents = Retrieve.__load_documents(directory_path, vector_store_name)
        #print("Documents retrieved from the pdf:\n",documents[0]) 
    
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        doc_chunks = splitter.split_documents(documents)
        #print("Doc chunks after split:\n",doc_chunks) 
        
        embedding_model = Retrieve.__load_embedding_model()
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=vector_store_name
        )
        print(">>>Embedding and chunking process completed successfully<<<")
        return vectordb

    def __init__(self, vector_store_name):
        self.vector_store_name = vector_store_name
        self.embedding_model = self.__load_embedding_model()
        # need to load the embedding model before calling load_vector_store
        vector_store = self.load_vector_store(vector_store_name)
        print("Vector store successfully loaded.")
        self.base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        
    @staticmethod
    def __load_embedding_model():
        voyage_embedding_model = VoyageAIEmbeddings(
                                voyage_api_key=os.environ.get("VOYAGE_API_KEY"), 
                                model="voyage-finance-2"
                            )
        openai_embedding_model = OpenAIEmbeddings(  model="text-embedding-ada-002", 
                                            openai_api_key=os.environ.get("OPENAI_API_KEY"))
        
        mpnet_embedding_model = STEmbedding('sentence-transformers/all-mpnet-base-v2')


        return mpnet_embedding_model

    def load_vector_store(self,vector_store_name):
        vector_store = Chroma(persist_directory=vector_store_name, embedding_function=self.embedding_model)
        return vector_store
    
    @Utils.time_it
    def get_response(self, query):
        all = True
        filtered_docs = self.__filter_docs(self.vector_store_name, query, all)
        filtered_docs = self.__keyword_search(filtered_docs, query)

        filtered_vector_store = Chroma.from_documents(
                    documents=filtered_docs,
                    embedding=self.embedding_model
                )
        
        # Use the new vector store with a retriever
        new_base_retriever = filtered_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 30})
        
        mqr_prompt = multiquery_response_prompt.format(query)
        sub_queries = self.__decompose_query(query)
        answers, metadata_lst,mqr_prompt = self.__process_subqueries(sub_queries, new_base_retriever,mqr_prompt)
        #print("Multi query response:\n",mqr_prompt)
        
        #print("Final Query:\n", final_query)

        answer = self.__get_openai_response(mqr_prompt)
        #answer, metadata = self.__get_response(query, new_base_retriever)

        return answer, metadata_lst
    
    @Utils.time_it
    def __get_response(self, query, retriever):

        llm = ChatOpenAI(   #model="gpt-3.5-turbo",
                            model = "gpt-4o-mini",
                            temperature=0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2 
                        )
        
        #mq_retriever_from_llm = MultiQueryRetriever.from_llm(
        #                   retriever=retriever, llm=llm
        #                        )
        
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        crossenc = CrossEncoderReranker(model=model, top_n=6)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=crossenc, base_retriever=retriever
        )

        start_time = time.perf_counter()
        ranked_docs = compression_retriever.invoke(query)

        end_time = time.perf_counter()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Time taken by compression_retriever.invoke: {elapsed_time:.4f} seconds")
    
        parsed_content, metadata = self.__get_content(ranked_docs)
        prompt = prompt_template.format(parsed_content, query)
        llm_response = self.__get_openai_response(prompt)
        
        return llm_response, metadata

    @Utils.time_it
    def __filter_docs(self, vector_store_name, query, all = False):
        filtered_docs = []
        # retrieve the list of filenames
        filtered_filename, entries = self.__get_filter_list(vector_store_name, query, all)
        vector_store = self.load_vector_store(vector_store_name)
        
        # for each file name retrieve the
        for filename in filtered_filename:
            for entry in entries:
                if entry.get("filename") == filename:
                    metadata = entry.get("metadata")
                    collection = vector_store.get(where={"source": metadata.get("source")}, include = ["metadatas", "documents"])
                    # dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data', 'included'])
                    doc_contents = collection.get("documents")
                    metadatas = collection.get("metadatas")
                    for doc_content,metadata in zip(doc_contents,metadatas):
                        document = Document( page_content=doc_content, metadata=metadata)
                        filtered_docs.append(document)
                    #for doc in documents:
                    #    #print("Doc:\n",doc)
                    #    if doc.metadata.get("source") == metadata.get("source"):
                    #        filtered_docs.append(doc) 
        return filtered_docs
    
    @Utils.time_it
    def __get_filter_list(self, vector_store_name, query, all):
        dir_path, file_name = vector_store_name.rsplit('/', 1)
        json_filename = dir_path + "/metadata/"+ file_name + "_metadata.json"
        with open(json_filename, 'r') as json_file:
            entries = json.load(json_file)
        
        filenames = ""
        for entry in entries:
            filenames += "\n\n" + entry["filename"]

        # if we need to return all the filnames then we return the filnames in 
        if all == True:
            # Split the string by \n\n to get a list of filenames
            filename_lst = filenames.split('\n\n')

            # strip any extra whitespace or newline characters around each filename
            filename_lst = [filename.strip() for filename in filename_lst]

            return filename_lst,entries
        
        prompt = filter_filename_prompt.format(query, filenames)
        with open('./data/prompt.txt', 'w') as file:
            # Write the string to the file
            file.write(prompt)
        #print("Before initiating openai request")
        llm_response = self.__get_openai_response(prompt)
        #print("LLM response:\n", llm_response)

        start = llm_response.find('[')  # Find the first occurrence of '['
        end = llm_response.find(']')    # Find the first occurrence of ']'

        # Extract the portion between '[' and ']'
        cleaned_input = llm_response[start:end+1]
        #print("Cleaned input:\n", cleaned_input)

        import ast
        filenames_list = ast.literal_eval(cleaned_input)
        print("Filename list:\n",filenames_list)
        
        return filenames_list, entries
    
    @Utils.time_it
    def __keyword_search(self, documents, query):
        #print("Total documents before keyword based search:", len(documents))

        # Initialize BM25 retriever for keyword-based search
        bm25_retriever = BM25Retriever.from_documents(documents)
        keyword_lst = self.__filter_keywords(query)

        doc_lst = []
        # Perform a keyword-based search
        for keyword in keyword_lst:
            #print("keyword:",keyword)
            docs = bm25_retriever.invoke(keyword)
            doc_lst.extend(docs)
            #for doc in docs:
            #    #print(doc.metadata)
            #    print("file:{},\n page:{}".format(doc.metadata.get("file_path"),doc.metadata.get("page")))
        
        #print("No. of filtered documents after keyword search:", len(doc_lst))
        return doc_lst
    
    @Utils.time_it
    def __filter_keywords(self, query):
        keywords_prompt = extract_information.format(query)
        keywords = self.__get_openai_response(keywords_prompt) 
 
        start = keywords.find('[')  # Find the first occurrence of '['
        end = keywords.find(']')    # Find the first occurrence of ']'

        # Extract the portion between '[' and ']'
        keywords = keywords[start:end+1]
        import ast
        from sklearn.metrics.pairwise import cosine_similarity

        keywords = ast.literal_eval(keywords)
        #print("Keywords extracted:\n", keywords)
        return keywords

    @Utils.time_it
    def __decompose_query(self, query):
        final_query = decompose_prompt.format(query)

        sub_queries = self.__get_openai_response(final_query)  
        #print("Sub queries:\n",sub_queries)
        start = sub_queries.find('[')  # Find the first occurrence of '['
        end = sub_queries.find(']')    # Find the first occurrence of ']'

        # Extract the portion between '[' and ']'
        sub_queries = sub_queries[start:end+1]
        import ast
        sub_queries = ast.literal_eval(sub_queries)
        #print("Sub-queries:\n", sub_queries)
        return sub_queries
    
    @Utils.time_it
    def __process_subqueries(self, sub_queries, retriever, mqr_prompt) -> Tuple[List[str], List[dict]]:
        answers = []
        metadata_lst = []

        for sub_query in sub_queries:
            #print("Sub-query:",sub_query)
            if sub_query == "":
                print("Empty query")
                continue
            answer, metadata = self.__get_response(sub_query, retriever)
            meta_info = {}
            meta_info["metadata"] = metadata
            meta_info["SubQuery"] = sub_query
            mqr_prompt += "\n\n subquery:" + sub_query
            mqr_prompt += "\n\n answer:" + answer
            answers.append(answer)
            metadata_lst.append(meta_info)

        return answers, metadata_lst, mqr_prompt
    
    @staticmethod
    def __save_meta(vector_store_name, metadata):
        dir_path, file_name = vector_store_name.rsplit('/', 1)

        # Save the collected data to a JSON file
        json_filename = dir_path + "/metadata/"+ file_name + "_metadata.json"
        with open(json_filename, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f"Document metadata saved to {json_filename}")

    @staticmethod
    @Utils.time_it
    def __load_documents(directory_path, vector_store_name):
        documents = []
        metadata = []
  
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".pdf"):  # Check if the file is a PDF
                    filename = os.path.join(root, file)
                    #print("Loading:", filename)
                    loader = PyMuPDFLoader(filename)
                    docs = loader.load()
                    meta = {
                    "filename": file,  # Store the filename
                    "metadata": docs[0].metadata,  # Store the metadata
                    }
                    meta = {
                    "filename": file,  # Store the filename
                    "metadata": {"source": docs[0].metadata["source"],
                                 "file_path": docs[0].metadata["file_path"],
                                 "subject": docs[0].metadata["subject"]},  # Store the metadata
                    }
                    metadata.append(meta)
                    documents.extend(docs)

        Retrieve.__save_meta(vector_store_name, metadata)
        return documents
    
    @Utils.time_it
    def __get_openai_response( self, prompt):
        messages = [{"role": "user", "content": prompt}]
        model_name = "gpt-4o-mini"
        #model_name = "gpt-3.5-turbo"
        response = openai.chat.completions.create(
            model= model_name,
            messages=messages,
            temperature=0,
        )
        #print("\nOpenAI Response:****************************\n", response)
        return response.choices[0].message.content
        #return response.choices[0].text.strip()

    @Utils.time_it
    def __get_content(self, docs):
        doc_content = ""
        metadata = []
        for doc in docs:
            #print("doc:\n",doc)
            #page_nos += str(doc.metadata["page_no"]) + ","
            #print("Content:\n", doc.page_content)
            doc_content += "\n\n" + "".join(doc.page_content)
            meta = {}
            meta["filepath"] = str(doc.metadata["file_path"])
            meta["title"] = str(doc.metadata["subject"])
            meta["pageno"] = str(doc.metadata["page"]+1)
            meta["total_pages"] = str(doc.metadata["total_pages"])
            metadata.append(meta)
        return doc_content, metadata
    
    def get_parent_retriever(self, vector_store_name, dir_path):
        from langchain.storage import InMemoryStore
        from langchain.retrievers import ParentDocumentRetriever

        vector_store = Chroma(persist_directory=vector_store_name, embedding_function=self.embedding_model)
        print("Vector store successfully loaded.")
        
        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, add_start_index=True)
        # This text splitter is used to create the child documents
        # It should create documents smaller than the parent
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, add_start_index=True)
        # The storage layer for the parent documents
        store = InMemoryStore()

        # Initialize the retriever
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        docs = self.__load_documents(dir_path)
        parent_retriever.add_documents(docs, ids=None)

        return parent_retriever
    