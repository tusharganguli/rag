
import os

from src.rag import Retrieve
from src.utils import Utils
from src.eval import Evaluate

env_path = os.path.join(os.path.dirname(__file__), '../config/.env')
Utils.load_env_vars(env_path)

@Utils.time_it
def test_apple10k():
        directory_path = "../Dataset/apple10k"
        vector_store_name = "./data/vectorstore/apple10k"
        override = True
        #Retrieve.create_vectorstore(directory_path, vector_store_name, override)
        
        query = """What percentage of the Company's net sales in 
                2022, 2021 and 2020 were from the U.S. and China?"""

        #rag = Retrieve(vector_store_name)
        #response,metadata = rag.get_response(query)
        #print("Answer:\n", response)
        #print("Metadata:\n",metadata)

        file_path = "../Dataset/apple10k/rag_benchmark_apple_10k_2022_with_context.csv"
        response_file = "./data/apple10k.csv"
        #Utils.create_test_data(vector_store_name, file_path, response_file)
        eval_file = "./data/apple10k_eval.csv"
        Evaluate.evaluate(response_file, eval_file)

def test_sec10q():
        directory_path = "../Dataset/KG-RAG-datasets-main/sec-10-q/data/v1/docs"
        vector_store_name = "./data/vectorstore/sec10q_all-mpnet-base-v2"
        override = True
        
        #Retrieve.create_vectorstore(directory_path, vector_store_name, override)
        
        query = """Examine how Intel's effective tax rate in the most recent 10-Q 
                compares with the tax-related discussions in the notes section."""

        #response,metadata = rag.get_response(query)
        #print("Answer:\n", response)
        #print("Metadata:\n",metadata)

        file_path = "../Dataset/KG-RAG-datasets-main/sec-10-q/data/v1/qna_data_mini.csv"
        response_file = "./data/sec10q_all-mpnet-base-v2.csv"
        Utils.create_test_data(vector_store_name, file_path, response_file)
        eval_file = "./data/sec10q_all-mpnet-base-v2_base_retriever_30_eval.csv"
        Evaluate.evaluate(response_file, eval_file)

test_apple10k()
