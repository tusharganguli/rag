
import pandas as pd
from dotenv import load_dotenv
import os
import openai
import time
from functools import wraps


class Utils:

    def get_openai_response(prompt):
        messages = [{"role": "user", "content": prompt}]
        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            temperature=0,
        )
        #print("\nOpenAI Response:****************************\n", response)
        return response.choices[0].message.content
        #return response.choices[0].text.strip()


    def load_env_vars(env_path = "./"):
        # Load environment variables from .env file
        load_dotenv(dotenv_path = env_path)

    def format_meta(meta_lst):
        meta_content = ""
        for metainfo in meta_lst:
            meta_content += "\n\n" + "Sub Query:" + metainfo.get("SubQuery")
            metadata_lst = metainfo.get("metadata")
            for m in metadata_lst:
                meta_content += "\n\n" + "File path:" + m.get("filepath")
                meta_content += "\n" + "Title:" + m.get("title")
                meta_content += "\n" + "Page No:" + m.get("pageno")
                meta_content += "\n" + "Total Pages:" + m.get("total_pages")
        return meta_content

    def print_meta(meta):
        for m in meta:
            print("File path:" + m.get("filepath"))
            print("Title:" + m.get("title"))
            print("\n" + "Page No:" + m.get("pageno")+1)
            print("\n" + "Total Pages:" + m.get("total_pages"))

    def get_meta( retriever, question):
        docs = retriever.invoke(question)
        metadata = []
        for doc in docs:
            meta = {}
            meta["filepath"] = str(doc.metadata["file_path"])
            meta["title"] = str(doc.metadata["subject"])
            meta["pageno"] = str(doc.metadata["page"]+1)
            meta["total_pages"] = str(doc.metadata["total_pages"])
            metadata.append(meta)
        return metadata

    def store_meta( query, base_retriever, multi_query_retriever, compression_retriever):
        import pandas as pd

        print("Retrieving Meta")
        base_meta = Utils.get_meta(base_retriever, query)
        multi_query_meta = Utils.get_meta(multi_query_retriever, query)
        compression_meta = Utils.get_meta(compression_retriever, query)
        df = pd.DataFrame()

        # Finding the maximum length of the lists
        max_length = max(len(base_meta), len(multi_query_meta), len(compression_meta))

        # Padding the lists with None to make them the same length
        base_meta.extend([None] * (max_length - len(base_meta)))
        multi_query_meta.extend([None] * (max_length - len(multi_query_meta)))
        compression_meta.extend([None] * (max_length - len(compression_meta)))

        df["Base"] = base_meta
        df["Multi Query"] = multi_query_meta
        df["Compression"] = compression_meta
        df.to_csv("metadata.csv", index=False)
        print("Meta stored successfully")

    def get_last_processed_row(file_path):
        """ Reads the last processed row number from a file """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return int(f.read().strip())
        return 0  # Start from the beginning if no log exists

    def save_last_processed_row(file_path, row_number):
        """ Saves the last processed row number to a file """
        with open(file_path, "w") as f:
            f.write(str(row_number))

    def create_test_data(vector_store_name: str, 
                        file_path: str, 
                        test_file: str) -> None:
        
        from src.rag import Retrieve

        if not os.path.exists(test_file):
            df = pd.read_csv(file_path)
            df["Generated Answer"] = ""  # Add new columns for generated answer and metadata
            df["Metadata"] = ""
            df.to_csv(test_file, index=False)

        # read the csv file
        df = pd.read_csv(test_file)
        
        rag = Retrieve(vector_store_name)

        last_processed_file = test_file.replace(".csv",".log")
        last_processed_row = Utils.get_last_processed_row(last_processed_file)

        for index, row in df.iterrows():
            if index < last_processed_row:
                continue  # Skip rows that have already been processed

            question = row["Query"]
            print("Question {}: {}".format(index+1,question))

            try:
                response, metadata = rag.get_response(question)
                df.at[index, "Generated Answer"] = response  # Update answer
                df.at[index, "Metadata"] = Utils.format_meta(metadata)  # Update metadata
                df.to_csv(test_file, index=False)
            except Exception as e:
                print(f"exception caught: {e}")
                Utils.save_last_processed_row(last_processed_file, index)
                break

    # Define the decorator to measure time
    def time_it(func):
        @wraps(func)  # This preserves the original function's metadata
        def wrapper(*args, **kwargs):
            # Start time
            start_time = time.perf_counter()

            # Call the original function
            result = func(*args, **kwargs)

            # End time
            end_time = time.perf_counter()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Time taken by '{func.__name__}': {elapsed_time:.4f} seconds")

            return result

        return wrapper


    """

    def create_test_data(vector_store_name: str, 
                        file_path: str, 
                        test_file: str) -> None:
        
        from src.rag import Retrieve

        # read the csv file
        df = pd.read_csv(test_file)
        
        rag = Retrieve(vector_store_name)
        generated_response = []
        compression_metadata = []

        for index, row in df.iterrows():
            question = row["Query"]
            print("Question:", question)
            response, metadata = rag.get_response(question)
            generated_response.append(response)
            compression_metadata.append(Utils.format_meta(metadata))

        df["Generated Answer"] = generated_response
        df["Metadata"] = compression_metadata
        
        df.to_csv(test_file, index=False)

    def create_test_data(vector_store_name, data_dir_path, file_path, test_file):
        # read the csv file
        df = pd.read_csv(file_path)
        rag = Rag()
        parent_retriever = rag.get_parent_retriever(vector_store_name, data_dir_path)
        generated_response = []
        metadata = []
        base_metadata = []
        for index, row in df.iterrows():
            question = row["Question"]
            print("Question:", question)
            response, meta = rag.get_response(parent_retriever, question)
            generated_response.append(response)
            metadata.append(format_meta(meta))
        df["Generated Answer"] = generated_response
        df["Metadata"] = metadata
        df.to_csv(test_file, index=False)

    """