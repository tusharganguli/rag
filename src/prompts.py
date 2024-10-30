
multiquery_response_prompt = """
                                The main query is specified after the "query" tag. Following is the list of 
                                sub-queries and their corresponding answers after "subquery" and "answer" tag.
                                Use the sub-queries and their corresponding answers to generate a final answer
                                for the query in the "query" tag.

                                query: {}
 
                            """

decompose_prompt = """
                    Decompose the following query into multiple independent sub-queries, 
                    ensuring there is no redundancy. Each sub-query should be distinct and focus on a 
                    specific aspect of the original query. Return the response as a Python list.

                    if the sub-query needs to make a calculation, do not pose it as a question,
                    use data to calculate such value.

                    query: {}

                    """

extract_information = """
                    Based on the information we are looking for in the query,
                    extract the most important keywords. 
                    Generate a final list in the form of a python list, do not explain it:
                    Use the following rules:
                    1. If specific information is required which is mentioned in the query,
                    consider it to generate keywords.
                    2. If the query contains specific location in the document to be searched then
                    include that location in the keywords list.
                    3. if the query contains the type of filename that should be queried then 
                    exclude that from the keywords list.

                    query: {}
                    """
filter_filename_prompt = """
                        Follow the following steps:

                        1. From the query below, identify proper nouns, including company names 
                        and ticker symbols. 
                        
                        2. Use these proper nouns and their synonyms 
                        (such as ticker symbols) to determine which filenames are relevant 
                        by comparing them to the filenames and their associated metadata. 
                        
                        3. If any of the proper nouns or their synonyms (e.g., "Apple" and "AAPL") 
                        are associated with the filename or metadata, include the filename in 
                        the filtered list.

                        4. Retrieve and compare the most important contexts of the query to the filtered list 
                        and if needed filter the filtered list based on the context of the query.

                        5. If the context is too generic include all filenames.
                        
                        6. Ensure that the final response only contains a python list of filenames 
                        that were filtered based on relevance to the query in the following format:

                        ''' filenames: ['filename1','filename2']'''
                        
                        query: {}

                        filenames and metadata: {}

                        """

filter_filename_prompt2 = """
                    From the json list of dictionary that contain filenames and their metadata in json format, 
                    Filter those filenames which might contribute to answering the query.
                    Use the filenames and their associated metadata to make the determination. 
                    Return list of only file names.

                    query: {}

                    filenames and metadata: {}

                    """

prompt_template = """
                Answer the question only based on the context provided. If the context is insufficient display
                "Insufficient information to answer the question." Response should start in a new line with
                the tag "answer:"
                
                context: {}

                question: {}

            """
