
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction
import re, time, json

from src.utils import Utils

class Evaluate:
    
    def evaluate(response_file, eval_file):
        df = pd.read_csv(response_file)

        results = {
            'bs': [],
            'r1': [],
            'rL': [],
            'bert_p': [],
            'bert_r': [],
            'bert_f1': [],
            'sim': [],
            'llm_recall': [],
            'llm_precision': [],
            'llm_f1': [],
            'g_cnt':[],
            'cand_cnt':[],
            'co_cnt':[],
            'g_claims':[],
            'cand_claims':[],
            'co_claims':[]
        }

        for index, row in df.iterrows():

            question = row["Query"]
            print("Question {}: {}".format(index,question))
            golden_resp = row["Response"]
            cand_resp = row["Generated Answer"]

            bs, r1, rL, bert_p, bert_r, bert_f1 = Evaluate.__evaluate_metrics(golden_resp,cand_resp)
            sim = Evaluate.__evaluate_similarity([golden_resp],[cand_resp])
            results['bs'].append(bs)
            results['r1'].append(r1)
            results['rL'].append(rL)
            results['bert_p'].append(bert_p)
            results['bert_r'].append(bert_r)
            results['bert_f1'].append(bert_f1)
            results['sim'].append(sim)

            llm_recall, llm_precision, llm_f1, llm_response = Evaluate.evaluate_via_llm(question, golden_resp, cand_resp)
            results['llm_recall'].append(llm_recall)
            results['llm_precision'].append(llm_precision)
            results['llm_f1'].append(llm_f1)
            g_claims = llm_response["Golden Response Claims"]
            cand_claims = llm_response["Candidate Response Claims"]
            co_claims = llm_response["Common Claims"]
            g_cnt = llm_response["No of Golden Response Claims"]
            cand_cnt = llm_response["No of Candidate Response Claims"]
            co_cnt = llm_response["No of Common Claims"]
            # this means that the candidate claims cover all the common claims which are part of the 
            # golden claims.
            if co_cnt > cand_cnt:
                cand_cnt = co_cnt
            results['g_claims'].append(g_claims)
            results['cand_claims'].append(cand_claims)
            results['co_claims'].append(co_claims)
            results['g_cnt'].append(g_cnt)
            results['cand_cnt'].append(cand_cnt)
            results['co_cnt'].append(co_cnt)                        
            
        df['Bleu Score'] = results['bs']
        df['Rouge-1'] = results['r1']
        df['Rouge-L'] = results['rL']
        df['Bert Precision'] = results['bert_p']
        df['Bert Recall'] = results['bert_r']
        df['Bert Score F1'] = results['bert_f1']
        df['Similarity Score'] = results['sim']
        df['LLM Recall'] = results['llm_recall']
        df['LLM Precision'] = results['llm_precision']
        df['LLM F1'] = results['llm_f1']
        df['Golden Response Claim Count'] = results['g_cnt']
        df['Candidate Response Claim Count'] = results['cand_cnt']
        df['Common Claim Count'] = results['co_cnt']
        df['Golden Response Claims'] = results['g_claims']
        df['Candidate Response Claims'] = results['cand_claims']
        df['Common Claims'] = results['co_claims']
        
        df.to_csv(eval_file, index=False)
    
    def __evaluate_similarity(reference_sentence,candidate_sentence):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Compute embeddings for both lists
        embeddings1 = model.encode(reference_sentence)
        embeddings2 = model.encode(candidate_sentence)
        # Compute cosine similarities
        similarities = model.similarity(embeddings1, embeddings2)
        return similarities[0][0].item()

    def __evaluate_metrics(reference, candidate):
        
        # Calculate BLEU score
        ref_bleu = reference.split()
        cand_bleu = candidate.split()
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu([ref_bleu], cand_bleu, smoothing_function=chencherry.method1)
        #print(f'BLEU score: {bleu_score:.4f}')

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        #print(f'ROUGE-1 score: {scores["rouge1"].fmeasure:.4f}')
        #print(f'ROUGE-L score: {scores["rougeL"].fmeasure:.4f}')

        # Calculate BERT score
        P, R, F1 = bert_score.score([candidate], [reference], lang="en", rescale_with_baseline=True)
        #print(f'BERT Precision: {P.mean().item():.4f}')
        #print(f'BERT Recall: {R.mean().item():.4f}')
        #print(f'BERT F1: {F1.mean().item():.4f}')

        return bleu_score, scores["rouge1"].fmeasure, scores["rougeL"].fmeasure, P.item(), R.item(), F1.item()
    
    def evaluate_via_llm(question, golden_response, candidate_response):
        
        prompt_template5 = """
            Given the following question:

            ###  Start Question:
            {}
            End Question

            and a golden response and a candidate response respectively. 

            ### Start Golden Response:
            {}
            End Golden Response

            ### Start Candidate Response:
            {}
            End Candidate Response

            ### Evaluate the two responses using the Evaluation Method below. 
            The responses could be numerical, specific (e.g., names or dates), or descriptive.

            ### Evaluation Method:
            1. Create a list of individual claims that can be inferred from the golden response with respect to the question.
            2. Create a list of individual claims that can be inferred from the candidate response with respect to the question.
            3. Calculate the total number of claims of the golden response present in the candidate response based on the following rules:
                - the complete statement of each claim in golden response should be checked against the complete statement of each claim in candidate response. 
                - If the golden response claim is specific in nature like numerical, names or dates then the candidate response claim should contains the exact value present in the golden response.
            
            ### For creating the individual claims follow the following instructions:
             - Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
             - Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
             - For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
             - Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.

            ### After creating the list, perform the following:
            1. In the golden response claims, if any claim can be directly inferred from the question only then remove it from the list.
            2. In the candidate repsonse claims, if any claim can be directly inferred from the question, only then remove it from the list.
            
            ### The final output should contain the explanation of the evaluation method and the numerical value in the following json format:
            
            {{
                Golden Response Claims: {{ <list of claims from the golden response> }}
                Candidate Response Claims: {{ <list of claims from the candidate response> }}
                Common Claims: {{ <list of claims from golden response present in candidate > }}
                No of Golden Response Claims: <value>
                No of Candidate Response Claims: <value>
                No of Common Claims: <value>
            }}
            
            ### Example:
            {{
                "Golden Response Claims": {{ 
                                                "1": The Mac line includes laptops.,
                                                "2": The laptops mentioned are MacBook Air and MacBook Pro.,
                                                "3": The Mac line includes desktops.,
                                                "4": The desktops mentioned are iMac, Mac mini, Mac Studio, and Mac Pro.
                                            }},
                "Candidate Response Claims":   {{
                                                "1": The company's line of personal computers is called Mac.,
                                                "2": It includes laptops.,
                                                "3": The laptops included are MacBook Air and MacBook Pro.,
                                                "4": It includes desktops.,
                                                "5": The desktops included are iMac, Mac mini, Mac Studio, and Mac Pro.,
                                                }},
                "No of Golden Response Claims": 4,
                "No of Candidate Response Claims": 5,
                "No of Common Claims": 4
            }}

            ### Please strictly adhere to the json format specified above. please provide the complete response
            in json format.

            """

        prompt = prompt_template5.format(question, golden_response, candidate_response)
        #print("Prompt**************************\n",prompt)
        #response = get_amazon_response(prompt)
        response = Utils.get_openai_response(prompt)
        json_response = None
        try:
            #print("\nResponse before extract:****************************\n",response)
            response = Evaluate.__extract_json(response)
            #print("\nResponse after extract:******************************\n",response)
            json_response = json.loads(response)
            #print(json_response)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print("Raw output:", response)
            return

        if json_response is None or json_response == '':
            print("Empty response ***************\n")
        #print("Response:\n", response)

        golden_cnt = json_response["No of Golden Response Claims"]
        candidate_cnt = json_response["No of Candidate Response Claims"]
        common_cnt = json_response["No of Common Claims"]
        # this means that the cnadidate claims cover all the common claims which are part of the 
        # golden claims.
        if common_cnt > candidate_cnt:
            candidate_cnt = common_cnt
        recall, precision, f1 = Evaluate.__calculate_llm_metrics(golden_cnt, candidate_cnt, common_cnt)
        
        time.sleep(1)

        return recall, precision, f1, json_response

    def __calculate_llm_metrics(golden_cnt, candidate_cnt, common_cnt):
        if golden_cnt == 0:
            recall = 0
        else:
            recall = common_cnt/golden_cnt
        if candidate_cnt == 0:
            precision = 0
        else:
            precision = common_cnt/candidate_cnt
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = (2*recall*precision)/(precision+recall)
        return recall, precision, f1

    def __extract_json(response):
            start_tag = '```json'
            end_tag = '```'
            start_tag_idx = response.find(start_tag)
            end_tag_idx = response.rfind(end_tag)
            if start_tag_idx != -1 and end_tag_idx != -1:
                return response[start_tag_idx+len(start_tag):end_tag_idx].strip()
            else:
                return response

