from sklearn.metrics import ndcg_score
import numpy as np
import logging
import os
from typing import List, Dict, Set
from tqdm import trange
from tqdm import tqdm
import torch
from multiprocessing import Pool
import heapq
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim
import os
import threading
import torch
import multiprocessing
import torch.nn.functional as F
from src.utils import create_ada_embedding
import json
import jieba
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class APIEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document.
    """

    def __init__(
        self,
        queries_list: List[str], # query
        queries_id_list: List[str], # qid
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]], 
        id_corpus_emb_dict=None,  # cid => emb
        model=None,
        bm25=None,
        tf_idf=None,
        retriever_type=None,
        output_path=None,
        write_csv=True,
        phase= 'test'
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.queries_id = queries_id_list
        self.queries_list = queries_list
        self.cid_corpus_dict = corpus
        self.corpus_ids = list(corpus.keys())
        self.corpus_list = [corpus[cid] for cid in self.corpus_ids]
        # 每一个cid对应在列表中的idx
        self.cid_to_idx = {cid: idx for idx,cid in enumerate(self.corpus_ids)}
        self.relevant_docs = relevant_docs
        self.score_function = cos_sim      
        self.id_corpus_emb_dict = id_corpus_emb_dict
        self.retriever_type = retriever_type
        self.output_path = output_path  
        self.write_csv = write_csv
        self.phase = phase
        if self.phase == 'test':
            if self.retriever_type == 'bm25':
                self.bm25 = bm25
            elif self.retriever_type == 'TF-IDF':
                self.tf_idf = tf_idf
            else:
                if self.retriever_type in ["domain_bert",'ToolRetriever']:
                    self.model = model
                    self.query_embeddings = self.model.encode(
                        self.queries_list,
                        show_progress_bar=False,
                        batch_size=10,
                        convert_to_tensor=True,
                        device=self.device
                    )
                elif self.retriever_type == "ada":
                    self.query_qid_dict = {query:int(qid) for query,qid in zip(self.queries_list,self.queries_id)}
                    if not self.write_csv:
                        self.queries_list,self.query_embeddings = create_ada_embedding(self.queries_list,output_path=None,type='no_save')
                    else:
                        self.queries_list,self.query_embeddings = create_ada_embedding(self.queries_list,output_path=f'{self.output_path}/{self.phase}_ada_emb.json',type='save')        
                    if not isinstance(self.query_embeddings, torch.Tensor):
                        self.query_embeddings = torch.tensor(self.query_embeddings, dtype=torch.float32)
                        self.query_embeddings = self.query_embeddings.to(self.device)
                    self.queries_id = [self.query_qid_dict[query] for query in self.queries_list]

                with torch.no_grad():
                    corpus_ids = []
                    corpus_vals = []
                    for cid, emb in self.id_corpus_emb_dict.items():
                        corpus_ids.append(cid)
                        if isinstance(emb, torch.Tensor):
                            emb_cpu = emb.cpu().numpy()  
                        elif isinstance(emb, (np.ndarray, list)):
                            emb_cpu = np.array(emb)
                        else:
                            raise TypeError(f"Unsupported embedding type: {type(emb)}")
                            
                        corpus_vals.append(emb_cpu)  

                    stacked_embeddings = np.stack(corpus_vals, axis=0) 
                    self.corpus_embeddings = torch.tensor(
                        stacked_embeddings,
                        dtype=torch.float32,
                        device=self.device
                    )
                    self.corpus_ids = corpus_ids
                    self.corpus_list = [corpus[cid] for cid in self.corpus_ids]

                
    def __call__(self,model=None,epoch=-1,steps=-1,*args, **kwargs) :
        if self.phase == 'train':
            self.model = model
            self.query_embeddings = self.model.encode(
                self.queries_list,
                show_progress_bar=False,
                batch_size=5,
                convert_to_tensor=True,
            )
            self.corpus_embeddings = self.model.encode(
                self.corpus_list,
                show_progress_bar=False,
                batch_size=5,
                convert_to_tensor=True,
            )
            self.id_corpus_emb_dict = {}
            for cid,cemb in zip(self.corpus_ids,self.corpus_embeddings):
                self.id_corpus_emb_dict[cid] = cemb
            with torch.no_grad():
                corpus_tensors = []
                corpus_ids = []
                for corpus_id, emb in self.id_corpus_emb_dict.items():
                    tensor_emb = torch.tensor(emb, dtype=torch.float32)
                    corpus_tensors.append(tensor_emb)
                    corpus_ids.append(corpus_id)  
                self.corpus_ids = corpus_ids             
                self.corpus_embeddings = torch.stack(corpus_tensors).to('cuda')
                self.corpus_list = [self.cid_corpus_dict[cid] for cid in self.corpus_ids]

        queries_result_list = [[] for _ in range(len(self.queries_list))]
        scores_list = []
        if self.retriever_type in ['domain_bert','ada','ToolRetriever']:
            with torch.no_grad():
                all_scores = self.score_function(self.query_embeddings, self.corpus_embeddings)
                all_scores = all_scores.cpu().numpy()
                sorted_indices = np.argsort(-all_scores, axis=1)
                for query_idx in range(len(self.queries_list)):
                    query_sorted_indices = sorted_indices[query_idx] 
                    queries_result_list[query_idx] = [
                        {
                            "corpus_id": self.corpus_ids[doc_idx],
                            "score": float(all_scores[query_idx][doc_idx])
                        }
                        for doc_idx in query_sorted_indices
                    ]          

        elif self.retriever_type == 'bm25':
            for query_idx, query in enumerate(self.queries_list):
                tokenized_query = query.split()
                doc_scores = self.bm25.get_scores(tokenized_query)
                for doc_idx, score in enumerate(doc_scores):
                    corpus_id = self.corpus_ids[doc_idx]
                    queries_result_list[query_idx].append({
                        "corpus_id": corpus_id,
                        "score": round(float(score),2)
                    })
        elif self.retriever_type == 'TF-IDF':
            for query_idx, query in enumerate(self.queries_list):
                doc_scores = self.tf_idf.get_score(query)
                for doc_idx, score in enumerate(doc_scores):
                    corpus_id = self.corpus_ids[doc_idx]
                    queries_result_list[query_idx].append({
                        "corpus_id": corpus_id,
                        "score": round(float(score),2)
                    })

        scores_list = self.compute_mertrics(queries_result_list)
        if self.write_csv:
            ndcg = self.save_outcome(scores_list,epoch,steps)
        if self.phase == 'test':
            return scores_list
        elif self.phase == 'train':
            print(ndcg)
            return np.mean(ndcg)    
        
    def compute_mertrics(self,queries_result_list):    
        scores_list = []
        for query_idx in range(len(queries_result_list)):
            query_id = self.queries_id[query_idx]
            top_hits = queries_result_list[query_idx] 
            sig_dict = self.compute_ndcg_for_query((query_idx,int(query_id),top_hits))
            scores_list.append(sig_dict)
        return scores_list
        
            

    def compute_ndcg_for_query(self,query_tuple):
        k_list = [5, 10]
        sig_dict = {}     
        q_idx, query_id, top_hits = query_tuple
        query_relevant_docs = self.relevant_docs[query_id]
        true_relevance = np.zeros(len(self.corpus_ids))
        predicted_scores = np.zeros(len(self.corpus_ids))
        for hit in top_hits:
            predicted_scores[self.cid_to_idx[int(hit["corpus_id"])]] = hit["score"]
            if int(hit["corpus_id"]) in query_relevant_docs or hit["corpus_id"] in query_relevant_docs:
                true_relevance[self.cid_to_idx[int(hit["corpus_id"])]] = 1
        sig_dict['id'] = query_id
        sig_dict['query'] = self.queries_list[q_idx]
        for k in k_list:
            sig_dict[f'ndcg{k}'] = ndcg_score([true_relevance], [predicted_scores], k=k)
        sig_dict['ndcg'] = np.mean(sig_dict['ndcg5'] + sig_dict['ndcg10'])
        return sig_dict
    
    
    def save_outcome(self,scores_list,epoch,steps):
        ndcg5,ndcg10 = [],[]
        for score in scores_list:
            ndcg5.append(score['ndcg5'])
            ndcg10.append(score['ndcg10'])
        avg_ndcg = [sum(ndcg5)/len(ndcg5),sum(ndcg10)/len(ndcg10)]
        json_file = os.path.join(self.output_path, f"{self.phase}_Retrieval_evaluation_results.json")
        with open(json_file, 'w') as f:
            average_ndcg = sum(avg_ndcg) / len(avg_ndcg)
            json_data = {"average_ndcg": average_ndcg}
            json.dump(json_data,f, indent=4)
        csv_path = os.path.join(self.output_path, f"{self.phase}_Retrieval_evaluation_results.csv")
        csv_headers = [
            "epoch",
            "steps",
            "Average NDCG@5",
            "Average NDCG@10"
        ]
        if not os.path.isfile(csv_path):
            fOut = open(csv_path, mode="w", encoding="utf-8")
            fOut.write(",".join(csv_headers))
            fOut.write("\n")
        else:
            fOut = open(csv_path, mode="a", encoding="utf-8")
        output_data = [epoch, steps]
        output_data.append(avg_ndcg)
        fOut.write(",".join(map(str, output_data)))
        fOut.write("\n")
        fOut.close()
        return avg_ndcg