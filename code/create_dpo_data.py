import json
import argparse
from src.utils import *
import threading
from tqdm import tqdm
from collections import defaultdict
import ijson
from multiprocessing import Pool, Manager
from functools import partial
from src.api_evaluator import APIEvaluator
import os
import torch
import multiprocessing
from multiprocessing import Pool
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_HOME"] = "/usr/local/cuda-11.1"
# os.environ["CUDA_PATH"] = "/usr/local/cuda-11.1"
def my_function(batch, dpo_data, n_sample, score_data, ir_corpus, ir_relevant_docs,retriever_type,bm25=None, if_idf=None,id_corpus_emb_dict=None, reward_model=None,output_path=None):
    """
    处理单个批次的数据
    Args:
        batch: 当前批次的数据
        dpo_data: 共享的DPO数据列表
        n_sample: 每个样本的生成数量
        score_data: 共享的分数数据列表
        ir_corpus: {tool_id:content}
        ir_relevant_docs: {qid: (tool_id)}
        BM25 :bm25模式使用
        id_corpus_emb_dict: {tool_id:embedding}(domain_bert和ada模式使用)
        reward_model: 奖励模型（domain_bert模式使用）
    """
    try:
        sentence = []
        query_id = []
        instruction = []
        id_instruction = {}

        for d in batch:
            sentence.append(d['orig_output'])
            query_id.append(int(d['id']))
            instruction.append(d['instruction'])
            # id->instruction(本质上是与fuzzy对应)
            id_instruction[int(d['id'])] = d['instruction']
            
            if n_sample == 1:
                sentence.append(d['new_output'])
                query_id.append(d['id'])
                instruction.append(d['instruction'])
            else:
                for i in range(len(d['new_output'])):                               
                    new_sentence = d['new_output'][i]
                    sentence.append(new_sentence)
                    query_id.append(d['id'])
                    instruction.append(d['instruction'])
        ir_evaluator=APIEvaluator(sentence,query_id,ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,if_idf,retriever_type,output_path,write_csv=False,phase='test')
        score_list = ir_evaluator()
        grouped_scores = defaultdict(list)
        for item in score_list:
            query_id = int(item['id'])
            grouped_scores[query_id].append(item)
        # 处理每个组的分数
        torch.cuda.empty_cache()
        for query_id, group in grouped_scores.items():
            orig_score = group[0]['ndcg']
            orig_query = group[0]['query']
            max_score = orig_score
            min_score = orig_score
            best_query = orig_query
            worst_query = orig_query
            for item in group:  
                score_data.append({
                    "query_id": query_id,
                    "query": item['query'],
                    "score": item['ndcg']
                })
                
                ndcg_score = item['ndcg']
                query_text = item['query']

                if ndcg_score > max_score:
                    max_score = ndcg_score
                    best_query = query_text

                if ndcg_score < min_score:
                    min_score = ndcg_score
                    worst_query = query_text

            if best_query != worst_query or max_score != min_score:
            
                dpo_sample = {
                    "query_id": query_id,
                    "conversations": [
                        {
                            "from": "human",
                            "value": id_instruction[query_id]
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": best_query
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": worst_query
                    }
                }

                dpo_data.append(dpo_sample)
    except Exception as e:
        print(e)




def create_dpo_data(args):
    """
    创建DPO数据集
    Args:
        args: 命令行参数
    Returns:
        tuple: (dpo_data, score_data)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if device == 'cpu':
    with Manager() as manager:
        # 创建共享数据结构
        dpo_data = manager.list()
        score_data = manager.list()
        
        ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,if_idf = initialize_retriever(
            args.model_path, 
            retriever_type=args.retriever_type,
            output_dir=args.retriever_dir,
            phase='train'
        )
        if args.retriever_type not in  ['bm25','TF-IDF'] :
            cpu_id_corpus_emb_dict = {
            cid: emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.array(emb)
            for cid, emb in id_corpus_emb_dict.items()
        }

            # 使用普通字典传递数据
            shared_data = {
                'ir_corpus': dict(ir_corpus),
                'ir_relevant_docs': dict(ir_relevant_docs),
                'id_corpus_emb_dict': cpu_id_corpus_emb_dict
            }
        else:
            shared_data = {
                'ir_corpus': dict(ir_corpus),
                'ir_relevant_docs': dict(ir_relevant_docs),
                'id_corpus_emb_dict': id_corpus_emb_dict
            }
        print("Counting total samples...")
        with open(args.inference_json, 'rb') as f:
            total_count = sum(1 for _ in ijson.items(f, 'item'))
        total_batches = (total_count + args.batch_size - 1) // args.batch_size
        print(f"Total samples: {total_count}, Total batches: {total_batches}")

        process_func = partial(
            my_function,
            dpo_data=dpo_data,
            n_sample=args.n_sample,
            score_data=score_data,
            ir_corpus=shared_data['ir_corpus'],
            ir_relevant_docs=shared_data['ir_relevant_docs'],
            retriever_type=args.retriever_type,
            bm25=bm25,
            if_idf=if_idf,
            id_corpus_emb_dict=shared_data['id_corpus_emb_dict'],
            reward_model=reward_model,
            output_path=os.path.dirname(args.dpo_json),
        )

        with Pool(processes=60) as pool:
            list(tqdm(
                pool.imap(
                    process_func,
                    batch_read_json(args.inference_json, args.batch_size),
                ),
                total=total_batches,
                position=0,
                leave=True, 
                desc="Processing batches"
            ))
        print("Saving results...")
        with open(args.dpo_json, 'w') as out_f:
            json.dump(list(dpo_data), out_f, ensure_ascii=False, indent=4)
        print(f"DPO dataset has been saved to {args.dpo_json}")
        
        with open(args.score_json, 'w') as out_f:
            json.dump(list(score_data), out_f, ensure_ascii=False, indent=4)
        print(f"Score dataset has been saved to {args.score_json}")
            
    # else:
    #     dpo_data = []
    #     score_data = []       
    #     ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25 = initialize_retriever(
    #         args.model_path,
    #         retriever_type=args.retriever_type,
    #         output_dir=args.retriever_dir,
    #         phase='train'
    #     )        
    #     print("Counting total samples...")
    #     with open(args.inference_json, 'rb') as f:
    #         total_count = sum(1 for _ in ijson.items(f, 'item'))
    #     total_batches = (total_count + args.batch_size - 1) // args.batch_size
    #     for batch in tqdm(batch_read_json(args.inference_json, args.batch_size),
    #                         total=total_batches,
    #                         desc="Processing batches"):
    #         my_function(
    #             batch=batch,
    #             dpo_data=dpo_data,
    #             n_sample=args.n_sample,
    #             score_data=score_data,
    #             ir_corpus=ir_corpus,
    #             ir_relevant_docs=ir_relevant_docs,
    #             retriever_type=args.retriever_type,
    #             bm25=bm25,
    #             id_corpus_emb_dict=id_corpus_emb_dict,
    #             reward_model=reward_model,
    #             output_path=os.path.dirname(args.dpo_json)
    #         )
    #     print("Saving results...")
    #     with open(args.dpo_json, 'w') as out_f:
    #         json.dump(list(dpo_data), out_f, ensure_ascii=False, indent=4)
    #     print(f"DPO dataset has been saved to {args.dpo_json}")
        
    #     with open(args.score_json, 'w') as out_f:
    #         json.dump(list(score_data), out_f, ensure_ascii=False, indent=4)
    #     print(f"Score dataset has been saved to {args.score_json}")
        

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_dir', type=str, 
                       default='BridgeRetrieval/data/retriever/G3/hybrid')
    parser.add_argument('--inference_json', type=str, 
                       default='/hpc2hdd/home/yingz/zly/Toolplanner/BridgeRetrieval/data/bridge_model/G3/fuzzy_hybrid/Llama-3.2-3B/train_4.json',
                       help='input file path')
    parser.add_argument('--dpo_json', type=str,
                       default='BridgeRetrieval/data/bridge_model/G3/fuzzy_hybrid/Llama-3.2-3B/train_4/bm25_dpo_data/retriever_dpo_train.json',
                       help='output file path')
    parser.add_argument('--score_json', type=str,
                       default='BridgeRetrieval/data/bridge_model/G3/fuzzy_hybrid/Llama-3.2-3B/train_4/bm25_dpo_data/retriever_score.json')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--model_path', type=str,
                       default='/dat03/zly/ToolPlanner/zly_data/bridge_model/G3/fuzzy_llmhybrid_new/llama_2_7b_15000')
    parser.add_argument('--n_sample', type=int, default=4)
    parser.add_argument('--retriever_type', type=str, default='bm25')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.dpo_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.score_json), exist_ok=True)
    create_dpo_data(args)