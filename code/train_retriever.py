import logging
import os
import json
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
from src.api_evaluator import APIEvaluator
import argparse
from src.utils import process_retrieval_ducoment,initialize_retriever,read_query
import csv
import gc
# os.environ["CUDA_HOME"] = "/usr/local/cuda-11.1"
# os.environ["CUDA_PATH"] = "/usr/local/cuda-11.1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/dat03/zly/ToolPlanner/BridgeRetrieval/data/retriever/G3/fuzzy_hybrid/inference/sft_llama_2_7b_lr_5e-5/checkpoint-5000_1')
    parser.add_argument("--model_path", default='/dat03/zly/ToolPlanner/model/orig/bert-base-uncased', type=str)
    parser.add_argument("--output_path", default='/dat03/zly/ToolPlanner/BridgeRetrieval/output/G3/retriever/fuzzy_hybrid/inference/sft_llama_2_7b_lr_5e-5/checkpoint-5000_1', type=str)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_steps", default=500, type=float)
    parser.add_argument("--max_seq_length", default=256, type=int)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    lr = args.learning_rate
    warmup_steps = args.warmup_steps
    # data_path = args.data_dir
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    model_save_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(model_save_path, exist_ok=True)
    train_samples = []  
    for data_path in args.data_dir:
        ir_train_queries = {}       
        train_queries_df = pd.read_csv(os.path.join(data_path, 'train.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
        ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25 = initialize_retriever(
                    args.model_path, 
                    retriever_type="domain_bert",
                    output_dir=args.data_dir,
                    phase='train',
                )
        for row in train_queries_df.itertuples():
            ir_train_queries[row.qid] = row.query
        labels_df = pd.read_csv(os.path.join(data_path, 'qrels.train.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
        for row in labels_df.itertuples():
            try:
                sample = InputExample(texts=[ir_train_queries[row.qid], ir_corpus[row.docid]], label=row.label)
                train_samples.append(sample)
            except Exception as e:
                continue
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    train_loss = losses.MultipleNegativesRankingLoss(reward_model)
    ir_corpus,ir_relevant_docs,id_corpus_emb_dict,_,_ = initialize_retriever(
                args.model_path, 
                retriever_type="domain_bert",
                output_dir=args.data_dir,
                phase='test',
            )
    test_qid_list,test_query_list=read_query(os.path.join(data_path, 'test.query.txt'))    
    ir_evaluator = APIEvaluator(test_query_list,test_qid_list, ir_corpus, ir_relevant_docs,_,retriever_type='domain_bert',write_csv=True,output_path=args.output_path,phase='train')

    reward_model.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=ir_evaluator,
                    epochs=num_epochs,
                    warmup_steps=warmup_steps,
                    optimizer_params={'lr': lr},
                    output_path=model_save_path
                    )