import argparse
import torch
import random
import numpy as np
import logging
import os
logger = logging.getLogger(__name__)
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import get_vocab_SST2, get_vocab_CliniSTS, get_vocab_QNLI, word_normalize
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
# from SanText import SanText_plus,SanText_plus_init

from SanText import SanText_TEM_init, SanText_TEM
import copy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # 实验室GPU选0或者1

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def negative_2D(matrix):                      # 矩阵内元素变成负数
    matrix = copy.deepcopy(matrix)            # 不对可变对象的原参数值进行修改    
    arr = np.array(matrix)
    return -arr

def cal_Lw(all_word_embed, sensitive_word_embed, gamma):                   # 计算单词的Lw和其概率
    distance = euclidean_distances(all_word_embed, sensitive_word_embed)   # 先计算距离
    score = negative_2D(distance)
    Lw = []                             # 用来包含距离小于gamma的单词的索引
    __Lw = []                           # 用来包含距离大于gamma的单词的索引
    score_list = []                     # 用来记录Lw的分数
    Lw_dict = {}                        # 单词与其对应的Lw
    __Lw_dict = {}                      # 单词与其对应的W\Lw
    Lw_score = {}                       # Lw_dict与其对应的分数
    print("Start Lw ......")            # distance的格式为(len(all_word_embed),len(sensitive_word_embed)) 
    for i in range(len(distance)):           # len(all_word_embed)
        for j in range(len(distance[i])):    # len(sensitive_word_embed)
            if distance[i][j] <= gamma :
                Lw.append(j)
                score_list.append(score[i][j])
            else:
                __Lw.append(j)
        Lw_dict[i] = copy.deepcopy(Lw) 
        __Lw_dict[i] = copy.deepcopy(__Lw)
        Lw_score[i] = copy.deepcopy(score_list)
        Lw.clear()
        __Lw.clear()
        score_list.clear()
    return Lw_dict, __Lw_dict, Lw_score

def main():
    import datetime
    starttime = datetime.datetime.now()

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/SST-2/",
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_SanText/QNLI/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/glove.840B.300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert'],
        default='glove',
        help='embedding used for sanitization'
    )

    parser.add_argument('--task',
                        choices=['CliniSTS', "SST-2", "QNLI"],
                        default='SST-2',
                        help='NLP eval tasks')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=15, help="privacy parameter epsilon")
    parser.add_argument("--p", type=float, default=0.2, help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.5,
                        help="SanText+: how many words are treated as sensitive")

    parser.add_argument("--threads", type=int, default=12, help="number of processors")

    parser.add_argument("--gamma", type=float, default=7, help="distance parameter gamma")

    args = parser.parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Running method: %s, task: %s,  epsilon = %s, random_seed: %d" % (
    args.method, args.task, args.epsilon, args.seed))

    if args.method == "SanText":
        args.sensitive_word_percentage = 1.0
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)
    else:
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon, "sword_%.2f_p_%.2f"%(args.sensitive_word_percentage,args.p), "gam_%.2f" % args.gamma)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type=="glove":
        tokenizer = English()
        tokenizer_type="word"
    else:
        tokenizer  = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
    if args.task == "SST-2":   # 目录，分词器，分词类型
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "CliniSTS":
        vocab = get_vocab_CliniSTS(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        raise NotImplementedError

    # vocab包含数据集中每个单词及其数量，如'profession': 3

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]         # 把每个单词由多到少排列（出现频次多在前）
    sensitive_words = words[-sensitive_word_count - 1:]     # 把敏感单词由多到少排列（出现频次多在前）

    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}  # 把敏感单词附上从0到n的id（n+1为敏感单词总数量）
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words),len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed=[]

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0
    if args.embedding_type == "glove":
        num_lines = sum(1 for _ in open(args.word_embedding_path))

        # args.word_embedding_path 的路径为 ./data/glove.840B.300d.txt
        # 这里用于循环的_由单个token和其300维词向量拼接而成
        # num_lines=2196017，是glove.840B.300d.txt的行数

        logger.info("Loading Word Embedding File: %s" % args.word_embedding_path)

        with open(args.word_embedding_path) as f:
            # Skip first line if of form count/dim.    # 如表格数目增加/减少，请跳过第一行。
            line = f.readline().rstrip().split(' ')    # 读取f中的一整行，再删除行末尾的指定字符，最后用空格划分
            if len(line) != 2:
                f.seek(0)                              # 重新设置文件读取指针到开头
            for row in tqdm(f, total=num_lines - 1):
                content = row.rstrip().split(' ')      # row由单词和其300维词向量拼接而成，content删除了row末尾的指定字符并用空格划分
                cur_word=word_normalize(content[0])    # cur-word为row的单词
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count                   # word2id表示数据集中的所有单词及其下标，如'my': 41（无重复单词）
                    all_count += 1
                    emb=[float(i) for i in content[1:]]             # emb表示单词的300维嵌入
                    all_word_embed.append(emb)                      # all_word_embed表示word2id的嵌入，并且一一对应
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count        # sword2id表示word2id中所有敏感单词及其下标，如'area': 28（无重复单词）
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)            # sensitive_word_emded表示sword2id的嵌入
                assert len(word2id)==len(all_word_embed)            # 当word2id的长度不等于all_word_embed时触发异常
                assert len(sword2id) == len(sensitive_word_embed)   # 当sword2id的长度不等于sensitive_word_embed时触发异常
            f.close()
    else:
        logger.info("Loading BERT Embedding File: %s" % args.bert_model_path)
        model=BertForMaskedLM.from_pretrained(args.bert_model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)

    all_word_embed=np.array(all_word_embed, dtype='f')                # 创建数组 all_word_embed，规模如(14283, 300)
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')  # 创建数组 sensitive_word_embed，规模如(12816, 300)

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    logger.info("Calculating Prob Matrix for Exponential Mechanism...")
    # prob_matrix = cal_probability(all_word_embed,sensitive_word_embed, args.epsilon) # 概率矩阵prob_matrix，规模如(14283, 12816)

    threads = min(args.threads, cpu_count())

    Lw_dict, __Lw_dict, Lw_score = cal_Lw(all_word_embed, sensitive_word_embed, args.gamma)
    print("Lw Complete ......")
    
    for file_name in ['train.tsv','dev.tsv']:
        data_file = os.path.join(args.data_dir, file_name)
        out_file = open(os.path.join(args.output_dir, file_name), 'w')
        logger.info("Processing file: %s. Will write to: %s" % (data_file,os.path.join(args.output_dir, file_name)))

        num_lines = sum(1 for _ in open(data_file))

        # 这里循环的_表示train.tsv（或dev.tsv）中每一行
        # num_lines为当前文件的行数

        with open(data_file, 'r') as rf:
            # header
            header = next(rf)                               # header为数据集最上方sentence label
            out_file.write(header)                          # 在输出的文档最上方加上sentence label
            labels = []
            docs = []
            if args.task == "SST-2":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[0]
                    label = int(content[1])
                    if args.embedding_type == "glove":
                        doc = [token.text for token in tokenizer(text)]     # doc就是text分词后得到的token组成的数组
                    else:
                        doc = tokenizer.tokenize(text)
                    docs.append(doc)                                        # docs就是doc的集合
                    labels.append(label)                                    # labels就是label的集合
            elif args.task == "CliniSTS":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[7]
                    text2 = content[8]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)
                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)
            elif args.task == "QNLI":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[1]
                    text2 = content[2]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)

                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)

            rf.close()

        # with Pool(threads, initializer=SanText_plus_init, initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
        #     annotate_ = partial(
        #         SanText_plus,
        #     )
        #     results = list(
        #         tqdm(
        #             p.imap(annotate_, docs, chunksize=32),
        #             total=len(docs),
        #             desc="Sanitize docs using SanText",
        #         )
        #     )
        #     p.close()
        
        with Pool(threads, initializer=SanText_TEM_init, initargs=(word2id, args.gamma, Lw_dict, __Lw_dict, Lw_score, args.epsilon, words, tokenizer, args.p, sword2id)) as p:
            annotate_ = partial(
                SanText_TEM,
            )
            results = list(
                tqdm(
                    p.imap(annotate_, docs, chunksize=32),
                    total=len(docs),
                    desc="Sanitize docs using SanText_TEM",
                )
            )
            p.close()
#         print("TEM_init")
#         TEM_init(word2id, all_word_embed, args.gamma, args.epsilon, words, tokenizer)
#         print("init complate")
#         Truncated_Exponential_Mechanism(docs)

        if args.task == "SST-2":
            for i, predicted_text in enumerate(results):
                write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "CliniSTS":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + "none\t" * 6 + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "QNLI":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)

        out_file.close()

    endtime = datetime.datetime.now()
    sec_time = (endtime - starttime).seconds
    min_time, sec_time = divmod(sec_time, 60)
    hou_time, min_time = divmod(min_time, 60)
    logger.info("运行时间：%02d:%02d:%02d" % (hou_time, min_time, sec_time))

if __name__ == "__main__":
    main()
















