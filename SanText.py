
import random
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from functools import partial
from multiprocessing import Pool, cpu_count
import sympy 
from sympy import *

##############################       SanText       ##############################

# def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
#     distance = euclidean_distances(word_embed_1, word_embed_2)
#     sim_matrix = -distance
#     prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
#     return prob_matrix


# def SanText_init(prob_matrix_init,):
#     global prob_matrix
#     prob_matrix = prob_matrix_init

# def SanText(doc):
#     new_doc = []
#     for token in doc:
#         sampling_prob = prob_matrix[token]
#         sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
#         new_doc.append(sampling_index[0])
#     return new_doc


# def SanText_plus_init(prob_matrix_init, word2id_init, sword2id_init, all_words_init, p_init, tokenizer_init):
#     global prob_matrix
#     global word2id
#     global sword2id
#     global id2sword
#     global all_words
#     global p
#     global tokenizer

#     prob_matrix = prob_matrix_init
#     word2id = word2id_init
#     sword2id=sword2id_init                            # 前面是单词后面是序号

#     id2sword = {v: k for k, v in sword2id.items()}    # 把sword2id转过来了，前面是序号后面是单词

#     all_words = all_words_init
#     p=p_init
#     tokenizer=tokenizer_init

# def SanText_plus(doc):
#     new_doc = []
#     for word in doc:
#         if word in word2id:
#             # In-vocab
#             if word in sword2id:
#                 #Sensitive Words
#                 index = word2id[word]                                                     # 查看当前单词的id
#                 sampling_prob = prob_matrix[index]                                        # 用id查找概率矩阵中该单词与其他单词的概率
#                 sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob) # 按概率随机选择一个id
#                 new_doc.append(id2sword[sampling_index[0]])                               # 在new_doc上添加该id转变成的单词
#             else:
#                 #Non-sensitive words
#                 flip_p=random.random()
#                 if flip_p<=p:
#                     #sample a word from Vs based on prob matrix
#                     index = word2id[word]
#                     sampling_prob = prob_matrix[index]
#                     sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
#                     new_doc.append(id2sword[sampling_index[0]])
#                 else:
#                     #keep as the original
#                     new_doc.append(word)
#         else:
#             #Out-of-Vocab words
#             sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )                # 采样概率为1/n，n为vocab中全部单词数量
#             sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
#             new_doc.append(all_words[sampling_index[0]])

#     new_doc = " ".join(new_doc)
#     return new_doc


# def get_sanitized_doc(docs, embedding_matrix, epsilon=2.0, threads=12):
#     threads = min(threads, cpu_count())

#     prob_matrix=cal_probability(embedding_matrix, embedding_matrix, epsilon=epsilon,)

#     with Pool(threads, initializer=SanText_init, initargs=(prob_matrix,)) as p:
#         annotate_ = partial(
#             SanText,
#         )
#         results = list(
#             tqdm(
#                 p.imap(annotate_, docs, chunksize=32),
#                 total=len(docs),
#                 desc="Sanitize docs using SanText",
#             )
#         )
#         p.close()
    
#     return results


##############################       TEM       ##############################
# import copy
# def negative_2D(matrix):                    # 矩阵内元素变成负数
#     matrix = copy.deepcopy(matrix)            # 不对可变对象的原参数值进行修改    
#     arr = np.array(matrix)
#     return -arr

# def cal_Lw(all_word_embed, gamma=7.0):              # 计算单词的Lw和其概率
#     distance = euclidean_distances(all_word_embed, all_word_embed)   # 先计算距离
#     score = negative_2D(distance)
#     Lw = []                             # 用来包含距离小于gamma的单词的索引
#     __Lw = []                           # 用来包含距离大于gamma的单词的索引
#     Lw_dict = {}                        # 单词与其对应的Lw
#     __Lw_dict = {}                      # 单词与其对应的W\Lw
#     print("Start Lw ......")
#     for i in range(len(distance)):
#         for j in range(len(distance[i])):  
#             if distance[i][j] <= gamma :
#                 Lw.append(j)
#             else:
#                 __Lw.append(j)
#         Lw_dict[i] = Lw
#         __Lw_dict[i] = __Lw
#         Lw = []
#         __Lw = []
#     return Lw_dict, __Lw_dict, score

# def Gumbel_Sample(miu, epsilon, shape):
#     return np.random.gumbel(miu, 2/epsilon, shape)

# def TEM_init(word2id_init, gamma_init, Lw_dict_init, __Lw_dict_init, score_init, epsilon_init, all_words_init, tokenizer_init):
#     global word2id
#     global id2word
#     global gamma
#     global epsilon
#     global eps
#     global all_words
#     global tokenizer
#     global Lw_dict
#     global __Lw_dict
#     global score 
#     global miu

    
#     word2id = word2id_init
#     id2word = {v: k for k, v in word2id.items()}
#     gamma = gamma_init
#     epsilon = epsilon_init
#     eps = 1e-20
#     all_words = all_words_init
#     tokenizer = tokenizer_init
#     Lw_dict = Lw_dict_init
#     __Lw_dict = __Lw_dict_init
#     score = score_init
#     miu=-1*(2/epsilon)*(S.EulerGamma.n(8))


# def Truncated_Exponential_Mechanism(doc):  #TEM
#     new_doc = []
#     for word in doc:
#         if word in word2id:
#             # In-vocab

#             index = word2id[word]       # 当前单词的索引
#             Lw_score = Lw_dict[index]       # 得到gamma范围内各个单词的分数 
    
#             vertical_score = -1*gamma+2*np.log(len(score)-len(Lw_score)+eps)/epsilon
#             vertical_noise_score = vertical_score + Gumbel_Sample(miu, epsilon, 1 )   # 垂直元素⊥
#             # Lw_score.append(perpendicular_score)
            
#             Noise = Gumbel_Sample(miu, epsilon, [len(Lw_score),] )           
#             Lw_noise_score = np.sum([Noise,Lw_score],axis=0)
            
#             if vertical_noise_score<=max(Lw_noise_score):            # 输出范围内单词
#                 _id = Lw_noise_score.tolist().index(max(Lw_noise_score)) # 分数最大时当前单词的索引
#                 new_doc.append(id2word[Lw_dict[index][_id]]) 
#             else:                                      # 输出范围外单词
#                 sampling_prob = 1 / len(__Lw_dict[index]) * np.ones(len(__Lw_dict[index]), ) 
#                 sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
#                 word_id = __Lw_dict[index][sampling_index[0]]
#                 new_doc.append(id2word[word_id])                    
#         else:
#             #Out-of-Vocab words
#             sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )             # 采样概率为1/n，n为vocab中全部单词数量
#             sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
#             new_doc.append(all_words[sampling_index[0]])

#     new_doc = " ".join(new_doc)
#     return new_doc

##############################   SanText+TEM   ##############################
import copy
def negative_2D(matrix):                    # 矩阵内元素变成负数
    matrix = copy.deepcopy(matrix)            # 不对可变对象的原参数值进行修改    
    arr = np.array(matrix)
    return -arr

def cal_Lw(all_word_embed, gamma=7.0):              # 计算单词的Lw和其概率
    distance = euclidean_distances(all_word_embed, all_word_embed)   # 先计算距离
    score = negative_2D(distance)
    Lw = []                             # 用来包含距离小于gamma的单词的索引
    __Lw = []                           # 用来包含距离大于gamma的单词的索引
    Lw_dict = {}                        # 单词与其对应的Lw
    __Lw_dict = {}                      # 单词与其对应的W\Lw
    print("Start Lw ......")
    for i in range(len(distance)): 
        for j in range(len(distance[i])):  
            if distance[i][j] <= gamma :
                Lw.append(j)
            else:
                __Lw.append(j)
        Lw_dict[i] = Lw   
        __Lw_dict[i] = __Lw 
        Lw = []
        __Lw = []
    return Lw_dict, __Lw_dict, score

def Gumbel_Sample(miu, epsilon, shape):
    return np.random.gumbel(miu, 2/epsilon, shape)

def SanText_TEM_init(word2id_init, gamma_init, Lw_dict_init, __Lw_dict_init, Lw_score_init, epsilon_init, all_words_init, tokenizer_init, p_init, sword2id_init):
    global word2id
    global id2word
    global gamma
    global epsilon
    global eps
    global all_words
    global tokenizer
    global Lw_dict
    global __Lw_dict
    global Lw_score 
    global miu
    global p
    global sword2id

    word2id = word2id_init
    id2word = {v: k for k, v in word2id.items()}
    gamma = gamma_init
    epsilon = epsilon_init
    eps = 1e-20
    all_words = all_words_init
    tokenizer = tokenizer_init
    Lw_dict = Lw_dict_init
    __Lw_dict = __Lw_dict_init
    Lw_score = Lw_score_init
    miu=-1*(2/epsilon)*(S.EulerGamma.n(8))
    p = p_init
    sword2id = sword2id_init




def SanText_TEM(doc):  #TEM
    new_doc = []
    for word in doc:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                # sensitive words
                index = word2id[word]
                index_score = Lw_score[index]
                
                vertical_score = -1*gamma+2*np.log(len(word2id)-len(index_score)+eps)/epsilon
                vertical_noise_score = vertical_score + Gumbel_Sample(miu, epsilon, 1 )   # 垂直元素⊥
            
                Noise = Gumbel_Sample(miu, epsilon, [len(index_score),] )           
                Lw_noise_score = np.sum([Noise,index_score],axis=0)
            
                if vertical_noise_score<=max(Lw_noise_score):                # 输出范围内单词
                    _id = Lw_noise_score.tolist().index(max(Lw_noise_score)) # 分数最大时当前单词的索引
                    new_doc.append(id2word[Lw_dict[index][_id]])
                else:                                                        # 输出范围外单词
                    sampling_prob = 1 / len(__Lw_dict[index]) * np.ones(len(__Lw_dict[index]), ) 
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    word_id = __Lw_dict[index][sampling_index[0]]
                    new_doc.append(id2word[word_id])
            else:
                #Non-sensitive words
                flip_p=random.random()
                if flip_p<=p:

                    index = word2id[word]
                    index_score = Lw_score[index]       # 得到gamma范围内各个单词的分数 

                    if len(index_score)==0 :           # Lw不包含单词
                        sampling_prob = 1 / len(__Lw_dict[index]) * np.ones(len(__Lw_dict[index]), ) 
                        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                        word_id = __Lw_dict[index][sampling_index[0]]
                        new_doc.append(id2word[word_id])
                        print("3",id2word[word_id])
                    else :                          # Lw包含单词
                        vertical_score = -1*gamma+2*np.log(len(word2id)-len(index_score)+eps)/epsilon
                        vertical_noise_score = vertical_score + Gumbel_Sample(miu, epsilon, 1 )   # 垂直元素⊥
            
                        Noise = Gumbel_Sample(miu, epsilon, [len(index_score),] )           
                        Lw_noise_score = np.sum([Noise,index_score],axis=0)
            
                        if vertical_noise_score<=max(Lw_noise_score):                # 输出范围内单词
                            _id = Lw_noise_score.tolist().index(max(Lw_noise_score)) # 分数最大时当前单词的索引
                            new_doc.append(id2word[Lw_dict[index][_id]])
                        else:                                                        # 输出范围外单词
                            sampling_prob = 1 / len(__Lw_dict[index]) * np.ones(len(__Lw_dict[index]), ) 
                            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                            word_id = __Lw_dict[index][sampling_index[0]]
                            new_doc.append(id2word[word_id])
                else:
                    #keep as the original
                    new_doc.append(word)
           
        else:
            #Out-of-Vocab words
            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )            # 采样概率为1/n，n为vocab中全部单词数量
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(all_words[sampling_index[0]])

    new_doc = " ".join(new_doc)
    return new_doc








