





from math import degrees
import os
import sys
import torch
from turtle import distance
import nltk
import math
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool


# nltk.download('averaged_perceptron_tagger')
# sys.path.append('/data/private/gaohongcheng/BenchmarkRobustness-NLP-main/AttackMethod')
from AttackMethod.ModelBased.searching import SearchingMethod
from AttackMethod.RuleBased.Char.typo_transform import TypoTransform
from AttackMethod.RuleBased.Char.glyph_transform import GlyphTransform
# from AttackMethod.RuleBased.Char.phonetic_transform import PhoneticTransform
from AttackMethod.RuleBased.Word.synonym_transform import SynonymTransform
import random
from AttackMethod.RuleBased.Word.synonym_transform import SynonymTransform
from AttackMethod.RuleBased.Word.contextual_transform import ContextualTransform
from AttackMethod.RuleBased.Sentence.distraction_transform import DistractionTransform
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluator import EditDistance
import multiprocessing

transformer_dict = {
    # "phonetic": PhoneticTransform,
    "typo": TypoTransform,
    "glyph": GlyphTransform,
    "phonetic": PhoneticTransform,
    "synonym": SynonymTransform,
    "contextual": ContextualTransform,
    "distraction": DistractionTransform
}




def load_rule_transformer(name, degree, dataset='', dis_type="char",sentence=''):
    if name=="phonetic":
        return transformer_dict[name](degree=degree,distance_type =dis_type,ori_sent=sentence)
    elif name == "distraction":
        return transformer_dict[name](degree = degree, dataset = dataset)
    else:
        return transformer_dict[name](degree=degree,ori_sent=sentence)

class ModelTransform():
    def __init__(self, victim_model, tokenizer, attacker ='word',dataset = "", degree=0.1, aug_num=1,batch_ratio=0.15,dis_type='char'):
        self.model = victim_model
        self.tokenizer = tokenizer
        self.degree = degree
        self.aug_num = aug_num
        self.attacker = attacker
        self.batch_ratio = batch_ratio
        self.dis_type = dis_type
        self.distance = EditDistance()
        self.dataset = dataset


    def transform(self, sentence,access_info='score', searching_method='greedy'):
        sentence=sentence.strip()
        search = SearchingMethod(self.model, self.tokenizer, access_info, searching_method)
        if self.dis_type=='char':
            self.transform_num = math.ceil(len(sentence) * self.degree)
        else:
            self.transform_num = math.ceil(len(sentence.split()) * self.degree)
        

        # sentence_tokens = self.tokenizer.tokenize(sentence)
        if self.attacker in ["distraction"]:
            sents = self.sentence_transform(sentence,search)
        else:
            sents = self.char_word_transform(sentence,search)

        # new_sentence = self.tokenizer.convert_tokens_to_string(new_sentence_tokens)
        return sents



    def char_word_transform(self,sentence,search):
        batch_size = math.ceil(len(sentence.split()) * self.batch_ratio)
        transformer = load_rule_transformer(name=self.attacker, degree=self.degree, dis_type=self.dis_type,sentence=sentence)

        # print(sentence)
        split = sentence.strip().split()
        sent_len = len(sentence.strip())
        if self.dis_type=="char":
            sent_len = len(sentence.strip())
        else:
            sent_len = len(split)
        sents = []
        sent = deepcopy(sentence)
        for _ in range(self.aug_num):
            for __ in range(int(self.transform_num*3/batch_size)):
                if self.distance(sentence, sent)/sent_len >= self.degree:
                    break 
                loc = search.search(sent)
                for i in range(batch_size):
                    candidate_sent = transformer.transform_target(sent,loc[i]) if self.dis_type=='word' else transformer.transform_char(sent,loc[i])
                    if candidate_sent is None:
                        continue   
                    sent = candidate_sent
                    if self.distance(sentence, sent)/sent_len >= self.degree:
                        break 
            if sent not in sents:
                sents.append(sent)
        del transformer
        return sents

    def ge(self,sentence):
        
        sent = deepcopy(sentence)
        candidate_sent = deepcopy(sentence)
        max_trial = 10
        for __ in range(max_trial):
            with torch.no_grad():
                use_score = self.tr.use.get_sim(sentence, candidate_sent)
            if use_score > self.use_thred:
                sent = candidate_sent
                # print(candidate_sent, use_score)
            else:
                break
                # random select one type of transformation
            new_sents=[]
            for ___ in range(10):
                transform_type = random.choice(self.tr.transform_types)
                self.tr.transformation = self.tr.TRANSFORMATION[transform_type]
                new_sents.append(self.tr.transformation(sent))
            word_losses = {}
            tempoutput = self.se.get_prob(new_sents)
            for i in range(len(new_sents)):
                word_losses[i] = tempoutput[i][self.ta] 
            loc = [k for k, v in sorted(word_losses.items(), key=lambda item: item[1])]
            candidate_sent = new_sents[loc[0]]
        return sent
        self.sents.append(sent)
    
    def sentence_transform(self,sentence,search):
        transformer = load_rule_transformer(name=self.attacker, degree=self.degree, dis_type=self.dis_type,sentence=sentence,dataset=self.dataset)
        self.sents = []        
        self.use_thred = 1 - self.degree
        
        target = search.get_pred([sentence])[0]
        self.se = search

        self.tr = transformer
        self.us = self.use_thred
        self.ta = target
        
        # pool_obj = multiprocessing.Pool()

        # sents = pool_obj.map(self.ge,[range(0,self.aug_num),target,search])
        # ctx = multiprocessing.get_context("spawn")
        mp.set_start_method('spawn',True)
        # mp.set_start_method('forkserver', force=True)
        # processes = []
        # for rank in range(self.aug_num):
        #     p = mp.Process(target=self.ge, args=(sentence,target,search,transformer))
        #     # We first train the model across `num_processes` processes
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        # print(self.sents)
        with Pool(processes=10) as pool:
            sents = pool.map(self.ge,[sentence]*self.aug_num)
        
        # for _ in range(self.aug_num):

            
            # if sent not in sents:
            #     sents.append(sent)
        # del transformer
        return sents


if __name__ == '__main__':
    bert_type = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    model = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=2)
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # if torch.cuda.is_available():
    model.cuda()
    mt = ModelTransform(model,tokenizer,0.2,2)
    sent = mt.transform("OK, it is very nice!","typo")
    print(sent)
    
