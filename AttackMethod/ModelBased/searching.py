from encodings import search_function
import torch
import torch.nn as nn
import numpy as np
# from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from AttackMethod.PackDataset import packDataset_util


class SearchingMethod():
    def __init__(self, victim, tokenizer,access_info='score', searching_method='greedy'):
        self.access_info = access_info
        self.search_method = searching_method
        self.victim = victim
#         self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer = tokenizer


    def search(self, sample):
        '''

        :param victim:
        :param sample:
        :return: a position that should be perturbed
        '''
        
        # bert_type = "bert-base-cased"

        
        saliency_score = self.access_saliency_score(sample)
        if self.search_method == 'greedy':
            # return min(saliency_score, key=saliency_score.get)
            # print("-------------------")
            return [k for k, _ in sorted(saliency_score.items(), key=lambda item: item[1])]
        elif self.search_method == 'pso':
            pass
        else:
            print("Invalid searching method")




    def access_saliency_score(self, sample):
        '''

        :param sample: a sentence
        :param victim: the victim model (Huggingface Classifier)
        :return: a list, containing the saliency score of every words in the sample
        '''
        # sentence_tokens = self.tokenizer.tokenize(sample)
        sentence_tokens = sample.split()
        
        if self.access_info == 'gradient':
            pass
        elif self.access_info == 'score' or 'decision':
            return self.sentence_score(sentence_tokens,self.access_info)
#         elif self.access_info == 'decision':
#             pass
        else:
            print("Invalid Access Information")
            return None

        
    def sentence_score(self, sentence_tokens,access_info):
        # target = self.get_pred([self.tokenizer.convert_tokens_to_string(sentence_tokens)])[0]
        target = self.get_pred([' '.join(sentence_tokens)])[0]
        word_losses = {}
        sentence_without = ['']*len(sentence_tokens)
        for i in range(len(sentence_tokens)):
            sentence_tokens_without =  sentence_tokens[:i] +["[MASK]"]+ sentence_tokens[i + 1:]
            sentence_without[i] = ' '.join(sentence_tokens_without)
        if access_info == 'score':
            tempoutput = self.get_prob(sentence_without)
        elif access_info == 'decision':
            tempoutput = self.get_decision(sentence_without)
        elif access_info == 'gradient':
            tempoutput = self.get_gradient(sentence_without)
        for i in range(len(sentence_tokens)):
            word_losses[i] = tempoutput[i][target] 


        return word_losses


    # access to the classification probability scores with respect input sentences
    # def get_prob(self, sentences):
    #     inputs = self.tokenizer(sentences, return_tensors='pt', return_length=512, truncation=True, padding=True)
    #     input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    #     if torch.cuda.is_available():
    #         input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
    #     # print(inputs_ids.shape, attention_mask.shape)

    #     # print(self.model.device)
    #     outputs = self.victim(input_ids, attention_mask).logits      
    #     outputs = outputs.detach().cpu().numpy()
    #     predicts = outputs.argmax(axis=1).tolist()
    #     return outputs
    
    def get_pred(self, sentences):
            return self.get_prob(sentences).argmax(axis=1)


    def get_prob(self, sentences):
        batch_size = 200
        # print(len(aug_sents))
        sents_list = [(each,0) for each in sentences]
        
        pack_util = packDataset_util(self.tokenizer)
        sents_loader = pack_util.get_loader(sents_list, shuffle=False, batch_size=batch_size)
        
        outputs =  []
        self.victim.eval()
        with torch.no_grad():
            for padded_text, attention_masks, labels in sents_loader:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels =  padded_text.cuda(),  attention_masks.cuda(),  labels.cuda()
                # print(victim_model.device)
                # print(padded_text.device,attention_masks.device, labels.device)
                output = self.victim(padded_text, attention_masks).logits
                output = output.detach().cpu().tolist()
                # print(output)
                outputs.extend(output)
        # print(outputs[0])
        self.victim.zero_grad()
        return np.array(outputs)
    
    def get_gradient(self,sentences):
        model = self.victim
        model.eval()

        # 将模型的参数设为 "不需要梯度"， 节省空间存储和加快计算速度
        for param in model.parameters():
            param.requires_grad = False
        # 设置word-embedding 层的参数为 "需要梯度"
        for param in model.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = True

        inputs = self.tokenizer(inputs, return_tensors='pt')
        output = model(**inputs).logits



        labels = torch.tensor([1])
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, labels)
        model.zero_grad()
        loss.backward()

        # obtain gradients in word-embedding layer

        for param in model.bert.embeddings.word_embeddings.parameters():
            embedding_grad = param.data.grad






    def get_decision(self, sentences):
        pred = self.get_pred(sentences)
        decisions = [0] * len(self.get_prob([sentences])[0])
        decisions[pred] = 1 
        return decisions






