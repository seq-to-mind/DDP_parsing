import numpy as np
import re
import global_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class NeuralParser(nn.Module):
    def __init__(self):
        super(NeuralParser, self).__init__()

        self.language_backbone = AutoModel.from_pretrained(global_config.pretrained_model, output_hidden_states=True)

        self.tokenizer = AutoTokenizer.from_pretrained(global_config.pretrained_tokenizer, use_fast=True)

        self.sentence_level_gru = nn.GRU(input_size=global_config.hidden_size, hidden_size=int(global_config.hidden_size), batch_first=True, bidirectional=True)

        self.linear_for_link = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(global_config.hidden_size * 4, global_config.hidden_size), nn.Dropout(p=0.3),
                                             nn.Tanh(), nn.Linear(global_config.hidden_size, 1))
        self.linear_for_relation = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(global_config.hidden_size * 4, global_config.hidden_size), nn.Dropout(p=0.3),
                                                 nn.Linear(global_config.hidden_size, global_config.num_relations, bias=False))

    def encoding_all_utterances(self, input_sequence_list, edu_number_list):
        input_sequence_list = [re.sub("\s+", " ", i) for i in input_sequence_list]
        current_batch_size = len(edu_number_list)

        batch_tokenized = self.tokenizer(input_sequence_list, return_tensors="pt", padding=True)
        batch_input_tensor = batch_tokenized.data["input_ids"].cuda()
        attention_mask = batch_tokenized["attention_mask"].cuda()
        batch_embedding_after_bert = self.language_backbone(batch_input_tensor, attention_mask=attention_mask)[0]

        batch_token_len_list = torch.sum(batch_tokenized["attention_mask"], dim=1).detach().numpy().tolist()
        batch_token_list = [batch_tokenized.encodings[i].tokens[:batch_token_len_list[i]] for i in range(current_batch_size)]

        output_embedding_list = []
        for i in range(current_batch_size):
            embedding_after_bert = batch_embedding_after_bert[i, :batch_token_len_list[i], :].unsqueeze(0)
            input_sentence = batch_token_list[i]
            assert len(input_sentence) < 512

            # selected_idx_list = [k for k, v in enumerate(input_sentence) if v in ["[CLS]", "[SEP]"]]
            # selected_idx_list = selected_idx_list[:-1]
            selected_idx_list = [k for k, v in enumerate(input_sentence) if v in ["[CLS]", "<s>"]]
            selected_idx_list = selected_idx_list

            assert len(selected_idx_list) == edu_number_list[i]

            if global_config.sentence_embedding_mode == "average":
                """ get utterance with average """
                selected_idx_list.append(embedding_after_bert.size(1))
                embedding_after_bert = torch.cat([torch.mean(embedding_after_bert[:, selected_idx_list[i]: selected_idx_list[i + 1], :], dim=1, keepdim=True)
                                                  for i in range(0, len(selected_idx_list) - 1)], dim=1)
            elif global_config.sentence_embedding_mode == "first":
                """ get utterance embedded representation by index selection """
                embedding_after_bert = embedding_after_bert[:, selected_idx_list, :]
            elif global_config.sentence_embedding_mode == "first_last":
                embedding_after_bert = (embedding_after_bert[:, selected_idx_list, :] + embedding_after_bert[:, [k for k, v in enumerate(input_sentence) if v in ["</s>"]], :]) / 2
            else:
                print("#ERROR# The global_config.sentence_embedding_mode is invalid:", global_config.sentence_embedding_mode)
                exit()

            embedding_after_gru = None
            if global_config.use_gru_after_sentence:
                tmp_encoded, _ = self.sentence_level_gru(embedding_after_bert)
                embedding_after_gru = tmp_encoded[:, :, : embedding_after_bert.size(2)] + tmp_encoded[:, :, embedding_after_bert.size(2):]

            output_embedding_list.append(torch.cat([embedding_after_bert, embedding_after_gru], dim=2))

        return output_embedding_list


class Model:
    def __init__(self):
        self.agent = NeuralParser()
        if global_config.use_cuda:
            self.agent.cuda()

        self.link_loss_function = nn.CrossEntropyLoss()
        self.relation_loss_function = nn.CrossEntropyLoss()

        if global_config.different_learning_rate:
            bert_param_ids = list(map(id, self.agent.language_backbone.parameters()))
            self.backbone_params = filter(lambda p: id(p) in bert_param_ids, self.agent.parameters())
            self.other_params = filter(lambda p: id(p) not in bert_param_ids, self.agent.parameters())
            self.optimizer = torch.optim.AdamW([{'params': self.backbone_params, 'lr': global_config.learning_rate},
                                                {'params': self.other_params, 'lr': 0.001}], lr=global_config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(params=self.agent.parameters(), lr=global_config.learning_rate)

        if global_config.freeze_some_bert_layer:
            for name, param in self.agent.language_backbone.named_parameters():
                layer_num = re.findall("layer\.(\d+)\.", name)
                if len(layer_num) > 0 and int(layer_num[0]) > 2:
                    print("Unfreeze layer:", int(layer_num[0]))
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def adjust_learning_rate(self, backbone_lr, other_lr):
        assert global_config.different_learning_rate
        print("learning rate is changed to:", backbone_lr, other_lr)
        self.optimizer.param_groups[0]["lr"] = backbone_lr
        self.optimizer.param_groups[1]["lr"] = other_lr

    def forward(self, batch, eval_mode=False):
        target_relation, link_start, link_end, input_text, edu_number = [], [], [], [], []
        for i in range(len(batch)):
            target_relation.append([j["type"] for j in batch[i]["relations"]])
            link_start.append([j["y"] for j in batch[i]["relations"]])
            link_end.append([j["x"] for j in batch[i]["relations"]])
            edu_number.append(len(batch[i]["edus"]))
            input_text.append([j["speaker"][:6] + ": " + " ".join(j["tokens"]) for j in batch[i]["edus"]])

        link_loss, relation_loss = None, None

        loss_accumulate_number = 0
        link_prediction, relation_prediction = [], []

        batch_sample_input_text = [global_config.sentence_seg_token.join(input_text[i]) for i in range(len(batch))]
        batch_sample_edu_reps = self.agent.encoding_all_utterances(batch_sample_input_text, edu_number_list=edu_number)

        for i in range(len(batch)):

            one_sample_target_relation = target_relation[i]
            one_sample_target_link = link_end[i]
            assert edu_number[i] == len(one_sample_target_relation) + 1 and edu_number[i] == len(one_sample_target_link) + 1

            one_sample_edu_reps = batch_sample_edu_reps[i]

            """ limit_the_link_length """
            if global_config.limit_the_link_length:
                if global_config.relink_to_root:
                    one_sample_target_link = [0 if one_sample_target_link[k - 1] == 0 else k - one_sample_target_link[k - 1] for k in range(1, edu_number[i])]
                else:
                    one_sample_target_link = [k - (one_sample_target_link[k - 1] + 1) for k in range(1, edu_number[i])]

            """ Link and Relation prediction """
            target_link_tensor = torch.tensor(one_sample_target_link).unsqueeze(0).cuda()
            target_relation_tensor = torch.tensor(one_sample_target_relation).unsqueeze(0).cuda()

            tmp_point_list = []
            tmp_relation_list = []
            for j in range(1, edu_number[i]):
                if global_config.limit_the_link_length:
                    """ limit_the_link_length """
                    if global_config.relink_to_root:
                        tmp_flipped_reps = one_sample_edu_reps[:, [0, ] + [i for i in range(max(1, j - global_config.limit_link_range), j)][::-1], :]
                    else:
                        tmp_flipped_reps = one_sample_edu_reps[:, [i for i in range(max(0, j - global_config.limit_link_range), j)][::-1], :]
                    tmp_tensor = torch.cat([one_sample_edu_reps[:, j, :].unsqueeze(1).expand(-1, tmp_flipped_reps.size(1), -1), tmp_flipped_reps], dim=2)

                else:
                    """ no link length limit """
                    tmp_tensor = torch.cat([one_sample_edu_reps[:, j, :].unsqueeze(1).expand(-1, j, -1), one_sample_edu_reps[:, :j, :]], dim=2)

                tmp_link_point_res = self.agent.linear_for_link(tmp_tensor)

                if link_loss:
                    link_loss += self.link_loss_function(input=tmp_link_point_res, target=target_link_tensor[:, j - 1].unsqueeze(0))
                else:
                    link_loss = self.link_loss_function(input=tmp_link_point_res, target=target_link_tensor[:, j - 1].unsqueeze(0))

                loss_accumulate_number += 1

                one_step_link_to_point = torch.argmax(tmp_link_point_res, dim=1).detach().cpu().numpy().tolist()[0][0]

                if global_config.limit_the_link_length:
                    if global_config.relink_to_root:
                        if one_step_link_to_point != 0:
                            one_step_link_to_point = j - one_step_link_to_point
                    else:
                        one_step_link_to_point = j - (one_step_link_to_point + 1)

                if not eval_mode:
                    # use gold links for relation training.
                    node_from_relation = one_sample_edu_reps[:, j, :].unsqueeze(1)
                    node_to_relation = one_sample_edu_reps[:, one_sample_target_link[j - 1], :].unsqueeze(1)

                else:
                    # use relation prediction for evaluation
                    node_from_relation = one_sample_edu_reps[:, j, :].unsqueeze(1)
                    node_to_relation = one_sample_edu_reps[:, one_step_link_to_point, :].unsqueeze(1)

                tmp_relation_res = self.agent.linear_for_relation(torch.cat([node_from_relation, node_to_relation], dim=2)).transpose(1, 2)
                if relation_loss:
                    relation_loss += self.relation_loss_function(input=tmp_relation_res, target=target_relation_tensor[:, j - 1].unsqueeze(0))
                else:
                    relation_loss = self.relation_loss_function(input=tmp_relation_res, target=target_relation_tensor[:, j - 1].unsqueeze(0))

                one_step_relation = torch.argmax(tmp_relation_res, dim=1).detach().cpu().numpy().tolist()[0][0]

                tmp_point_list.append([j, one_step_link_to_point])
                tmp_relation_list.append(one_step_relation)

            link_prediction.append(tmp_point_list)
            relation_prediction.append(tmp_relation_list)

        link_loss = link_loss / loss_accumulate_number
        relation_loss = relation_loss / loss_accumulate_number

        return link_loss, link_prediction, link_end, relation_loss, relation_prediction, target_relation

    def batch_infer(self, batch):
        input_text = [i.strip().split("<utterance>") for i in batch]
        edu_number = [len(i) for i in input_text]
        link_prediction, relation_prediction = [], []

        batch_sample_input_text = [global_config.sentence_seg_token.join(input_text[i]) for i in range(len(batch))]
        batch_sample_edu_reps = self.agent.encoding_all_utterances(batch_sample_input_text, edu_number_list=edu_number)

        for i in range(len(batch)):
            one_sample_edu_reps = batch_sample_edu_reps[i]

            """ Link and Relation prediction """
            tmp_point_list = []
            tmp_relation_list = []
            for j in range(1, edu_number[i]):
                if global_config.limit_the_link_length:
                    """ limit_the_link_length """
                    if global_config.relink_to_root:
                        tmp_flipped_reps = one_sample_edu_reps[:, [0, ] + [i for i in range(max(1, j - global_config.limit_link_range), j)][::-1], :]
                    else:
                        tmp_flipped_reps = one_sample_edu_reps[:, [i for i in range(max(0, j - global_config.limit_link_range), j)][::-1], :]
                    tmp_tensor = torch.cat([one_sample_edu_reps[:, j, :].unsqueeze(1).expand(-1, tmp_flipped_reps.size(1), -1), tmp_flipped_reps], dim=2)

                else:
                    """ no link length limit """
                    tmp_tensor = torch.cat([one_sample_edu_reps[:, j, :].unsqueeze(1).expand(-1, j, -1), one_sample_edu_reps[:, :j, :]], dim=2)

                tmp_link_point_res = self.agent.linear_for_link(tmp_tensor)

                one_step_link_to_point = torch.argmax(tmp_link_point_res, dim=1).detach().cpu().numpy().tolist()[0][0]

                if global_config.limit_the_link_length:
                    if global_config.relink_to_root:
                        if one_step_link_to_point != 0:
                            one_step_link_to_point = j - one_step_link_to_point
                    else:
                        one_step_link_to_point = j - (one_step_link_to_point + 1)

                """ use relation prediction for evaluation """
                node_from_relation = one_sample_edu_reps[:, j, :].unsqueeze(1)
                node_to_relation = one_sample_edu_reps[:, one_step_link_to_point, :].unsqueeze(1)

                tmp_relation_res = self.agent.linear_for_relation(torch.cat([node_from_relation, node_to_relation], dim=2)).transpose(1, 2)

                one_step_relation = torch.argmax(tmp_relation_res, dim=1).detach().cpu().numpy().tolist()[0][0]

                tmp_point_list.append([j, one_step_link_to_point])
                tmp_relation_list.append(one_step_relation)

            link_prediction.append(tmp_point_list)
            relation_prediction.append(tmp_relation_list)

        return input_text, link_prediction, relation_prediction

    def batch_train(self, batch):
        self.agent.train()
        self.optimizer.zero_grad()
        link_loss, link_prediction, target_link, relation_loss, relation_prediction, target_relation = self.forward(batch)

        (link_loss + 5.0 * relation_loss).backward()

        self.optimizer.step()
        return link_loss.item(), link_prediction, target_link, relation_loss.item(), relation_prediction, target_relation

    def batch_eval(self, batch):
        self.agent.eval()
        link_loss, link_prediction, target_link, relation_loss, relation_prediction, target_relation = self.forward(batch)
        return link_loss.item(), link_prediction, target_link, relation_loss.item(), relation_prediction, target_relation

    def save_model(self, save_path):
        """ save model """
        print("Saving model to:", save_path)
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, load_path):
        """ save model """
        print("Loading model from:", load_path)
        self.agent.load_state_dict(torch.load(load_path))
