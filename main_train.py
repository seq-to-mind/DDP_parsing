import os
import random
import time
from collections import Counter

import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from utils import load_data, get_batches
from Model import Model

import global_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
running_random_number = random.randint(1000, 9999)

running_log_name = "./running_log/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_" + str(running_random_number) + ".txt"
open(running_log_name, "w")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_name_collection():
    name_collection = open("./data/name_collection/female_english_names", encoding="utf-8").readlines() + \
                      open("./data/name_collection/male_english_names", encoding="utf-8").readlines()
    name_collection = [i.strip()[:6] for i in name_collection if len(i.strip()) > 2]
    name_collection = list(set(name_collection))
    return name_collection


def unify_dependency_annotation(sample_list):
    all_edu_number = 0
    all_no_link_edu_number = 0
    all_relation_distribution = []
    all_link_distribution = []

    tmp_list = []
    for tmp_i in range(len(sample_list)):
        # add ROOT node
        sample_list[tmp_i]["edus"] = [{"speaker": "#ROOT#", "text_raw": " ", "text": " ", "tokens": [" "], 'turn': 0}] + sample_list[tmp_i]["edus"]
        sample_list[tmp_i]["relations"] = [{"type": 16, "x": -1, "y": 0}] + sample_list[tmp_i]["relations"]
        for i in range(len(sample_list[tmp_i]["relations"])):
            sample_list[tmp_i]["relations"][i]["x"] += 1
            sample_list[tmp_i]["relations"][i]["y"] += 1

        one_sample = sample_list[tmp_i]
        one_edu_number = len(one_sample["edus"])
        all_edu_number += one_edu_number
        original_links = one_sample["relations"]

        """ Swap the reverse links with distance 1, this will affect the data """
        for k, v in enumerate(original_links):
            if v["x"] - v["y"] == 1:
                tmp = original_links[k]["y"]
                original_links[k]["y"] = original_links[k]["x"]
                original_links[k]["x"] = tmp

        if global_config.rebuild_speaker_names:
            speaker_list = list(set([one_sample["edus"][i]["speaker"] for i in range(one_edu_number)]))
            new_speaker_list = np.random.choice(global_name_collection, size=len(speaker_list), replace=False)
            replace_speaker_dict = {}
            for k, v in enumerate(speaker_list):
                replace_speaker_dict[v] = new_speaker_list[k]
            replace_speaker_dict["#ROOT#"] = "#ROOT#"
            for i in range(one_edu_number):
                sample_list[tmp_i]["edus"][i]["speaker"] = replace_speaker_dict[sample_list[tmp_i]["edus"][i]["speaker"]]


        if global_config.limit_the_link_length:
            # process too long links
            original_links = [v for k, v in enumerate(original_links) if v["y"] - v["x"] < global_config.limit_link_range]

        # remove the reverse links
        original_links = [v for k, v in enumerate(original_links) if v["y"] > v["x"]]
        all_link_distribution += [i["x"] for i in original_links]

        no_link_edu_set = set(range(1, one_edu_number)) - set([i["y"] for i in original_links])
        all_no_link_edu_number += len(no_link_edu_set)
        all_relation_distribution += [i["type"] for i in original_links]

        """ remove multiple precedent links """
        original_links = original_links[::-1]
        new_links = [v for k, v in enumerate(original_links) if k == 0 or v["y"] != original_links[k - 1]["y"]][::-1]

        if global_config.limit_the_link_length:
            if global_config.relink_to_root:
                new_links += [{"type": 16, "x": 0, "y": i} for i in no_link_edu_set]
            else:
                # add the unlinked EDU with Continues Link to precedent EDU
                new_links += [{"type": 16, "x": i - 1, "y": i} for i in no_link_edu_set]

        else:
            # point to the ROOT node for unlinked EDUs
            new_links += [{"type": 16, "x": 0, "y": i} for i in no_link_edu_set]

        new_links = sorted(new_links, key=lambda k: k["y"])

        sample_list[tmp_i]["relations"] = new_links

    # print(Counter(tmp_list))

    """ remove the samples with empty relation """
    sample_list = [i for i in sample_list if len(i["relations"]) > 0]

    """ remove the samples that are too long """
    sample_list = [i for i in sample_list if len(i["edus"]) < 30]
    # sample_list = [i for i in sample_list if len(i["edus"]) > 15]

    return sample_list


def get_summary_sum(s, length):
    loss_bi, loss_multi = s[0] / length, s[1] / length
    prec_bi, recall_bi = s[4] * 1. / s[3], s[4] * 1. / s[2]
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = s[5] * 1. / s[3], s[5] * 1. / s[2]
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return [loss_bi, loss_multi, f1_bi, f1_multi]


def train_process(model, train_batches):
    train_epoch = 0

    best_stac_link_f1, best_stac_merge_f1 = {"epoch": 0, "link_f1": 0, "relation_f1": 0, "merge_f1": 0}, {"epoch": 0, "link_f1": 0, "relation_f1": 0, "merge_f1": 0}
    best_mol_link_f1, best_mol_merge_f1 = {"epoch": 0, "link_f1": 0, "relation_f1": 0, "merge_f1": 0}, {"epoch": 0, "link_f1": 0, "relation_f1": 0, "merge_f1": 0}

    while train_epoch < global_config.num_epochs:
        summary_steps = 0

        random.seed(500 + train_epoch)
        random.shuffle(train_batches)

        if global_config.scheduling_learning_rate:
            if train_epoch < 1:
                print("Running with warm up learning rate.")
                model.adjust_learning_rate(backbone_lr=0.00001, other_lr=0.003)
            else:
                lr_decay = 0.98 ** (train_epoch - 1)
                model.adjust_learning_rate(backbone_lr=0.00002 * lr_decay, other_lr=0.001 * lr_decay)
            if train_epoch > 1:
                model.agent.linear_for_relation[-1].weight.requires_grad = True

        for batch in train_batches:
            loss_link, _, _, loss_rel, _, _ = model.batch_train(batch)
            summary_steps += 1
            if summary_steps % 50 == 0:
                print(train_epoch, summary_steps, "training loss", loss_link + loss_rel, "loss_link", loss_link, "loss_rel", loss_rel)

        best_stac_link_f1, best_stac_merge_f1 = evaluate_process(model, stac_test_batches, train_epoch, best_stac_link_f1, best_stac_merge_f1)
        best_mol_link_f1, best_mol_merge_f1 = evaluate_process(model, mol_test_batches, train_epoch, best_mol_link_f1, best_mol_merge_f1)

        train_epoch += 1

        if global_config.save_model and train_epoch > 7:
            model.save_model("./saved_models/best_model_" + str(running_random_number) + "_epoch" + str(train_epoch) + ".pth")

        with open(running_log_name, "a") as fp:
            fp.write("\n")

    with open("./running_log/tmp_best_result", "a") as fp:
        save_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fp.write(save_time + " Best stac link result: " + str(best_stac_link_f1) + "\n")
        fp.write(save_time + " Best stac merge result: " + str(best_stac_merge_f1) + "\n")
        fp.write(save_time + " Best mol link result: " + str(best_mol_link_f1) + "\n")
        fp.write(save_time + " Best mol merge result: " + str(best_mol_merge_f1) + "\n\n")


def evaluate_process(model, test_batches, train_epoch, best_link_f1=None, best_merge_f1=None, preview=False):
    all_predict_link, all_target_link = [], []
    all_predict_rel, all_target_rel = [], []
    all_test_loss = []
    for batch in test_batches:
        loss_link, pred_link, target_link, loss_rel, pred_rel, target_rel = model.batch_eval(batch)
        all_predict_link.extend(pred_link)
        all_target_link.extend(target_link)
        all_predict_rel.extend(pred_rel)
        all_target_rel.extend(target_rel)
        all_test_loss.append(loss_link + loss_rel)

    if preview:
        pass

    all_target_link_flatten, all_predict_link_flatten = [j for i in all_target_link for j in i], [j for i in all_predict_link for j in i]
    print("\nTest Result in epoch:", train_epoch)
    link_f1 = f1_score(all_target_link_flatten, all_predict_link_flatten, average="micro")
    print("Link F1:", link_f1)
    print(all_predict_link[:5], "\n", all_target_link[:5])

    all_target_rel_flatten, all_predict_rel_flatten = [j for i in all_target_rel for j in i], [j for i in all_predict_rel for j in i]
    relation_f1 = f1_score(all_target_rel_flatten, all_predict_rel_flatten, average="micro")
    print("Relation F1:", relation_f1)
    print(all_predict_rel[:5], "\n", all_target_rel[:5])

    merge_f1 = [1 if (all_target_link_flatten[i] == all_predict_link_flatten[i]) and (all_target_rel_flatten[i] == all_predict_rel_flatten[i]) else 0 for i in
                range(len(all_target_link_flatten))]
    merge_f1 = sum(merge_f1) / len(merge_f1)
    print("Merge F1:", merge_f1)
    print("Test loss:", np.mean(all_test_loss))

    with open(running_log_name, "a") as fp:
        fp.write("Epoch: " + str(train_epoch) + " Link F1: " + str(link_f1) + " Relation F1: " + str(relation_f1) + " Merge F1: " + str(merge_f1) + " Test loss: " + str(
            np.mean(all_test_loss)) + "\n")

    if best_link_f1 and best_merge_f1:
        if link_f1 > best_link_f1["link_f1"]:
            best_link_f1 = {"epoch": train_epoch, "link_f1": link_f1, "relation_f1": relation_f1, "merge_f1": merge_f1}
        if merge_f1 > best_merge_f1["merge_f1"]:
            best_merge_f1 = {"epoch": train_epoch, "link_f1": link_f1, "relation_f1": relation_f1, "merge_f1": merge_f1}

    if train_epoch == 999:
        print(confusion_matrix(y_true=all_target_rel_flatten, y_pred=all_predict_rel_flatten, labels=range(0, 17)))

    return best_link_f1, best_merge_f1


if __name__ == '__main__':

    setup_seed(100)
    global_name_collection = get_name_collection()

    map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
                     'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
                     'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Background': 15, 'Break': 16}

    mol_data_train = load_data('./data/molweni_data/train_data.json', map_relations)
    mol_data_test = load_data('./data/molweni_data/test_data.json', map_relations)

    stac_data_train = load_data('./data/stac_data/train_data.json', map_relations)
    stac_data_test = load_data('./data/stac_data/test_data.json', map_relations)
    print(map_relations)

    assert len(map_relations.keys()) == 18

    mol_data_train = unify_dependency_annotation(mol_data_train)
    mol_data_test = unify_dependency_annotation(mol_data_test)

    stac_data_train = unify_dependency_annotation(stac_data_train)
    stac_data_test = unify_dependency_annotation(stac_data_test)

    data_train = mol_data_train + stac_data_train

    print('Train Dataset sizes: %d' % (len(data_train)))
    print('Test Dataset sizes: %d %d' % (len(stac_data_test), len(mol_data_test)))

    random.seed(100)
    random.shuffle(data_train)

    train_batches = get_batches(data_train, global_config.batch_size)
    mol_test_batches = get_batches(mol_data_test, global_config.batch_size)
    stac_test_batches = get_batches(stac_data_test, global_config.batch_size)

    model = Model()

    train_mode = True

    if train_mode:
        train_process(model, train_batches)
    else:
        load_model_path = "./saved_models/XXX.pth"
        model.load_model(load_model_path)

        with torch.no_grad():
            evaluate_process(model, mol_test_batches, 999)
            evaluate_process(model, stac_test_batches, 999)
