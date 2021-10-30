import os
import torch
from utils import get_batches
from Model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_process(model, infer_batches):
    predict_link = []
    predict_rel = []
    sample_utter_level = []
    for batch in infer_batches:
        utter_list, pred_link, pred_rel = model.batch_infer(batch)
        predict_link.extend(pred_link)
        predict_rel.extend(pred_rel)
        sample_utter_level.extend(utter_list)

    return sample_utter_level, predict_link, predict_rel


if __name__ == '__main__':
    map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
                     'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
                     'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Background': 15, 'Break': 16}

    assert len(map_relations.keys()) == 18

    infer_sample_list = open("./data/text_for_inference.txt").readlines()
    infer_batches = get_batches(infer_sample_list, 8)

    model = Model()
    load_model_path = "./saved_models/one_checkpoint.pth"
    model.load_model(load_model_path)

    with torch.no_grad():
        all_sample_utter_level, all_predict_link, all_predict_relation = infer_process(model, infer_batches)
        print(all_sample_utter_level[0])
        print(all_predict_link[0])
        print(all_predict_relation[0])
