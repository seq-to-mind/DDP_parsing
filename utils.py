import json
import re
import numpy as np
from nltk import word_tokenize
from data.filter_out_list import global_filter_out_token_set
import global_config


def cmp_relation(a, b):
    if a["x"] == b["x"] and a["y"] == b["y"]: return 0
    if a["y"] < b["y"] or (a["y"] == b["y"] and a["x"] < b["x"]): return -1
    return 1


def load_data(filename, map_relations):
    print("Loading data:", filename)
    with open(filename, "r") as f_in:
        inp = f_in.read()
        data = json.loads(inp)
        cnt_multi_parents = 0

        for dialog in data:
            last_speaker = None
            turn = 0

            for edu in dialog["edus"]:
                edu["text_raw"] = edu["text"] + " "
                text = edu["text"]

                while text.find("http") >= 0:
                    i = text.find("http")
                    j = i

                    while j < len(text) and text[j] != ' ':
                        j += 1
                    text = text[:i] + " [url] " + text[j + 1:]

                invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
                for ch in invalid_chars:
                    text = re.sub(ch, "", text)

                """ add the processed data to key 'text' """
                if global_config.vocab_refining:
                    tokens = word_tokenize(edu["text"])
                    tokens = [tmp_t for tmp_t in tokens if tmp_t not in global_filter_out_token_set]
                    edu["text"] = " ".join(tokens)
                    edu["tokens"] = tokens
                else:
                    tokens = word_tokenize(edu["text"])
                    edu["text"] = " ".join(tokens)
                    edu["tokens"] = tokens

                if edu["speaker"] != last_speaker:
                    last_speaker = edu["speaker"]
                    turn += 1
                edu["turn"] = turn

            have_relation = {}
            relations = []

            for relation in dialog["relations"]:
                if (relation["x"], relation["y"]) in have_relation:
                    continue
                relations.append(relation)
                have_relation[(relation["x"], relation["y"])] = True

            dialog["relations"] = relations

            for relation in dialog["relations"]:
                if not relation["type"] in map_relations:
                    map_relations[relation["type"]] = len(map_relations)
                relation["type"] = map_relations[relation["type"]]

            dialog["relations"] = sorted(dialog["relations"], key=lambda l: (l["y"], l["x"]))
            cnt = [0] * len(dialog["edus"])

            for r in dialog["relations"]:
                cnt[r["y"]] += 1

            for i in range(len(dialog["edus"])):
                if cnt[i] > 1:
                    cnt_multi_parents += 1

    cnt_edus, cnt_relations, cnt_relations_backward = 0, 0, 0
    for dialog in data:
        cnt_edus += len(dialog["edus"])
        cnt_relations += len(dialog["relations"])

        for r in dialog["relations"]:
            if r["x"] > r["y"]:
                cnt_relations_backward += 1

    print("%d dialogs, %d edus, %d relations, %d backward relations" % (len(data), cnt_edus, cnt_relations, cnt_relations_backward))
    print("%d edus have multiple parents" % cnt_multi_parents)

    return data


def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])

    return batches
