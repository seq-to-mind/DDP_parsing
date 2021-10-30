use_cuda = True

pretrained_model = "roberta-base"
pretrained_tokenizer = "roberta-base"

sentence_seg_token = " </s> <s> "
# sentence_seg_token = " [SEP] [CLS] "

sentence_embedding_mode = "average"
freeze_some_bert_layer = False
use_gru_after_sentence = True
different_learning_rate = True
scheduling_learning_rate = True

limit_the_link_length = False
limit_link_range = 15
relink_to_root = False

rebuild_speaker_names = True
vocab_refining = True

using_sliding_window_bert = False

train = True
save_model = False

hidden_size = 768
num_relations = 17

batch_size = 32
num_epochs = 20
learning_rate = 0.00002
