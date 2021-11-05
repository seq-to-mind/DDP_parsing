## Introduction
* One implementation of the paper "Improving Multi-Party Dialogue Discourse Parsing via Domain Integration". <br>
* The parsing follows the Segmented Discourse Representation Theory (SDRT) scheme. <br>
* Users can apply it to parse the input dialogue text, and get dependency-parsing structure and relation prediction. <br>
* This repo and the pre-trained model are only for research use. <br>

## Package Requirements
1. pytorch==1.7.1
2. transformers==4.8.2

## Supported Languages
We trained and evaluated the model with two English dialogue discourse parsing corpora: STAC and Molweni. <br>

## Data Format
* [Inference Input] `InputSentence`: The input dialogue content with the #ROOT# head token and utterance split `\<utterance\>`. The raw text will be tokenized and encoded by the `roberta-base` language backbone. <br>
    * Raw Sequence Example: <br>
    *#ROOT# \<utterance\> A: Hi Tom, are you busy tomorrow’s afternoon? \<utterance\> B: I’m pretty sure I am. What’s up? \<utterance\> A: Can you go with me to the animal shelter?. \<utterance\> B: What do you want to do? ... ... \<utterance\> B: I wonder what he'll name it. \<utterance\> A: He said he’d name it after his hamster –Lemmy- he's a great Motorhead fan :-)*

* [Inference Output] `all_sample_utter_level`: The list of utterances, each utterance is taken as one EDU in the parsing process. <br>
    
* [Inference Output] `all_predict_link`: Predictions of the discourse links, in the form of `[from_utterance_id, to_utterance_id]`, where the first node is the #ROOT# token. <br>
    * Output Example: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 3], [6, 5], [7, 6], [8, 7], [9, 8], [10, 8], [11, 10], [12, 10], [13, 12], [14, 13], [15, 14], [16, 14], [17, 16]] <br>
    
* [Inference Output] `all_predict_relation`: Predictions of the discourse relation classification result. See relation type mapping is defined in the code. <br>
   * Output Example: [16, 12, 12, 12, 3, 0, 5, 0, 0, 0, 9, 7, 3, 0, 10, 7, 3]

## How to use it for parsing
* Put the dialogue content to the file `./data/text_for_inference.txt`. See sample data in the file. <br>
* Run the script `main_infer.py` to obtain the dialogue discourse parsing result. See the script for detailed model output. <br>
* We recommend users to run the parser on a GPU-equipped environment. <br>

## Citation

```
@inproceedings{liu-chen-2021-improving,
    title = "Improving Multi-Party Dialogue Discourse Parsing via Domain Integration",
    author = "Liu, Zhengyuan and Chen, Nancy",
    booktitle = "Proceedings of the 2nd Workshop on Computational Approaches to Discourse",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.codi-main.11",
    pages = "122--127",
}
```
