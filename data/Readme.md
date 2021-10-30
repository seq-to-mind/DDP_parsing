#### Training Data
The training samples from Molweni corpus is located in `molweni_data` folder.
The training samples from STAC corpus is located in `stac_data` folder.

#### Construct the Inference Input
* The input dialogue content with the #ROOT# head token and utterance split `\<utterance\>`. 
* The raw text will be tokenized and encoded by the `roberta-base` language backbone. <br>
* Raw Sequence Example: <br>
 *#ROOT# \<utterance\> A: Hi Tom, are you busy tomorrow’s afternoon? \<utterance\> B: I’m pretty sure I am. What’s up? \<utterance\> A: Can you go with me to the animal shelter?. \<utterance\> B: What do you want to do? ... ... \<utterance\> B: I wonder what he'll name it. \<utterance\> A: He said he’d name it after his hamster –Lemmy- he's a great Motorhead fan :-)
