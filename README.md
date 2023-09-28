# Natural Language Understanding System


## Model
The natural language understanding model is divided into 2 sub-models: an entity recognizer and an intent predictor. The entity recognizer identifies named entities in the text input, such as people, places, and devices. The intent predictor predicts the user's intent, such as whether they are asking a question, giving a command, or making a request.

### Entity Recognizer
In the dataset, we label each input sentence on each of its tokens with the entity type ID. Due to the fact that the probability a token being part of a named entity is affected by the input context and also the entity types of the previous tokens. Conditioning on both these conditions helps to reduce the variance of the predicted distribution, thus makes it easier for the model to learn.

Mathematically, we have to estimate the probability of a label sequence given an input context assumed as
```math
\begin{align*}
P(L_{1:N} | T_{1:N}) = P(L_1 | T_{1:N}) \prod_{i=2}^N P(L_i | T_{1:N}, L_{1:i-1})
\end{align*}
```
Where $T_i$ is the i-ith token in the input sequence and $L_i$ is its corresponding label.

The probability of a token being part of a named entity is modeled as a conditional categorical distribution. Parameterizing this distribution using a neural network, we obtain
```math
\begin{align*}
P(L_{1:N} | T_{1:N}) &= P(L_1 | T_{1:N}) \prod_{i=2}^N P(L_i | T_{1:N}, L_{1:i-1}) \\
&= \text{Cat} (L_1, \lambda = \text{NN}_\theta (T_{1:N}, L_{0})) \prod_{i=2}^N \text{Cat} (L_i, \lambda = \text{NN}_\theta (T_{1:N}, L_{1:i-1}))
\end{align*}
```
For the convenience of computing in the neural network, we defined a special label $L_0$. In the scope of this project, we choose it to be a `neutral` label.

The neural network used is a simple vanilla Transformer. For more details about the model architecture, take a look at the original paper [here](https://arxiv.org/abs/1706.03762).

### Intent predictor
The intent predictor, much like the entity recognizer, is also a semi-autoregressive sequence-to-sequence model. Therefore, we employ the exact same model architecture as that of the enity recognizer. The mathematics behind the intent predictor and the entity recognizer are just identical.

### Pretrained Models
Download the pretrained models and the configuration file [here](https://drive.google.com/drive/folders/1XsebKkHYT5psveAne_3FhUHayLKClsxT?usp=sharing) and place it anywhere for the convenience of retrieving later.

## Tokenizer
In this project, we use a byte-pair encoding tokenizer implemented in the `sentencepiece` library. After installing all the modules and dependencies, run the script `build_tokenizer.py` to build a new tokenizer on your own dataset
```
python build_tokenizer.py -f <path-to-the-jsonl-dataset> -s <the-vocabulary-size>
```
Note: The whole corpus will be automatically lowered and the default vocabulary size is 768.\
The trained tokenizer will be in the [tokenizer/tok.model](tokenizer/tok.model) directory by default. In the scope of this track in which the target task is understanding the Vietnamese spoken language to control smarthome devices, it is recommended to use the tokenizer pretrained with the vocab size of 768 on the given dataset from BKAI (placed under the [tokenizer](tokenizer/) directory).

## Dataset

There are 2 parts in the dataset: `audio` and `labels`. The `audio` part includes recorded voice files in `wav` format. The `labels` part includes a `jsonl` file in which there are samples labeling the audio files. An example of a label (unflattend to be similar to `json` format) from the dataset
```json
{
    "id": "64831a87d9f56915da41ef35",
    "sentence": "cái đèn tranh trong nhà giữ đồ Trường Sa có còn không ấy nhờ đi kiểm tra ngay nhé",
    "intent": "Kiểm tra tình trạng thiết bị",
    "sentence_annotation": "cái [ device : đèn tranh ] trong [ location : nhà giữ đồ Trường Sa ] có còn không ấy nhờ đi [ command : kiểm tra ] ngay nhé",
    "entities": [
        {
            "type": "device",
            "filler": "đèn tranh"
        },
        {
            "type": "location",
            "filler": "nhà giữ đồ Trường Sa"
        },
        {
            "type": "command",
            "filler": "kiểm tra"
        }
    ],
    "file": "64831a87d9f56915da41ef35.wav"
}
```
### Entities

The structure of each list of entities
```json
"entities": [
    {
        "type": "...",
        "filler": "..."
    },
    // ...
]
```

Types of entities with the correspoding IDs:
```
0: neutral
1: changing value
2: command
3: device
4: duration
5: location
6: scene
7: target number
8: time at
```

`filler` is the pattern detected in the sentence.\
There are 8 entity types appearing in the dataset and it can be turned into a per-token classification problem. Here, we add a special type, namely `neutral`, indexed by `0`, which specifies patterns that are not actually entities.

### Intents

Examples of intents appearing in the dataset
```
Bật thiết bị
Giảm mức độ của thiết bị
Kiểm tra tình trạng thiết bị
Kích hoạt cảnh
Mở thiết bị
...
```
The intents are represented in natural language, thus has no explicit class-form like the entities. Therefore, it should be turned into a sequence-to-sequence task.

## Train

### Important Update
In our latest experiments, we have found that the best training procedure is to train the entity recognizer along with the base encoder first and then freeze the encoder while training the intent predictor. The issue with training the intent predictor first is that there are multiple sentences with exactly the same intent. As a result, the encoder does not extract useful features for these sentences, and the extracted features are mostly the same. However, the entity patterns for each sentence are quite different from each other. Therefore, training the entity along with the base encoder first allows the encoder to learn useful information about the sentence. The training procedure can be executed by running the following commands
```
# train the entity recognizer
python experimental/train_entity_first/train_entity.py -b <batch_size> -e <epochs>
```
Once the entiry recognizer is trained, it yields a checkpoint of the base model, which consists of an intent predictor with the trained encoder. Retrieve the base checkpoint path and execute the following command
```
# train the intent predictor
python experimental/train_entity_first/train_intent.py -bc "<base_checkpoint_path>" -b <batch_size> -e <epochs>
```


<!-- ### Old training procedure
The training process is divided into two stages. First, we train the intent predictor, which functions as a sentence summarizer and is capable of capturing the context of the sentence. Once the intent predictor is trained, it is then frozen. In the second stage, we train the entity recognizer. In this stage, the encoder from the intent predictor is utilized as a feature extractor for the entity recognizer.

To train the intent predictor, execute the following command
```
python train_indent.py -b <batch_size> -e <number_of_epochs> 
```
There are several more training options which are listed in [train_intent.py](train_intent.py).

To train the entity recognizer, execute the following command
```
python train_entity.py -ic <intent_predictor_checkpoint_path> -b <batch_size> -e <number_of_epochs>
```
For more details about other training options, take a look at [train_entity.py](train_entity.py). -->


## Inference
To perform inference, execute the following command
```
python inference.py -mi <path_to_the_intent_predictor_checkpoint> -me <path_to_the_entity_predictor_checkpoint> -s <input_sentence>
```
For more option details, take a look at [inference.py](inference.py).
