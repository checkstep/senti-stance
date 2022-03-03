# Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training

 The goal of stance detection is to determine the viewpoint expressed in a piece of text towards a target. These viewpoints or contexts are often expressed in many different languages depending on the user and the platform, which can be a local news outlet, a social media platform, a news forum, etc. Most research in stance detection, however, has been limited to working with a single language and on a few limited targets, with little work on cross-lingual stance detection. Moreover, non-English sources of labelled data are often scarce and present additional challenges. Recently, large multilingual language models have substantially improved the performance on many non-English tasks, especially such with limited numbers of examples. This highlights the importance of model pre-training and its ability to learn from few examples. In this paper, we present the most comprehensive study of cross-lingual stance detection to date: we experiment with 15 diverse datasets in 12 languages from 6 language families, and with 6 low-resource evaluation settings each. For our experiments, we build on pattern-exploiting training, proposing the addition of a novel label encoder to simplify the verbalisation procedure. We further propose sentiment-based generation of stance data for pre-training, which shows sizeable improvement of more than 6% F1 absolute in low-shot settings compared to several strong baselines. 

## Setup

```console
$ python3 -m venv ~/.virtualenvs/stance-detection
$ source ~/.virtualenvs/stance-detection/bin/activate
```

### Updating project dependencies

```console
# And to install the packages
$ pip install -r requirements.txt
```

## Getting the datasets
### Our splits

We release our few-shot splits (32, 64, 128, 256) in the [data/fewshow](data/fewshow) folder. Moreover, we release the sentiment annotated Wiki snipets in the [data/wikipedia](data/wikipedia) folder. The full training, dev and test sets can be obtained from the links below.

### Multilingual
* Stance Prediction and Claim Verification: An Arabic Perspective [Data](https://github.com/latynt/ans) (ans)
* Integrating Stance Detection and Fact Checking in a Unified Corpus [Data](http://groups.csail.mit.edu/sls/downloads/factchecking/) (arabicfc)
* Detecting Stance in Czech News Commentaries [Data](http://nlp.kiv.zcu.cz/research/sentiment#stance) (czech)
* Stance Evolution and Twitter Interactions in an Italian Political Debate [Data](https://github.com/mirkolai/Stance-Evolution-and-Twitter-Interactions) (conref)
* (Danish) Joint Rumour Stance and Veracity Prediction [Data](https://github.com/danish-stance-detectors/Stance/tree/master/data) (dast)
* Multilingual stance detection in social media political debates [Data](https://www.sciencedirect.com/science/article/abs/pii/S0885230820300085) (e-fra, r-ita)
* An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System [Data](https://github.com/sahilswami96/StanceDetection_CodeMixed) (hindi)
* Overview of NLPCC Shared Task 4: Stance Detection in Chinese Microblogs [Data](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html) (nlpcc)
* Stance and Gender Detection in Tweets on Catalan Independence@Ibereval 2017 [Data](https://stel.ub.edu/Stance-IberEval2017/data.html) (iberval)
* Stance Prediction for Russian: Data and Analysis [Data](https://github.com/lozhn/rustance) (rustance)
* SardiStance @ EVALITA2020 [Data](http://www.di.unito.it/~tutreeb/sardistance-evalita2020/index.html) (sardistance)
* X-Stance: A Multilingual Multi-Target Dataset for Stance Detection [Data](https://zenodo.org/record/3831317)

Some datasets may require additional steps to aquire, e.g., in order to obtain `SardiStance` you need to fill out a form, `IberEval`'s test sets need to be obtained from the competition organizers.

### English
* Stance Detection Benchmark [Data](https://github.com/UKPLab/mdl-stance-robustness#preprocessing) (arc argmin fnc1 iac1 ibmcs perspectrum scd semeval2016t6 semeval2019t7 snopes)
* Will-They-Won't-They [Data](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) (wtwt)
* Emergent [Data](https://www.dropbox.com/sh/9t7fd7xfahb0e1v/AABHcvt9dSH6RNFpnSoYqlZra/emergent?) (emergent)
* Rumor has it [Data](https://github.com/vahedq/rumors/tree/master/data) (rumor)
* Multi-Target Stance Dataset [Data](http://www.site.uottawa.ca/~diana/resources/stance_data/) (mtsd)
* Political Debates [Data](http://mpqa.cs.pitt.edu/corpora/political_debates/) (poldeb)
* VAried Stance Topics [Data](https://github.com/emilyallaway/zero-shot-stance) (vast)

We used the data splits as described in [Cross-Domain Label-Adaptive Stance Detection](https://aclanthology.org/2021.emnlp-main.710/) ([code](https://github.com/checkstep/mole-stance)).


## Running the models

```shell

DATASETS=(arc argmin fnc1 iac1 ibmcs perspectrum scd semeval2016t6 semeval2019t7 snopes emergent mtsd poldeb rumor vast wtwt)
CROSS_LINGUAL_DATASETS=(conref-ita arabicfc ans nlpcc czech dast e-fra hindi iberval2017-ca iberval2017-es r-ita rustance sardistance xstance-de xstance-fr) 

python src/stancedetection/models/trainer_le.py --data_dir "data/all/" \
                                      --model_name_or_path ${MODEL_NAME} \
                                      --output_dir ${OUTPUT_DIR} \
                                      --task_names ${DATASET_NAME} \
                                      --model_type xlm-r \
                                      --replace_classification \
                                      --do_train \
                                      --do_eval \
                                      --learning_rate ${LEARNING_RATE} \
                                      --weight_decay 0.01 \
                                      --per_gpu_train_batch_size 16 \
                                      --per_gpu_eval_batch_size 128 \
                                      --num_train_epochs 50000 \
                                      --warmup_proportion ${WARMUP} \
                                      --adam_epsilon 1e-08 \
                                      --logging_steps 200 \
                                      --max_steps ${MAX_STEPS} \
                                      --max_seq_length ${MAX_SEQ_LEN} \
                                      --evaluate_during_training \
                                      --gradient_accumulation_steps 1 \
                                      --seed ${SEED} \
                                      --dataset_suffix "_${SHOTS}_${i}" \
                                      --fp16 \
                                      --cache_dir cache \
                                      --balanced \
                                      --lambda_mlm ${LAMBDA_MLM} \
                                      --positive_samples_synonyms ${POSITIVE_SAMPLES_SYNONYMS} \
                                      --negative_samples_synonyms ${NEGATIVE_SAMPLES_SYNONYMS} \
                                      --negative_samples_rand ${NEGATIVE_SAMPLES_RAND} \
                                      --p_replace_pos_label ${P_REPLACE_POS_LABEL} \
                                      --p_replace_neg_label ${P_REPLACE_NEG_LABEL} \
                                      --p_mask ${P_MASK} \
                                      --p_random ${P_RANDOM} \
                                      --p_delete 0.0 \
                                      --p_split 0.0 \
                                      --p_swap 0.0 \
                                      --p_label_cond 0.0 \
                                      --overwrite_output_dir
```

## References

Please cite as [1]. There is also an [arXiv version](https://arxiv.org/abs/2109.06050).


[1] Hardalov, M., Arora, A., Nakov, P., & Augenstein, I. (2022).  [*"Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training"*](https://arxiv.org/abs/2109.06050), Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22).



```
@article{hardalov-etal-2022-fewshot,
	title        = {Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training},
	author       = {Hardalov, Momchil and Arora, Arnav and Nakov, Preslav and Augenstein, Isabelle},
	year         = 2022,
	month        = {Feb},
	journal      = {Proceedings of the AAAI Conference on Artificial Intelligence},
	volume       = 36
}
```
## License

The code in this repository is licenced under the [CC-BY-NC-SA 4.0](LICENSE). The datasets are licensed under [CC-BY-SA 4.0](LICENSE.data).
