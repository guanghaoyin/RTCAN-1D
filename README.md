# RTCAN-1D

The code of paper "A Efficient Multimodal framework for large scale Emotion Recognition by Fusing Music and Electrodermal Activity Signals"

RTCAN-1D is a multimodel framework to fuse music and EDA signals for emotion recognition.

The code is tested on Ubuntu 14.04/16.04 /Windows 10 environment (Python3.7, PyTorch_1.1.0, CUDA9.0, cuDNN9.0) with Titan X/1080Ti/Xp GPUs. 

## Contents

1. [Introduction](https://github.com/guanghaoyin/RTCAN-1D#Introduction)
2. [Run](https://github.com/guanghaoyin/RTCAN-1D#Run)
3. [Citation](https://github.com/guanghaoyin/RTCAN-1D#Citation)

## Introduction

Considerable attention has been paid for physiological signal-based emotion recognition in field of affective computing. For the reliability and user friendly acquisition, Electrodermal Activity (EDA) has great advantage in practical applications. However, the EDA-based emotion recognition with hundreds of subjects still lacks effective solution. In this paper, our work makes an attempt to fuse the subject individual EDA features and the external evoked music features. And we propose an end-to-end multimodal framework, the 1-dimensional residual temporal and channel attention network (RTCAN-1D). For EDA features, the novel convex optimization-based EDA (CvxEDA) method is applied to decompose EDA signals into pahsic and tonic signals for mining the dynamic and steady features. The channel-temporal attention mechanism for EDA-based emotion recognition is firstly involved to improve the temporal- and channel-wise representation. For music features, we process the music signal with the open source toolkit openSMILE to obtain external feature vectors. The individual emotion features from EDA signals and external emotion benchmarks from music are fused in the classifing layers. We have conducted systematic comparisons on three multimodal datasets (PMEmo, DEAP, AMIGOS) for 2-classes valance/arousal emotion recognition. Our proposed RTCAN-1D outperforms the existing state-of-the-art models, which also validate that our work provides an reliable and efficient solution for large scale emotion recognition.


Whole Architecture of RTCAN-1D

[![architecture](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/architecture.png)](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/architecture.png)

Architecture of RTCAG for EDA feature extraction

[![architecture](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/RTCAG.png)](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/RTCAG.png)



## Run

### Prepare training data

1. Download DEAP, AMIGOS and PMEmo datasets from [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html), [AMIGOS](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html) and [PMEmo](https://drive.google.com/drive/folders/1NhN4KaLQPFg9nRNOwne-Lnkxi3nlJHR3)
   The downloading of DEAP and AMIGOS requires the User License Agreement. Our previous PMEmo has been uploaded in Google Cloud for free downloading
2. Getting the EDA(GSR) signals from the preprocessed data of three datasets. You can use the Matlab or Python to get the specific EDA channel, depending on the format of your downloading data.
3. Getting the music features from the preprocessed data of PMEmo. The static music feature extraction has been accomplished by OpenSMILE in our previous PMEmo dataset.
4. Conducting the CvxEDA decomposition to get phasic and tonic signals. Please refer to the searched CvxEDA.py in Github
5. Conducting annotation recreation. As followed figure illustrates, (a) The annotation collection from a subject in the V/A emotional space; (b) The k-means clustering to calculate two cluster centers and their midpoint; (c) The high and low arousal state separated by threshold; (d) The high and low valence state separated by threshold.
   [![architecture](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/threshold.png)](https://github.com/guanghaoyin/RTCAN-1D/tree/master/figs/threshold.png)
6. Saving the phasic, tonic and original signals as .txt file for reading.  Saving the label as .csv file. 



### Begin to train



1. (optional) Download models for our paper and place them in '/save/'
   The models for DEAP/AMIGOS/PMEmo can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1JRiyfJUnNrMepKxUqD3BYfgCKLfTM32U?usp=sharing)
2. Run the following scripts to start run in different dataset.

```
#PMEmo arousal music+EDA temporal+channel attention 
python PMEmo_run.py run_train('Arousal','tonic', use_music = True, use_attention = True, use_non_local = True)

#PMEmo valence music+EDA temporal+channel attention 
python PMEmo_run.py run_train('Valence','tonic', use_music = True, use_attention = True, use_non_local = True)

#DEAP arousal music+EDA temporal+channel attention 
python DEAP_run.py run_train('Arousal','tonic', use_attention = True, use_non_local = True)

#DEAP valence music+EDA temporal+channel attention 
python DEAP_run.py run_train('Valence','tonic', use_attention = True, use_non_local = True)

#AMIGOS arousal music+EDA temporal+channel attention 
python AMIGOS_run.py run_train('Arousal','tonic', use_attention = True, use_non_local = True)

#AMIGOS valence music+EDA temporal+channel attention 
python AMIGOS_run.py run_train('Valence','tonic', use_attention = True, use_non_local = True)

```

## Citation

If you find the code helpful in your resarch or work, please cite the following papers.

```
@inproceedings{Zhang:2018:PDM:3206025.3206037,
 author = {Zhang, Kejun and Zhang, Hui and Li, Simeng and Yang, Changyuan and Sun, Lingyun},
 title = {The PMEmo Dataset for Music Emotion Recognition},
 booktitle = {Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval},
 series = {ICMR '18},
 year = {2018},
 isbn = {978-1-4503-5046-4},
 location = {Yokohama, Japan},
 pages = {135--142},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/3206025.3206037},
 doi = {10.1145/3206025.3206037},
 acmid = {3206037},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {dataset, eda, experiment, music emotion recognition},
} 


```
