# Description
In this study, the Diffusion model and Transformer model are combined to solve the problem of insufficient data in flood prediction. The diffusion model is used for data expansion and the Transformer model is used for flood flow prediction. We use the flood prediction data of Wanan Reservoir, Zhexi Reservoir and Pankou Reservoir as the data set to forecast, and the results show that the prediction effect is better than the traditional model in most cases.
![image](https://github.com/user-attachments/assets/08d204b0-51a3-4e2a-950f-50e0841fad16)
# Code
## Dataset Preparation
The data set we use is the flood data collected from Wan 'an Reservoir, Zhe Xi Reservoir and Pankou Reservoir, including the hydrological and rainfall information of the basin.Due to the nature of this study, the participants did not consent to public sharing of their data.
## Training & Sampling
For training, you can reproduce the experimental results of all benchmarks by runing

'(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --train'
