# SR2024FinTone
### Analyizing BOK Meeting's tone by using FinBERT ###

FinTone is a model to analyize tone of BOK metting.
It predict each sentence of meeting to 1 if it is hawkish 0 if it is dorvish.
To predict this we conduct labeling from lots of bond report(i.e. "hawkishdorvishlabeling.csv").
Note that we labeled sensitively for a little change in stance. For example, "채권 금리의 상승폭이 제한되었다", "채권 금리의 상승폭이 축소되었다" is labeled to 0(dorvish).

We use Ko-FinBERT-SC as a baseline made by SNU to make FinTone.
We trained it by changing classification layer and fine-tuning.
Accuracy and Loss is shown below.

![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/3a291978-b1f8-4473-a479-5a80537ca50c)

We anlyize this model with kospi and policy rate.(left is baseline right is our model)

![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/3abc229e-de0c-4c24-ad22-c87b053773be)
![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/38b19d01-2a07-41f5-8090-9f88107b1d09)
![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/0b54c361-06c0-4097-9861-7428c29ecb69)
![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/2f4bacc5-583c-4ca0-ae71-e653f6708140)

Especially with kospi, we found that our model's prediction precedes one month ahead of the kospi
The image below shows the number of delayed month(X) and the correlation(Y).

![image](https://github.com/jiwon103203/SR2024FinTone/assets/127197114/92208958-755c-4810-b3c8-f2f5ea79a929)
