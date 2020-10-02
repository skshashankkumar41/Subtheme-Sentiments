# Subtheme Sentiments 

Idea is to develop an approach that given a sample will identify the sub themes along with their respective sentiments.


## Approach
I considered this problem as a Multi-Label classification and used pre-trained BERT models with fine-tuning to train. By doing [Data Exploration](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/Data-Exploration.pdf) I came to know there are around 10k datapoints and around 90 unique labels but most of them are noisy and are present in very low frequency. So, after doing some preprocessing and undersampling some more frequently occuring at the end we have 23 unique labels and around 6k datapoints. Look [Data Exploration](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/Data-Exploration.pdf) for more details.

I choosed Pretrained BERT models to leverage the information of Language models and as the data is mostly consist of reviews, Language models would work fine and also It is very easy to Implement.

## Performance Metrics 
**Micro f1 score:**
Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.

**Macro f1 score:**
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

[https://www.kaggle.com/wiki/MeanFScore](https://www.kaggle.com/wiki/MeanFScore)

[http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

**Hamming loss:** The Hamming loss is the fraction of labels that are incorrectly predicted.

[https://www.kaggle.com/wiki/HammingLoss](https://www.kaggle.com/wiki/HammingLoss)

## Results 
After 5 Epochs model started overfitting. More Detail in [Models Analysis](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/Model-Analysis.pdf) 
### Evaluation 
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"><span style="font-weight:bold">Metric</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Training</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Validation</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">BCE Loss</span></td>
    <td class="tg-0pky">0.019</td>
    <td class="tg-0pky">0.025</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">F1-Micro-Score</span></td>
    <td class="tg-0pky">0.821</td>
    <td class="tg-0pky">0.737</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">F1-Macro-Score</span></td>
    <td class="tg-0pky">0.618</td>
    <td class="tg-0pky">0.536</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">Hamming Loss</span></td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.046</td>
  </tr>
</tbody>
</table>


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

