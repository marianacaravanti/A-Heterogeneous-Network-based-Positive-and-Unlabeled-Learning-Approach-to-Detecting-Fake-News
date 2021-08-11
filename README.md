# A-Heterogeneous-Network-based-Positive-and-Unlabeled-Learning-Approach-to-Detecting-Fake-News

- Mariana Caravanti (ICMC/USP) | mariana.caravanti@usp.br
- Bruno Nogueira (Facom/UFMS) | bruno@facom.ufms.br
- Rafael Rossi (UFMS) | rafael.g.rossi@ufms.br
- Ricardo Marcacini | ricardo.marcacini@icmc.usp.br
- Solange Rezende | solange@icmc.usp.br

# Abstract

The dynamism of fake news evolution and dissemination plays a crucial role in influencing and confirming personal beliefs. To minimize the spread of disinformation approaches proposed in the literature, automatic fake news detection generally learns models through binary supervised algorithms considering textual and contextual information. However, labeling significant amounts of real news to build accurate classifiers is difficult and time-consuming due to their broad spectrum. Positive and unlabeled learning (PUL) can be a good alternative in this scenario. PUL algorithms learn models considering little labeled data of the interest class and use unlabeled data to increase classification performance. This paper proposes a heterogeneous network variant of the PU-LP algorithm, a PUL algorithm based on similarity networks. Our network incorporates different linguistic features to characterize fake news, such as representative terms, emotiveness, pausality, and average sentence size. Also, we considered two representations of the news to compute similarity: term frequency-inverse document frequency and Doc2Vec, which aims to create a numerical representation of a document, regardless of its length. We evaluated our approach in six datasets written in Portuguese or English, comparing its performance with a binary semi-supervised baseline algorithm, using two well-established label propagation algorithms: LPHN and GNetMine. The results indicate that PU-LP with heterogeneous networks can be competitive to binary semi-supervised learning. Also, linguistic features such as representative terms and pausality improved the classification performance, especially when there is a small amount of labeled news.

# Datasets

Os datasets estão disponíveis em um link do google Drive, disponível na pasta pasta Heterogeneous PU-LP/Datasets.

# Heterogeneous PU-LP

![methodology](https://github.com/marianacaravanti/A-Heterogeneous-Network-based-Positive-and-Unlabeled-Learning-Approach-to-Detecting-Fake-News/blob/main/Figures/methodology.png)

# Table of features used in each heterogeneous network
![features table](https://github.com/marianacaravanti/A-Heterogeneous-Network-based-Positive-and-Unlabeled-Learning-Approach-to-Detecting-Fake-News/blob/main/Figures/features%20table.png)

# Features of each news Dataset
![datasets](https://github.com/marianacaravanti/A-Heterogeneous-Network-based-Positive-and-Unlabeled-Learning-Approach-to-Detecting-Fake-News/blob/main/Figures/datasets%20information.png)

# Results
![results](https://github.com/marianacaravanti/A-Heterogeneous-Network-based-Positive-and-Unlabeled-Learning-Approach-to-Detecting-Fake-News/blob/main/Figures/results.png)

# References
[PU-LP]: Ma, S., Zhang, R.: Pu-lp: A novel approach for positive and unlabeled learning by label propagation. In: 2017 IEEE International Conference on Multimedia & Expo
Workshops (ICMEW). pp. 537–542. IEEE (2017).

[FBR]: Silva, R.M., Santos, R.L., Almeida, T.A., Pardo, T.A.: Towards automatically filtering fake news in portuguese. Expert Systems with Applications 146, 113–199
(2020).

[FNN]: Shu, Kai, et al. Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data 8.3 (2020): 171-188.

[FNC0]: https://github.com/several27/FakeNewsCorpus
[FNC1]: https://github.com/several27/FakeNewsCorpus
[FNC2]: https://github.com/several27/FakeNewsCorpus
