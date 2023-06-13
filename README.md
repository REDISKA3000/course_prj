# Structural breaks identification in sound signals
## Introduction
Sound processing is considered as a modern
field in deep learning, however many
approaches from CV and NLP are well
adopted and widely used there. Hence, they
are used to detect anomalies in weather
changes (rain, wind, storm, hail), malfunctions
in the operation of mechanisms, unusual road
traffic, out-of-tune musical instrument,
echolocation of ships and submarines,
anomalies of air flows in air conditioning
systems.
In this study anomaly detection in sound
signals is explored. I am going to discuss
existing solutions, demonstrate my own one
and compare them
## Dataset description
The dataset consists of 2 ones released earlier:
ToyADMOS and MIMII. The dataset includes
development dataset, additional training dataset and
evaluation dataset. This dataset consists of
10-seconds recordings of 6 distinct mechanism types:
Fan, Pump, Slider, Valve, Toy-Car and Toy-Conveyor.
For each mechanism there was chosen 4 machines,
which normal and abnormal sounds were recorded.
Training data contains around 1000 samples for each
machine in development and additional datasets,
about 400 test recording in development and
evaluation datasets.

Link on the original dataset: https://zenodo.org/record/3678171

## Existing solutions 
Here you can enjoy my detailed report about existing solutions for this task: https://drive.google.com/file/d/1n3pOZw1yYKsznO3HC8BSL1l8mx5kRMlx/view?usp=sharing
## The pipeline of my solution
![model_pipeline drawio](https://github.com/REDISKA3000/course_prj/assets/49620289/2d53b8c4-143c-470c-b765-21678687e622)
## Comparison of metric scores
![загруженное (2)](https://github.com/REDISKA3000/course_prj/assets/49620289/5c7e6b53-0426-46c6-ae06-8f15e081d66f)
## References
[1] Pawel Daniluk, Marcin Gozdziewski, Slawomir Kapka, and Michal Kos-
mider. Ensemble of auto-encoder based systems for anomaly detection.
Technical report, DCASE2020 Challenge, July 2020.

[2] Jing Gao and Pang-ning Tan. Converting output scores from outlier
detection algorithms into probability estimates. In Sixth International
Conference on Data Mining (ICDM’06), pages 212–221, 2006.

[3] Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle.
MADE: Masked Autoencoder for Distribution Estimation. In Proceed-
ings of the 32nd International Conference on Machine Learning, vol-
ume 37 of JMLR Proceedings, pages 881–889. JMLR.org, 2015.

[4] Ritwik Giri, Srikanth V. Tenneti, Karim Helwani, Fangzhou Cheng,
Umut Isik, and Arvindh Krishnaswamy. Unsupervised anomalous sound
detection using self-supervised classification and group masked autoen-
coder for density estimation. Technical report, DCASE2020 Challenge,
July 2020.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep resid-
ual learning for image recognition. In 2016 IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 770–778, 2016.
[6] Diederik P. Kingma and Max Welling. An introduction to variational au-
toencoders. Foundations and Trends® in Machine Learning, 12(4):307–
392, 2019.

[7] Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Noboru Harada, and
Keisuke Imoto. Toyadmos: A dataset of miniature-machine operating
sounds for anomalous sound detection. In 2019 IEEE Workshop on
Applications of Signal Processing to Audio and Acoustics (WASPAA),
pages 313–317, 2019.

[8] Hans-Peter Kriegel, Peer Kroger, Erich Schubert, and Arthur Zimek.
Interpreting and Unifying Outlier Scores, pages 13–24.

[9] Poojan Oza and Vishal M. Patel. C2ae: Class conditioned auto-encoder
for open-set recognition. In 2019 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 2302–2311, 2019.

[10] Paul Primus. Reframing unsupervised machine condition monitoring as
a supervised classification task with outlier-exposed classifiers. Technical
report, DCASE2020 Challenge, July 2020.

[11] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido,
Kaori Suefusa, and Yohei Kawaguchi. MIMII dataset: Sound dataset for
malfunctioning industrial machine investigation and inspection. CoRR,
abs/1909.09347, 2019.

[12] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and
Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottle-
necks, 2019.

[13] A ̈aron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan,
Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and
Koray Kavukcuoglu. Wavenet: A generative model for raw audio. CoRR,
abs/1609.03499, 2016.

[14] Phongtharin Vinayavekhin, Tadanobu Inoue, Shu Morikuni, Shiqiang
Wang, Tuan Hoang Trong, David Wood, Michiaki Tatsubori, and Ryuki
Tachibana. Detection of anomalous sounds for machine condition mon-
itoring using classification confidence. Technical report, DCASE2020
Challenge, July 2020.
