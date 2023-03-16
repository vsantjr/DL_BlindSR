# Blind Image Super-Resolution via Deep Learning
----
In this repository, there are all relevant information and files created for a controlled experiment designed and carried out for blind image super-resolution (SR) via deep learning (DL) and deep neural networks (DNNs).

This research was developed within the project *Classificação de imagens via redes neurais profundas e grandes bases de dados para aplicações aeroespaciais* (Image classification via Deep neural networks and large databases for aeroSpace applications - [**IDeepS**](https://github.com/vsantjr/IDeepS)) which is supported by the Laboratório Nacional de Computação Científica (LNCC/MCTI, Brazil) via resources of the [SDumont](http://sdumont.lncc.br) supercomputer.


## Datasets
----
Altogether, 14 small low-resolution (LR) image datasets were created from five different broader domains. They are available as a [Kaggle dataset](https://www.kaggle.com/datasets/valdivinosantiago/dl-blindsr-datasets). The high-resolution (HR) images created by five deep DL techniques (see Section *DL Techniques and DNN-based Metric* below) considering all 14 LR datasets can also be accessed in this very same location. 

## DL Techniques and DNN-based Metric
----
The selected DL techniques to make part of this evaluation are (click in the links below to go to the respository of the DL techniques):

- Adaptive Pseudo Augmentation ([APA](https://github.com/EndlessSora/DeceiveD));
- Blind Image SR with Spatially Variant Degradations ([BlindSR](https://github.com/sunreef/BlindSR));
- Deep Alternating Network ([DAN](https://github.com/greatlog/DAN));
- [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch);
- Mixture of Experts Super-Resolution ([MoESR](https://github.com/memad73/MoESR)).

In addition, a DNN-based no-reference image quality assessment (NR-IQA) metric was used in the experiment: Multi-dimension attention network for no-reference image quality assessment ([MANIQA](https://github.com/IIGROUP/MANIQA)) score.

## Supporting Code
----
All the supporting code developed is explained below (instructions to run each code is given in the file):

- ```calculate_NIQE.py```: This code allows the calculation of the NR-IQA metric NIQE. It was adapted based on the NIQE's implementation of [Praful Gupta](https://github.com/guptapraful/niqe);
- ```down_HR_MANIQA.py```: This code performs the downsampling of the HR images produced by the DL techniques so that the MANIQA DNN can be properly used. The bicubic interpolation method was used (default of the resize - PIL's Image module);
- ```calculate_MANIQA.py```: This code handles the values of the MANIQA scores of the HR images obtained by executing the DL techniques for blind image SR;
- ```behaviours_DL.py```: This code helps perceiving whether the two top approaches present similar behaviours when deriving HR images. There is also a correlation analysis related to this question;
- ```calculate_sharpness.ipynb```: This code calculates the sharpness of the HR images based on the maximum overall contrast.



## Additional Files

There are two folders with additional files:

- *json*: Main output files genearated by ```calculate_NIQE.py``` and ```calculate_MANIQA.py```;
- *csv_MANIQA*: These are ```.csv``` files created by the MANIQA DNN when calculating the scores of the HR images.

## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE](LICENSE) file for details.


