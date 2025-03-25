# MedPix 2.0
MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset for Advanced AI Applications.

### images
Folder with the images of the data set

### images_overview.csv
CSV file with info of the images in images folder

### splitted_dataset folder
Folder with splits of the data set.
Refer to /splitted_dataset/README.md for further informations. 

### Case_topic.json
Contains a list of JSON, each of these provide the information of a single clinical case.
The structure of each element is reported below:

- `U_id` the UID code idenifies a clinical case
- `TAC` list of names of the .png files containing the CT scans (if present). Images are under the image folder. 
- `MRI` list of names of the .png files containing the MR scans (if present). Images are under the image folder. 
- `Case` dictionary with the information of the clinical case. It contains the following information:
  *   _Title_ the diagnosis
  *   _History_ patient's history
  *   _Exam_
  *   _Findings_
  *   _Differential Diagnosis_
  *   _Case Diagnosis_
  *   _Diagnosis By_
  
- `Topic` Dictionary with the general information about the disease. It contains the following information:
  *  _Title_ the diagnosis
  *  _Disease Discussion_
  *  _ACR Code_
  *  _Category_

### Descriptions.json
Contains a list of JSON, each of these provide the textual information about a single image, stored in the image folder.
The structure of each element is reported below:

- `Type` Can be CT or MR, identifies teh scanning modality of the image.
- `U_id` The UID code of the clinical case the image belongs to.
- `image` name of the image file
- `location` fine-grained information about the body part location of the given image
- `location category` macro-location of the body-part showen in the given image
- `Description` Dictionary with the decriptive information of the image. It contains the following information:
  *   _ACR codes_
  *   _Age_ age of the patient
  *   _Sex_ sex of the patient
  *   _Caption_ refers to the specific caption of the image
  *   _Figure part_
  *   _Modality_ scanning modality of the image
  *   _Plane_


Please cite our work as follows if you use MedPix 2.0 
```
@misc{siragusa2025medpix20comprehensivemultimodal,
      title={MedPix 2.0: A Comprehensive Multimodal Biomedical Data set for Advanced AI Applications with Retrieval Augmented Generation and Knowledge Graphs}, 
      author={Irene Siragusa and Salvatore Contino and Massimo La Ciura and Rosario Alicata and Roberto Pirrone},
      year={2025},
      eprint={2407.02994},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2407.02994}, 
}
```
