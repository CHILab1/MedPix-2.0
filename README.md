# MedPix-2.0
MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset for Advanced AI Applications

### Case_topic.json
Contains a list of JSON, each of these provide the information of a single clinical case.
The structure of each element is reported below:

- U_id -> UID code idenifies a clinical case
- TAC -> list of names of the .png files containing the CT scans (if present). Images are under the image folder. 
- MRI -> list of names of the .png files containing the MR scans (if present). Images are under the image folder. 
- Case -> Dictionary with the information of the clinical case. It contains the following information:
  *   Title -> the diagnosis
  *   History -> patient's history
  *   Exam
  *   Findings
  *   Differential Diagnosis
  *   Case Diagnosis
  *   Diagnosis By
  
- Topic -> Dictionary with the general information about the disease. It contains the following information:
  *  Title -> the diagnosis
  *  Disease Discussion
  *  ACR Code
  *  Category

  ### Descriptions.json
Contains a list of JSON, each of these provide the textual information about a single image, stored in the image folder.
The structure of each element is reported below:

- Type -> Can be CT or MR, identifies teh scanning modality of the image.
- U_id -> The UID code of the clinical case the image belongs to.
- image -> name of the image file
- location -> fine-grained information about the body part location of the given image
- location category -> macro-location of the body-part showen in the given image
- Description -> Dictionary with the decriptive information of the image. It contains the following information:
  *   ACR codes
  *   Age -> age of the patient
  *   Sex -> sex of the patient
  *   Caption -> refers to the specific caption of the image
  *   Figura part
  *   Modality -> scanning modality of the image
  *   Plane
