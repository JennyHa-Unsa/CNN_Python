## DATASETS

### SimpsomMNIST
- Descripción: Es una versión reducida y divertida del famoso MNIST, pero en lugar de dígitos manuscritos, contiene imágenes de personajes de la serie Los Simpson.
- Training set : 8000
- Test set : 2000
- Original Type: .JPG (Grayscale)
- Size: 28 * 28
- Categories: 10
    - 0: "bart_simpson",
    - 1: "charles_montgomery_burns",
    - 2: "homer_simpson",
    - 3: "krusty_the_clown",
    - 4: "lisa_simpson",
    - 5: "marge_simpson",
    - 6: "milhouse_van_houten",
    - 7: "moe_szyslak",
    - 8: "ned_flanders",
    - 9: "principal_skinner"
- Converted type: .CSV
    - {label, pixel0, pixel1,...,pixel183}

### BreastMNIST
- Description: Es una colección de imágenes médicas diseñada para tareas de clasificación binaria en imágenes biomédicas, especialmente en el ámbito de la imagenología mamaria
- Training set : 546
- Test set : 156
- Type: .CSV (Grayscale)
    - {label, pixel0, pixel1,...,pixel183}
- Size: 28 * 28
- Categories: 2 (Binary)
    - 0: "maligno",
    - 1: "normal/beningno",
    
### HAM10000
- Descripción:  Es una colección de imágenes dermatoscópicas de lesiones cutáneas pigmentadas, diseñada para la investigación y desarrollo de modelos de aprendizaje automático en el diagnóstico del **cáncer de piel**.
- Original Type: .JPG (RGB)
- Original Size: 600 × 450 
- Categories:7
    - 0: akiec Actinic Keratoses 
    - 1: bcc Basal Cell Carcinoma: 
    - 2: bkl Benign Keratosis-like Lesions
    - 3: df Dermatofibroma
    - 4: mel Melanoma
    - 5: nv Melanocytic Nevi
    - 6: vasc Vascular Lesions
    
    - 0: akiec
    - 1: bcc 
    - 2: bkl 
    - 3: df 
    - 4: mel 
    - 5: nv 
    - 6: vasc 

- Converted type: .CSV
    - {label, pixel0, pixel1,...,pixel183}  
- Training set : 546
- Test set : 156
