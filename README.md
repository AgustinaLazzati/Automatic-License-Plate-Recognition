# Automatic-License-Plate-Recognition
This is the repository that contains an algorithm that automatically detects and recognize license plates from vehicle images.


### Code Sections
- initial data data exploration at `DataExplorations`
- protocal testing and definition at `protocol`
- YOLO test code and models at `YOLO` 
- character segmentation pipelines at `CharacterSegmentation`
- character descriptors and classification at `CharacterDescriptors`


### Plots

- Most plots can be found at `plots`.

- Images used to test the YOLO model can be seen under its own directory, goint to `runs/detect/`

- The HSV plots used in the Fréchet + FFT pipeline can be seen at `plots/hsv_plots`

- Plots related to the characters classification and validation are in the same folder as the code, `CharacterDescriptors`

> it seems that more than 100 plots where done during the development of this project

### Dataset Description

- **`real_plates.zip`** – Contains images of real vehicles, captured from both front and side angles.  
- **`example_fonts.zip`** – A synthetic dataset generated using Spanish license plate fonts.  
  - Files included:
    - `digitsIms.pkl` and `alphabetIms.pkl`: Cropped images of numeric and alphabetic characters.  
    - `digitsLabels` and `alphabetLabels`: Label files corresponding to each character.  
- **Captured Images** – A supplementary image set gathered under a standardized capture protocol.
