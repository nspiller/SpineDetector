
# Spine detection
Detect dendritic spines from a two dimensional image

Developed by Chancellor Gary

Minor modification by Tetsuya Watabe and Ryohei Yasuda


#Installation
The following example can be used for installation. 

Tested in python 3.11.3. 

```bash
git clone https://github.com/ryoheiyasuda/SpineDetector.git
cd spinedetect
python(or python3) -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

# Usage
The following example demonstrates the use of the code for detecting spines from images in the "sample" folder.

```bash
python SpineDetection.py
```

# Creating network
The following example will create a network model from a training set from the "New_Dataset3" folder.

```bash
python Train.py
```
