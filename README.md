# 🛰️ Satellite Image Quality Classifier

This project provides an automated pipeline to evaluate and classify satellite images based on visual quality and scientific relevance. It mimics intelligent onboard systems that filter and prioritize data before transmission to Earth.

The system evaluates each image using custom quality metrics and detection logic, then sorts them into structured output directories.

## 📦 Project Structure

```
.
├── src/
│   ├── classifier.py               # Main classification pipeline
│   ├── image_quality_detector.py   # Image quality and content detection logic
├── DataSet/
│   ├── Input_Image/                # Folder for raw, unclassified images
│   ├── Output_Image/
│   │   ├── Good_Image/             # High-quality and relevant images
│   │   ├── Bad_Image/              # Low-quality or irrelevant images
│   │   ├── Final_Output/           # Final filtered results used for output
```

## 🧠 Key Features

- **Black image detection** (over/underexposed, empty, or corrupted)
- **Blur analysis**
- **Noise detection**
- **Earth vs. space classification**
- **Sunburn artifact recognition**
- **Horizon curve fitting (to ensure Earth visibility)**

Images are automatically annotated and labeled. Classification results determine whether an image is considered usable (Good) or discardable (Bad).

The final decision set is saved in:  
```
DataSet/Output_Image/Final_Output/
```

## ⚙️ Dependencies

This project requires Python 3 and the following libraries:

```bash
pip install opencv-python numpy
```

## 🚀 Getting Started

### 1. Place your input images:

Copy your raw satellite images into:

```bash
DataSet/Input_Image/
```

### 2. Run the classifier:

```bash
python3 src/classifier.py
```

### 3. Optional: Clear previous results

```bash
rm DataSet/Output_Image/Bad_Image/*
rm DataSet/Output_Image/Good_Image/*
rm DataSet/Output_Image/Final_Output/*
```

## 🔧 Customization

You can easily adjust:
- The **quality thresholds** and detection criteria in `image_quality_detector.py`
- The **final classification logic** in `classifier.py`
- **Compression logic** (if integrated), e.g., based on resolution or relevance

This makes the system flexible for different mission profiles, such as planetary observation, urban monitoring, or astronomical surveys.

## 📊 Output

Each processed image is:
- Analyzed for multiple quality issues
- Labeled and annotated
- Stored in the appropriate subfolder based on classification
- If accepted, included in the final export directory (`Final_Output/`)

This structure is ideal for integration with image transmission pipelines or further post-processing.

## 📌 Use Case

This project was developed as part of a space engineering course project, simulating onboard filtering logic for low-bandwidth satellite missions. The system ensures that only the most valuable and clean images are prioritized for transmission to Earth.

## 👨‍💻 Author

**Natane Partouche**  
Final Year Computer Science & Space Engineering Student  
