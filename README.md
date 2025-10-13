# Towards Extremely Lightweight Distracted Driver Recognition with Distillation-Based Neural Architecture Search and Knowledge Transfer



The folder `Image-based Recognition` contains the source code and description of the image-based DDR, and the folder `Video-based Recognition` contains the source code and description of the video-based DDR.  




### Supplement: Average Accuracy on the Drive&Act Dataset

> **Bold = best in column** (smallest parameter size; highest accuracy per column)

| Model            | Parameter Size (M) | Fine-grained Activities (Val) | Fine-grained Activities (Test) | Atomic Action (Val) | Atomic Action (Test) | Object (Val) | Object (Test) | Location (Val) | Location (Test) | Action-Object-Location (Val) | Action-Object-Location (Test) |
|------------------|--------------------|-------------------------------|---------------------------------|---------------------|----------------------|--------------|---------------|----------------|-----------------|------------------------------|-------------------------------|
| **3D ResNet-50** | 47.02              | 42.40%                        | **43.78%**                      | 41.67%              | **38.82%**           | 49.91%       | **47.64%**    | **37.33%**     | **44.52%**      | **14.62%**                   | **12.73%**                    |
| **Student3D**    | **2.03**           | **44.01%**                    | 43.46%                          | **45.83%**          | 36.66%               | **49.93%**   | 45.51%        | 35.43%         | 37.15%          | 12.53%                       | 11.41%                        |
