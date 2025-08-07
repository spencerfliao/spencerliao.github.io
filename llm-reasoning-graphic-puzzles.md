# Enhancing LLM Visual Reasoning
               

## Corpus + explanation

### Overview

[View Presentation Slides](https://github.com/spencerfliao/llm-reasoning-graphic-puzzles/blob/ed60cff9f7180a37906a4e79e1a0af96ca977586/Slide%20Deck.pdf)

[View Presentation Video](https://drive.google.com/file/d/1A67JKig9ujmueDwPy8qMMmdqBzqMdZ4s/view?usp=share_link)



#### Motivation

Large language models (LLMs) struggle with tasks that require abstract, few-shot reasoning—especially in visual domains. The Abstraction and Reasoning Corpus (ARC) provides a benchmark for such tasks, mimicking the kind of generalization humans excel at but machines often fail to replicate.
<img width="976" height="856" alt="Screenshot 2025-07-31 at 18 18 59" src="https://github.com/user-attachments/assets/6b4e98ff-c85f-4f7e-b0db-f17adbb556f9" />

#### Our Approach

We augment the ARC dataset with structured natural language annotations that describe the underlying logic of each task. Each annotation includes reflections, pixel/object transformations, helper functions, and program instructions. To support this, we convert ARC JSON files into images for easier interpretation and annotation.

#### Quality Assurance

To maintain consistency, we conducted an inter-annotator agreement study with a binary correctness evaluation, reaching an agreement score of **85%** across annotators.

#### Interface and Applications

We also built a searchable interface to explore tasks by reasoning patterns or helper functions. The resulting annotated corpus will be used to fine-tune an LLM capable of solving and explaining ARC tasks autonomously.

---

### Sources
1. ARC Challenge Dataset Source: [Kaggle ARC Challenge Dataset](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview) (Original source of ARC challenge problems)
2. Visual representations of the ARC Dataset: [GitHub Repository](https://github.ubc.ca/MDS-CL-2023-24/COLX_523_ARCOT/tree/master/data/images) (Images generated from the ARC dataset to make the task of annotations easier)

### Corpus Format
The original ARC dataset is stored in JSON format, designed for compatibility with various machine-learning tools. It is divided into three folders: training, evaluation and test. Each contains multiple JSON files representing distinct reasoning problems, including inputs and outputs designed to test abstract reasoning abilities. The training and evaluation examples also have images corresponding to each JSON file. Each image is a visual representation of a reasoning problem. The human annotations are still a work in progress, so the final storage format and database structure will be determined based on the scalability needs and the ease of access for both machine learning models and researchers. For now we are storing them in TSV files.

## Corpus collection

### Overview
The code is used to convert JSON files into PNG files, which, along with the human annotations, form a part of the corpus. It processes the training and evaluation data, generates images based on this data, combines these images with specific configurations, and saves the resulting images along with metadata about the data. The format conversion is necessary to provide a visual aid for manually writing the annotations. The code can be found in a Python file named utils.py, located inside the src directory at the root level.

### Libraries
The code utilizes several libraries such as json, os, shutil, numpy, pandas, PIL (Python Imaging Library), and matplotlib for file handling, data manipulation, image processing, and image generation, respectively.

### Functions
1. `load_json(file_path)`: Loads and returns data from a JSON file. It takes a file path as an argument and returns the parsed JSON data.
2. `add_margin(pil_img, top, right, bottom, left, color)`: Adds a margin around a PIL image and returns the new image with the margin. It takes various aspects of the margin as arguments.
3. `delete_directory(path)`: Deletes a directory at the given path. It handles exceptions to avoid crashes if the directory does not exist or other errors occur during deletion.
4. `generate_image(array, file_path, title)`: Generates and saves an image based on a 2D array where each element represents a color coded by integers. It maps these integers to specific colors, creates a plotted image with these colors, and saves it to a file path with a given title.
5. `generate_and_combine_images(data, output_dir, filename, temp_dir='./temp')`: Processes input data to generate images for training and test datasets, adds margins, combines them horizontally and vertically, and saves the final combined image. It cleans up by deleting temporary images and directories used during processing.
6. `get_metadata(input_data)`: Extracts and returns metadata from the input data, such as the number of rows and columns in the input and output data.

### Algorithm
1. Set up directories for output, training, and evaluation images.
2. Iterate over JSON files in the training and evaluation directories.
3. For each JSON file:
   1. Load the data.
   2. Generate and combine images based on the data.
   3. Extract metadata from the data.
4. Save the metadata for training and evaluation datasets as CSV files in the specified output directory.


## Annotation

### Description of Annotations
We provide a high-level overview of our annotation process, focusing on human natural language descriptions for each ARC task. Specifically, we generate six sentence-level annotations for each task: Reflection, Pixel Changes, Object Changes, Helper Functions, Overall Pattern, and Program Instructions. Here's a breakdown of each section:

- Reflection: Offers a concise abstract overview of the task.
- Pixel Changes: Describes pixel-level relationships between input and output pairs, including movements, color alterations, or pattern variations.
- Object Changes: Defines objects as connected pixel sets, often sharing the same color. This section outlines object-level relationships, covering movements, shapes, counts, sizes, positions, or colors.
- Helper Functions: Lists Python helper functions applicable to each task and suggests their potential use.
- Overall Pattern: Summarizes pixel and object changes, providing a simpler description of the input-output relationship.
- Program Instructions: Offers a plan or pseudocode for writing a Python function to solve the ARC task.

Informed by Kumar et al.'s findings on neural models learning human inductive bias through natural language abstraction [1], we prioritize abstract descriptions. For instance, if a pattern resembles an axe, we describe it as such rather than focusing on pixel, row, or column specifics. This approach aims to enhance accuracy in solving ARC tasks, leveraging the handcrafted nature of the dataset. However, this method may not be effective for automatically generated ARC tasks.

### Tools for Annotation
We've developed a script to convert all ARC tasks from JSON to images, which are then equally distributed among annotators. Annotation occurs on a shared Google Sheet, facilitating collaboration. Upon completion, the annotations will be exported to a TSV file.

### Annotators
Project team members are responsible for annotations, with each member handling a quarter of the total workload. We opt not to utilize external resources like Mechanical Turk due to the technical nature of the task, expecting higher quality and alignment with our goals from team members' involvement.

### Expectations
We anticipate annotating approximately 200 ARC tasks in total, with each annotator responsible for 50 annotations. Estimated at 10 minutes per annotation, each annotator will spend approximately 8 hours in total. This estimate is based on a pilot study involving four annotations. Combined with paired data from GPT4 in future work, we believe this dataset will be adequate for bootstrapping a Reinforcement Learning from Human Feedback (RLHF) training on a Large Language Model (LLM).

### Data Quality
Pilot studies have been conducted on a few annotations, guiding our annotation approach. Annotators will cross-check each other's work to maintain high quality. Additionally, annotator training will cover technical details, such as the meaning of helper functions, and ensure consistency in writing style.

### Pilot Study Report
We conducted two phases of pilot studies, annotating 14 samples in total. Initially, we focused solely on reflections, but subsequent discussions highlighted issues like data alignment and color representation. In the second trial, we annotated all specified sections, streamlined the process with Google Sheets and image conversion scripts, and refined our annotation style to better suit our goals.

### Annotating the materials

Annotation guidelines for a task are provided in the form of various sections of instructions, and we will be annotating the sections ourselves.
Some examples of data processed into visually appropriate format and their annotations are shown below: <br>

#### Data
The `data` directory contains the following subdirectories:
1. `references`: Partial dataset comprising of gpt4 and human annotations that is used as a basis to form the corpus.
2. `original`: Original Kaggle dataset comprising of images in the form of .json files, that we use to form our main corpus containing images and the annotations.
3. `images`: Images regenerated from the .json files in the `original` folder, that we use for writing the human annotations.
4. `annotations`: Annotations for the images in the form of .tsv files.

#### Example

The original ARC dataset is stored in JSON format and designed to be compatible with various machine-learning tools. It is divided into three folders: training, evaluation and test. Each contains multiple JSON files representing distinct reasoning problems, including inputs and outputs designed to test abstract reasoning abilities. To make the annotation task more accessible, we converted the JSON files to PNG images. The code for that is present in the `src/utils.py` file. The images, on the other hand, are located in the `data/images` directory. To prevent any data leakage problems, we only converted the training and evaluation  JSON files to images since these are the only files we will be annotating. This aligns with our overall goal: to fine-tune an LLM to automatically generate the annotations for the test set and then use them to solve the ARC reasoning tasks. 
Out of the 800 examples (400 training and 400 evaluation), we took 200 and annotated 50. Each annotation consisted of four components: reflections, pixel/object changes, helper functions, and program instructions.
1. Reflections: This part outlines the underlying logic or pattern of the problem, providing a high-level understanding of the solution. It describes how a human might generally approach solving the problem based on the observations made from the inputs and outputs.
2. Pixel/Object Changes: Here, we detail the specific alterations made to the objects or pixels within the grid. This includes identifying relevant objects or groups of pixels and describing the changes they undergo, such as movements, colour changes, expansions, or contractions, to achieve the desired output.
3. Helper Functions: We list the predefined helper functions used in the solution. These functions perform object detection, colour replacement, grid manipulation, object merging, and so on. These functions serve as building blocks for constructing the step-by-step solution for the reasoning task.
4. Program Instructions: This section provides a sequential guide for executing the solution using the previously mentioned helper functions. It is a detailed algorithm that, when followed, would lead from the input grid to the output grid.

<img width="1503" height="768" alt="Screenshot 2025-07-31 at 18 19 20" src="https://github.com/user-attachments/assets/8bfa716e-15b1-4d53-8f42-9145b11f84ae" />

We decided to store the annotations as TSV files. They are located in the `data/annotations` directory.
These annotations will fine-tune an LLM using Chain of Thought Prompting and Thought Cloning. The goal is that for the test set, it can generate its annotations and use them to solve each reasoning task. 

## Interannotator agreement study

Interannotator agreement data files are located at `/data/interannotator_agreement`.

### Interannotator agreement measure
We calculate our interannotator agreement by manually inspecting each data point and giving a correct/incorrect label to each annotation. For each annotation we assign two other annotators for doing the inspection. The interannotator agreement score is calculated by the percentage of correct labels.

This form of agreement is chosen because the nature of our annotation are natural language sentences. We cannot use n-gram or language model based similarities because the focus of our data is in logical deduction, not statistical similarities or semantic similarities. There are no existing model that can evaluate the logical similarity of two annotations realiably. Therefore we choose to annotate a binary agreement score to each annotation, and use the average as the overall score. Moreover the annotators are already well trained and familiar with the dataset, so we can confidently trust the scores given by the annotators.

### Interannotator score
Using the script at `/src/inter_annotator_agreement.py`, we obtained a score of 0.85 out of 1. Which means the annotators agree 85% of our data is accurate.

### Annotation realiability
85% is fairly reasonable as our data annotation involves mentally demanding logical deductions in writing appropriate psuedocode using predefined functions. Some details in the thought process might be missed, or the usage of predefined functions might introduce bugs.


## Experimenting with annotation options

We experimented different forms of annotation during milestone2. Originally our annotation contain the fields `Filename`, `Reflections`, `Pixel Changes`, `Object Changes`, `Helper Functions`, `Overall Pattern` and `Program Instructions`. However during our initial annotations, as seen at `/data/annotations/archived`,  we discovered some of the fields are redundant or unclear in purpose. For example, `Reflections` and `Overall Pattern` are quite similar. They differ in detailness but the difference is not important in downstream tasks. Therefore we choosed to merge them together. We also merged `Pixel Changes` and `Objects Changes` to `Pixel/Object Changes` because most tasks is best described by either one of them. 

## The Interface

1. **Search by Description of Reflection:**
   - **Note:** Users are able to search the corpus by entering keywords or phrases related to the ARC tasks.
2. **Search by Helper Function:**
   - **Note:** Another feature is accessing detailed annotations for ARC tasks by helper functions. This allows users to find tasks that are of similar patterns and solutions, which aids in understanding the reasoning behind each task's solution and to see examples of how to programmatically approach problem-solving.

### Implementation

   - **User Interface:** A simple text box will enable users to enter their search terms, with a toggle indicating whether the user wants to search by either options. Results that include the original task image and annotations would be displayed in the order of relevance. (Optional filters, e.g., difficulty level, if we deem it necessary and have the time to develop, could refine searches, enhancing discoverability of relevant tasks.)
<img width="1680" height="940" alt="Screenshot 2025-07-31 at 18 19 26" src="https://github.com/user-attachments/assets/3b64e0fe-07d8-4ac7-9ba5-b89176c809ba" />

### Justification of Choices

Our choice to focus on reflection and helper functions for the annotated search functionality comes from the desire to showcase the depth of human understanding and computational strategies required to solve ARC tasks. "Reflections" give a quick, abstract insight into the problem-solving required, appealing to users interested in the cognitive aspects of the dataset. In contrast, helper functions cater to those more interested in the programming challenge, providing a bridge between abstract reasoning and practical coding exercises.


## References
1. Kumar, Sreejan, et al. "Using natural language and program abstractions to instill human inductive biases in machines." Advances in Neural Information Processing Systems 35 (2022): 167-180.
