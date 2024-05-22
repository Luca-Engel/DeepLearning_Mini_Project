# DeepLearning_Project

## Information for TAs
![Info](image.png)

### Leveraging Audio Transformers to Detect Hatespeech
We will implement an audio transformer for hate speech classification. The dataset will be a text-based Hugging Face dataset that we synthesize to audio files.

As a dataset we are using the following text dataset: [Measuring Hate Speech - Hugging Face](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/viewer?row=99)
These Elements are then synthesized to audio files using the [gTTS](https://gtts.readthedocs.io/en/latest/) synthesizer.

------------------------

## Dataset
[Measuring Hate Speech - Hugging Face](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/viewer?row=99)

## Synthesizer
[gTTS](https://gtts.readthedocs.io/en/latest/)


## Timeline
### Week 8.04. - 14.04
**Task: Project Planning and Dataset Preparation**
- Familiarize yourself with existing literature on hate speech classification and audio processing techniques.
- Begin experimenting with the audio synthesizer
- generate audio dataset from the text data.

### Week 15.04. - 21.04
**Task: Data Preprocessing and Feature Extraction**
- Develop scripts to preprocess the audio data generated from text.
- Extract relevant features from the audio samples, considering aspects like spectrograms, MFCCs, or Mel-spectrograms.
- Explore techniques for handling imbalanced classes in the dataset if applicable.
- Split the dataset into training, validation, and test sets.
- Start with baseline CNN

### Week 22.04. - 28.04
**Task: Baseline Model Development (CNN)**
- Implement a baseline CNN model for hate speech classification using the preprocessed audio features.
- Train the CNN model on the training dataset and validate its performance using the validation set.
- Experiment with different architectures, hyperparameters, and regularization techniques to optimize model performance.
- Document the baseline model's architecture and performance metrics.

### Week 29.04. - 05.05
**Task: Advanced Model Research and Design (Audio Transformer)**
- Dive deeper into the audio transformer architecture and related research papers.
- Understand the intricacies of implementing an audio transformer for hate speech classification.
- Design the architecture of the advanced model based on the insights gained from the research.
- Set up the necessary infrastructure and libraries required for implementing the audio transformer.
- Begin Implementation of the audio transformer architecture

### Week 06.05. - 12.05
**Task: Advanced Model Implementation and Training**
- Finish the implementation of the transformer architecture
- Train the advanced model on the preprocessed audio data.
- Monitor the training process, tune hyperparameters if necessary, and address any issues that arise.
- Evaluate the performance of the advanced model on the validation set and compare it with the baseline CNN model.

### Week 13.05. - 19.05
**Task: Poster Creation**
- Buffer time for the transformer architecture
- Buffer time for the hyperparameter tuning
- Document the advanced model's architecture and performance metrics.
- Summarize the project's objectives, methodologies, datasets, and model architectures in a concise manner.
- Design a visually appealing poster layout using appropriate graphics and text.
- Incorporate key findings, results, and performance metrics from both the baseline CNN and advanced audio transformer models.
- Review and refine the poster content to ensure clarity and coherence.

### Week 20.05. - 26.05
**Task: Presentation Preparation and Finalization**
- Prepare a slide deck for the presentation based on the content of the poster.
- Practice delivering the presentation to ensure smooth delivery and adherence to time constraints.
- Gather feedback from peers or mentors and make necessary revisions to both the poster and presentation.
- Finalize the poster and presentation materials for submission and presentation.
- Rehearse the presentation multiple times to build confidence and familiarity with the content.
