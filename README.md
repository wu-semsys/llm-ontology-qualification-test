# LLM Ontology Qualification Test

This project is designed to process and evaluate prompts using various Language Model (LLM) APIs.

## Project Context

This project is based on a qualification test for ontology modeling developed by Tsaneva and Sabou (2023). The test classifies examinees into four categories based on their scores across questions of varying difficulty:

- Novice
- Beginner
- Intermediate
- Expert

The original research aimed to enhance human-in-the-loop ontology curation through task design. This project adapts the qualification test to evaluate the performance of various Language Models in ontology modeling tasks.

Reference:
```
@article{HEROJournal,
author = {Tsaneva, Stefani and Sabou, Marta},
title = {Enhancing Human-in-the-Loop Ontology Curation Results through Task Design},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626960},
doi = {10.1145/3626960},
journal = {J. Data and Information Quality},
month = {oct},
keywords = {human computation, human-in-the-loop, ontology evaluation}
}
```

## Project Structure

- `llm_qualification.py`: Main script to run the qualification process
- `ai_models.py`: Helper module for AI model interactions
- `utils.py`: Utility functions for file handling and data processing
- `requirements.txt`: List of required Python packages
- `.env.example`: Configuration file for API keys and other settings
- `qualification_questions/` and `qualification_questions_stepbystep/`: Folder containing the qualification questions developed from previous research

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/wu-semsys/llm-ontology-qualification-test.git
   cd llm-ontology-qualification-test
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API keys and configuration:
   - Rename the `.env.example` file to `.env`
   - Edit the `.env` file and add your API keys and other configuration settings

## Configuration

Edit the `.env` file in the project root with the following structure:

```
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GOOGLE_API_KEY=your_google_api_key
OLLAMA_HOST=http://localhost:11434 (for example)
ANTHROPIC_API_KEY=your_anthropic_api_key
TOGETHER_API_KEY=your_together_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder values with your actual API keys and adjust any other settings as needed.

## Usage

To run the LLM qualification process:

1. Ensure your qualification questions are in the folder
   - These questions are based on previous research and should already be present in the repository

2. Run the main script:
   ```
   python llm_qualification.py
   ```

3. Check the results:
   - Output will be saved in the `qualification_questions_(stepbystep_)outputs` folder
   - Each model's responses will be in a separate subfolder

## Customization

- To add or remove models, modify the `models` list in `llm_qualification.py`
- Adjust input and output folder names in `llm_qualification.py` if needed

## Contributing

Contributions to improve the project are welcome.

## License

MIT

## Contact

For more information about the team members involved in this project, please visit:
[Semantic Systems Research Group](https://semantic-systems.org/team/)