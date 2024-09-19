import os
from tqdm import tqdm
from colorama import init
from ai_models import AIModelFactory
from config import Config
from utils import print_color, save_to_file

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

def read_prompt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def process_prompts(input_folder, output_base_folder, models):
    config = Config()
    
    for model_type in models:
        print_color(f"\nProcessing with model: {model_type}", color="cyan", style="bright")
        ai_model = AIModelFactory.create_model(model_type, config)
        model_output_folder = os.path.join(output_base_folder, model_type.replace(':', '_'))
        
        prompt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        
        for prompt_file in tqdm(prompt_files, desc=f"Processing prompts for {model_type}"):
            input_path = os.path.join(input_folder, prompt_file)
            prompt = read_prompt_file(input_path)
            
            try:
                response = ai_model.get_response([{"role": "user", "content": prompt}])
                
                if response:
                    output_file = f"response_{prompt_file}"
                    save_to_file(model_output_folder, output_file, response)
                    print_color(f"  Saved response for {prompt_file}", color="green")
                else:
                    print_color(f"  No response generated for {prompt_file}", color="yellow")
            
            except Exception as e:
                print_color(f"  Error processing {prompt_file}: {str(e)}", color="red")

if __name__ == "__main__":
    input_folder = "qualification_questions_stepbystep"  # Folder containing input prompt text files
    output_base_folder = "qualification_questions_stepbystep_outputs"  # Base folder for outputs
    
    models = [
        'openai:gpt-4o',
        'deepseek:deepseek-coder',
        'deepseek:deepseek-chat',
        'gemini:gemini-1.5-flash',
        'gemini:gemma2-27b-it',
        'ollama:llama3',
        'ollama:gemma',
        'anthropic:claude-3-5-sonnet-20240620',
        'together:Qwen/Qwen2-72B-Instruct',
        'together:mistralai/Mixtral-8x7B-Instruct-v0.1',
        'groq:llama3-70b-8192',
    ]
    
    process_prompts(input_folder, output_base_folder, models)