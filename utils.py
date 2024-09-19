import re
import os
import json
from collections import Counter
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

def sanitize_filename(filename):
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[^\w\-_\. ]', '_', filename).replace(' ', '_')

def get_safe_path(base_path, model_type, num_runs):
    # Sanitize the model type for use in folder and file names
    safe_model_type = sanitize_filename(model_type.replace(':', '_'))
    
    # Create the folder path
    folder_path = os.path.join(base_path, f"Prock_LLM_{safe_model_type}")
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the file name
    file_name = f'analysis_results_{safe_model_type}_runs_{num_runs}.json'
    
    # Join the folder path and file name
    return os.path.join(folder_path, file_name)

def print_color(text, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def extract_letter_answer(response):
    # First, try to match "Answer: X)" pattern
    match = re.search(r'(?:Answer:\s*)([A-E])\)', response)
    if match:
        return match.group(1)
    
    # If not found, try to match "Answer: X" pattern without parenthesis
    match = re.search(r'(?:Answer:\s*)([A-E])', response)
    if match:
        return match.group(1)
    
    # If still not found, look for any A-E option followed by closing parenthesis
    matches = re.findall(r'([A-E])\)', response)
    if matches:
        return matches[-1]
    
    # If still not found, look for any standalone A-E at the end of a line
    matches = re.findall(r'^([A-E])$', response, re.MULTILINE)
    if matches:
        return matches[-1]
    
    return 'N/A'

def save_to_file(folder_path, file_name, content):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path
    
def sanitize_folder_name(name):
    return re.sub(r'[^\w\-]', '_', name.replace(':', '_'))

def save_to_json(folder_path, file_name, data):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_to_excel(data, file_path):
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False, engine='openpyxl')

def get_majority_vote(votes):
    vote_counts = Counter(votes)
    majority_vote, count = vote_counts.most_common(1)[0]
    is_tie = len([v for v in vote_counts.values() if v == count]) > 1
    return majority_vote, is_tie

def save_progress(output_folder, progress_data):
    progress_file = os.path.join(output_folder, 'progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress(output_folder):
    progress_file = os.path.join(output_folder, 'progress.json')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print_color(f"Error decoding progress file: {str(e)}", Fore.RED)
            print_color("Progress file might be corrupted. Starting fresh.", Fore.YELLOW)
            return None
        except Exception as e:
            print_color(f"Unexpected error reading progress file: {str(e)}", Fore.RED)
            print_color("Starting fresh.", Fore.YELLOW)
            return None
    return None


ONTOLOGY_EVALUATION_INSTRUCTION = """
You are an experienced ontology engineer. The user will send you a number of tasks following the template below. 

#Start Task Template#

Context: <Context>

Ontology Axioms: <Axioms>

Question: Which of the following statements holds for the given class(es): <ClassNames>? 
A) A disjointness axiom is missing for the given class(es).
B) The given class(es) contain(s) a confusion between the logical and linguistic meaning of "and".
C) The given class(es) contain(s) a trivially satisfiable universal (allValuesFrom) restriction.
D) The given class(es) is/are missing a closure axiom (missing allValuesFrom restriction).
E) The given class(es) do(es) not contain any of the given modeling errors, or is/are modelled correctly.

#End Task Template#

Consider the instructions provided next and think step by step for answering each task. Your response should be one of the options [A|B|C|D|E] without any further explanations.

Types of Modeling Errors:

A) Missing Disjointness Axiom
In OWL, classes are overlapping by default and disjointness between classes needs to be explicitly asserted using disjointness axioms. If there is no disjointness axiom between two classes, an individual can be a member of both of them simultaneously. If, on the other hand, there is a disjointness axiom between two classes, no individual can be a member of both of them at the same time. The omission of disjointness axioms is a common modeling error, for example because disjointness is assumed to be by default.

B) Confusion between logical and linguistic "and"
In common linguistic usage, "and" and "or" do not correspond consistently to logical conjunction and disjunction, respectively. Modeling errors can be made when a concept formulated in natural language is modelled too "literally" in an ontology.

C) Trivially satisifiable universal (allValuesFrom) Restrictions
Modeling errors arising from the incorrect assumption that a universal (allValuesFrom) restriction also implies an existential (someValuesFrom) restriction. A universal restriction is satisfied in case either all asserted property values conform to it, or no property values are asserted at all. The second option may get overlooked, leading to a modeling error.
                       
D) Missing Closure Axiom (missing allValuesFrom Restriction)
Another common modeling error is to model a concept using existential (someValuesFrom) restrictions to state that the concept has certain property values, but not stating that these are all the property values allowed via universial (allValuesFrom) restrictions. These modeling errors may often come from disregarding the Open World Assumption.
"""
